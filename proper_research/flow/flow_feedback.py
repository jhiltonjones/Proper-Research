"""Simulated flow-ring feedback for the closed loop.

Given the true beam-tip position (from the plant) and the planned centreline,
this reconstructs the tip-position *estimate* a four-element circumferential
flow ring would produce, so the controller can track it *instead of vision*.

Pipeline (one call per control step)
------------------------------------
1. nearest centreline point ``q`` and local vessel radius ``R_local``;
2. local tangent -> transverse projection ``P``;
3. true transverse offset ``e_true = P (p_tip - q)``;
4. build the local Womersley field (``alpha`` from ``R_local``) and a
   :class:`SensorRing`; read four channels at the current cardiac phase and add
   sensor noise ``sigma_s`` (referenced to the cycle-mean flow via
   ``sigma_over_vmean``);
5. invert the two differential channels with the *known* instantaneous gain
   ``g(phase) = 2 u'(a, phase)``; when ``|g|`` drops into the low-gain window
   (D3) hold the previous estimate;
6. return ``p_meas = p*_desired + P^T e_hat`` -- "the tip is at the planned
   point, offset transversely by the estimate".

The quadrupole residual (D5) is logged as a model-validity monitor but is not
fed back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .flow_field import (
    FlowField, Waveform, womersley_number, vmean_phasor,
    OMEGA_BLOOD, RHO_BLOOD, MU_BLOOD,
)
from .sensor_model import SensorRing, quadrupole, quadrupole_second_order
from .riccati_inverse_jacobian import transverse_projection


def _default_waveform() -> Waveform:
    # steady-dominant physiological-style trace, PI ~ 1 (documented assumption)
    return Waveform(steady=2.5, harmonics=(1.0 + 0j,
                                           0.40 * np.exp(-1j * 1.1),
                                           0.20 * np.exp(-1j * 2.2)))


@dataclass
class FlowFeedbackConfig:
    a_ring_m: float = 0.9e-3               # sensor circle radius (inside a ~1 mm catheter)
    sigma_over_vmean: float = 0.02         # ASSUMPTION: sigma_s is uncharacterised
    heart_rate_bpm: float = 72.0
    rho: float = RHO_BLOOD
    mu: float = MU_BLOOD
    waveform: Waveform = field(default_factory=_default_waveform)
    low_gain_fraction: float = 0.08        # hold the estimate when |g| < this * peak |g|
    average_window: int = 1             # box-average the last N raw estimates.
    #                                     Default 1 (off): the flow-fed DARE loop
    #                                     has thin phase margin against the
    #                                     plant/model Jacobian mismatch, and even
    #                                     a 2-step measurement lag tips it into a
    #                                     limit cycle.  A prompt, noisier estimate
    #                                     tracks better than a smooth, lagged one.
    ema_tau_s: float = 0.0                # optional extra exponential smoothing
    deadband_m: float = 3.0e-5           # report 0 below this (do not chase noise)
    seed: int = 0
    log_quadrupole: bool = True

    @property
    def omega(self) -> float:
        return 2.0 * np.pi * self.heart_rate_bpm / 60.0


@dataclass
class FlowSample:
    e_true_m: np.ndarray
    e_hat_m: np.ndarray
    R_local_m: float
    alpha: float
    a_over_R: float
    phase_rad: float
    instantaneous_gain: float
    peak_gain: float
    held: bool
    wall_contact: bool
    quadrupole_residual: float
    channels: np.ndarray

    def as_row(self) -> dict[str, float]:
        return {
            "flow_e_true_y_m": float(self.e_true_m[0]),
            "flow_e_true_z_m": float(self.e_true_m[1]),
            "flow_e_hat_y_m": float(self.e_hat_m[0]),
            "flow_e_hat_z_m": float(self.e_hat_m[1]),
            "flow_e_true_norm_m": float(np.linalg.norm(self.e_true_m)),
            "flow_e_hat_norm_m": float(np.linalg.norm(self.e_hat_m)),
            "flow_estimate_error_m": float(
                np.linalg.norm(self.e_hat_m - self.e_true_m)
            ),
            "flow_R_local_m": self.R_local_m,
            "flow_alpha": self.alpha,
            "flow_a_over_R": self.a_over_R,
            "flow_phase_rad": self.phase_rad,
            "flow_instantaneous_gain": self.instantaneous_gain,
            "flow_low_gain_held": int(self.held),
            "flow_wall_contact": int(self.wall_contact),
            "flow_quadrupole_residual": self.quadrupole_residual,
        }


class FlowTipEstimator:
    """Turns the true tip position into the flow ring's tip-position estimate."""

    def __init__(self, lumen_C: Any, lumen_R: Any,
                 config: FlowFeedbackConfig | None = None) -> None:
        self.C = np.asarray(lumen_C, dtype=float)[:, :3].copy()
        self.R = np.asarray(lumen_R, dtype=float).reshape(-1).copy()
        if self.C.shape[0] != self.R.size or self.C.shape[0] < 2:
            raise ValueError("lumen_C and lumen_R must align and have >= 2 points.")
        self.cfg = config or FlowFeedbackConfig()
        self._kdt = cKDTree(self.C)
        from collections import deque
        self._rng = np.random.default_rng(self.cfg.seed)
        self._raw_e_hat = np.zeros(2, dtype=float)     # last un-held raw estimate
        self._window: deque = deque(maxlen=max(1, int(self.cfg.average_window)))
        self._ema = np.zeros(2, dtype=float)           # smoothed estimate
        self._last_t: float | None = None
        self._phase_grid = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)

    # -- geometry ------------------------------------------------------
    def _local_frame(self, tip: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        _, i = self._kdt.query(np.asarray(tip, dtype=float).reshape(3))
        i = int(i)
        q = self.C[i]
        lo = max(i - 1, 0)
        hi = min(i + 1, self.C.shape[0] - 1)
        tangent = self.C[hi] - self.C[lo]
        return q, float(self.R[i]), tangent

    # -- one measurement --------------------------------------------
    def estimate(
        self, true_tip: Any, desired_point: Any, time_s: float
    ) -> tuple[np.ndarray, FlowSample]:
        true_tip = np.asarray(true_tip, dtype=float).reshape(3)
        desired_point = np.asarray(desired_point, dtype=float).reshape(3)
        q, R_local, tangent = self._local_frame(true_tip)
        P = transverse_projection(tangent)                     # 2 x 3
        e_true = P @ (true_tip - q)                            # 2, metres

        a_ring = float(self.cfg.a_ring_m)
        # keep the ring physically inside the vessel for the simulation
        max_off = 0.9 * (R_local - a_ring)
        wall_contact = bool(np.linalg.norm(e_true) >= max_off)
        e_sim = e_true.copy()
        if wall_contact and np.linalg.norm(e_true) > 0:
            e_sim = e_true * (max_off / np.linalg.norm(e_true))

        alpha = womersley_number(R_local, self.cfg.omega, self.cfg.rho, self.cfg.mu)
        a_over_R = a_ring / R_local
        field = FlowField(alpha=alpha, waveform=self.cfg.waveform)
        ring = SensorRing(a_ring=a_ring, vessel_R=R_local, field=field)

        phase = float((self.cfg.omega * float(time_s)) % (2.0 * np.pi))

        # noise scale: sigma_s referenced to the cycle-mean |mean flow|
        vmean_cycle = float(np.mean(np.abs(
            [field.mean_flow(p) for p in self._phase_grid]
        )))
        sigma_s = self.cfg.sigma_over_vmean * max(vmean_cycle, 1e-12)

        s = ring.read(e_sim, phase) + self._rng.normal(0.0, sigma_s, size=4)
        lat_y = s[0] - s[2]
        lat_z = s[1] - s[3]

        # known instantaneous differential gain g(phase) = 2 u'(a, phase)
        g = 2.0 * float(field.radial_gradient(a_over_R, phase)) / R_local
        gains = np.abs(
            2.0 * np.array([field.radial_gradient(a_over_R, p)
                            for p in self._phase_grid]) / R_local
        )
        peak_gain = float(gains.max())
        held = abs(g) < self.cfg.low_gain_fraction * peak_gain

        bound = 0.45 * R_local        # the ring cannot resolve past roughly here
        if held or abs(g) < 1e-12:
            raw = self._raw_e_hat.copy()
        else:
            # linear guess, then a few Gauss-Newton steps against the exact
            # differential readout (the estimator owns the sensor model); each
            # step is accepted only if it reduces the channel residual, so a
            # noisy Jacobian cannot make the estimate diverge
            linear = np.clip(np.array([lat_y / g, lat_z / g]), -bound, bound)
            raw = linear.copy()
            s0 = ring.read(np.zeros(2), phase)
            target = np.array([lat_y, lat_z])

            def residual(e: np.ndarray) -> float:
                b = ring.read(e, phase) - s0
                return float(np.linalg.norm(
                    target - np.array([b[0] - b[2], b[1] - b[3]])))

            best_res = residual(raw)
            for _ in range(3):
                b = ring.read(raw, phase) - s0
                pred = np.array([b[0] - b[2], b[1] - b[3]])
                Jm = np.empty((2, 2))
                for ax in range(2):
                    st = np.zeros(2)
                    st[ax] = 1e-6
                    bp = ring.read(raw + st, phase) - s0
                    bm = ring.read(raw - st, phase) - s0
                    Jm[:, ax] = (np.array([bp[0] - bp[2], bp[1] - bp[3]])
                                 - np.array([bm[0] - bm[2], bm[1] - bm[3]])) / 2e-6
                if abs(np.linalg.det(Jm)) < 1e-12:
                    break
                cand = np.clip(raw + np.linalg.solve(Jm, target - pred),
                               -bound, bound)
                r = residual(cand)
                if r < best_res:
                    raw, best_res = cand, r
                else:
                    break
            self._raw_e_hat = raw.copy()

        # box-average the last N raw estimates (finite lag, ~sqrt(N) noise cut)
        self._window.append(raw.copy())
        raw = np.mean(np.stack(self._window, axis=0), axis=0)

        # optional extra exponential smoothing of the (already averaged) estimate
        if self._last_t is None:
            alpha_ema = 1.0
        else:
            step_dt = max(float(time_s) - self._last_t, 1e-6)
            alpha_ema = 1.0 - np.exp(-step_dt / max(self.cfg.ema_tau_s, 1e-6))
        self._last_t = float(time_s)
        self._ema = (1.0 - alpha_ema) * self._ema + alpha_ema * raw

        # hard deadband: do not chase sub-noise wobble, but do not bias a real
        # offset once it clears the threshold
        e_hat = self._ema.copy()
        if float(np.linalg.norm(e_hat)) <= self.cfg.deadband_m:
            e_hat = np.zeros(2)

        quad_resid = np.nan
        if self.cfg.log_quadrupole:
            q_meas = quadrupole(ring.read_phasor(e_sim))
            q_pred = quadrupole_second_order(ring, np.array([e_hat[0], e_hat[1]]))
            quad_resid = float(abs(q_meas - q_pred))

        p_meas = desired_point + P.T @ e_hat
        sample = FlowSample(
            e_true_m=e_true, e_hat_m=e_hat, R_local_m=R_local, alpha=alpha,
            a_over_R=a_over_R, phase_rad=phase, instantaneous_gain=g,
            peak_gain=peak_gain, held=held, wall_contact=wall_contact,
            quadrupole_residual=quad_resid, channels=np.asarray(s, dtype=float),
        )
        return p_meas, sample

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.cfg.seed)
        self._raw_e_hat = np.zeros(2, dtype=float)
        self._window.clear()
        self._ema = np.zeros(2, dtype=float)
        self._last_t = None


# --------------------------------------------------------------------------
# self-test -- straight lumen, known offset, no project sim imports
# --------------------------------------------------------------------------
def _self_test() -> bool:
    ok = True
    # straight lumen along +x, radius 3 mm
    s = np.linspace(0.0, 0.05, 120)
    C = np.column_stack([s, np.zeros_like(s), np.zeros_like(s)])
    R = np.full_like(s, 3.0e-3)
    est = FlowTipEstimator(C, R, FlowFeedbackConfig(sigma_over_vmean=0.0, seed=1))

    # tip at x=0.025, transversely offset by (0.4 mm, -0.2 mm)
    offset = np.array([0.0, 0.4e-3, -0.2e-3])
    true_tip = np.array([0.025, 0.0, 0.0]) + offset
    desired = np.array([0.025, 0.0, 0.0])

    errs = []
    for k in range(200):
        p_meas, samp = est.estimate(true_tip, desired, time_s=k * 0.01)
        if not samp.held and k > 40:          # skip EMA warm-up
            errs.append(np.linalg.norm(samp.e_hat_m - samp.e_true_m))
    t1 = np.median(errs) < 1e-6
    print(f"  [{'ok  ' if t1 else 'FAIL'}] noiseless estimate recovers the offset  "
          f"median err {1e6*np.median(errs):.4f} um over {len(errs)} settled samples")
    ok &= t1

    # p_meas reconstruction: desired + P^T e_hat  ~  true_tip
    p_meas, samp = est.estimate(true_tip, desired, time_s=0.13)
    t2 = np.linalg.norm(p_meas - true_tip) < 1e-5 or samp.held
    print(f"  [{'ok  ' if t2 else 'FAIL'}] p_meas ~ true tip  "
          f"({1e6*np.linalg.norm(p_meas - true_tip):.2f} um)")
    ok &= t2

    # noise raises the estimate error but stays bounded (compare in the P frame)
    est2 = FlowTipEstimator(C, R, FlowFeedbackConfig(sigma_over_vmean=0.02, seed=2))
    errs2 = []
    for k in range(400):
        _, sm = est2.estimate(true_tip, desired, k * 0.01)
        errs2.append(np.linalg.norm(sm.e_hat_m - sm.e_true_m))
    t3 = np.median(errs2) < 1.0e-4
    print(f"  [{'ok  ' if t3 else 'FAIL'}] with 2% noise: median estimate err "
          f"{1e6*np.median(errs2):.2f} um (offset {1e6*np.linalg.norm(offset):.0f} um)")
    ok &= t3

    # a purely oscillatory waveform DOES have a low-gain window (D3)
    est3 = FlowTipEstimator(C, R, FlowFeedbackConfig(
        sigma_over_vmean=0.0, seed=3,
        waveform=Waveform(steady=0.0, harmonics=(1.0 + 0j,))))
    held_any = any(est3.estimate(true_tip, desired, k * 0.005)[1].held
                   for k in range(400))
    print(f"  [{'ok  ' if held_any else 'FAIL'}] pure-oscillatory waveform hits "
          f"the low-gain window (estimate held)")
    ok &= held_any

    print("\n  PASS" if ok else "\n  FAILURES ABOVE")
    return ok


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    ok = _self_test()
    if args.self_test and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = ["FlowFeedbackConfig", "FlowSample", "FlowTipEstimator"]
