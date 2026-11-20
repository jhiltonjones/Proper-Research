"""STEP 2.3 -- flow measurement Jacobian M = d(measurement) / d(offset).

Built by central differences, using the *same* differencing convention as the
existing 3x7 beam Jacobian (``SensitivityOptions.difference_scheme = "central"``
in ``simulation/magnetic_beam/sensitivity_optimized.py``).

Provenance is tracked exactly as ``controllers/beam_jacobian_providers.py`` does:
every provider carries a ``name`` and a ``describe()`` so a run records *which*
measurement map it differentiated.  A comparison whose provenance is not recorded
is not a comparison.

Self-tests (``python -m proper_research.flow.flow_jacobian --self-test``):

  * differential M agrees with the analytic ``8 a/R`` to < 1e-6 in the
    Poiseuille limit;
  * quadrupole row is < 1e-14 for a PARABOLIC profile at any offset;
  * measured quadrupole matches ``quadrupole_second_order()`` to within 2% under
    Womersley flow.

  (Do NOT test the quadrupole against zero under Womersley -- it is not zero.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .sensor_model import (
    SensorRing,
    channel_decomposition,
    common_mode,
    quadrupole,
    quadrupole_second_order,
)

Array = np.ndarray

# Matches SensitivityOptions.eps_theta / central differencing in the beam code.
DEFAULT_EPS = 1e-6
DIFFERENCE_SCHEME = "central"


def central_difference_jacobian(
    measurement: Callable[[Array], Array],
    e: Array,
    *,
    eps: float = DEFAULT_EPS,
) -> Array:
    """d(measurement)/d(e) by central differences; e is the 2-vector offset."""
    e = np.asarray(e, dtype=float).reshape(2)
    m0 = np.atleast_1d(measurement(e))
    columns = []
    for axis in range(2):
        step = np.zeros(2)
        step[axis] = eps
        plus = np.atleast_1d(measurement(e + step))
        minus = np.atleast_1d(measurement(e - step))
        columns.append((plus - minus) / (2.0 * eps))
    return np.stack(columns, axis=1).astype(m0.dtype)


# --------------------------------------------------------------------------
# provider with provenance (mirrors BeamJacobianProvider)
# --------------------------------------------------------------------------
@dataclass
class FlowMeasurementProvider:
    """A named measurement map ``m(e)`` and its Jacobian, with provenance attached.

    ``name`` and ``describe()`` exist so a run can record which pipeline it
    differentiated -- the direct analogue of ``BeamJacobianProvider``.
    """

    name: str
    measurement_fn: Callable[[Array], Array]
    ring: SensorRing
    source: str
    channel_names: tuple[str, ...]
    eps: float = DEFAULT_EPS

    def measurement_at(self, e: Array) -> Array:
        return np.atleast_1d(self.measurement_fn(np.asarray(e, dtype=float).reshape(2)))

    def jacobian_at(self, e: Array) -> Array:
        """M = d(measurement)/d(offset), shape (k, 2)."""
        return central_difference_jacobian(self.measurement_fn, e, eps=self.eps)

    def __call__(self, e: Array) -> Array:
        return self.jacobian_at(e)

    # -- normalised sensitivity / conditioning ----------------------------
    def _v_mean_amplitude(self) -> float:
        from .flow_field import vmean_phasor
        wf = self.ring.field.waveform
        amp = abs(0.5 * wf.steady)
        if wf.harmonics:
            amp += abs(wf.harmonics[0]) * abs(vmean_phasor(self.ring.alpha))
        return float(amp) if amp > 0 else 1.0

    def normalised_jacobian(self, e: Array) -> Array:
        """M in units of ``v_mean / R`` -- directly comparable with 4|e|/R, 8a/R."""
        return self.jacobian_at(e) * self.ring.vessel_R / self._v_mean_amplitude()

    def conditioning(self, e: Array) -> dict[str, Any]:
        m = self.normalised_jacobian(e)
        sv = np.linalg.svd(np.atleast_2d(m), compute_uv=False)
        sv = np.asarray(sv, dtype=float)
        tol = max(m.shape) * np.finfo(float).eps * (sv[0] if sv.size else 0.0)
        rank = int(np.sum(sv > max(tol, 1e-12)))
        sigma_min = float(sv[sv > 1e-12].min()) if rank else 0.0
        cond = float(sv[0] / sigma_min) if sigma_min > 0 else np.inf
        return {
            "sigma_max": float(sv[0]) if sv.size else 0.0,
            "sigma_min": sigma_min,
            "rank": rank,
            "condition_number": cond,
            "singular_values": sv.tolist(),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "channels": list(self.channel_names),
            "n_channels": len(self.channel_names),
            "a_over_R": self.ring.a_over_R,
            "alpha": self.ring.alpha,
            "vessel_R_m": self.ring.vessel_R,
            "difference_scheme": DIFFERENCE_SCHEME,
            "eps": self.eps,
            "ring_id": hex(id(self.ring)),
        }

    def summary_line(self) -> str:
        info = self.describe()
        return (f"{info['name']}: {info['source']} | channels={info['channels']} "
                f"alpha={info['alpha']:.2f} a/R={info['a_over_R']:.3f} "
                f"scheme={info['difference_scheme']}")


# --------------------------------------------------------------------------
# composition into the existing stack  (section 3.2)
# --------------------------------------------------------------------------
def compose_with_beam_jacobian(M: Array, P: Array, J: Array) -> Array:
    """H = M . P . J.

    ``M`` is (k, 2), ``P`` projects tip position onto the local transverse plane
    of the centreline (2, 3), ``J`` is the existing 3x7 beam-position Jacobian,
    untouched.  ``rank(H) <= min(rank M, rank J)`` -- if ``rank M`` is deficient
    the composition is pointless and this raises.
    """
    M = np.atleast_2d(M)
    if np.linalg.matrix_rank(M) < 2:
        raise ValueError(
            "rank(M) < 2: the flow map cannot observe both transverse "
            "directions, so H = M.P.J is rank-deficient regardless of the beam. "
            "Do not form H."
        )
    return M @ np.asarray(P, dtype=float) @ np.asarray(J, dtype=float)


def transverse_projection(tangent: Array) -> Array:
    """Build P (2x3): rows span the plane perpendicular to the centreline tangent.

    The theory doc notes no local frame is stored after L1-L3, so P is built from
    the tangent here.
    """
    t = np.asarray(tangent, dtype=float).reshape(3)
    t = t / np.linalg.norm(t)
    seed = np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = seed - np.dot(seed, t) * t
    u /= np.linalg.norm(u)
    v = np.cross(t, u)
    return np.stack([u, v], axis=0)


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------
def _self_test() -> bool:
    from .flow_field import FlowField, Waveform, Gamma
    ok = True

    # 1. differential M vs analytic 8 a/R in the Poiseuille limit
    a_over_R = 0.2
    vessel_R = 4e-3
    a_ring = a_over_R * vessel_R
    ring = SensorRing(a_ring=a_ring, vessel_R=vessel_R,
                      field=FlowField(alpha=1e-4, waveform=Waveform.single_harmonic()))
    prov = FlowMeasurementProvider(
        name="differential", ring=ring, source="self-test",
        channel_names=("lateral_y", "lateral_z", "common_mode", "quadrupole"),
        measurement_fn=lambda e: np.array(list(channel_decomposition(ring.read_phasor(e)).values())),
    )
    m = prov.normalised_jacobian(np.array([0.03 * vessel_R, 0.0]))
    # rows 0,1 are lateral_y wrt (ey,ez) and lateral_z wrt (ey,ez)
    gain_y = abs(m[0, 0])
    gain_z = abs(m[1, 1])
    err = max(abs(gain_y - 8 * a_over_R), abs(gain_z - 8 * a_over_R))
    t1 = err < 1e-6
    print(f"  [{'ok  ' if t1 else 'FAIL'}] differential M == 8a/R (Poiseuille)  "
          f"gain_y={gain_y:.6f} gain_z={gain_z:.6f} vs {8*a_over_R:.6f}  err {err:.2e}")
    ok &= t1

    # 2. quadrupole row < 1e-14 for a parabolic (pure steady) profile, any offset
    ring_p = SensorRing(a_ring=a_ring, vessel_R=vessel_R,
                        field=FlowField(alpha=3.0, waveform=Waveform(steady=1.0, harmonics=())))
    worst_q = 0.0
    worst_dq = 0.0
    for e in ([0.0, 0.0], [0.1 * vessel_R, 0.0], [0.13 * vessel_R, -0.07 * vessel_R]):
        worst_q = max(worst_q, abs(quadrupole(ring_p.read_phasor(np.array(e)))))
        jac = central_difference_jacobian(
            lambda ee: np.atleast_1d(quadrupole(ring_p.read_phasor(ee))),
            np.array(e), eps=1e-4)
        worst_dq = max(worst_dq, float(np.max(np.abs(jac))))
    t2 = worst_q < 1e-14 and worst_dq < 1e-8
    print(f"  [{'ok  ' if t2 else 'FAIL'}] quadrupole == 0 for parabolic profile  "
          f"worst |Q| {worst_q:.2e}, worst |dQ/de| (FD) {worst_dq:.2e}")
    ok &= t2

    # 3. measured quadrupole matches quadrupole_second_order() within 2% (Womersley)
    ring_w = SensorRing(a_ring=a_ring, vessel_R=vessel_R,
                        field=FlowField(alpha=5.0, waveform=Waveform.single_harmonic()))
    worst_rel = 0.0
    for eR in (0.02, 0.05):
        e = np.array([eR * vessel_R, 0.0])
        measured = quadrupole(ring_w.read_phasor(e))
        predicted = quadrupole_second_order(ring_w, e)
        worst_rel = max(worst_rel, abs(measured - predicted) / abs(measured))
    t3 = worst_rel < 0.02
    print(f"  [{'ok  ' if t3 else 'FAIL'}] measured Q vs 2nd-order prediction (Womersley)  "
          f"worst rel {worst_rel:.4f}")
    ok &= t3

    # 4. central difference matches the analytic phasor gradient of common_mode
    prov_s = FlowMeasurementProvider(
        name="scalar", ring=ring_w, source="self-test", channel_names=("common_mode",),
        measurement_fn=lambda e: np.atleast_1d(common_mode(ring_w.read_phasor(e))))
    e = np.array([0.05 * vessel_R, 0.0])
    m_s = prov_s.jacobian_at(e)
    # scalar map gradient is parallel to e (radial only) -> second column ~ 0
    t4 = abs(m_s[0, 1]) < 1e-3 * abs(m_s[0, 0])
    print(f"  [{'ok  ' if t4 else 'FAIL'}] scalar map gradient is radial only  "
          f"|dm/dez|/|dm/dey| = {abs(m_s[0,1])/abs(m_s[0,0]):.2e}")
    ok &= t4

    print("\n  PASS" if ok else "\n  FAILURES ABOVE")
    return ok


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    ok = _self_test()
    if args.self_test and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
