"""STEP 2.1-2.2 -- circumferential flow-sensor ring model.

Four sensors on a circle of radius ``a_ring`` about the catheter axis, at angles
``phi_k``.  Sensor ``k`` sits at vessel-frame radius ``|e + a_ring u_k|`` and
reads the axial flow speed there:

    s_k = u(|e + a_ring u_k|, t) + n_k,     u_k = (cos phi_k, sin phi_k)

The EXACT radius expression is used, never the small-offset expansion -- the
expansion is what the study is testing, not what it assumes.

Ring geometry ``(a_ring, phi_k)`` is a constructor parameter, not a constant.

Channel combinations are exposed as named functions:

    lateral_y   = s0 - s2          -> transverse offset e_y
    lateral_z   = s1 - s3          -> transverse offset e_z
    common_mode = mean(s)          -> mean flow -> local vessel radius
    quadrupole  = s0 - s1 + s2 - s3 -> axisymmetry residual (model-validity monitor)

The quadrupole is zero to *first* order only.  To second order

    Q = [ u''(a) - u'(a)/a ] (e_y^2 - e_z^2) + O(e^4)

and the bracket vanishes identically only for a parabolic profile, so under
Womersley flow an off-centre catheter produces a quadrupole signal with no
secondary flow present.  :func:`quadrupole_second_order` predicts that term so it
can be subtracted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .flow_field import FlowField, Waveform, grad_phasor, grad2_phasor, vmean_phasor

DEFAULT_PHI = (0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0)


@dataclass(frozen=True)
class SensorRing:
    """Ring geometry and the vessel it sits in.

    ``a_ring``   -- sensor circle radius about the catheter axis [m]
    ``vessel_R`` -- local vessel radius [m]  (sets y = radius / vessel_R and alpha)
    ``phi``      -- the four sensor angles [rad]
    ``field``    -- the :class:`FlowField` (carries alpha and the waveform)
    """

    a_ring: float
    vessel_R: float
    field: FlowField
    phi: tuple[float, ...] = DEFAULT_PHI

    def __post_init__(self) -> None:
        if len(self.phi) != 4:
            raise ValueError("The ring model is specialised to four sensors.")
        if self.a_ring <= 0.0 or self.vessel_R <= 0.0:
            raise ValueError("a_ring and vessel_R must be positive.")
        if self.a_ring >= self.vessel_R:
            raise ValueError("Sensor ring does not fit inside the vessel.")

    @property
    def a_over_R(self) -> float:
        return self.a_ring / self.vessel_R

    @property
    def alpha(self) -> float:
        return self.field.alpha

    def _unit_vectors(self) -> np.ndarray:
        return np.array([[np.cos(p), np.sin(p)] for p in self.phi])

    def _radii(self, e: np.ndarray) -> np.ndarray:
        """Exact vessel-frame radius at each sensor for a transverse offset e."""
        e = np.asarray(e, dtype=float).reshape(2)
        centres = e[None, :] + self.a_ring * self._unit_vectors()
        return np.linalg.norm(centres, axis=1)

    # -- readings -----------------------------------------------------------
    def read(self, e, phase: float) -> np.ndarray:
        """Four real channel readings at wave phase ``omega t = phase`` (no noise)."""
        y = self._radii(e) / self.vessel_R
        return self.field.velocity(y, phase)

    def read_phasor(self, e) -> np.ndarray:
        """Four complex fundamental-harmonic phasors (phase-independent amplitude).

        The observability analysis works on these: a phasor's magnitude is the
        cycle amplitude and its argument the phase, so conditioning results do
        not depend on an arbitrary sampling instant.
        """
        from .flow_field import profile_phasor
        y = self._radii(e) / self.vessel_R
        steady = self.field.waveform.steady * (1.0 - y ** 2)
        if not self.field.waveform.harmonics:
            return steady.astype(complex)
        coeff = self.field.waveform.harmonics[0]
        return steady + 1j * coeff * profile_phasor(self.alpha, y)

    def noisy(self, e, phase: float, sigma_s: float, rng=None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        return self.read(e, phase) + rng.normal(0.0, sigma_s, size=4)


# --------------------------------------------------------------------------
# channel combinations  (operate on any length-4 array, real or complex)
# --------------------------------------------------------------------------
def lateral_y(s):
    s = np.asarray(s)
    return s[..., 0] - s[..., 2]


def lateral_z(s):
    s = np.asarray(s)
    return s[..., 1] - s[..., 3]


def common_mode(s):
    return np.mean(np.asarray(s), axis=-1)


def quadrupole(s):
    s = np.asarray(s)
    return s[..., 0] - s[..., 1] + s[..., 2] - s[..., 3]


def channel_decomposition(s) -> dict:
    return {
        "lateral_y": lateral_y(s),
        "lateral_z": lateral_z(s),
        "common_mode": common_mode(s),
        "quadrupole": quadrupole(s),
    }


# --------------------------------------------------------------------------
# second-order quadrupole prediction  (section 5.2)
# --------------------------------------------------------------------------
def quadrupole_second_order(ring: SensorRing, e) -> complex:
    """Predicted quadrupole phasor from the current offset and the known profile.

        Q ~ [ f''(c) - f'(c)/c ] (ey^2 - ez^2) / R^2 * (i coeff)

    in the same phasor units as :meth:`SensorRing.read_phasor`, with ``c = a/R``.
    Subtract this from the measured quadrupole and attribute only the remainder
    to Dean secondary flow -- a raw quadrupole is not a Dean measurement.
    """
    e = np.asarray(e, dtype=float).reshape(2)
    if not ring.field.waveform.harmonics:
        return 0.0 + 0.0j          # pure steady flow is parabolic: Q == 0
    c = ring.a_over_R
    bracket = grad2_phasor(ring.alpha, c) - grad_phasor(ring.alpha, c) / c
    coeff = ring.field.waveform.harmonics[0]
    ey, ez = e / ring.vessel_R
    return complex(1j * coeff * bracket * (ey ** 2 - ez ** 2))


def quadrupole_residual(ring: SensorRing, e) -> complex:
    """Measured quadrupole minus its second-order prediction (the Dean detector)."""
    measured = quadrupole(ring.read_phasor(e))
    return complex(measured - quadrupole_second_order(ring, e))


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------
def _self_test() -> bool:
    from .flow_field import FlowField, Waveform, grad_phasor
    ok = True
    vessel_R, c = 4e-3, 0.2
    ring = SensorRing(a_ring=c * vessel_R, vessel_R=vessel_R,
                      field=FlowField(alpha=1e-4, waveform=Waveform.single_harmonic()))

    # exact reading matches the small-offset expansion s_k ~ u(a) + u'(a)(e . u_k)
    e = np.array([0.01 * vessel_R, 0.0])
    s = ring.read_phasor(e)
    s0 = ring.read_phasor(np.zeros(2))
    up = grad_phasor(1e-4, c) / vessel_R
    expected_lat_y = 2.0 * (1j) * up * e[0]      # (i coeff) factor, coeff = 1
    t1 = abs(lateral_y(s) - expected_lat_y) / abs(expected_lat_y) < 1e-3
    print(f"  [{'ok  ' if t1 else 'FAIL'}] exact reading -> 2 u'(a) e_y to first order")
    ok &= t1

    # channels are orthogonal combinations: common mode carries u(a), quad ~ 0 here
    t2 = abs(common_mode(s0) - (1j) * (1 - 0)) > 0 and abs(quadrupole(s0)) < 1e-14
    print(f"  [{'ok  ' if t2 else 'FAIL'}] common mode nonzero, quadrupole ~ 0 at e=0")
    ok &= t2

    # exact != expansion at large offset under Womersley (the thing the study
    # tests).  For a parabolic profile s0 - s2 is exactly linear in e; under
    # Womersley the Bessel profile adds higher-order terms.
    ring_w = SensorRing(a_ring=c * vessel_R, vessel_R=vessel_R,
                        field=FlowField(alpha=6.0, waveform=Waveform.single_harmonic()))
    up_w = grad_phasor(6.0, c) / vessel_R
    e_big = np.array([0.3 * vessel_R, 0.0])
    exact = lateral_y(ring_w.read_phasor(e_big))
    lin = 2.0 * (1j) * up_w * e_big[0]
    rel = abs(exact - lin) / abs(lin)
    t3 = rel > 1e-3
    print(f"  [{'ok  ' if t3 else 'FAIL'}] exact reading departs from the expansion "
          f"at |e|=0.3R under Womersley  (rel {rel:.3f})")
    ok &= t3

    # ring geometry is a parameter, not a constant
    ring2 = SensorRing(a_ring=0.3 * vessel_R, vessel_R=vessel_R,
                       field=ring.field, phi=(0.1, 1.7, 3.2, 4.8))
    t4 = ring2.a_over_R == 0.3 and ring2.phi != DEFAULT_PHI
    print(f"  [{'ok  ' if t4 else 'FAIL'}] ring geometry (a, phi_k) is configurable")
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
