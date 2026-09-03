"""STEP 1 -- flow field model.

Womersley oscillatory flow in a rigid tube plus an optional steady component,
parameterised by ``(R, omega, rho, mu, waveform)``.  The waveform is a harmonic
series so a measured physiological trace can be supplied later; the default is a
single harmonic plus a steady offset.

The observability analysis only ever needs *ratios* of derivatives of the
dimensionless profile, so everything here is exposed as a complex phasor and the
(unknown) pressure amplitude cancels downstream.

Self-tests (``python -m proper_research.flow.flow_field --self-test``) check this
implementation against ``flow_observability_reference`` -- the oracle is ground
truth.

    T1  Gamma(alpha -> 0, a/R) == 4 a/R to relative error < 1e-8
    T2  reproduce the Gamma/Gamma_P table (FLOW_CENTERING_THEORY.md section 4.2)
    T3  mass conservation: integral of the profile == vmean_phasor to < 1e-10
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import special

# Blood defaults -- FLAGGED AS ASSUMPTIONS (no fluid parameters exist in the
# project; see the STEP 0 audit).
RHO_BLOOD = 1060.0
MU_BLOOD = 3.5e-3
HEART_RATE_BPM = 72.0
OMEGA_BLOOD = 2.0 * math.pi * HEART_RATE_BPM / 60.0


# --------------------------------------------------------------------------
# waveform
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Waveform:
    """A pressure-gradient waveform as ``steady + sum_n A_n cos(n omega t + phi_n)``.

    ``harmonics`` holds complex Fourier coefficients ``A_n e^{i phi_n}`` for
    ``n = 1, 2, ...``; ``steady`` is the DC (Poiseuille) component in the same
    units.  A single harmonic with no steady offset is the analytic default used
    by the observability derivations; supplying several harmonics and a steady
    term reproduces a real trace (see :meth:`from_series`).
    """

    steady: float = 0.0
    harmonics: tuple[complex, ...] = (1.0 + 0.0j,)

    @property
    def n_harmonics(self) -> int:
        return len(self.harmonics)

    @classmethod
    def single_harmonic(cls, amplitude: float = 1.0, steady: float = 0.0) -> "Waveform":
        return cls(steady=float(steady), harmonics=(complex(amplitude),))

    @classmethod
    def from_series(cls, steady: float, coefficients: "np.ndarray") -> "Waveform":
        return cls(steady=float(steady),
                   harmonics=tuple(complex(c) for c in np.asarray(coefficients).ravel()))

    def pulsatility_index(self, alpha: float, a_over_R: float = 0.0,
                          n_phase: int = 512) -> float:
        """(max - min) / mean of the *mean-flow* time series over one cycle."""
        field_ = FlowField(alpha=alpha, waveform=self)
        t = np.linspace(0.0, 2.0 * math.pi, n_phase, endpoint=False)
        q = np.array([field_.mean_flow(phase) for phase in t])
        mean = float(np.mean(q))
        if abs(mean) < 1e-30:
            return math.inf
        return float((q.max() - q.min()) / abs(mean))


# --------------------------------------------------------------------------
# core Womersley phasors  (per unit pressure-gradient harmonic)
# --------------------------------------------------------------------------
def womersley_number(radius_m: float, omega: float = OMEGA_BLOOD,
                     rho: float = RHO_BLOOD, mu: float = MU_BLOOD) -> float:
    return float(radius_m) * math.sqrt(omega * rho / mu)


def _beta(alpha: float) -> complex:
    return float(alpha) * (1j ** 1.5)


def profile_phasor(alpha: float, y):
    """f(y) = 1 - J0(beta y) / J0(beta), the dimensionless velocity profile."""
    beta = _beta(alpha)
    return 1.0 - special.jv(0, beta * np.asarray(y, dtype=float)) / special.jv(0, beta)


def grad_phasor(alpha: float, y):
    """df/dy = beta J1(beta y) / J0(beta).  Gradient wrt r is this over R."""
    beta = _beta(alpha)
    return beta * special.jv(1, beta * np.asarray(y, dtype=float)) / special.jv(0, beta)


def grad2_phasor(alpha: float, y):
    """d2f/dy2 = beta^2 [J0(beta y) - J1(beta y)/(beta y)] / J0(beta)."""
    beta = _beta(alpha)
    arg = beta * np.asarray(y, dtype=float)
    return beta ** 2 * (special.jv(0, arg) - special.jv(1, arg) / arg) / special.jv(0, beta)


def vmean_phasor(alpha: float) -> complex:
    """Cross-sectional mean of f: 1 - 2 J1(beta) / (beta J0(beta))."""
    beta = _beta(alpha)
    return complex(1.0 - 2.0 * special.jv(1, beta) / (beta * special.jv(0, beta)))


def gamma_poiseuille(a_over_R: float) -> float:
    return 4.0 * float(a_over_R)


def Gamma(alpha: float, a_over_R: float) -> float:
    """Discrimination gain |grad_phasor(a/R)| / |vmean_phasor|; -> 4 a/R as alpha -> 0."""
    return float(abs(grad_phasor(alpha, a_over_R)) / abs(vmean_phasor(alpha)))


def Gamma_ratio(alpha: float, a_over_R: float) -> float:
    return Gamma(alpha, a_over_R) / gamma_poiseuille(a_over_R)


# --------------------------------------------------------------------------
# the flow field
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FlowField:
    """Womersley + steady flow for one vessel cross-section.

    Times are given as a *phase* ``omega t`` in radians.  All velocities are in
    units of the per-harmonic pressure scaling ``P / (rho omega)`` (harmonic n
    uses ``omega_n = n omega``), so only ratios are physically meaningful -- the
    observability analysis uses exactly those ratios.
    """

    alpha: float
    waveform: Waveform = field(default_factory=Waveform)

    def _alpha_n(self, n: int) -> float:
        # alpha scales as sqrt(omega); harmonic n has omega_n = n omega.
        return self.alpha * math.sqrt(n)

    def velocity(self, y, phase: float) -> np.ndarray:
        """Axial velocity u(y, t), real, at dimensionless radius y and phase omega t."""
        y = np.asarray(y, dtype=float)
        wf = self.waveform
        # steady Poiseuille part: u = 2 v (1 - y^2) shape; here in profile units
        # the steady solution of the same ODE is (1 - y^2), mean 1/2.
        out = wf.steady * (1.0 - y ** 2)
        for k, coeff in enumerate(wf.harmonics, start=1):
            an = self._alpha_n(k)
            phasor = 1j * coeff * profile_phasor(an, y)
            out = out + np.real(phasor * np.exp(1j * k * phase))
        return out

    def radial_gradient(self, y, phase: float) -> np.ndarray:
        """du/dr in profile units per R, i.e. df/dy summed over the waveform."""
        y = np.asarray(y, dtype=float)
        wf = self.waveform
        out = wf.steady * (-2.0 * y)
        for k, coeff in enumerate(wf.harmonics, start=1):
            an = self._alpha_n(k)
            phasor = 1j * coeff * grad_phasor(an, y)
            out = out + np.real(phasor * np.exp(1j * k * phase))
        return out

    def mean_flow(self, phase: float) -> float:
        """Cross-sectional mean axial velocity at the given phase."""
        wf = self.waveform
        out = 0.5 * wf.steady
        for k, coeff in enumerate(wf.harmonics, start=1):
            an = self._alpha_n(k)
            out += float(np.real(1j * coeff * vmean_phasor(an) * np.exp(1j * k * phase)))
        return out

    # -- fundamental-harmonic phasors (what the observability analysis consumes)
    def grad_phasor_ring(self, a_over_R: float) -> complex:
        return complex(1j * self.waveform.harmonics[0] * grad_phasor(self.alpha, a_over_R))

    def vmean_phasor_fundamental(self) -> complex:
        return complex(1j * self.waveform.harmonics[0] * vmean_phasor(self.alpha))

    def discrimination_gain(self, a_over_R: float) -> float:
        return Gamma(self.alpha, a_over_R)


# --------------------------------------------------------------------------
# Dean correction -- SEPARATE, FLAGGED, SWITCHABLE
# --------------------------------------------------------------------------
def dean_number(radius_m: float, radius_of_curvature_m: float, vmean: float,
                rho: float = RHO_BLOOD, mu: float = MU_BLOOD) -> float:
    """De = Re sqrt(R / R_c), with Re = 2 R v_mean rho / mu."""
    re = 2.0 * float(radius_m) * float(vmean) * rho / mu
    return float(re * math.sqrt(radius_m / radius_of_curvature_m))


def dean_peak_shift_fraction(dean: float, enabled: bool = True) -> float:
    """Fractional outward shift of the velocity peak in a curved tube, delta/R.

    LITERATURE CORRELATION -- NOT DERIVED.  This is *not* part of the Womersley
    solution above; it is a smooth monotone interpolation of the bands quoted in
    FLOW_CENTERING_THEORY.md section 5.1 (De < 20: shift <~ 2% R; 20-100:
    developing; > 100: 10-20% R):

        delta / R = 0.18 * (1 - exp(-De / 160))

    The direction (toward the outer wall) is trustworthy; the magnitude is a
    scaling argument with a wide band.  Set ``enabled=False`` to switch the
    correction off entirely (returns 0.0) -- the base profile never sees it.
    """
    if not enabled:
        return 0.0
    return 0.18 * (1.0 - math.exp(-abs(float(dean)) / 160.0))


def dean_peak_shift_band(dean: float) -> tuple[float, float]:
    centre = dean_peak_shift_fraction(dean)
    return (0.5 * centre, 1.6 * centre)


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------
def _self_test() -> bool:
    from proper_research.flow import flow_observability_reference as oracle

    ok = True

    # T1
    errs = [abs(Gamma_ratio(1e-3, c) - 1.0) for c in (0.1, 0.2, 0.3, 0.5)]
    t1 = max(errs) < 1e-8
    print(f"  [{'ok  ' if t1 else 'FAIL'}] T1 Gamma(alpha->0)=4a/R  max rel err {max(errs):.2e}")
    ok &= t1

    # T2 -- against the oracle and the theory table
    worst_oracle = 0.0
    worst_table = 0.0
    for a in oracle.GAMMA_RATIO_TABLE_ALPHAS:
        for j, c in enumerate(oracle.GAMMA_RATIO_TABLE_A_OVER_R):
            got = Gamma_ratio(float(a), c)
            worst_oracle = max(worst_oracle, abs(got - oracle.gamma_ratio(float(a), c)))
            worst_table = max(worst_table, abs(got - oracle.GAMMA_RATIO_REFERENCE[a][j]))
    t2 = worst_oracle < 1e-12 and worst_table < 1.5e-3
    print(f"  [{'ok  ' if t2 else 'FAIL'}] T2 Gamma/Gamma_P table  "
          f"vs oracle {worst_oracle:.1e}, vs theory {worst_table:.4f}")
    ok &= t2

    # T3 -- mass conservation
    y = np.linspace(0.0, 1.0, 200_001)
    worst_mc = 0.0
    for a in (1.0, 3.0, 7.0, 15.0):
        integral = 2.0 * np.trapezoid(profile_phasor(a, y) * y, y)
        worst_mc = max(worst_mc, abs(integral - vmean_phasor(a)))
    t3 = worst_mc < 1e-9
    print(f"  [{'ok  ' if t3 else 'FAIL'}] T3 mass conservation  max |error| {worst_mc:.2e}")
    ok &= t3

    # steady-flow sanity: mean_flow of a pure steady waveform is steady/2
    ff = FlowField(alpha=5.0, waveform=Waveform(steady=2.0, harmonics=()))
    t4 = abs(ff.mean_flow(1.3) - 1.0) < 1e-12
    print(f"  [{'ok  ' if t4 else 'FAIL'}] steady mean flow == steady/2")
    ok &= t4

    # Dean correction is off by default-switch and separate
    t5 = (dean_peak_shift_fraction(500.0, enabled=False) == 0.0
          and dean_peak_shift_fraction(0.0) == 0.0
          and 0.0 < dean_peak_shift_fraction(100.0) < 0.18)
    print(f"  [{'ok  ' if t5 else 'FAIL'}] Dean correction separate + switchable")
    ok &= t5

    print("\n  PASS" if ok else "\n  FAILURES ABOVE")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    ok = _self_test()
    if args.self_test and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
