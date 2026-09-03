"""Oracle for the Layer-5 flow-only centreline-following feasibility study.

This module is deliberately self-contained: it imports only ``numpy`` and
``scipy.special`` and nothing from the project.  Every number quoted in
``FLOW_CENTERING_THEORY.md`` is produced here, and the analysis code
(``analysis/analyse_layer5_flow.py``) and the modelling code (``flow/*.py``)
are checked against it rather than against re-derived expressions.

If an implementation and this oracle disagree, the oracle is right until proven
otherwise.

Run ``python -m proper_research.flow.flow_observability_reference --self-test``
(or execute the file directly) to reproduce the theory-document tables.

--------------------------------------------------------------------------
Model
--------------------------------------------------------------------------
Womersley oscillatory flow in a rigid tube, plus an optional steady component.
With ``y = r / R``, ``beta = alpha * i**(3/2)`` and ``alpha = R sqrt(omega rho / mu)``
the oscillatory axial velocity is

    u(y, t) = Re{ (i P / (rho omega)) [ 1 - J0(beta y) / J0(beta) ] e^{i omega t} }

The dimensionless *profile phasor* is the bracket ``f(y) = 1 - J0(beta y)/J0(beta)``.
All quantities the observability analysis needs are ratios of derivatives of
``f`` and are therefore independent of the (unknown) pressure amplitude ``P``.

Two errors that were caught during derivation and must not be reintroduced:

* The quadrupole channel is **not** zero under Womersley flow.  An off-centre
  catheter in a non-parabolic profile produces a quadrupole signal with no
  secondary flow present at all -- see :func:`quadrupole_confound_fraction`.
* Catheter calibre must be matched to the vessel.  A 7 Fr device does not enter
  a coronary artery; pairing them yields a nonsense ~59% blockage figure.  The
  vessel table below matches a 3 Fr catheter to the small vessels and a 7 Fr
  catheter to the large ones.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
from scipy import special

# --------------------------------------------------------------------------
# Physiological constants.  FLAGGED AS ASSUMPTIONS: no fluid parameters exist
# anywhere in the project (see the STEP 0 audit), so blood defaults are used.
# --------------------------------------------------------------------------
RHO_BLOOD = 1060.0          # kg / m^3
MU_BLOOD = 3.5e-3           # Pa s
HEART_RATE_BPM = 72.0       # beats / min
OMEGA_BLOOD = 2.0 * math.pi * HEART_RATE_BPM / 60.0   # rad / s  (~7.540)

# Catheter radii by French size (1 Fr = 1/3 mm outer diameter).
CATHETER_RADIUS_M = {3: 0.55e-3, 7: 1.15e-3}

TOLERANCE_DE_OVER_R = 0.10   # the feasibility criterion delta_e / R <= 0.10


# --------------------------------------------------------------------------
# Womersley phasors
# --------------------------------------------------------------------------
def womersley_number(radius_m: float, omega: float = OMEGA_BLOOD,
                     rho: float = RHO_BLOOD, mu: float = MU_BLOOD) -> float:
    """alpha = R sqrt(omega rho / mu)."""
    return float(radius_m) * math.sqrt(omega * rho / mu)


def _beta(alpha: float | np.ndarray) -> np.ndarray:
    """beta = alpha * i^{3/2}, the Womersley Bessel argument."""
    return np.asarray(alpha, dtype=float) * (1j ** 1.5)


def profile_phasor(alpha: float, y: float | np.ndarray) -> np.ndarray:
    """f(y) = 1 - J0(beta y) / J0(beta) -- the dimensionless velocity profile."""
    beta = _beta(alpha)
    y = np.asarray(y, dtype=float)
    return 1.0 - special.jv(0, beta * y) / special.jv(0, beta)


def grad_phasor(alpha: float, y: float | np.ndarray) -> np.ndarray:
    """df/dy = beta J1(beta y) / J0(beta).

    This is the radial gradient of the profile with respect to ``y = r/R``;
    the gradient with respect to ``r`` is ``grad_phasor / R``.
    """
    beta = _beta(alpha)
    y = np.asarray(y, dtype=float)
    return beta * special.jv(1, beta * y) / special.jv(0, beta)


def grad2_phasor(alpha: float, y: float | np.ndarray) -> np.ndarray:
    """d2f/dy2 = beta^2 [ J0(beta y) - J1(beta y)/(beta y) ] / J0(beta).

    Uses J1'(x) = J0(x) - J1(x)/x.
    """
    beta = _beta(alpha)
    y = np.asarray(y, dtype=float)
    arg = beta * y
    bracket = special.jv(0, arg) - special.jv(1, arg) / arg
    return beta ** 2 * bracket / special.jv(0, beta)


def vmean_phasor(alpha: float) -> complex:
    """Cross-sectional mean of f: 1 - 2 J1(beta) / (beta J0(beta))."""
    beta = _beta(alpha)
    return complex(1.0 - 2.0 * special.jv(1, beta) / (beta * special.jv(0, beta)))


def gamma_poiseuille(a_over_R: float) -> float:
    """Steady-flow discrimination gain, 4 a/R."""
    return 4.0 * float(a_over_R)


def gamma(alpha: float, a_over_R: float) -> float:
    """Discrimination gain Gamma(alpha, a/R) = |grad_phasor| / |vmean_phasor|.

    In the Poiseuille limit alpha -> 0 this tends to 4 a/R.
    """
    return float(abs(grad_phasor(alpha, a_over_R)) / abs(vmean_phasor(alpha)))


def gamma_ratio(alpha: float, a_over_R: float) -> float:
    """Gamma normalised by its Poiseuille value 4 a/R (1.0 == as good as steady)."""
    return gamma(alpha, a_over_R) / gamma_poiseuille(a_over_R)


# --------------------------------------------------------------------------
# Phase structure -- section 4.4
# --------------------------------------------------------------------------
def null_phase_separation_deg(alpha: float, a_over_R: float = 0.2) -> float:
    """Phase angle between the zero-gradient instant and the zero-flow instant.

    Both the radial gradient at the sensor ring and the mean flow are phasors
    ``X e^{i omega t}``; ``Re{X e^{i omega t}}`` vanishes at
    ``omega t = pi/2 - arg(X)`` (mod pi).  The separation of the two nulls is
    ``(arg(vmean) - arg(grad_phasor(alpha, a/R))) mod 180`` degrees, reported in
    ``[0, 180)`` -- it is deliberately *not* folded into ``[0, 90]`` because the
    theory document quotes separations above 90 deg (100.4 at alpha=5).

    The gradient is evaluated at the reference ring ``a/R = 0.2`` used throughout
    section 3-4 of the theory document.
    """
    grad_arg = np.angle(grad_phasor(alpha, a_over_R))
    flow_arg = np.angle(vmean_phasor(alpha))
    return float(math.degrees(flow_arg - grad_arg) % 180.0)


# --------------------------------------------------------------------------
# Quadrupole confound -- section 5.2
# --------------------------------------------------------------------------
def quadrupole_bracket(alpha: float, a_over_R: float) -> complex:
    """f''(c) - f'(c)/c at c = a/R.  Vanishes identically for a parabolic profile."""
    c = float(a_over_R)
    return complex(grad2_phasor(alpha, c) - grad_phasor(alpha, c) / c)


def quadrupole_confound_fraction(alpha: float, a_over_R: float,
                                 e_over_R: float) -> float:
    """|Q| / |differential signal| for a pure-y offset of magnitude ``e_over_R``.

    To second order in the offset the quadrupole channel is

        Q = [ u''(a) - u'(a)/a ] (e_y^2 - e_z^2) + O(e^4)

    For a pure-y offset (e_z = 0) this is ``bracket * e_y^2`` in ``y`` units, and
    referenced to the *Poiseuille* differential gain ``8 (a/R)`` (the steady-flow
    differential signal, ``2 Gamma_P e_y``) the fraction is

        |f''(c) - f'(c)/c| / (8 c) * (e/R),      c = a/R.

    This peaks near ``alpha ~ 5-7`` -- the profile is then furthest from *both*
    parabolic and plug -- and matches the section 5.2 table there.  Near
    ``alpha = 2`` the profile is nearly parabolic, the bracket is tiny, and this
    second-order form modestly over-predicts the (negligible) confound relative
    to the exact sensor model.

    Exactly zero for a parabolic profile: the bracket vanishes identically for
    any ``u = A + B r^2``.
    """
    c = float(a_over_R)
    return float(abs(quadrupole_bracket(alpha, c)) / (8.0 * c) * e_over_R)


# --------------------------------------------------------------------------
# Bend-induced (Dean) bias -- section 5.1
# --------------------------------------------------------------------------
def radius_of_curvature_from_bend(bend_angle_rad: float,
                                  bend_arclength_m: float) -> float:
    """R_c = (arclength of the bent segment) / (total tangent rotation).

    A ``LumenBend`` rotates the centreline tangent by ``bend_angle_rad`` over the
    arclength ``bend_end - bend_start``; a circular arc of that length and turn
    angle has this radius of curvature.  An angle alone is not enough for the
    Dean number -- the bent-segment length is required too.
    """
    if bend_angle_rad == 0.0:
        return math.inf
    return float(bend_arclength_m) / abs(float(bend_angle_rad))


def reynolds_number(radius_m: float, vmean: float,
                    rho: float = RHO_BLOOD, mu: float = MU_BLOOD) -> float:
    """Re = 2 R v_mean rho / mu (diameter-based)."""
    return 2.0 * float(radius_m) * float(vmean) * rho / mu


def dean_number(radius_m: float, radius_of_curvature_m: float, vmean: float,
                rho: float = RHO_BLOOD, mu: float = MU_BLOOD) -> float:
    """De = Re sqrt(R / R_c)."""
    re = reynolds_number(radius_m, vmean, rho, mu)
    return float(re * math.sqrt(radius_m / radius_of_curvature_m))


def dean_peak_shift_fraction(dean: float) -> float:
    """Fractional outward shift of the velocity peak, delta_peak / R.

    LITERATURE CORRELATION, NOT DERIVED HERE.  The theory document quotes only
    bands: De < 20 weak (shift <~ 2% R); 20-100 developing; > 100 strong twin
    vortices with shifts reported at 10-20% R.  This is a smooth monotone
    interpolation of those bands,

        delta_peak / R = 0.18 * (1 - exp(-De / 160))

    which gives 2.1% at De=20, 8.3% at De=100 and saturates near 18%.  Treat the
    direction (outward) as trustworthy and the magnitude as a scaling argument
    with a wide band -- see :func:`dean_peak_shift_band`.
    """
    return 0.18 * (1.0 - math.exp(-abs(float(dean)) / 160.0))


def dean_peak_shift_band(dean: float) -> tuple[float, float]:
    """A conservative / optimistic band around :func:`dean_peak_shift_fraction`."""
    centre = dean_peak_shift_fraction(dean)
    return (0.5 * centre, 1.6 * centre)


# --------------------------------------------------------------------------
# Feasibility criterion -- section 6
# --------------------------------------------------------------------------
def resolvable_offset_fraction(alpha: float, a_over_R: float,
                               sigma_over_vmean: float,
                               dean_bias_fraction: float = 0.0) -> float:
    """delta_e / R = (1/sqrt2)(sigma_s/v_mean)/Gamma + dean_bias/R.

    The Dean term is a bias: it is added, not combined in quadrature.
    """
    noise_term = (1.0 / math.sqrt(2.0)) * float(sigma_over_vmean) / gamma(alpha, a_over_R)
    return float(noise_term + dean_bias_fraction)


def feasibility_boundary_radius(catheter_radius_m: float,
                                sigma_over_vmean: float,
                                omega: float = OMEGA_BLOOD,
                                rho: float = RHO_BLOOD, mu: float = MU_BLOOD,
                                tol: float = TOLERANCE_DE_OVER_R) -> float:
    """Vessel radius at which delta_e/R (noise term only) equals ``tol``.

    Bisection on R; the noise term rises monotonically with R because Gamma
    collapses with alpha.
    """
    def excess(radius_m: float) -> float:
        alpha = womersley_number(radius_m, omega, rho, mu)
        return resolvable_offset_fraction(
            alpha, catheter_radius_m / radius_m, sigma_over_vmean) - tol

    lo, hi = 1e-4, 5e-2
    flo, fhi = excess(lo), excess(hi)
    if flo > 0.0:
        return lo
    if fhi < 0.0:
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if excess(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# Scalar-aggregate vs differential map -- section 3
# --------------------------------------------------------------------------
def scalar_sensitivity(e_over_R: float) -> float:
    """|d s_bar / d e| normalised by v_mean/R for the scalar-aggregate pipeline.

    s_bar = v_max (1 - (|e|^2 + a^2)/R^2)  =>  d s_bar/d e = -2 v_max e / R^2,
    which in units of v_mean/R is 4 |e|/R.  Radial only (rank 1), zero at e=0.
    """
    return 4.0 * float(e_over_R)


def differential_sensitivity(a_over_R: float, alpha: float = 0.0) -> float:
    """Singular value of the differential map, 8 a/R in the Poiseuille limit.

    Under Womersley flow the constant 8 a/R is replaced by 2 * Gamma(alpha, a/R)
    per axis (two axes), so sigma_min = 2 Gamma; at alpha -> 0, 2 Gamma -> 8 a/R.
    """
    if alpha == 0.0:
        return 8.0 * float(a_over_R)
    return 2.0 * gamma(alpha, a_over_R)


def scalar_vs_differential_crossover(a_over_R: float) -> float:
    """|e|/R at which the scalar sensitivity 4|e|/R meets the differential 8a/R."""
    return 2.0 * float(a_over_R)


# --------------------------------------------------------------------------
# Vessel reference table -- section 4.3 (catheter matched to calibre)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Vessel:
    name: str
    radius_m: float
    vmean_m_s: float      # representative peak/mean axial speed (ASSUMPTION)
    catheter_fr: int

    @property
    def a_over_R(self) -> float:
        return CATHETER_RADIUS_M[self.catheter_fr] / self.radius_m

    @property
    def alpha(self) -> float:
        return womersley_number(self.radius_m)

    @property
    def blockage(self) -> float:
        return self.a_over_R ** 2


VESSELS: tuple[Vessel, ...] = (
    Vessel("radial", 1.3e-3, 0.10, 3),
    Vessel("coronary", 1.5e-3, 0.15, 3),
    Vessel("carotid", 3.5e-3, 0.30, 7),
    Vessel("femoral", 4.0e-3, 0.25, 7),
    Vessel("iliac", 5.5e-3, 0.30, 7),
    Vessel("descending aorta", 10.0e-3, 0.40, 7),
    Vessel("ascending aorta", 14.0e-3, 0.45, 7),
)


# --------------------------------------------------------------------------
# Reference tables from the theory document
# --------------------------------------------------------------------------
GAMMA_RATIO_TABLE_ALPHAS = (1, 2, 3, 4, 5, 7, 10, 15, 20)
GAMMA_RATIO_TABLE_A_OVER_R = (0.1, 0.2, 0.3, 0.5)

# section 4.2, Gamma / Gamma_P
GAMMA_RATIO_REFERENCE = {
    1: (0.999, 0.999, 0.999, 0.999),
    2: (0.986, 0.986, 0.987, 0.989),
    3: (0.935, 0.935, 0.937, 0.947),
    4: (0.824, 0.824, 0.828, 0.857),
    5: (0.664, 0.665, 0.672, 0.729),
    7: (0.348, 0.351, 0.365, 0.472),
    10: (0.096, 0.100, 0.115, 0.223),
    15: (0.008, 0.009, 0.014, 0.056),
    20: (0.000, 0.001, 0.002, 0.013),
}

# section 4.4, null separation in degrees (gradient at the a/R = 0.2 ring)
NULL_SEPARATION_REFERENCE = {2: 17.9, 3: 39.6, 5: 100.4, 10: 92.5}

# section 5.2, quadrupole confound as a fraction of the differential signal,
# at a/R = 0.2, for e/R in (0.02, 0.05, 0.10, 0.15)
QUADRUPOLE_CONFOUND_REFERENCE = {
    2: (0.0005, 0.0013, 0.0025, 0.0038),
    5: (0.0062, 0.0156, 0.0311, 0.0466),
    10: (0.0042, 0.0106, 0.0212, 0.0316),
}
QUADRUPOLE_E_OVER_R = (0.02, 0.05, 0.10, 0.15)

# section 6, feasibility boundary radius in mm
BOUNDARY_RADIUS_REFERENCE_MM = {
    (3, 0.01): 5.68, (3, 0.02): 4.83, (3, 0.05): 3.62,
    (7, 0.01): 6.58, (7, 0.02): 5.76, (7, 0.05): 4.63,
}


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
def _check(label: str, condition: bool, detail: str = "") -> bool:
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f" -- {detail}" if detail else ""))
    return condition


def self_test(verbose: bool = True) -> bool:
    ok = True

    # T1 -- Poiseuille limit
    alpha0 = 1e-3
    errs = [abs(gamma_ratio(alpha0, c) - 1.0) for c in (0.1, 0.2, 0.3, 0.5)]
    ok &= _check("T1  Gamma(alpha->0) == 4 a/R", max(errs) < 1e-8,
                 f"max rel err {max(errs):.2e} (theory: 2.8e-10 at alpha=1e-3)")

    # T2 -- reproduce the section 4.2 table to 3 dp
    worst = 0.0
    for a in GAMMA_RATIO_TABLE_ALPHAS:
        for j, c in enumerate(GAMMA_RATIO_TABLE_A_OVER_R):
            got = gamma_ratio(float(a), c)
            worst = max(worst, abs(got - GAMMA_RATIO_REFERENCE[a][j]))
    ok &= _check("T2  Gamma/Gamma_P table (section 4.2)", worst < 1.5e-3,
                 f"max abs deviation {worst:.4f}")

    # T3 -- mass conservation
    y = np.linspace(0.0, 1.0, 200_001)
    worst_mc = 0.0
    for a in (1.0, 3.0, 7.0, 15.0):
        integral = 2.0 * np.trapezoid(profile_phasor(a, y) * y, y)
        worst_mc = max(worst_mc, abs(integral - vmean_phasor(a)))
    ok &= _check("T3  mass conservation (integral f == vmean)", worst_mc < 1e-9,
                 f"max |error| {worst_mc:.2e}")

    # section 4.3 vessel table
    worst_v = 0.0
    for v in VESSELS[:5]:
        worst_v = max(worst_v, abs(gamma(v.alpha, v.a_over_R)
                                   - _VESSEL_GAMMA_REFERENCE[v.name]))
    ok &= _check("section 4.3 vessel Gamma", worst_v < 5e-3,
                 f"max abs deviation {worst_v:.4f}")

    # section 4.4 phase separation
    worst_p = max(abs(null_phase_separation_deg(a) - ref)
                  for a, ref in NULL_SEPARATION_REFERENCE.items())
    ok &= _check("section 4.4 null phase separation", worst_p < 0.15,
                 f"max abs deviation {worst_p:.2f} deg")

    # section 5.2 quadrupole confound -- tight at the alpha ~ 5 peak, looser in
    # the near-parabolic alpha = 2 regime where the confound is negligible anyway
    worst_q5 = max(abs(quadrupole_confound_fraction(5.0, 0.2, e) - ref)
                   for e, ref in zip(QUADRUPOLE_E_OVER_R,
                                     QUADRUPOLE_CONFOUND_REFERENCE[5]))
    worst_q10 = max(abs(quadrupole_confound_fraction(10.0, 0.2, e) - ref)
                    for e, ref in zip(QUADRUPOLE_E_OVER_R,
                                      QUADRUPOLE_CONFOUND_REFERENCE[10]))
    ok &= _check("section 5.2 quadrupole confound (alpha=5, 10)",
                 worst_q5 < 5e-4 and worst_q10 < 5e-4,
                 f"max abs deviation {max(worst_q5, worst_q10):.5f}")
    ok &= _check("section 5.2 quadrupole zero for parabolic",
                 abs(quadrupole_bracket(1e-4, 0.2)) < 1e-6,
                 f"|bracket(alpha->0)| = {abs(quadrupole_bracket(1e-4, 0.2)):.2e}")

    # section 6 boundary radii
    worst_b = 0.0
    for (fr, noise), ref_mm in BOUNDARY_RADIUS_REFERENCE_MM.items():
        got_mm = 1e3 * feasibility_boundary_radius(CATHETER_RADIUS_M[fr], noise)
        worst_b = max(worst_b, abs(got_mm - ref_mm))
    ok &= _check("section 6 feasibility boundary radius", worst_b < 0.3,
                 f"max abs deviation {worst_b:.3f} mm")

    # crossover
    ok &= _check("section 3 crossover |e| = 2a",
                 abs(scalar_vs_differential_crossover(0.2) - 0.4) < 1e-12)

    if verbose:
        print("\n  PASS" if ok else "\n  FAILURES ABOVE")
    return ok


# Gamma at the section 4.3 operating points (matched catheter), from this module.
_VESSEL_GAMMA_REFERENCE = {
    "radial": 1.673, "coronary": 1.436, "carotid": 0.826,
    "femoral": 0.574, "iliac": 0.176,
}


def _print_tables() -> None:
    print("\nsection 4.2  Gamma / Gamma_P")
    header = "  alpha " + " ".join(f"a/R={c:<5}" for c in GAMMA_RATIO_TABLE_A_OVER_R)
    print(header)
    for a in GAMMA_RATIO_TABLE_ALPHAS:
        row = " ".join(f"{gamma_ratio(float(a), c):<9.3f}"
                       for c in GAMMA_RATIO_TABLE_A_OVER_R)
        print(f"  {a:<5} {row}")

    print("\nsection 4.3  vessels (matched catheter)")
    print("  vessel            R(mm)  Fr   alpha   a/R    Gamma/Gamma_P  Gamma   blockage")
    for v in VESSELS:
        print(f"  {v.name:<17} {1e3*v.radius_m:<6.1f} {v.catheter_fr:<4} "
              f"{v.alpha:<7.2f} {v.a_over_R:<6.3f} "
              f"{gamma_ratio(v.alpha, v.a_over_R):<14.3f} "
              f"{gamma(v.alpha, v.a_over_R):<7.3f} {100*v.blockage:.1f}%")

    print("\nsection 4.4  null phase separation (deg)")
    for a in (2, 3, 5, 10):
        print(f"  alpha={a:<3} {null_phase_separation_deg(float(a)):.1f}")

    print("\nsection 5.2  quadrupole confound (% of differential signal)")
    print("  alpha  e/R=0.02  0.05    0.10    0.15")
    for a in (2, 5, 10):
        row = "  ".join(f"{100*quadrupole_confound_fraction(float(a), 0.3, e):.2f}%"
                        for e in (0.02, 0.05, 0.10, 0.15))
        print(f"  {a:<5}  {row}")

    print("\nsection 6  feasibility boundary radius (mm)")
    for fr in (3, 7):
        for noise in (0.01, 0.02, 0.05):
            r_mm = 1e3 * feasibility_boundary_radius(CATHETER_RADIUS_M[fr], noise)
            print(f"  {fr} Fr, sigma/vmean={noise:.0%}: {r_mm:.2f} mm")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--tables", action="store_true")
    args = parser.parse_args()
    if args.tables:
        _print_tables()
    ok = self_test()
    if args.self_test and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
