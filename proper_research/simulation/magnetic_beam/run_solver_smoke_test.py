from __future__ import annotations

import importlib
from pathlib import Path
import numpy as np

PACKAGE_NAME = __package__ or (
    f"proper_research.simulation.{Path(__file__).resolve().parent.name}"
)

_config = importlib.import_module(f"{PACKAGE_NAME}.config")
_contact = importlib.import_module(f"{PACKAGE_NAME}.contact")
_problem = importlib.import_module(f"{PACKAGE_NAME}.problem")
_forward_model = importlib.import_module(f"{PACKAGE_NAME}.forward_model")
_diagnostics = importlib.import_module(f"{PACKAGE_NAME}.diagnostics")
_lumen_geometry = importlib.import_module(f"{PACKAGE_NAME}.lumen_geometry")

BeamModelConfig = _config.BeamModelConfig
ContactConfig = _config.ContactConfig
ContactParams = _contact.ContactParams
LumenQuery = _contact.LumenQuery
BaseFrameConfig = _problem.BaseFrameConfig
MagneticBeamForwardModel = _forward_model.MagneticBeamForwardModel
summarise_solve_result = _diagnostics.summarise_solve_result
LumenBend = _lumen_geometry.LumenBend
LumenConfig = _lumen_geometry.LumenConfig
build_lumen_query = _lumen_geometry.build_lumen_query

run_plot_initial = np.array(
    [
        -0.06,
        0.13,
        0.0,
        np.pi/ np.sqrt(2.0),
        0.0,
        np.pi / np.sqrt(2.0),
        0.02,
    ],
    dtype=float,
)
def rotation_matrices_z(rot_vec, angle):
    rot_mat = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0,0,1]
    ])
    return rot_mat @ rot_vec
run_plot_initial[3:6] = rotation_matrices_z(run_plot_initial[3:6], 90)
RUN_PLOT_P7 = run_plot_initial
BENCHMARK_P7 = RUN_PLOT_P7.copy()



def build_benchmark_model(
    *,
    contact_enabled: bool = True,
) -> MagneticBeamForwardModel:
    return build_run_plot_model(contact_enabled=contact_enabled)


def make_run_plot_lumen_config() -> LumenConfig:
    """Return the exact lumen shape used by ``run_plot.py``."""
    return LumenConfig(
        length=0.06,
        n_pts=240,
        n_ref_pts=100,
        radius=0.006,
        bends=(
            LumenBend(
                bend_axis=(0.0, 0.0, 1.0),
                bend_angle_rad=np.deg2rad(0.0),
                bend_start=0.0,
                bend_end=0.015,
            ),
            LumenBend(
                bend_axis=(0.0, 0.0, 1.0),
                bend_angle_rad=np.deg2rad(-100.0),
                bend_start=0.015,
                bend_end=0.025,
            ),
        ),
    )


def make_run_plot_p7() -> np.ndarray:
    """
    Return the exact final source pose used by ``run_plot.py``.

    The rotation vector [pi, 0, 0] has had its axis rotated 45 degrees about
    world y, reproducing ``R_y(45 deg) @ [pi, 0, 0]``.
    """
    return RUN_PLOT_P7.copy()
def make_test_lumen(
    *,
    lumen_cfg: LumenConfig | None = None,
    p0_ur_lumen: np.ndarray | None = None,
    q0_ur_lumen: np.ndarray | None = None,
) -> LumenQuery:
    """
    Build a configurable lumen from its world pose and shape configuration.

    The default remains a straight, constant-radius lumen along world -x.
    """
    if lumen_cfg is None:
        lumen_cfg = LumenConfig(
            length=0.12,
            n_pts=160,
            n_ref_pts=80,
            radius=0.010,
        )
    if p0_ur_lumen is None:
        p0_ur_lumen = np.zeros(3)
    if q0_ur_lumen is None:
        q0_ur_lumen = np.array([1.0, 0.0, 0.0, 0.0])

    return build_lumen_query(
        p0_ur_lumen=p0_ur_lumen,
        q0_ur_lumen=q0_ur_lumen,
        lumen_cfg=lumen_cfg,
    )


def make_curved_test_lumen_config() -> LumenConfig:
    """
    Example 3D lumen with two smooth turns and a local narrowing.

    Remove the second bend for a planar lumen. Reverse an angle to create an
    S-bend. Change the second axis to obtain a different out-of-plane turn.
    """
    return LumenConfig(
        length=0.12,
        n_pts=240,
        n_ref_pts=100,
        radius=0.010,
        bends=(
            LumenBend(
                bend_axis=(0.0, 0.0, 1.0),
                bend_angle_rad=np.deg2rad(35.0),
                bend_start=0.020,
                bend_end=0.065,
            ),
            LumenBend(
                bend_axis=(0.0, 1.0, 0.0),
                bend_angle_rad=np.deg2rad(-20.0),
                bend_start=0.060,
                bend_end=0.105,
            ),
        ),
        radius_profile=(
            (0.000, 0.010),
            (0.045, 0.010),
            (0.070, 0.007),
            (0.090, 0.007),
            (0.120, 0.010),
        ),
    )


def calculate_composite_beam_properties(
    *,
    beam_diameter: float,
    particle_mass_fraction: float,
    particle_density: float,
    particle_specific_moment: float,
    silicone_density: float,
    remanence_fraction: float,
) -> dict[str, float]:
    """
    Convert constituent data into density and distributed magnetic moment.

    ``particle_specific_moment`` is in A m^2/kg, numerically equal to emu/g.
    ``remanence_fraction`` converts a saturation-specific moment into the
    programmed remanent fraction. Use one only when 80.6 emu/g is itself a
    measured remanent value.
    """
    diameter = float(beam_diameter)
    w_particle = float(particle_mass_fraction)
    rho_particle = float(particle_density)
    rho_silicone = float(silicone_density)
    sigma_particle = float(particle_specific_moment)
    eta_remanence = float(remanence_fraction)

    if diameter <= 0:
        raise ValueError("beam_diameter must be positive.")
    if not 0.0 < w_particle < 1.0:
        raise ValueError("particle_mass_fraction must lie strictly between 0 and 1.")
    if rho_particle <= 0 or rho_silicone <= 0:
        raise ValueError("Constituent densities must be positive.")
    if sigma_particle < 0:
        raise ValueError("particle_specific_moment must be non-negative.")
    if not 0.0 <= eta_remanence <= 1.0:
        raise ValueError("remanence_fraction must lie between 0 and 1.")

    specific_volume = (
        w_particle / rho_particle
        + (1.0 - w_particle) / rho_silicone
    )
    composite_density = 1.0 / specific_volume
    particle_volume_fraction = (
        (w_particle / rho_particle) / specific_volume
    )

    area = 0.25 * np.pi * diameter**2
    particle_mass_per_length = w_particle * composite_density * area
    moment_per_length = (
        eta_remanence
        * sigma_particle
        * particle_mass_per_length
    )
    composite_magnetization = moment_per_length / area

    return {
        "area": float(area),
        "composite_density": float(composite_density),
        "particle_volume_fraction": float(particle_volume_fraction),
        "particle_mass_per_length": float(particle_mass_per_length),
        "mass_per_length": float(composite_density * area),
        "moment_per_length": float(moment_per_length),
        "composite_magnetization": float(composite_magnetization),
        "equivalent_polarization": float(
            4.0 * np.pi * 1e-7 * composite_magnetization
        ),
    }


def make_uniform_axial_m_local_factory(
    *,
    moment_per_length: float,
    local_axis: np.ndarray | tuple[float, float, float] = (-1.0, 0.0, 0.0),
):
    """
    Build a length-aware factory for a uniformly magnetized composite beam.

    The returned ``m_local_fun`` supplies magnetic dipole moment per unit beam
    length in A m. The energy routine performs the arclength integration, so it
    must not be multiplied by segment length here.
    """
    moment_per_length = float(moment_per_length)
    local_axis = np.asarray(local_axis, float).reshape(3)
    axis_norm = float(np.linalg.norm(local_axis))

    if moment_per_length < 0:
        raise ValueError("moment_per_length must be non-negative.")
    if axis_norm < 1e-12:
        raise ValueError("local_axis must be nonzero.")

    local_axis = local_axis / axis_norm

    def m_local_factory(
        *,
        L_model: float,
        wire_len: float,
        tip_len: float,
    ):
        # The complete modelled beam is the magnetic composite. These arguments
        # are retained so the same factory interface also supports tip-only or
        # piecewise magnetization profiles later.
        del L_model, wire_len, tip_len

        def m_local_fun(s_mid, unused_parameter):
            del unused_parameter
            n_segments = np.asarray(s_mid, float).reshape(-1).size
            return (
                moment_per_length
                * np.repeat(local_axis[:, None], n_segments, axis=1)
            )

        return m_local_fun

    return m_local_factory


def make_Kinv_fun(
    *,
    youngs_modulus: float,
    poisson_ratio: float,
    outer_diameter: float,
    inner_diameter: float = 0.0,
):
    """Build the constant Cosserat compliance profile from beam mechanics."""
    E = float(youngs_modulus)
    nu = float(poisson_ratio)
    d_outer = float(outer_diameter)
    d_inner = float(inner_diameter)

    if E <= 0:
        raise ValueError("youngs_modulus must be positive.")
    if not -1.0 < nu < 0.5:
        raise ValueError("poisson_ratio must lie between -1 and 0.5.")
    if d_outer <= 0:
        raise ValueError("outer_diameter must be positive.")
    if d_inner < 0 or d_inner >= d_outer:
        raise ValueError(
            "inner_diameter must be non-negative and smaller than outer_diameter."
        )

    G = E / (2.0 * (1.0 + nu))
    I = np.pi * (d_outer**4 - d_inner**4) / 64.0
    J = np.pi * (d_outer**4 - d_inner**4) / 32.0

    EI = E * I
    GJ = G * J
    Kinv_single = np.diag([1.0 / GJ, 1.0 / EI, 1.0 / EI])

    def Kinv_fun(smid, wire_len):
        del wire_len
        number_of_segments = np.asarray(smid, float).reshape(-1).size
        return np.repeat(
            Kinv_single[:, :, None],
            number_of_segments,
            axis=2,
        )

    return Kinv_fun


def build_test_model(
    *,
    contact_enabled: bool = True,
    effective_youngs_modulus: float = 1.0e6,
    remanence_fraction: float = 1.0,
    lumen_cfg: LumenConfig | None = None,
    p0_ur_lumen: np.ndarray | None = None,
    q0_ur_lumen: np.ndarray | None = None,
) -> MagneticBeamForwardModel:
    """
    Build the physical test model.

    ``effective_youngs_modulus=1 MPa`` is a provisional runnable value. Replace
    it with the measured modulus of the cured 80 wt% particle/Ecoflex composite.
    """
    beam_diameter = 2.0e-3
    particle_mass_fraction = 0.8
    particle_density = 7450.0
    particle_specific_moment = 80.6
    silicone_density = 1070.0
    poisson_ratio = 0.49

    composite = calculate_composite_beam_properties(
        beam_diameter=beam_diameter,
        particle_mass_fraction=particle_mass_fraction,
        particle_density=particle_density,
        particle_specific_moment=particle_specific_moment,
        silicone_density=silicone_density,
        remanence_fraction=remanence_fraction,
    )

    base = BaseFrameConfig(
        p0_ur=np.array([0.0, 0.0, 0.0]),
        q0_ur=np.array([1.0, 0.0, 0.0, 0.0]),
        u_star=np.zeros(3),
    )

    beam = BeamModelConfig(
        N_nodes=10,
        maxiter=100,
        L0_init=0.01,
        dL_internal=0.02,
        L_tip_full=0.04,
        L_tip_min=0.01,
        energy_scale=1e-8,
        u_scale=30.0,
    )

    if contact_enabled:
        contact = ContactConfig(
            enabled=True,
            use_in_jacobian=True,
            params=ContactParams(
                r_beam=0.5 * beam_diameter,
                k=1e8,
                pen_switch=5e-5,
                k_hard=1e10,
                smooth=True,
                smooth_eps=1e-5,
                window=3,
            ),
        )
        lumen_query = make_test_lumen(
            lumen_cfg=lumen_cfg,
            p0_ur_lumen=p0_ur_lumen,
            q0_ur_lumen=q0_ur_lumen,
        )
    else:
        contact = ContactConfig.disabled()
        lumen_query = None

    Kinv_fun = make_Kinv_fun(
        youngs_modulus=effective_youngs_modulus,
        poisson_ratio=poisson_ratio,
        outer_diameter=beam_diameter,
    )
    m_local_factory = make_uniform_axial_m_local_factory(
        moment_per_length=composite["moment_per_length"],
    )

    mu0 = 4.0 * np.pi * 1e-7
    source_diameter = 68.0e-3
    source_length = 62.4e-3
    source_remanence = 1.2
    source_volume = np.pi * (0.5 * source_diameter) ** 2 * source_length
    source_moment = source_remanence * source_volume / mu0
    m_body = np.array([0.0, 0.0, source_moment])

    model = MagneticBeamForwardModel(
        base=base,
        beam=beam,
        contact=contact,
        Kinv_fun=Kinv_fun,
        m_body=m_body,
        lumen_query=lumen_query,
        m_local_factory=m_local_factory,
    )
    model.composite_properties = composite
    model.source_magnet_parameters = {
        "diameter": source_diameter,
        "length": source_length,
        "remanence": source_remanence,
        "dipole_moment": float(source_moment),
    }
    model.effective_youngs_modulus = float(effective_youngs_modulus)
    return model


def build_run_plot_model(
    *,
    contact_enabled: bool = True,
) -> MagneticBeamForwardModel:
    """Build the exact material and lumen case displayed by ``run_plot.py``."""
    return build_test_model(
        contact_enabled=contact_enabled,
        lumen_cfg=make_run_plot_lumen_config(),
        effective_youngs_modulus=1.0e6,
        remanence_fraction=1.0,
        p0_ur_lumen=np.array([0.0, -0.005, 0.0]),
    )


def run_source_geometry_guard_test() -> None:
    model = build_test_model(contact_enabled=False)
    if not hasattr(model, "_validate_source_magnet_clearance"):
        raise AssertionError(
            "The imported MagneticBeamForwardModel does not contain the "
            "source-magnet geometry guard. Imported from "
            f"{_forward_model.__file__!r}."
        )
    invalid_p7 = np.array(
        [0.03, 0.0, 0.03, 0.0, 0.0, 0.0, 0.06],
        dtype=float,
    )
    try:
        model.solve(invalid_p7, commit=False)
    except ValueError as exc:
        assert "intersects the finite source magnet" in str(exc)
    else:
        raise AssertionError(
            "Expected the source-cylinder intersection guard to reject "
            "the old singular test pose."
        )
    print("Source geometry guard smoke test passed.")


def run_forward_only() -> None:
    model = build_run_plot_model(contact_enabled=False)

    p7 = make_run_plot_p7()

    result = model.solve(p7, commit=True)

    print("\n=== FORWARD SOLVE TEST ===")
    summarise_solve_result(result)

    assert result.tip.shape == (3,)
    assert result.success, (
        "The forward equilibrium did not converge; failed solutions are "
        "deliberately not returned as exact cache hits."
    )
    assert result.u_flat_opt.size == 3 * (model.beam.N_nodes - 1)
    assert model.cache.u_flat_opt is not None
    assert model.cache.tip is not None

    cached = model.solve(p7, commit=True)
    assert cached.info["cache_hit"] is True
    assert cached.info["solve_path"] == "exact_cache"

    p7_changed = p7.copy()
    p7_changed[0] += 1.0e-4
    changed = model.solve(p7_changed, commit=True)
    assert changed.info["solve_path"] in {
        "warm_direct",
        "warm_fallback_continuation",
    }

    print("Forward solve smoke test passed.")


def run_forward_with_contact() -> None:
    model = build_run_plot_model(contact_enabled=True)

    p7 = make_run_plot_p7()

    result = model.solve(p7, commit=True)

    print("\n=== FORWARD SOLVE WITH CONTACT TEST ===")
    summarise_solve_result(result)

    assert result.tip.shape == (3,)
    assert result.success, "The contact equilibrium did not converge."
    assert "W_cf" in result.parts
    assert model.cache.u_flat_opt is not None

    print("Forward solve with contact smoke test passed.")


def run_jacobian_test() -> None:
    model = build_run_plot_model(contact_enabled=True)

    p7 = make_run_plot_p7()

    tip = model(p7)
    J = model.jacobian_tip_pose7(
        p7,
        solve_if_needed=False,
        eps_theta=1e-6,
        eps_hess=1e-4,
        debug_jac=False,
    )

    print("\n=== JACOBIAN TEST ===")
    print("tip =", tip)
    print("J shape =", J.shape)
    print(J)

    assert J.shape == (3, 7)
    assert np.all(np.isfinite(J))

    # An identical request returns the cached matrix.
    J_cached = model.jacobian_tip_pose7(p7, mode="fast")
    assert np.array_equal(J, J_cached)

    # A nearby controller step retains the Hessian but recomputes G_theta.
    p7_changed = p7.copy()
    p7_changed[5] += 1.0e-3
    model.solve(p7_changed, commit=True)
    J_changed = model.jacobian_tip_pose7(
        p7_changed,
        mode="fast",
        reuse_cached=False,
    )
    assert J_changed.shape == (3, 7)
    assert np.all(np.isfinite(J_changed))
    assert model.last_sens_info["hessian_reused"] is True
    assert model.last_sens_info["gradient_evaluations"] == 8

    print("Jacobian smoke test passed.")


def main() -> None:
    print("Running magnetic_beam smoke tests...")
    print(f"Package under test: {PACKAGE_NAME}")
    print(f"Forward model file: {_forward_model.__file__}")

    run_source_geometry_guard_test()
    run_forward_only()
    run_forward_with_contact()


    run_jacobian_test()

    print("\nAll magnetic_beam smoke tests passed.")


if __name__ == "__main__":
    main()