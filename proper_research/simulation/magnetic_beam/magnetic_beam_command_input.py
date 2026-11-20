from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from proper_research.simulation.simulations.model_factory import (
    ModelBundle, build_forward_model, build_lumen_geometry, build_stiffness_profile, make_contact_params
)
from beam_direction_magnetisation.quarternions.quarternions_functions import (
    T_to_p_quat_wxyz,
)
from scipy.spatial.transform import Rotation as R
from proper_research.simulation.magnetic_beam.kinematics import effective_lengths, pose8_quat_to_pose7_rotvec
from proper_research.parameters import default_magnet_params
from beam_direction_magnetisation.cosserat_6d_pose import ur_pose6_to_T

mag_params = default_magnet_params()
@dataclass(frozen=True)
class ScenePoses:
    """
    All poses are UR-style pose6 values expressed in the robot-base frame.

    pose6:
        [x, y, z, rx, ry, rz]

    The final three entries are one rotation vector, not Euler angles.
    """

    beam_base_robot: np.ndarray
    lumen_base_robot: np.ndarray
    source_magnet_robot: np.ndarray
    insertion_m: float
    dt: float
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


def magnet_pose_and_insertion_to_p7(
    source_magnet_pose_robot: np.ndarray,
    insertion_m: float,
) -> np.ndarray:
    """
    Convert a robot-frame magnet-centre pose and insertion into the
    forward model's p7 format.

    Returns
    -------
    p7:
        [magnet_x, magnet_y, magnet_z, rx, ry, rz, L_inserted]
    """
    source_magnet_pose_robot = np.asarray(
        source_magnet_pose_robot,
        float,
    ).reshape(6)

    insertion_m = float(insertion_m)

    if not np.all(
        np.isfinite(
            source_magnet_pose_robot
        )
    ):
        raise ValueError(
            "Source magnet pose contains "
            "non-finite values."
        )

    if not np.isfinite(insertion_m):
        raise ValueError(
            "Insertion is not finite."
        )

    if insertion_m <= 0.0:
        raise ValueError(
            "Insertion must be positive."
        )

    return np.concatenate(
        [
            source_magnet_pose_robot,
            np.array(
                [insertion_m],
                dtype=float,
            ),
        ]
    )
RUN_PLOT_P7 = np.array(
    [
        -0.06,
        0.13,
        0.0,
        np.pi / np.sqrt(2.0),
        0.0,
        np.pi / np.sqrt(2.0),
        0.023,
    ],
    dtype=float,
)
BENCHMARK_P7 = RUN_PLOT_P7.copy()

def make_initial_poses(
    *,
    source_magnet_robot: np.ndarray,
    insertion_m: float,
    beam_base_robot: np.ndarray | None = None,
    lumen_base_robot: np.ndarray | None = None,
    dt: float = 0.01,
) -> ScenePoses:
    source_magnet_robot = np.asarray(
        source_magnet_robot,
        float,
    ).reshape(6)

    insertion_m = float(insertion_m)

    if beam_base_robot is None:
        beam_base_robot = np.array(
            [
                0.7981328220229531,
                -0.7099273166922002,
                -0.1000000000000000,
                np.pi,
                0.001,
                0.001,
            ],
            dtype=float,
        )

    if lumen_base_robot is None:
        lumen_base_robot = np.array(
            [
                0.7981328220229531,
                -0.7099273166922002,
                -0.1000000000000000,
                np.pi,
                0.001,
                0.001,
            ],
            dtype=float,
        )

    return ScenePoses(
        beam_base_robot=np.asarray(
            beam_base_robot,
            float,
        ).reshape(6),
        lumen_base_robot=np.asarray(
            lumen_base_robot,
            float,
        ).reshape(6),
        source_magnet_robot=(
            source_magnet_robot.copy()
        ),
        insertion_m=insertion_m,
        dt=float(dt),
    )


def build_model_bundle2(
    *,
    beam_base_pose_robot: np.ndarray,
    lumen_base_pose_robot: np.ndarray,
    L0: float,
    lumen_cfg,
    plant_contact: bool = True,
    use_composite_magnetisation: bool = True,
    use_physical_stiffness: bool = False,
    effective_youngs_modulus: float = 1.0e6,
    remanence_fraction: float = 1.0,
) -> ModelBundle:
    if hasattr(lumen_cfg, "validate"):
        lumen_cfg.validate()

    beam_base_pose_robot = np.asarray(
        beam_base_pose_robot,
        float,
    ).reshape(6)

    lumen_base_pose_robot = np.asarray(
        lumen_base_pose_robot,
        float,
    ).reshape(6)

    # ------------------------------------------------------------
    # Robot frame is being used as the model world frame.
    # ------------------------------------------------------------
    T_robot_beam = ur_pose6_to_T(
        beam_base_pose_robot
    )

    p0_robot_beam, q0_robot_beam = (
        T_to_p_quat_wxyz(
            T_robot_beam
        )
    )

    T_robot_lumen = ur_pose6_to_T(
        lumen_base_pose_robot
    )

    p0_robot_lumen, q0_robot_lumen = (
        T_to_p_quat_wxyz(
            T_robot_lumen
        )
    )

    # ------------------------------------------------------------
    # Beam lengths
    # ------------------------------------------------------------
    L_model, wire_len_model, tip_len_model = (
        effective_lengths(L0)
    )

    print(
        f"L_ins={float(L0):.6f} -> "
        f"L_model={L_model:.6f}, "
        f"wire_len={wire_len_model:.6f}, "
        f"tip_len={tip_len_model:.6f}"
    )

    # ------------------------------------------------------------
    # Lumen expressed in the same robot/world frame.
    # ------------------------------------------------------------
    lumen_C, lumen_R, lumen_path, s_path = (
        build_lumen_geometry(
            p0_ur_lumen=p0_robot_lumen,
            q0_ur_lumen=q0_robot_lumen,
            lumen_cfg=lumen_cfg,
        )
    )

    # --------------------------------------------------------
    # Physical composite-beam properties
    # --------------------------------------------------------
    beam_diameter = 2.0e-3
    particle_mass_fraction = 0.8
    particle_density = 7450.0
    particle_specific_moment = 80.6
    silicone_density = 1070.0
    poisson_ratio = 0.49

    composite = calculate_composite_beam_properties(
        beam_diameter=beam_diameter,
        particle_mass_fraction=(
            particle_mass_fraction
        ),
        particle_density=particle_density,
        particle_specific_moment=(
            particle_specific_moment
        ),
        silicone_density=silicone_density,
        remanence_fraction=(
            remanence_fraction
        ),
    )

    if use_composite_magnetisation:
        m_local_factory = (
            make_uniform_axial_m_local_factory(
                moment_per_length=(
                    composite[
                        "moment_per_length"
                    ]
                ),

                # Confirm this sign against the programmed
                # magnetisation direction of your beam.
                local_axis=(-1.0, 0.0, 0.0),
            )
        )
    else:
        # Legacy make_m_local_fun_wire_tip profile. The forward-model builder
        # supplies its own default factory when this value is None.
        m_local_factory = None

    if m_local_factory is not None:
        m_test_fun = m_local_factory(
            L_model=L_model,
            wire_len=wire_len_model,
            tip_len=tip_len_model,
        )
        s_test = np.linspace(0.0, L_model, 5)
        m_test = np.asarray(
            m_test_fun(s_test, wire_len_model),
            dtype=float,
        )

        print("\n[COMPOSITE BEAM MAGNETISATION]")
        print(
            "  moment_per_length:",
            composite["moment_per_length"],
            "A m",
        )
        print("  profile shape:", m_test.shape)
        print(
            "  profile norms:",
            np.linalg.norm(m_test, axis=0),
        )
        print(
            "  approximate total moment:",
            composite["moment_per_length"] * L_model,
            "A m^2",
        )
    # Change stiffness independently from magnetisation.
    if use_physical_stiffness:
        Kinv_fun = make_Kinv_fun(
            youngs_modulus=(
                effective_youngs_modulus
            ),
            poisson_ratio=poisson_ratio,
            outer_diameter=beam_diameter,
            inner_diameter=0.0,
        )
    else:
        # Preserve current simulator mechanics initially.
        Kinv_fun = build_stiffness_profile()

    contact = make_contact_params()

    # --------------------------------------------------------
    # External source magnet
    # --------------------------------------------------------
    # Keep the simulator's existing source-magnet moment and
    # body-axis convention. This is not beam magnetisation.
    m_body = np.array(
        [
            -mag_params.mag_epm,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    common_model_kwargs = dict(
        p0_ur=p0_robot_beam,
        q0_ur=q0_robot_beam,
        Kinv_fun=Kinv_fun,
        m_body=m_body,
        lumen_C=lumen_C,
        lumen_R=lumen_R,

        # New distributed beam-magnetisation factory.
        m_local_factory=m_local_factory,
    )

    forward_model_contact = (
        build_forward_model(
            **common_model_kwargs,
            contact_enabled=True,
            use_lumen_jac=True,
            contact_params=contact,
        )
    )

    forward_model_no_contact = (
        build_forward_model(
            **common_model_kwargs,
            contact_enabled=False,
            use_lumen_jac=False,
            contact_params=None,
        )
    )

    # Store physical metadata for logging and validation.
    for model in (
        forward_model_contact,
        forward_model_no_contact,
    ):
        model.composite_properties = dict(
            composite
        )
        model.effective_youngs_modulus = float(
            effective_youngs_modulus
        )
        model.beam_magnetisation_profile = (
            "uniform_axial_composite"
            if use_composite_magnetisation
            else "legacy_wire_tip"
        )
        model.beam_magnetisation_axis_local = (
            np.array(
                [-1.0, 0.0, 0.0],
                dtype=float,
            )
            if use_composite_magnetisation
            else None
        )

    plant_model = (
        forward_model_contact
        if plant_contact
        else forward_model_no_contact
    )

    models = {
        "plant": plant_model,
        "contact": forward_model_contact,
        "no_contact": forward_model_no_contact,
    }

    return ModelBundle(
        p0_ur=p0_robot_beam,
        q0_ur=q0_robot_beam,
        lumen_C=lumen_C,
        lumen_R=lumen_R,
        lumen_path=lumen_path,
        s_path=s_path,
        models=models,
    )
def build_benchmark_model(
    *,
    contact_enabled: bool = True,
) -> MagneticBeamForwardModel:
    return build_run_plot_model(contact_enabled=contact_enabled)


def make_run_plot_lumen_config() -> LumenConfig:
    """Return the exact lumen shape used by ``run_plot.py``."""
    return LumenConfig(
        length=0.04,
        n_pts=240,
        n_ref_pts=100,
        radius=0.004,
        bends=(
            LumenBend(
                bend_axis=(0.0, 0.0, 1.0),
                bend_angle_rad=np.deg2rad(00.0),
                bend_start=0.0,
                bend_end=0.01,
            ),
            LumenBend(
                bend_axis=(0.0, 0.0, 1.0),
                bend_angle_rad=np.deg2rad(0.0),
                bend_start=0.02,
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


def run_smoke_tests() -> None:
    print("Running magnetic_beam smoke tests...")
    print(f"Package under test: {PACKAGE_NAME}")
    print(f"Forward model file: {_forward_model.__file__}")

    run_source_geometry_guard_test()
    run_forward_only()
    run_forward_with_contact()


    run_jacobian_test()

    print("\nAll magnetic_beam smoke tests passed.")


def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve the magnetic-beam model from one robot-frame source-magnet "
            "pose and one insertion magnitude."
        )
    )

    parser.add_argument("x", type=float, help="Magnet-centre x coordinate.")
    parser.add_argument("y", type=float, help="Magnet-centre y coordinate.")
    parser.add_argument("z", type=float, help="Magnet-centre z coordinate.")
    parser.add_argument("rx", type=float, help="UR rotation-vector rx component.")
    parser.add_argument("ry", type=float, help="UR rotation-vector ry component.")
    parser.add_argument("rz", type=float, help="UR rotation-vector rz component.")
    parser.add_argument("insertion", type=float, help="Inserted beam length.")

    parser.add_argument(
        "--position-units",
        choices=("m", "mm"),
        default="m",
        help="Units for x, y and z. Default: m.",
    )
    parser.add_argument(
        "--rotation-units",
        choices=("rad", "deg"),
        default="rad",
        help="Units for rx, ry and rz. Default: rad.",
    )
    parser.add_argument(
        "--insertion-units",
        choices=("m", "mm"),
        default="m",
        help="Units for insertion. Default: m.",
    )
    parser.add_argument(
        "--model",
        choices=("contact", "no_contact", "both"),
        default="contact",
        help="Forward model to evaluate. Default: contact.",
    )
    parser.add_argument(
        "--beam-base",
        type=float,
        nargs=6,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        default=None,
        help=(
            "Optional beam-base pose in metres/radians. The fixed calibrated "
            "pose is used when omitted."
        ),
    )
    parser.add_argument(
        "--lumen-base",
        type=float,
        nargs=6,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        default=None,
        help=(
            "Optional lumen-entrance pose in metres/radians. The fixed calibrated "
            "pose is used when omitted."
        ),
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Controller/sample timestep in seconds. Default: 0.01.",
    )
    parser.add_argument(
        "--jacobian",
        action="store_true",
        help="Also calculate and print the 3x7 tip-position Jacobian.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not open the 3D beam/lumen/source plot.",
    )
    parser.add_argument(
        "--smoke-tests",
        action="store_true",
        help="Run the legacy internal smoke tests instead of the input command.",
    )

    return parser.parse_args()


def command_from_args(args: argparse.Namespace) -> tuple[np.ndarray, float]:
    source_pose = np.array(
        [args.x, args.y, args.z, args.rx, args.ry, args.rz],
        dtype=float,
    )

    if args.position_units == "mm":
        source_pose[:3] *= 1.0e-3
    if args.rotation_units == "deg":
        source_pose[3:6] = np.deg2rad(source_pose[3:6])

    insertion_m = float(args.insertion)
    if args.insertion_units == "mm":
        insertion_m *= 1.0e-3

    # This returns the exact public input expected by the forward model:
    # [x, y, z, rx, ry, rz, L].
    p7 = magnet_pose_and_insertion_to_p7(
        source_pose,
        insertion_m,
    )
    return p7, insertion_m


def solve_robot_command(args: argparse.Namespace) -> int:
    p7, insertion_m = command_from_args(args)

    scene = make_initial_poses(
        source_magnet_robot=p7[:6],
        insertion_m=insertion_m,
        beam_base_robot=args.beam_base,
        lumen_base_robot=args.lumen_base,
        dt=args.dt,
    )

    lumen_cfg = make_run_plot_lumen_config()

    bundle = build_model_bundle2(
        beam_base_pose_robot=scene.beam_base_robot,
        lumen_base_pose_robot=scene.lumen_base_robot,
        L0=scene.insertion_m,
        lumen_cfg=lumen_cfg,
        plant_contact=(args.model != "no_contact"),
    )

    print("\n[INPUT COMMAND]")
    print("  p7 =", p7)
    print("  convention = [x, y, z, rx, ry, rz, L]")
    print("  positions/insertion are in metres; rotation vector is in radians")

    if args.model == "both":
        selected_models = (
            ("contact", bundle.models["contact"]),
            ("no_contact", bundle.models["no_contact"]),
        )
    elif args.model == "no_contact":
        selected_models = (("no_contact", bundle.models["no_contact"]),)
    else:
        selected_models = (("contact", bundle.models["contact"]),)

    results = {}
    for name, model in selected_models:
        print(f"\n=== {name.upper()} FORWARD SOLUTION ===")
        result = model.solve(p7, commit=True)
        results[name] = result
        summarise_solve_result(result)

        centreline = np.asarray(result.p, dtype=float)
        tangent = np.full(3, np.nan, dtype=float)
        if centreline.ndim == 2 and centreline.shape[0] >= 2:
            tangent = centreline[-1] - centreline[-2]
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm > 1.0e-12:
                tangent /= tangent_norm

        print("tip [robot m] =", np.asarray(result.tip, dtype=float))
        print("tip tangent    =", tangent)

        if args.jacobian:
            J = model.jacobian_tip_pose7(
                p7,
                solve_if_needed=False,
                reuse_cached=False,
            )
            print("J_tip_pose7 shape =", J.shape)
            print(J)

        if not args.no_plot:
            model.plot_solution(
                p7,
                result=result,
                show=True,
            )

    if "contact" in results and "no_contact" in results:
        tip_contact = np.asarray(results["contact"].tip, dtype=float)
        tip_no_contact = np.asarray(results["no_contact"].tip, dtype=float)
        print("\n=== CONTACT / NO-CONTACT DIFFERENCE ===")
        print(
            "tip difference [mm] =",
            1.0e3 * np.linalg.norm(tip_contact - tip_no_contact),
        )

    return 0


def main() -> int:
    args = parse_command_line()
    if args.smoke_tests:
        run_smoke_tests()
        return 0
    return solve_robot_command(args)

from pathlib import Path


def run_single_robot_pose() -> None:
    # ============================================================
    # 1. INPUT COMMAND
    # ============================================================
    # Magnet-centre pose in the robot-base frame:
    # [x, y, z, rx, ry, rz]
    #
    # Position: metres
    # Rotation: UR rotation vector in radians
    source_magnet_robot = np.array(
        [
        7.85132822e-01, -7.09927317e-01,  1.00000000e-01,  2.22144147e+00,
        2.22144147e+00,  1.36024059e-16
        ],
        dtype=float,
    )
    current_rot = R.from_rotvec(source_magnet_robot[3:6])
    new_rot = R.from_euler('z', 0)
    final_rot = new_rot*current_rot
    source_magnet_robot[3:6] = final_rot.as_rotvec()
    # p8 = np.array([0.57,	-0.74,	-0.1,	0.01,	0.91,	0.41,	0,	0.02])
    # p7 = pose8_quat_to_pose7_rotvec(p8)
    # # Beam insertion in metres.
    # insertion_m = p7[-1]
    insertion_m = 0.013
    # Build the exact seven-value forward-model input.
    p7 = magnet_pose_and_insertion_to_p7(
        source_magnet_robot,
        insertion_m,
    )

    # ============================================================
    # 2. FIXED BEAM AND LUMEN POSES
    # ============================================================
    scene = make_initial_poses(
        source_magnet_robot=p7[0:-1],
        insertion_m=p7[-1],
    )

    lumen_cfg = make_run_plot_lumen_config()

    # ============================================================
    # 3. BUILD CONTACT AND NO-CONTACT MODELS
    # ============================================================
    bundle = build_model_bundle2(
        beam_base_pose_robot=(
            scene.beam_base_robot
        ),
        lumen_base_pose_robot=(
            scene.lumen_base_robot
        ),
        L0=insertion_m,
        lumen_cfg=lumen_cfg,
        plant_contact=True,
    )
    print(f"Beam shape is {scene.beam_base_robot.shape}")
    # Select the model to evaluate.
    model = bundle.models["contact"]

    # ============================================================
    # 4. SOLVE
    # ============================================================
    result = model.solve(
        p7,
        commit=True,
    )

    if not bool(result.success):
        raise RuntimeError(
            "The forward equilibrium solve did not converge."
        )

    print("\n=== FORWARD SOLUTION ===")
    print("p7:", p7)
    print("tip:", result.tip)
    print(
        "centreline shape:",
        np.asarray(result.p).shape,
    )
    print(f"vector of p7 is {p7[0:3]} and beam base is {scene.beam_base_robot[0:3]}")
    dot_prod = np.dot(p7[0:3],scene.beam_base_robot[0:3])
    norm_1 = np.linalg.norm(p7[0:3])
    norm_2 = np.linalg.norm(scene.beam_base_robot[0:3])
    angle_1 = dot_prod/(norm_1*norm_2)
    angle_rad = np.arccos(angle_1)
    angle_bet = np.degrees(angle_rad)

    print(f"Angle is {angle_bet}")
    # ============================================================
    # 5. PLOT AND SAVE
    # ============================================================
    output_path = Path(
        "contact_forward_solution.png"
    )

    model.plot_solution(
        p7,
        result=result,

        # Display an interactive figure.
        show=True,

        # Also save the same figure.
        save_path=output_path,
    )

    print(
        f"Saved plot to: "
        f"{output_path.resolve()}"
    )


def main() -> None:
    run_single_robot_pose()



if __name__ == "__main__":
    main()