from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from dataclasses import dataclass

import numpy as np

import time
from concurrent.futures import Future, ThreadPoolExecutor

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
from mpl_toolkits.mplot3d import proj3d
from scipy.spatial.transform import Rotation

from proper_research.simulation.simulations.model_factory import (
    ModelBundle, build_forward_model, build_lumen_geometry, build_stiffness_profile, make_contact_params
)
from beam_direction_magnetisation.quarternions.quarternions_functions import (
    T_to_p_quat_wxyz,
)
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
                -0.7149273166922002,
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
        length=0.06,
        n_pts=240,
        n_ref_pts=100,
        radius=0.008,
        bends=(
            LumenBend(
                bend_axis=(0.0, 0.0, 1.0),
                bend_angle_rad=np.deg2rad(0.0),
                bend_start=0.0,
                bend_end=0.005,
            ),
            LumenBend(
                bend_axis=(0.0, 0.0, 1.0),
                bend_angle_rad=np.deg2rad(-100.0),
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

def _normalise_rows(values: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def _lumen_tube_mesh(
    centreline: np.ndarray,
    radius: np.ndarray,
    *,
    n_circle: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct a light visualization mesh around the lumen centreline."""
    centreline = np.asarray(centreline, dtype=float)
    radius = np.asarray(radius, dtype=float).reshape(-1)

    if centreline.ndim != 2 or centreline.shape[1] < 3:
        raise ValueError(
            f"centreline must have shape (N, >=3), got {centreline.shape}."
        )
    centreline = centreline[:, :3]
    if radius.size != centreline.shape[0]:
        raise ValueError("radius and centreline must contain the same number of points.")

    tangent = np.gradient(centreline, axis=0)
    tangent = _normalise_rows(tangent)

    normal_1 = np.zeros_like(tangent)
    previous = None
    for index, tangent_i in enumerate(tangent):
        reference = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(tangent_i, reference))) > 0.90:
            reference = np.array([0.0, 1.0, 0.0], dtype=float)

        normal_i = np.cross(tangent_i, reference)
        norm_i = float(np.linalg.norm(normal_i))
        if norm_i <= 1.0e-12:
            reference = np.array([1.0, 0.0, 0.0], dtype=float)
            normal_i = np.cross(tangent_i, reference)
            norm_i = float(np.linalg.norm(normal_i))
        normal_i /= max(norm_i, 1.0e-12)

        if previous is not None and float(np.dot(normal_i, previous)) < 0.0:
            normal_i = -normal_i
        normal_1[index] = normal_i
        previous = normal_i

    normal_2 = _normalise_rows(np.cross(tangent, normal_1))
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_circle), endpoint=True)
    cosine = np.cos(theta)[None, :, None]
    sine = np.sin(theta)[None, :, None]

    surface = (
        centreline[:, None, :]
        + radius[:, None, None]
        * (
            cosine * normal_1[:, None, :]
            + sine * normal_2[:, None, :]
        )
    )
    return surface[:, :, 0], surface[:, :, 1], surface[:, :, 2]


def _as_centreline(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError(f"Beam centreline must be two-dimensional, got {points.shape}.")
    if points.shape[1] == 3:
        return points.copy()
    if points.shape[0] == 3:
        return points.T.copy()
    raise ValueError(f"Beam centreline must have one dimension of length 3, got {points.shape}.")


class InteractiveMagneticBeamExplorer:
    """
    Matplotlib GUI for moving the source magnet and resolving the beam model.

    The magnet can be moved using:
      * x/y/z sliders,
      * on-screen +/- translation buttons,
      * left-dragging the magnet in the current camera plane.

    Rotation buttons support two explicit operations:
      * local/body: compose an intrinsic rotation about the magnet's current
        local x, y, or z axis;
      * rotvec axis: rotate the UR rotation-vector itself about a selected
        robot/world x, y, or z axis while preserving its magnitude.

    The rotvec sliders expose the exact UR rotation-vector components passed
    to the model. Changing one component directly is different from either
    rotation-button operation.
    """

    def __init__(
        self,
        *,
        bundle: ModelBundle,
        initial_p7: np.ndarray,
        initial_model_mode: str = "contact",
        translation_span_m: float = 0.080,
        insertion_min_m: float = 0.005,
        insertion_max_m: float = 0.080,
        translation_step_m: float = 0.001,
        rotation_step_rad: float = np.deg2rad(2.0),
        insertion_step_m: float = 0.001,
        solve_delay_s: float = 0.20,
        show_lumen_surface: bool = True,
        beam_radius_m: float = 1.0e-3,
    ) -> None:
        self.bundle = bundle
        self.models = {
            "contact": bundle.models["contact"],
            "no_contact": bundle.models["no_contact"],
        }
        self.p7_initial = np.asarray(initial_p7, dtype=float).reshape(7).copy()
        self.p7 = self.p7_initial.copy()
        self.model_mode = str(initial_model_mode)

        if self.model_mode not in {"contact", "no_contact", "both"}:
            raise ValueError(f"Unknown model mode: {self.model_mode!r}.")

        self.translation_span_m = float(translation_span_m)
        self.translation_step_m = float(translation_step_m)
        self.rotation_step_rad = float(rotation_step_rad)
        self.insertion_step_m = float(insertion_step_m)
        self.solve_delay_s = max(float(solve_delay_s), 0.0)
        self.beam_radius_m = max(float(beam_radius_m), 0.0)

        insertion_min_m = float(insertion_min_m)
        insertion_max_m = float(insertion_max_m)
        if insertion_min_m <= 0.0:
            raise ValueError("insertion_min_m must be positive.")
        if insertion_max_m <= insertion_min_m:
            raise ValueError("insertion_max_m must exceed insertion_min_m.")
        self.insertion_min_m = insertion_min_m
        self.insertion_max_m = insertion_max_m

        self.lumen_C = np.asarray(bundle.lumen_C, dtype=float)[:, :3].copy()
        self.lumen_R = np.asarray(bundle.lumen_R, dtype=float).reshape(-1).copy()
        if self.lumen_R.size != self.lumen_C.shape[0]:
            raise ValueError("bundle.lumen_R and bundle.lumen_C are inconsistent.")

        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="beam-solve")
        self.future: Future | None = None
        self.pending_request: tuple[int, np.ndarray, str] | None = None
        self.latest_request_id = 0
        self.last_completed_id = -1
        self.last_request_time = 0.0
        self.force_next_solve = False
        self.closed = False
        self.dragging_magnet = False
        self.drag_projection_depth = None
        self.updating_widgets = False

        self.fig = plt.figure(figsize=(16.0, 9.5))
        # Reserve a dedicated strip under the 3D view for solution metrics.
        # Keeping these values out of the right-hand control column prevents
        # them from overlapping sliders, radio buttons, and help text.
        self.ax = self.fig.add_axes([0.04, 0.18, 0.64, 0.78], projection="3d")
        self.ax.set_xlabel("Robot x [m]")
        self.ax.set_ylabel("Robot y [m]")
        self.ax.set_zlabel("Robot z [m]")
        self.ax.set_title("Interactive magnetic continuum beam explorer")
        self.ax.mouse_init(rotate_btn=3, pan_btn=2)

        self.ax.plot(
            self.lumen_C[:, 0],
            self.lumen_C[:, 1],
            self.lumen_C[:, 2],
            linestyle="--",
            linewidth=1.2,
            color="0.35",
            label="Lumen centreline",
        )

        if show_lumen_surface:
            try:
                x_surface, y_surface, z_surface = _lumen_tube_mesh(
                    self.lumen_C,
                    self.lumen_R,
                    n_circle=20,
                )
                self.ax.plot_surface(
                    x_surface,
                    y_surface,
                    z_surface,
                    alpha=0.12,
                    linewidth=0.0,
                    antialiased=True,
                    color="0.55",
                )
            except Exception as exc:
                print(f"[INTERACTIVE] Lumen surface could not be drawn: {exc}")

        beam_base = np.asarray(bundle.p0_ur, dtype=float).reshape(3)
        self.ax.scatter(
            [beam_base[0]],
            [beam_base[1]],
            [beam_base[2]],
            marker="s",
            s=45,
            color="black",
            label="Beam base",
        )

        (self.contact_line,) = self.ax.plot(
            [], [], [],
            linewidth=3.0,
            color="tab:blue",
            label="Beam: contact",
        )
        (self.no_contact_line,) = self.ax.plot(
            [], [], [],
            linewidth=2.4,
            linestyle="--",
            color="tab:orange",
            label="Beam: no contact",
        )
        (self.contact_tip_marker,) = self.ax.plot(
            [], [], [],
            marker="o",
            markersize=7,
            linestyle="None",
            color="tab:blue",
        )
        (self.no_contact_tip_marker,) = self.ax.plot(
            [], [], [],
            marker="o",
            markersize=7,
            linestyle="None",
            color="tab:orange",
        )

        (self.magnet_center_marker,) = self.ax.plot(
            [], [], [],
            marker="o",
            markersize=9,
            linestyle="None",
            color="black",
            label="Source magnet centre",
            picker=10,
        )
        (self.magnet_axis_line,) = self.ax.plot(
            [], [], [],
            linewidth=7.0,
            alpha=0.65,
            color="tab:red",
            label="Source magnet axis",
        )
        triad_colours = ("tab:red", "tab:green", "tab:blue")
        self.magnet_triad_lines = [
            self.ax.plot([], [], [], linewidth=2.0, color=colour)[0]
            for colour in triad_colours
        ]
        self.magnet_triad_labels = [
            self.ax.text(
                0.0,
                0.0,
                0.0,
                label,
                color=colour,
                fontsize=8,
                weight="bold",
            )
            for label, colour in zip(("Xm", "Ym", "Zm"), triad_colours)
        ]

        self.status_text = self.fig.text(
            0.71,
            0.955,
            "Ready",
            ha="left",
            va="top",
            fontsize=10,
            family="monospace",
        )

        # Dedicated solution-information panel below the 3D plot. Contact and
        # no-contact results have separate columns, so their text cannot overlap.
        self.metrics_axis = self.fig.add_axes([0.04, 0.015, 0.64, 0.135])
        self.metrics_axis.set_xlim(0.0, 1.0)
        self.metrics_axis.set_ylim(0.0, 1.0)
        self.metrics_axis.set_xticks([])
        self.metrics_axis.set_yticks([])
        for spine in self.metrics_axis.spines.values():
            spine.set_color("0.75")
            spine.set_linewidth(0.8)

        self.contact_metrics_text = self.metrics_axis.text(
            0.015,
            0.92,
            "Contact: waiting for first solve...",
            ha="left",
            va="top",
            fontsize=8.6,
            family="monospace",
            transform=self.metrics_axis.transAxes,
        )
        self.no_contact_metrics_text = self.metrics_axis.text(
            0.515,
            0.92,
            "No contact: waiting for first solve...",
            ha="left",
            va="top",
            fontsize=8.6,
            family="monospace",
            transform=self.metrics_axis.transAxes,
        )

        self.help_text = self.fig.text(
            0.71,
            0.055,
            "Left-drag magnet: move in camera plane\n"
            "Right-drag: rotate camera | Middle-drag: pan\n"
            "Rotation mode local/body: intrinsic magnet-axis turns\n"
            "Rotation mode rotvec axis: rotate the UR vector direction",
            ha="left",
            va="bottom",
            fontsize=8.2,
            color="0.35",
        )

        self._build_controls()
        self._set_axis_limits()
        self._update_magnet_artists()
        self.ax.legend(loc="upper left", fontsize=8)

        self.fig.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_motion)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        self.poll_timer = self.fig.canvas.new_timer(interval=75)
        self.poll_timer.add_callback(self._poll_solver)
        self.poll_timer.start()

        self.request_solve(force=True)

    def _build_controls(self) -> None:
        x0, y0, z0, rx0, ry0, rz0, insertion0 = self.p7_initial
        slider_left = 0.75
        slider_width = 0.22
        slider_height = 0.024
        slider_y_values = [0.77, 0.725, 0.680, 0.635, 0.590, 0.545, 0.500]

        span = self.translation_span_m
        slider_specs = (
            ("x [m]", x0 - span, x0 + span, x0),
            ("y [m]", y0 - span, y0 + span, y0),
            ("z [m]", z0 - span, z0 + span, z0),
            ("rotvec x [rad]", -np.pi, np.pi, rx0),
            ("rotvec y [rad]", -np.pi, np.pi, ry0),
            ("rotvec z [rad]", -np.pi, np.pi, rz0),
            (
                "Insertion [m]",
                self.insertion_min_m,
                self.insertion_max_m,
                float(np.clip(insertion0, self.insertion_min_m, self.insertion_max_m)),
            ),
        )

        self.sliders: list[Slider] = []
        for y_position, (label, minimum, maximum, initial) in zip(
            slider_y_values,
            slider_specs,
        ):
            slider_axis = self.fig.add_axes(
                [slider_left, y_position, slider_width, slider_height]
            )
            slider = Slider(
                slider_axis,
                label,
                minimum,
                maximum,
                valinit=initial,
                valfmt="%.5f",
            )
            slider.on_changed(self._on_slider_changed)
            self.sliders.append(slider)

        self.model_axis = self.fig.add_axes([0.71, 0.805, 0.12, 0.095])
        self.model_radio = RadioButtons(
            self.model_axis,
            ("contact", "no_contact", "both"),
            active=("contact", "no_contact", "both").index(self.model_mode),
        )
        self.model_radio.on_clicked(self._on_model_changed)

        self.auto_axis = self.fig.add_axes([0.85, 0.82, 0.12, 0.05])
        self.auto_check = CheckButtons(self.auto_axis, ["Auto solve"], [True])

        self.translation_buttons = {}
        self.rotation_buttons = {}
        button_width = 0.055
        button_height = 0.038
        start_x = 0.71
        gap_x = 0.061

        self.fig.text(0.71, 0.455, "Translation nudges", fontsize=9, weight="bold")
        for row, axis_name in enumerate(("x", "y", "z")):
            y_position = 0.405 - row * 0.047
            for column, sign in enumerate((-1, 1)):
                label = f"{axis_name}{'-' if sign < 0 else '+'}"
                button_axis = self.fig.add_axes(
                    [start_x + column * gap_x, y_position, button_width, button_height]
                )
                button = Button(button_axis, label)
                button.on_clicked(
                    lambda _event, axis_index=row, direction=sign: self._nudge_translation(
                        axis_index, direction
                    )
                )
                self.translation_buttons[(row, sign)] = button

        # The same Rx/Ry/Rz buttons can perform either a physical local/body
        # composition or a rotation of the UR rotation-vector direction.
        self.fig.text(0.85, 0.475, "Rotation action", fontsize=9, weight="bold")
        self.rotation_mode_axis = self.fig.add_axes([0.845, 0.405, 0.13, 0.065])
        self.rotation_mode_radio = RadioButtons(
            self.rotation_mode_axis,
            ("local/body", "rotvec axis"),
            active=0,
        )
        self.rotation_mode_radio.on_clicked(self._on_rotation_mode_changed)

        rotation_start_x = 0.85
        for row, axis_name in enumerate(("Rx", "Ry", "Rz")):
            y_position = 0.345 - row * 0.047
            for column, sign in enumerate((-1, 1)):
                label = f"{axis_name}{'-' if sign < 0 else '+'}"
                button_axis = self.fig.add_axes(
                    [rotation_start_x + column * gap_x, y_position, button_width, button_height]
                )
                button = Button(button_axis, label)
                button.on_clicked(
                    lambda _event, axis_index=row, direction=sign: self._nudge_rotation(
                        axis_index, direction
                    )
                )
                self.rotation_buttons[(row, sign)] = button

        self.fig.text(0.71, 0.255, "Insertion", fontsize=9, weight="bold")
        insertion_minus_axis = self.fig.add_axes([0.71, 0.205, 0.075, 0.040])
        insertion_plus_axis = self.fig.add_axes([0.795, 0.205, 0.075, 0.040])
        self.insertion_minus_button = Button(insertion_minus_axis, "L -")
        self.insertion_plus_button = Button(insertion_plus_axis, "L +")
        self.insertion_minus_button.on_clicked(
            lambda _event: self._nudge_insertion(-1)
        )
        self.insertion_plus_button.on_clicked(
            lambda _event: self._nudge_insertion(1)
        )

        solve_axis = self.fig.add_axes([0.71, 0.135, 0.085, 0.045])
        reset_axis = self.fig.add_axes([0.805, 0.135, 0.075, 0.045])
        save_axis = self.fig.add_axes([0.890, 0.135, 0.075, 0.045])
        print_axis = self.fig.add_axes([0.71, 0.080, 0.12, 0.040])

        self.solve_button = Button(solve_axis, "Solve now")
        self.reset_button = Button(reset_axis, "Reset")
        self.save_button = Button(save_axis, "Save PNG")
        self.print_button = Button(print_axis, "Print p7")

        self.solve_button.on_clicked(lambda _event: self.request_solve(force=True))
        self.reset_button.on_clicked(lambda _event: self._reset_pose())
        self.save_button.on_clicked(lambda _event: self._save_figure())
        self.print_button.on_clicked(lambda _event: self._print_command())

    def _current_widgets_to_p7(self) -> np.ndarray:
        return np.array([slider.val for slider in self.sliders], dtype=float)

    def _set_widgets_from_p7(self, p7: np.ndarray) -> None:
        p7 = np.asarray(p7, dtype=float).reshape(7)
        self.updating_widgets = True
        try:
            for slider, value in zip(self.sliders, p7):
                value_clipped = float(np.clip(value, slider.valmin, slider.valmax))
                slider.set_val(value_clipped)
        finally:
            self.updating_widgets = False
        self.p7 = self._current_widgets_to_p7()
        self._update_magnet_artists()

    def _on_slider_changed(self, _value: float) -> None:
        if self.updating_widgets:
            return
        self.p7 = self._current_widgets_to_p7()
        self._update_magnet_artists()
        self.request_solve(force=False)

    def _on_model_changed(self, label: str) -> None:
        self.model_mode = str(label)
        self.request_solve(force=True)

    def _on_rotation_mode_changed(self, label: str) -> None:
        label = str(label)
        if label not in {"local/body", "rotvec axis"}:
            raise ValueError(f"Unknown rotation-button mode: {label!r}.")
        self._set_status(f"Rotation buttons: {label}")

    def _rotation_button_mode(self) -> str:
        return str(self.rotation_mode_radio.value_selected)

    def _nudge_translation(self, axis_index: int, direction: int) -> None:
        slider = self.sliders[int(axis_index)]
        target = slider.val + int(direction) * self.translation_step_m
        slider.set_val(float(np.clip(target, slider.valmin, slider.valmax)))

    def _nudge_rotation(self, axis_index: int, direction: int) -> None:
        """Apply the selected rotation operation about x, y, or z.

        ``local/body`` composes a physical intrinsic rotation:

            R_new = R_current @ R_increment_local

        ``rotvec axis`` rotates the UR rotation-vector as an ordinary 3-D
        vector in the robot/world frame:

            r_new = Q_axis @ r_current

        The second operation preserves ``||r||`` (the axis-angle magnitude)
        and changes only the direction of the represented rotation axis. It is
        equivalent to conjugating the orientation matrix:

            R_new = Q_axis @ R_current @ Q_axis.T

        It is not the same as applying an additional world-frame rotation.
        """
        axis_index = int(axis_index)
        direction = int(direction)
        if axis_index not in (0, 1, 2):
            raise ValueError(f"axis_index must be 0, 1, or 2; got {axis_index}.")
        if direction not in (-1, 1):
            raise ValueError(f"direction must be -1 or +1; got {direction}.")

        increment_rotvec = np.zeros(3, dtype=float)
        increment_rotvec[axis_index] = direction * self.rotation_step_rad
        increment = Rotation.from_rotvec(increment_rotvec)

        current_rotvec = np.asarray(self.p7[3:6], dtype=float).reshape(3)
        current_rotation = Rotation.from_rotvec(current_rotvec)
        mode = self._rotation_button_mode()

        if mode == "local/body":
            updated_rotation = current_rotation * increment
            updated_rotvec = updated_rotation.as_rotvec()
        elif mode == "rotvec axis":
            # Rotate the axis-angle vector itself around a fixed robot/world
            # axis. Rotation.apply performs Q_axis @ r_current.
            updated_rotvec = increment.apply(current_rotvec)

            # Protect against numerical drift in the represented angle.
            old_norm = float(np.linalg.norm(current_rotvec))
            new_norm = float(np.linalg.norm(updated_rotvec))
            if old_norm > 1.0e-12 and new_norm > 1.0e-12:
                updated_rotvec *= old_norm / new_norm
        else:
            raise RuntimeError(f"Unsupported rotation-button mode: {mode!r}.")

        p7_updated = self.p7.copy()
        p7_updated[3:6] = updated_rotvec
        self._set_widgets_from_p7(p7_updated)
        self._set_status(
            f"{mode}: rotated rotvec about robot "
            f"{'xyz'[axis_index]} by "
            f"{direction * np.rad2deg(self.rotation_step_rad):+.2f} deg"
        )
        self.request_solve(force=False)

    def _nudge_insertion(self, direction: int) -> None:
        slider = self.sliders[6]
        target = slider.val + int(direction) * self.insertion_step_m
        slider.set_val(float(np.clip(target, slider.valmin, slider.valmax)))

    def _reset_pose(self) -> None:
        self._set_widgets_from_p7(self.p7_initial)
        self.request_solve(force=True)

    def _save_figure(self) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"interactive_beam_{timestamp}.png")
        self.fig.savefig(output_path, dpi=180, bbox_inches="tight")
        self._set_status(f"Saved {output_path.resolve()}")

    def _print_command(self) -> None:
        print("\n[INTERACTIVE P7]")
        print(np.array2string(self.p7, precision=9, separator=", "))
        print("Convention: [x, y, z, rx, ry, rz, insertion]")
        self._set_status("Current p7 printed to the terminal")

    def _auto_solve_enabled(self) -> bool:
        return bool(self.auto_check.get_status()[0])

    def request_solve(self, *, force: bool) -> None:
        if self.closed:
            return
        if not force and not self._auto_solve_enabled():
            self._set_status("Pose changed; press Solve now")
            return

        self.latest_request_id += 1
        self.pending_request = (
            self.latest_request_id,
            self.p7.copy(),
            self.model_mode,
        )
        self.last_request_time = time.monotonic()
        self.force_next_solve = self.force_next_solve or bool(force)
        self._set_status("Solve queued")

    def _selected_model_names(self, mode: str) -> tuple[str, ...]:
        if mode == "both":
            return ("contact", "no_contact")
        return (mode,)

    def _solve_worker(
        self,
        request_id: int,
        p7: np.ndarray,
        mode: str,
    ) -> dict:
        start_time = time.perf_counter()
        outputs = {}

        for model_name in self._selected_model_names(mode):
            model = self.models[model_name]
            try:
                result = model.solve(p7, commit=True)
                success = bool(
                    getattr(
                        result,
                        "success",
                        (getattr(result, "info", {}) or {}).get("success", False),
                    )
                )
                centreline = _as_centreline(result.p)
                outputs[model_name] = {
                    "success": success,
                    "centreline": centreline,
                    "tip": np.asarray(result.tip, dtype=float).reshape(3).copy(),
                    "info": dict(getattr(result, "info", {}) or {}),
                    "error": None,
                }
            except Exception as exc:
                outputs[model_name] = {
                    "success": False,
                    "centreline": None,
                    "tip": None,
                    "info": {},
                    "error": f"{type(exc).__name__}: {exc}",
                }

        return {
            "request_id": int(request_id),
            "p7": np.asarray(p7, dtype=float).copy(),
            "mode": str(mode),
            "outputs": outputs,
            "elapsed_s": float(time.perf_counter() - start_time),
        }

    def _poll_solver(self) -> None:
        if self.closed:
            return

        if self.future is not None and self.future.done():
            future = self.future
            self.future = None
            try:
                payload = future.result()
            except Exception as exc:
                self._set_status(f"Worker failure: {type(exc).__name__}: {exc}")
            else:
                self.last_completed_id = int(payload["request_id"])
                self._apply_solution_payload(payload)

        if self.future is None and self.pending_request is not None:
            time_since_request = time.monotonic() - self.last_request_time
            if self.force_next_solve or time_since_request >= self.solve_delay_s:
                request_id, p7, mode = self.pending_request
                self.pending_request = None
                self.force_next_solve = False
                self._set_status(f"Solving {mode} model...")
                self.future = self.executor.submit(
                    self._solve_worker,
                    request_id,
                    p7,
                    mode,
                )

    def _apply_solution_payload(self, payload: dict) -> None:
        outputs = payload["outputs"]
        mode = payload["mode"]

        line_map = {
            "contact": (self.contact_line, self.contact_tip_marker),
            "no_contact": (self.no_contact_line, self.no_contact_tip_marker),
        }
        metrics_by_model: dict[str, str] = {
            "contact": "Contact model not selected",
            "no_contact": "No-contact model not selected",
        }

        for model_name, (line, tip_marker) in line_map.items():
            visible = model_name in self._selected_model_names(mode)
            output = outputs.get(model_name)

            if not visible or output is None:
                line.set_visible(False)
                tip_marker.set_visible(False)
                continue

            display_name = "Contact" if model_name == "contact" else "No contact"

            if not output["success"] or output["centreline"] is None:
                line.set_visible(False)
                tip_marker.set_visible(False)
                failure = output["error"] or "solve did not converge"
                metrics_by_model[model_name] = (
                    f"{display_name}: FAILED\n"
                    f"  {failure}"
                )
                continue

            centreline = output["centreline"]
            tip = output["tip"]
            line.set_data(centreline[:, 0], centreline[:, 1])
            line.set_3d_properties(centreline[:, 2])
            line.set_visible(True)
            tip_marker.set_data([tip[0]], [tip[1]])
            tip_marker.set_3d_properties([tip[2]])
            tip_marker.set_visible(True)

            metrics = self._calculate_geometry_metrics(centreline)
            metrics_by_model[model_name] = "\n".join(
                [
                    f"{display_name}",
                    f"tip [m]       [{tip[0]: .4f}, {tip[1]: .4f}, {tip[2]: .4f}]",
                    f"tip angle      {metrics['tip_angle_deg']:7.2f} deg",
                    f"tip clearance  {1e3 * metrics['tip_clearance_m']:7.3f} mm",
                    f"min clearance  {1e3 * metrics['minimum_clearance_m']:7.3f} mm",
                ]
            )

        self.contact_metrics_text.set_text(metrics_by_model["contact"])
        self.no_contact_metrics_text.set_text(metrics_by_model["no_contact"])
        self._set_status(
            f"Solved {mode} in {payload['elapsed_s']:.2f} s"
            + ("; newer pose queued" if self.pending_request is not None else "")
        )
        self._set_axis_limits(extra_outputs=outputs)
        self.fig.canvas.draw_idle()

    def _calculate_geometry_metrics(self, beam_centreline: np.ndarray) -> dict[str, float]:
        beam_centreline = _as_centreline(beam_centreline)
        tip = beam_centreline[-1]

        if beam_centreline.shape[0] >= 2:
            tip_tangent = beam_centreline[-1] - beam_centreline[-2]
            tip_tangent /= max(float(np.linalg.norm(tip_tangent)), 1.0e-12)
        else:
            tip_tangent = np.full(3, np.nan)

        difference = self.lumen_C - tip[None, :]
        distances = np.linalg.norm(difference, axis=1)
        nearest_index = int(np.argmin(distances))
        i0 = max(nearest_index - 1, 0)
        i1 = min(nearest_index + 1, self.lumen_C.shape[0] - 1)
        lumen_tangent = self.lumen_C[i1] - self.lumen_C[i0]
        lumen_tangent /= max(float(np.linalg.norm(lumen_tangent)), 1.0e-12)

        dot_value = float(np.clip(np.dot(tip_tangent, lumen_tangent), -1.0, 1.0))
        tip_angle_deg = float(np.degrees(np.arccos(dot_value)))
        tip_clearance_m = float(
            self.lumen_R[nearest_index]
            - distances[nearest_index]
            - self.beam_radius_m
        )

        pairwise = beam_centreline[:, None, :] - self.lumen_C[None, :, :]
        pairwise_distance = np.linalg.norm(pairwise, axis=2)
        nearest_indices = np.argmin(pairwise_distance, axis=1)
        node_distances = pairwise_distance[np.arange(beam_centreline.shape[0]), nearest_indices]
        node_clearance = self.lumen_R[nearest_indices] - node_distances - self.beam_radius_m

        return {
            "tip_angle_deg": tip_angle_deg,
            "tip_clearance_m": tip_clearance_m,
            "minimum_clearance_m": float(np.min(node_clearance)),
        }

    def _update_magnet_artists(self) -> None:
        position = self.p7[:3]
        rotation_matrix = Rotation.from_rotvec(self.p7[3:6]).as_matrix()

        self.magnet_center_marker.set_data([position[0]], [position[1]])
        self.magnet_center_marker.set_3d_properties([position[2]])

        triad_length = 0.020
        for axis_index, (line, label) in enumerate(
            zip(self.magnet_triad_lines, self.magnet_triad_labels)
        ):
            endpoint = position + triad_length * rotation_matrix[:, axis_index]
            line.set_data([position[0], endpoint[0]], [position[1], endpoint[1]])
            line.set_3d_properties([position[2], endpoint[2]])
            label.set_position((endpoint[0], endpoint[1]))
            label.set_3d_properties(endpoint[2])

        source_axis = rotation_matrix @ np.array([-1.0, 0.0, 0.0], dtype=float)
        source_length = 0.060
        endpoint_a = position - 0.5 * source_length * source_axis
        endpoint_b = position + 0.5 * source_length * source_axis
        self.magnet_axis_line.set_data(
            [endpoint_a[0], endpoint_b[0]],
            [endpoint_a[1], endpoint_b[1]],
        )
        self.magnet_axis_line.set_3d_properties([endpoint_a[2], endpoint_b[2]])

        self.fig.canvas.draw_idle()

    def _set_axis_limits(self, extra_outputs: dict | None = None) -> None:
        initial_position = self.p7_initial[:3]
        span = self.translation_span_m
        corner_low = initial_position - span
        corner_high = initial_position + span
        points = [self.lumen_C, corner_low[None, :], corner_high[None, :], self.p7[:3][None, :]]

        if extra_outputs:
            for output in extra_outputs.values():
                centreline = output.get("centreline") if isinstance(output, dict) else None
                if centreline is not None:
                    points.append(np.asarray(centreline, dtype=float))

        all_points = np.vstack(points)
        minimum = np.min(all_points, axis=0)
        maximum = np.max(all_points, axis=0)
        centre = 0.5 * (minimum + maximum)
        half_span = 0.5 * (maximum - minimum)
        half_span = np.maximum(half_span, 0.025)
        half_span *= 1.08

        self.ax.set_xlim(centre[0] - half_span[0], centre[0] + half_span[0])
        self.ax.set_ylim(centre[1] - half_span[1], centre[1] + half_span[1])
        self.ax.set_zlim(centre[2] - half_span[2], centre[2] + half_span[2])
        self.ax.set_box_aspect(2.0 * half_span)

    def _magnet_screen_position(self) -> tuple[float, float, float]:
        projected_x, projected_y, projected_z = proj3d.proj_transform(
            self.p7[0],
            self.p7[1],
            self.p7[2],
            self.ax.get_proj(),
        )
        display_x, display_y = self.ax.transData.transform((projected_x, projected_y))
        return float(display_x), float(display_y), float(projected_z)

    def _on_mouse_press(self, event) -> None:
        if event.inaxes is not self.ax or event.button != 1:
            return
        display_x, display_y, projected_z = self._magnet_screen_position()
        if np.hypot(event.x - display_x, event.y - display_y) <= 18.0:
            self.dragging_magnet = True
            self.drag_projection_depth = projected_z
            self._set_status("Dragging magnet in camera plane")

    def _on_mouse_motion(self, event) -> None:
        if not self.dragging_magnet or event.inaxes is not self.ax:
            return
        if event.x is None or event.y is None:
            return

        projected_x, projected_y = self.ax.transData.inverted().transform((event.x, event.y))
        inverse_projection = np.linalg.inv(self.ax.get_proj())
        world_x, world_y, world_z = proj3d.inv_transform(
            projected_x,
            projected_y,
            self.drag_projection_depth,
            inverse_projection,
        )
        translated = np.array(
            [float(np.asarray(world_x).squeeze()), float(np.asarray(world_y).squeeze()), float(np.asarray(world_z).squeeze())],
            dtype=float,
        )

        p7_updated = self.p7.copy()
        for axis_index in range(3):
            slider = self.sliders[axis_index]
            p7_updated[axis_index] = float(
                np.clip(translated[axis_index], slider.valmin, slider.valmax)
            )
        self._set_widgets_from_p7(p7_updated)
        self.request_solve(force=False)

    def _on_mouse_release(self, event) -> None:
        del event
        if self.dragging_magnet:
            self.dragging_magnet = False
            self.drag_projection_depth = None
            self.request_solve(force=True)

    def _set_status(self, message: str) -> None:
        self.status_text.set_text(str(message))
        self.fig.canvas.draw_idle()

    def _on_close(self, event) -> None:
        del event
        self.closed = True
        try:
            self.poll_timer.stop()
        except Exception:
            pass
        self.executor.shutdown(wait=False, cancel_futures=True)

    def show(self) -> None:
        plt.show()


DEFAULT_INTERACTIVE_P8 = np.array(
    [
    6.45132822e-01, -7.09927317e-01, -1.00000000e-01, -5.33680086e-04,
    -9.99999756e-01 ,-3.18309808e-04 ,-3.18309808e-04 , 1.30000000e-02
    ],
    dtype=float,
)


def parse_interactive_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open an interactive 3D magnetic-beam explorer. Move and rotate the "
            "source magnet, vary insertion, and resolve the contact/no-contact model."
        )
    )

    initial_group = parser.add_mutually_exclusive_group()
    initial_group.add_argument(
        "--p7",
        type=float,
        nargs=7,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ", "L"),
        help="Initial [x, y, z, rx, ry, rz, insertion] in metres/radians.",
    )
    initial_group.add_argument(
        "--p8",
        type=float,
        nargs=8,
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ", "L"),
        help="Initial quaternion pose [x, y, z, qw, qx, qy, qz, insertion].",
    )

    parser.add_argument(
        "--model",
        choices=("contact", "no_contact", "both"),
        default="contact",
        help="Model displayed initially. It can also be changed in the GUI.",
    )
    parser.add_argument(
        "--beam-base",
        type=float,
        nargs=6,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        default=None,
        help="Optional beam-base pose in robot coordinates [m, rad].",
    )
    parser.add_argument(
        "--lumen-base",
        type=float,
        nargs=6,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        default=None,
        help="Optional lumen-base pose in robot coordinates [m, rad].",
    )
    parser.add_argument(
        "--translation-span-mm",
        type=float,
        default=250.0,
        help="Slider range on either side of the initial source position.",
    )
    parser.add_argument(
        "--translation-step-mm",
        type=float,
        default=1.0,
        help="Distance moved by each translation button press.",
    )
    parser.add_argument(
        "--rotation-step-deg",
        type=float,
        default=2.0,
        help=(
            "Angle used by each rotation-button press. In local/body mode it "
            "is an intrinsic magnet-axis increment; in rotvec-axis mode it "
            "rotates the UR rotation-vector direction about a robot axis."
        ),
    )
    parser.add_argument(
        "--insertion-step-mm",
        type=float,
        default=1.0,
        help="Insertion change per L+ or L- button press.",
    )
    parser.add_argument(
        "--insertion-min-mm",
        type=float,
        default=5.0,
        help="Minimum insertion slider value.",
    )
    parser.add_argument(
        "--insertion-max-mm",
        type=float,
        default=80.0,
        help="Maximum insertion slider value.",
    )
    parser.add_argument(
        "--solve-delay-ms",
        type=float,
        default=200.0,
        help="Debounce time after a slider or drag update before solving.",
    )
    parser.add_argument(
        "--no-lumen-surface",
        action="store_true",
        help="Plot only the lumen centreline rather than the translucent tube.",
    )
    return parser.parse_args()


def _initial_p7_from_interactive_args(args: argparse.Namespace) -> np.ndarray:
    if args.p7 is not None:
        p7 = np.asarray(args.p7, dtype=float).reshape(7)
    else:
        p8 = (
            DEFAULT_INTERACTIVE_P8
            if args.p8 is None
            else np.asarray(args.p8, dtype=float).reshape(8)
        )
        p7 = np.asarray(pose8_quat_to_pose7_rotvec(p8), dtype=float).reshape(7)

    return magnet_pose_and_insertion_to_p7(p7[:6], float(p7[6]))


def run_interactive_explorer(args: argparse.Namespace) -> int:
    backend = str(plt.get_backend()).lower()
    if "agg" in backend and "nbagg" not in backend and "webagg" not in backend:
        raise RuntimeError(
            "Matplotlib is using a non-interactive Agg backend. Run this script "
            "from a desktop Python environment with QtAgg or TkAgg. For example, "
            "set MPLBACKEND=QtAgg before launching."
        )

    p7 = _initial_p7_from_interactive_args(args)
    scene = make_initial_poses(
        source_magnet_robot=p7[:6],
        insertion_m=float(p7[6]),
        beam_base_robot=args.beam_base,
        lumen_base_robot=args.lumen_base,
    )
    bundle = build_model_bundle2(
        beam_base_pose_robot=scene.beam_base_robot,
        lumen_base_pose_robot=scene.lumen_base_robot,
        L0=float(p7[6]),
        lumen_cfg=make_run_plot_lumen_config(),
        plant_contact=(args.model != "no_contact"),
    )

    print("\n[INTERACTIVE EXPLORER]")
    print("Initial p7:", p7)
    print("Left-drag the black magnet marker to translate it in the camera plane.")
    print("Use the sliders and buttons for exact translation, rotation and insertion.")
    print("Rotation buttons can be switched between local/body composition and rotvec-axis rotation.")

    explorer = InteractiveMagneticBeamExplorer(
        bundle=bundle,
        initial_p7=p7,
        initial_model_mode=args.model,
        translation_span_m=1.0e-3 * float(args.translation_span_mm),
        insertion_min_m=1.0e-3 * float(args.insertion_min_mm),
        insertion_max_m=1.0e-3 * float(args.insertion_max_mm),
        translation_step_m=1.0e-3 * float(args.translation_step_mm),
        rotation_step_rad=np.deg2rad(float(args.rotation_step_deg)),
        insertion_step_m=1.0e-3 * float(args.insertion_step_mm),
        solve_delay_s=1.0e-3 * float(args.solve_delay_ms),
        show_lumen_surface=not args.no_lumen_surface,
    )
    explorer.show()
    return 0


def main() -> int:
    args = parse_interactive_command_line()
    return run_interactive_explorer(args)


if __name__ == "__main__":
    raise SystemExit(main())
