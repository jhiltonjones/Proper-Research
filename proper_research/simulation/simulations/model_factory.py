from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from proper_research.simulation.simulations.scenario import (
 LumenConfig
)
from proper_research.parameters import default_magnet_params, default_beam_params

from proper_research.simulation.magnetic_beam.config import (
    BeamModelConfig,
    ContactConfig,
)
from proper_research.simulation.magnetic_beam.contact import (
    ContactParams,
    LumenQuery, 
)
from proper_research.simulation.magnetic_beam.problem import BaseFrameConfig
from proper_research.simulation.magnetic_beam.forward_model import MagneticBeamForwardModel
from proper_research.simulation.magnetic_beam.kinematics import effective_lengths

from beam_direction_magnetisation.cosserat_6d_pose import ur_pose6_to_T
from beam_direction_magnetisation.cosserat_w_minimal_energy import (
    make_lumen_centerline_turning,
)
from beam_direction_magnetisation.quarternions.quarternions_functions import (
    T_to_p_quat_wxyz,
)
from proper_research.simulation.magnetic_beam.run_solver_smoke_test import make_Kinv_fun,make_uniform_axial_m_local_factory, calculate_composite_beam_properties

mag_params = default_magnet_params()
beam_params = default_beam_params()


@dataclass(frozen=True)
class ModelBundle:
    """
    Geometry and forward-model bundle for one experiment.
    """
    p0_ur: np.ndarray
    q0_ur: np.ndarray
    lumen_C: np.ndarray
    lumen_R: np.ndarray
    lumen_path: np.ndarray
    s_path: np.ndarray
    models: dict[str, Any]


def arc_length_param(C: np.ndarray) -> np.ndarray:
    C = np.asarray(C, float)

    if C.ndim != 2 or C.shape[1] < 3:
        raise ValueError(f"C must have shape (M, >=3), got {C.shape}.")

    ds = np.linalg.norm(np.diff(C[:, :3], axis=0), axis=1)

    s = np.zeros(C.shape[0], float)
    s[1:] = np.cumsum(ds)

    return s


def resample_polyline(C: np.ndarray, ds_target: float = 1e-3):
    """
    Resample a 3D polyline to approximately uniform arc-length spacing.
    """
    C = np.asarray(C, float)

    if C.ndim != 2 or C.shape[1] < 3:
        raise ValueError(f"C must have shape (M, >=3), got {C.shape}.")

    ds_target = float(ds_target)

    if ds_target <= 0.0:
        raise ValueError("ds_target must be positive.")

    s = arc_length_param(C)
    total_length = float(s[-1])

    if total_length < 1e-12:
        return C[:, :3].copy(), s.copy()

    s_rs = np.arange(0.0, total_length + 0.5 * ds_target, ds_target)

    C_rs = np.zeros((s_rs.size, 3), float)

    for j in range(3):
        C_rs[:, j] = np.interp(s_rs, s, C[:, j])

    return C_rs, s_rs


def rod_section_stiffness(r: float, E: float, nu: float) -> dict[str, float]:
    """
    Circular rod axial, bending, and torsional stiffness values.
    """
    r = float(r)
    E = float(E)
    nu = float(nu)

    if r <= 0.0:
        raise ValueError("Rod radius r must be positive.")

    if E <= 0.0:
        raise ValueError("Young's modulus E must be positive.")

    if not (-1.0 < nu < 0.5):
        raise ValueError("Poisson ratio nu should satisfy -1 < nu < 0.5.")

    A = np.pi * r**2
    I = np.pi * r**4 / 4.0
    J = 0.5 * np.pi * r**4
    G = E / (2.0 * (1.0 + nu))

    return {
        "r": r,
        "E": E,
        "nu": nu,
        "A": A,
        "I": I,
        "J": J,
        "G": G,
        "EA": E * A,
        "EI": E * I,
        "GJ": G * J,
    }


def make_Kbt_inv_profile(
    *,
    EI_wire: float,
    EI_tip: float,
    GJ_wire: float,
    GJ_tip: float,
    bend_soft: float = 1.0,
    tors_soft: float = 1.0,
):
    """
    Build inverse bending/torsion stiffness profile.

    The returned function has signature:

        Kbt_inv_profile(s, len_wire)

    where:
        s:
            Arc-length coordinates.

        len_wire:
            Length of non-magnetized wire before the magnetic tip.
    """

    EI_wire = float(EI_wire)
    EI_tip = float(EI_tip)
    GJ_wire = float(GJ_wire)
    GJ_tip = float(GJ_tip)

    if EI_wire <= 0.0 or EI_tip <= 0.0:
        raise ValueError("EI values must be positive.")

    if GJ_wire <= 0.0 or GJ_tip <= 0.0:
        raise ValueError("GJ values must be positive.")

    def Kbt_inv_profile(s, len_wire):
        s = np.asarray(s, float)
        len_wire = float(len_wire)

        mask_tip = s >= len_wire

        EI_s = np.where(mask_tip, EI_tip, EI_wire)
        GJ_s = np.where(mask_tip, GJ_tip, GJ_wire)

        Kinv = np.zeros((3, 3, s.size), float)
        Kinv[0, 0, :] = tors_soft / GJ_s
        Kinv[1, 1, :] = bend_soft / EI_s
        Kinv[2, 2, :] = bend_soft / EI_s

        return Kinv

    return Kbt_inv_profile


def build_stiffness_profile():
    """
    Shared stiffness profile for contact and no-contact forward models.
    """
    wire = rod_section_stiffness(
        r=200e-6,
        E=50e6,
        nu=0.4,
    )

    tip = rod_section_stiffness(
        r=beam_params.r,
        E=beam_params.E,
        nu=0.49,
    )

    return make_Kbt_inv_profile(
        EI_wire=wire["EI"],
        EI_tip=tip["EI"],
        GJ_wire=wire["GJ"],
        GJ_tip=tip["GJ"],
        bend_soft=1.0,
        tors_soft=1.0,
    )


def effective_lengths(
    L_ins: float,
    *,
    L_tip_full: float = 0.04,
    L_tip_min: float = 0.01,
):
    """
    Compute model, wire, and magnetized-tip effective lengths.
    """
    L_ins = float(L_ins)

    if L_ins < 0.0:
        raise ValueError("Insertion length L_ins cannot be negative.")

    tip_len = min(L_ins, L_tip_full)
    wire_len = max(L_ins - L_tip_full, 0.0)
    L_model = max(L_ins, L_tip_min)

    tip_len = min(tip_len, L_model)

    return L_model, wire_len, tip_len


def make_contact_params() -> ContactParams:
    """
    Default smooth lumen-contact parameters.
    """
    return ContactParams(
        r_beam=0.001,
        k=1e8,
        pen_switch=5e-5,
        k_hard=1e10,
        smooth=True,
        smooth_eps=1e-5,
        window=10,
    )


def build_lumen_geometry(
    *,
    p0_ur_lumen: np.ndarray,
    q0_ur_lumen: np.ndarray,
    lumen_cfg: LumenConfig,
):
    """
    Build the lumen centreline and radius array from a
    multi-bend LumenConfig.
    """
    lumen_cfg.validate()

    p0_ur_lumen = np.asarray(
        p0_ur_lumen,
        float,
    ).reshape(3)

    q0_ur_lumen = np.asarray(
        q0_ur_lumen,
        float,
    ).reshape(4)

    R0 = Rot.from_quat(
        [
            q0_ur_lumen[1],
            q0_ur_lumen[2],
            q0_ur_lumen[3],
            q0_ur_lumen[0],
        ]
    ).as_matrix()

    t0 = R0 @ np.array(
        [-1.0, 0.0, 0.0],
        dtype=float,
    )

    lumen_C_raw = (
        make_lumen_centerline_multi_bend(
            p_start=p0_ur_lumen,
            t0=t0,
            length=float(
                lumen_cfg.length
            ),
            n_pts=int(
                lumen_cfg.n_pts
            ),
            bends=lumen_cfg.bends,
        )
    )

    lumen_C, s_path = (
        resample_polyline_fixed_count(
            lumen_C_raw,

            # Do not retain the previous hardcoded value of 40.
            n_ref_pts=int(
                lumen_cfg.n_ref_pts
            ),
        )
    )

    lumen_R = np.full(
        lumen_C.shape[0],
        float(lumen_cfg.radius),
        dtype=float,
    )

    lumen_path = lumen_C.copy()

    return (
        lumen_C,
        lumen_R,
        lumen_path,
        s_path,
    )
def make_lumen_centerline_multi_bend(
    p_start,
    t0,
    *,
    length: float,
    n_pts: int,
    bends,
):
    """
    Construct a centreline containing zero or more smooth bends.

    Bend axes are interpreted in world coordinates. Bend rotations
    accumulate in the order in which they appear in `bends`.
    """
    p_start = np.asarray(
        p_start,
        float,
    ).reshape(3)

    t0 = np.asarray(
        t0,
        float,
    ).reshape(3)

    t0_norm = np.linalg.norm(t0)

    if t0_norm <= 1e-12:
        raise ValueError("Initial lumen tangent cannot be zero.")

    t0 = t0 / t0_norm

    length = float(length)
    n_pts = int(n_pts)
    bends = tuple(bends)

    if length <= 0.0:
        raise ValueError("length must be positive.")

    if n_pts < 2:
        raise ValueError("n_pts must be at least two.")

    s = np.linspace(
        0.0,
        length,
        n_pts,
    )

    ds = np.diff(s)

    def smoothstep(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def rodrigues(vector, axis, angle):
        vector = np.asarray(
            vector,
            float,
        ).reshape(3)

        axis = np.asarray(
            axis,
            float,
        ).reshape(3)

        axis_norm = np.linalg.norm(axis)

        if axis_norm <= 1e-12:
            raise ValueError("Bend axis cannot be zero.")

        axis = axis / axis_norm

        return (
            vector * np.cos(angle)
            + np.cross(axis, vector) * np.sin(angle)
            + axis
            * np.dot(axis, vector)
            * (1.0 - np.cos(angle))
        )

    def tangent_at(s_query):
        # Begin from the initial tangent and apply every bend.
        tangent = t0.copy()

        for bend in bends:
            bend_start = float(
                bend.bend_start
            )
            bend_end = float(
                bend.bend_end
            )

            xi = (
                (s_query - bend_start)
                / (bend_end - bend_start)
            )

            progress = smoothstep(xi)

            angle = (
                float(bend.bend_angle_rad)
                * progress
            )

            tangent = rodrigues(
                tangent,
                bend.bend_axis,
                angle,
            )

        tangent_norm = np.linalg.norm(tangent)

        if tangent_norm <= 1e-12:
            raise FloatingPointError(
                "Calculated lumen tangent is zero."
            )

        return tangent / tangent_norm

    C = np.zeros(
        (n_pts, 3),
        dtype=float,
    )

    C[0] = p_start

    # Midpoint integration is more accurate than evaluating the
    # tangent only at the left endpoint.
    for i in range(n_pts - 1):
        s_mid = 0.5 * (
            s[i] + s[i + 1]
        )

        tangent_mid = tangent_at(s_mid)

        C[i + 1] = (
            C[i]
            + ds[i] * tangent_mid
        )

    return C
def resample_polyline_fixed_count(C: np.ndarray, n_ref_pts: int=60):
    """
    Resample a 3D polyline to exactly n_ref_pts equally spaced
    reference points by arclength.
    """
    C = np.asarray(C, float)

    if C.ndim != 2 or C.shape[1] < 3:
        raise ValueError(f"C must have shape (M, >=3), got {C.shape}.")

    n_ref_pts = int(n_ref_pts)

    if n_ref_pts < 2:
        raise ValueError("n_ref_pts must be at least 2.")

    s = arc_length_param(C)
    total_length = float(s[-1])

    if total_length < 1e-12:
        return C[:1, :3].repeat(n_ref_pts, axis=0), np.zeros(n_ref_pts)

    s_rs = np.linspace(0.0, total_length, n_ref_pts)

    C_rs = np.zeros((n_ref_pts, 3), float)

    for j in range(3):
        C_rs[:, j] = np.interp(s_rs, s, C[:, j])

    return C_rs, s_rs
def build_forward_model(
    *,
    p0_ur: np.ndarray,
    q0_ur: np.ndarray,
    Kinv_fun,
    m_body: np.ndarray,
    lumen_C: np.ndarray,
    lumen_R: np.ndarray,
    contact_enabled: bool,
    use_lumen_jac: bool,
    contact_params: ContactParams | None,
    m_local_factory=None
):
    """
    Build a new magnetic_beam forward model instance.

    Contact-aware model:
        contact_enabled=True
        use_lumen_jac=True
        contact_params=ContactParams(...)

    No-contact model:
        contact_enabled=False
        use_lumen_jac=False
        contact_params=None

    Notes
    -----
    use_lumen_jac is preserved for compatibility with the old experiment config.
    In the new code it maps to ContactConfig.use_in_jacobian.
    """
    p0_ur = np.asarray(p0_ur, float).reshape(3)
    q0_ur = np.asarray(q0_ur, float).reshape(4)
    m_body = np.asarray(m_body, float).reshape(3)

    lumen_C = np.asarray(lumen_C, float)
    lumen_R = np.asarray(lumen_R, float).reshape(-1)

    if lumen_C.ndim != 2 or lumen_C.shape[1] < 3:
        raise ValueError(f"lumen_C must have shape (M, >=3), got {lumen_C.shape}.")

    lumen_C = lumen_C[:, :3].copy()

    if lumen_R.size != lumen_C.shape[0]:
        raise ValueError(
            f"lumen_R length {lumen_R.size} does not match lumen_C length "
            f"{lumen_C.shape[0]}."
        )

    if contact_enabled and contact_params is None:
        raise ValueError("contact_params is required when contact_enabled=True.")

    base = BaseFrameConfig(
        p0_ur=p0_ur,
        q0_ur=q0_ur,
        u_star=np.zeros(3),
    )

    beam = BeamModelConfig(
        N_nodes=10,
        maxiter=100,
        L0_init=0.01,
        dL_internal=0.04,
        L_tip_full=0.04,
        L_tip_min=0.01,
        energy_scale=1e-8,
        u_scale=30.0,
    )
    if contact_enabled:
        contact_cfg = ContactConfig(
            enabled=True,
            use_in_jacobian=bool(use_lumen_jac),
            params=contact_params,
        )
        lumen_query = LumenQuery(lumen_C, lumen_R)
    else:
        contact_cfg = ContactConfig.disabled()
        lumen_query = None

    return MagneticBeamForwardModel(
        base=base,
        beam=beam,
        contact=contact_cfg,
        Kinv_fun=Kinv_fun,
        m_body=m_body,
        lumen_query=lumen_query,
        m_local_factory=m_local_factory,
    )


def build_model_bundle(
    *,
    pivot_point: np.ndarray,
    L0: float,
    lumen_cfg,
    plant_contact: bool = True,

    # Optional physical parameters.
    use_composite_magnetisation: bool = True,
    use_physical_stiffness: bool = False,
    effective_youngs_modulus: float = 1.0e6,
    remanence_fraction: float = 1.0,
) -> ModelBundle:
    if hasattr(lumen_cfg, "validate"):
        lumen_cfg.validate()

    pivot_point = np.asarray(
        pivot_point,
        float,
    ).reshape(6)

    T_ur_pivot = ur_pose6_to_T(
        pivot_point
    )
    p0_ur, q0_ur = T_to_p_quat_wxyz(
        T_ur_pivot
    )

    pivot_point_lumen = np.array(
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

    T_ur_pivot_lumen = ur_pose6_to_T(
        pivot_point_lumen
    )
    p0_ur_lumen, q0_ur_lumen = (
        T_to_p_quat_wxyz(
            T_ur_pivot_lumen
        )
    )

    L_model, wire_len_model, tip_len_model = (
        effective_lengths(L0)
    )

    print(
        f"L_ins={float(L0):.3f} -> "
        f"L_model={L_model:.3f}, "
        f"wire_len={wire_len_model:.3f}, "
        f"tip_len={tip_len_model:.3f}"
    )

    lumen_C, lumen_R, lumen_path, s_path = (
        build_lumen_geometry(
            p0_ur_lumen=p0_ur_lumen,
            q0_ur_lumen=q0_ur_lumen,
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
        # Legacy make_m_local_fun_wire_tip profile.
        m_local_factory = None
    m_test_fun = m_local_factory(
        L_model=L_model,
        wire_len=wire_len_model,
        tip_len=tip_len_model,
    )

    s_test = np.linspace(
        0.0,
        L_model,
        5,
    )

    m_test = np.asarray(
        m_test_fun(
            s_test,
            wire_len_model,
        ),
        float,
    )

    print("\n[COMPOSITE BEAM MAGNETISATION]")
    print(
        "  moment_per_length:",
        composite["moment_per_length"],
        "A m",
    )
    print(
        "  profile shape:",
        m_test.shape,
    )
    print(
        "  profile norms:",
        np.linalg.norm(
            m_test,
            axis=0,
        ),
    )
    print(
        "  approximate total moment:",
        composite["moment_per_length"]
        * L_model,
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
        p0_ur=p0_ur,
        q0_ur=q0_ur,
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
        p0_ur=p0_ur,
        q0_ur=q0_ur,
        lumen_C=lumen_C,
        lumen_R=lumen_R,
        lumen_path=lumen_path,
        s_path=s_path,
        models=models,
    )