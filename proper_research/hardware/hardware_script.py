from pathlib import Path
import numpy as np

from proper_research.simulation.simulations.controller_factory import build_controller
from proper_research.simulation.simulations.scenario import make_curvature_jacobian_grid
from proper_research.hardware.hierarchical_policy import (
    HierarchicalMPCPolicyConfig,
)
from proper_research.control.lab_ready_mpc import rotate_body_xy,build_initial_lumen_from_vision, pose8_quat_to_pose7_rotvec
from .hardware_hierarchical_runner import run_control
from proper_research.robot.live_hardware_control import LiveHardwareController
from proper_research.hardware.hardware_model_factory import build_hardware_model_bundle
from proper_research.parameters import default_magnet_params, default_beam_params
from proper_research.simulation.magnetic_beam.config import ContactParams
from proper_research.simulation_controller.sim_mpc_control import make_Kbt_inv_profile
import argparse
from beam_direction_magnetisation.quarternions.rotations import ur_pose6_to_T
from proper_research.simulation.minimal_energy import rod_section_stiffness
from beam_direction_magnetisation.quarternions.quarternions_functions import T_to_p_quat_wxyz
                                                                              
def smoke_test_hardware_setup(
    *,
    hw,
    mpc,
    model_bundle,
    robot_pose6,
    lumen_C_robot_m,
    lumen_R_robot_m,
    L0,
):
    print("\n" + "=" * 80)
    print("[SMOKE] Hardware/model setup test")
    print("=" * 80)

    print("[SMOKE] robot_pose6 =", np.asarray(robot_pose6, float))
    print("[SMOKE] lumen_C shape =", np.asarray(lumen_C_robot_m).shape)
    print("[SMOKE] lumen_R shape =", np.asarray(lumen_R_robot_m).shape)
    print("[SMOKE] lumen_R min/max [mm] =",
          1e3 * float(np.nanmin(lumen_R_robot_m)),
          1e3 * float(np.nanmax(lumen_R_robot_m)))

    p0 = np.asarray(mpc.p, float).reshape(-1) if getattr(mpc, "p", None) is not None else None
    print("[SMOKE] mpc.p exists:", p0 is not None)

    if p0 is None:
        print("[SMOKE] mpc.p is None; using controller initial p0 from controller_pack may be required.")
        return

    # Test prediction model forward call.
    p7 = np.asarray(p0[:7], float) if p0.size == 7 else None

    # If controller uses p8 quaternion internally, adapt to p7.
    if p0.size == 8:
        p7 = pose8_quat_to_pose7_rotvec(p0)
    elif p0.size == 7:
        p7 = p0.copy()
    else:
        raise ValueError(f"Unexpected mpc.p shape: {p0.shape}")

    y = np.asarray(model_bundle.plant_model(p7), float).reshape(-1)
    print("[SMOKE] forward output shape =", y.shape)
    print("[SMOKE] forward tip =", y[:3])

    J = np.asarray(
        model_bundle.jacobian_model.jacobian_tip_pose7(
            p7,
            solve_if_needed=True,
        ),
        float,
    )
    print("[SMOKE] J shape =", J.shape)
    print("[SMOKE] J finite =", bool(np.all(np.isfinite(J))))

    diag = model_bundle.jacobian_model.get_last_diag()
    print("[SMOKE] cond_H_beam =", diag.get("cond_H_beam", np.nan))
    print("[SMOKE] H_beam shape =",
          diag.get("H_beam_shape_0", None),
          diag.get("H_beam_shape_1", None))

    print("[SMOKE] PASS")
def run_hardware_test(
    *,
    mode: str,
    max_steps: int = 5,
    robot_ip: str = "192.168.56.101",
    pivot_hint=None,
    z_offset: float = 0.0,
    send_commands: bool = False,
):
    if mode.endswith("_live") and not send_commands:
        raise ValueError("Live mode requested but --send was not passed.")

    if mode.endswith("_shadow"):
        send_commands = False

    if mode in ("smoke", "lti_contact_shadow", "lti_contact_live"):
        jacobian_variant = "contact"
    elif mode in ("lti_no_contact_shadow", "lti_no_contact_live"):
        jacobian_variant = "no_contact"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    is_live_mode = mode in ("lti_contact_live", "lti_no_contact_live")

    if is_live_mode and not send_commands:
        raise ValueError("Live mode requested, but --send was not passed.")

    # Smoke and shadow modes still need robot pose, so dry_run must be False.
    # But they should not open/use the advancer because they are not sending commands.
    hw = LiveHardwareController(
        robot_ip=robot_ip,
        dry_run=False,
        use_advancer=bool(is_live_mode and send_commands),
        advancer_port="/dev/ttyACM0",
        advancer_baud=115200,
        advancer_delay_us=20,
        advancer_min_cmd_mm=0.166,
        xyz_min=(0.20, -1.50, -0.30),
        xyz_max=(1.20, +1.50, +1.50),
        max_trans_m=0.01,
        max_rot_rad=0.2,
        z_offset=float(z_offset),
        use_moveL_params=False,
        v=0.10,
        a=0.30,
    )

    robot_pose6 = hw.get_robot_pose_once()

    if robot_pose6 is None:
        raise RuntimeError(
            "Robot pose is None. This happens when LiveHardwareController.dry_run=True. "
            "For hardware smoke/shadow tests you need dry_run=False so the script can read "
            "the current UR pose."
        )

    robot_pose6 = hw.get_robot_pose_once()
    robot_pose6 = np.asarray(robot_pose6, dtype=float).reshape(6,)

    pivot_point = np.array([
        0.84813282,
        -0.68127317,
        -0.10000000,
        np.pi,
        1.0e-3,
        1.0e-3,
    ], dtype=float)

    lumen_C_robot_m, lumen_R_robot_m, lumen_base_robot_m = build_initial_lumen_from_vision(
        pivot_point=pivot_point,
        image_filename="focused_image.jpg",
        roi_polygon_path="/home/jack/Proper-Research/custom_area.json",
        blue_roi_path="blue_roi_box.json",
        green_roi_path="green_roi_box.json",
        pivot_hint = (309,330),
        show=False,
    )

    mag_params = default_magnet_params()
    beam_params = default_beam_params()


    T_ur_pivot = ur_pose6_to_T(np.asarray(pivot_point, dtype=float).reshape(6,))
    p0_ur, q0_ur = T_to_p_quat_wxyz(T_ur_pivot)
    MAG_YAW_CAL_DEG = -10.0

    m_body_nominal = np.array(
        [-float(mag_params.mag_epm), 0.0, 0.0],
        dtype=float,
    )

    m_body = rotate_body_xy(m_body_nominal, MAG_YAW_CAL_DEG)

    wire = rod_section_stiffness(
        r=200e-6,
        E=50e6,
        nu=0.4,
    )
    # tip = rod_section_stiffness(
    #     r=2e-3,
    #     E=3e6,
    #     nu=0.49,
    # )
    tip = rod_section_stiffness(
        r=beam_params.r,
        E=beam_params.E,
        nu=0.49,
    )
    EA_wire = wire["EA"]
    EI_wire = wire["EI"]
    GJ_wire = wire["GJ"]

    EA_tip = tip["EA"]
    EI_tip = tip["EI"]
    GJ_tip = tip["GJ"]

    Kinv_fun = make_Kbt_inv_profile(
        EI_wire=EI_wire,
        EI_tip=EI_tip,
        GJ_wire=GJ_wire,
        GJ_tip=GJ_tip,
        bend_soft=1.0,
        tors_soft=1.0,
    )
    contact_params = ContactParams(
        r_beam=0.001,
        k=1e5,
        pen_switch=5e-5,
        k_hard=1e10,
        smooth=True,
        smooth_eps=1e-5,
        window=3,
    )

    model_bundle = build_hardware_model_bundle(
        p0_ur=p0_ur,
        q0_ur=q0_ur,
        Kinv_fun=Kinv_fun,
        m_body=m_body,
        lumen_C_robot_m=lumen_C_robot_m,
        lumen_R_robot_m=lumen_R_robot_m,
        contact_params=contact_params,
        jacobian_variant=jacobian_variant,
    )

    L0 = 0.027
    dt = 0.01

    run_cfg = make_hardware_controller_config(
        Np=1,
        N_sqp=1,
        max_steps=max_steps,
        rollout_steps=1,
        solver_mode="lti",
    )

    controller_pack = build_controller(
        start_point=robot_pose6,
        L0=L0,
        dt=dt,
        plant_model=model_bundle.plant_model,
        jacobian_model=model_bundle.jacobian_model,
        lumen_C=model_bundle.lumen_C,
        lumen_R=model_bundle.lumen_R,
        run_cfg=run_cfg,
    )

    mpc = controller_pack["mpc"]

    if mode == "smoke":
        smoke_test_hardware_setup(
            hw=hw,
            mpc=mpc,
            model_bundle=model_bundle,
            robot_pose6=robot_pose6,
            lumen_C_robot_m=lumen_C_robot_m,
            lumen_R_robot_m=lumen_R_robot_m,
            L0=L0,
        )
        return

    history = run_control(
        mpc=mpc,
        pivot_point=pivot_point,

        image_filename="focused_image.jpg",
        red_roi_path="/home/jack/Proper-Research/custom_area.json",
        blue_roi_path="blue_roi_box.json",
        green_roi_path="green_roi_box.json",

        max_steps=max_steps,
        show=False,

        send_commands=send_commands,
        hw=hw,

        save_plots=True,
        plot_dir=f"hardware_debug_{mode}",

        lumen_C_robot_m=model_bundle.lumen_C,
        lumen_R_robot_m=model_bundle.lumen_R,

        csv_log_path=f"hardware_{mode}.csv",

        hierarchical_mpc_enabled=False,
        hierarchical_policy_config=None,
        rollout_steps_max=1,
        solver_mode="lti",
        pivot_hint = (309,330),
        enable_node_upgrade=False,
        node_upgrade_ref_idx=130,
        initial_N_nodes=10,
        upgraded_N_nodes=24,
    )

    print(f"[DONE] {mode}: {len(history)} steps")
def make_hardware_run_cfg(
    *,
    Np=5,
    N_sqp=5,
    max_steps=100,
    rollout_steps=2,
    solver_mode="lti",
):
    """
    Reuse the simulator ControllerConfig machinery without using the simulator plant.
    """
    exp_cfg = make_curvature_jacobian_grid(
        run_root=Path("dummy_hardware_config"),
        bend_angles_deg=(-30.0,),
        jacobian_variants=("contact",),
        solver_modes=(solver_mode,),
        rollout_steps_values=(rollout_steps,),
        Np=Np,
        N_sqp=N_sqp,
        max_steps=max_steps,
        plant_contact=True,
        adaptive_rollout_enabled=False,
    )[0]

    return exp_cfg.controller


def build_sim_controller_for_hardware(
    *,
    pivot_point,
    robot_pose6,
    L0,
    dt,
    lumen_C_robot_m,
    lumen_R_robot_m,
    prediction_model,
    jacobian_model,
    run_cfg,
):
    """
    Build the same MPCControllerTipXY used in simulation, but using
    the hardware lumen and hardware prediction model.
    """
    start_point = np.asarray(robot_pose6, dtype=float).reshape(6,)

    controller_pack = build_controller(
        start_point=start_point,
        L0=float(L0),
        dt=float(dt),
        plant_model=prediction_model,
        jacobian_model=jacobian_model,
        lumen_C=np.asarray(lumen_C_robot_m, dtype=float),
        lumen_R=np.asarray(lumen_R_robot_m, dtype=float),
        run_cfg=run_cfg,
    )

    return controller_pack
def make_hardware_controller_config(
    *,
    Np=5,
    N_sqp=5,
    max_steps=100,
    rollout_steps=2,
    solver_mode="lti",
):
    """
    Reuse the same ControllerConfig path as simulation without running a
    simulation grid.
    """
    exp_cfg = make_curvature_jacobian_grid(
        run_root=Path("dummy_hardware_config"),
        bend_angles_deg=(-30.0,),
        jacobian_variants=("contact",),
        solver_modes=(solver_mode,),
        rollout_steps_values=(rollout_steps,),
        Np=Np,
        N_sqp=N_sqp,
        max_steps=max_steps,
        plant_contact=True,
        adaptive_rollout_enabled=False,
    )[0]

    return exp_cfg.controller

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "smoke",
            "lti_contact_shadow",
            "lti_no_contact_shadow",
            "lti_contact_live",
            "lti_no_contact_live",
        ],
        default="smoke",
    )

    parser.add_argument("--max-steps", type=int, default=5)

    parser.add_argument(
        "--robot-ip",
        type=str,
        default="192.168.56.101",
        help="UR robot IP address.",
    )

    parser.add_argument(
        "--pivot-hint-x",
        type=int,
        default=318,
    )

    parser.add_argument(
        "--pivot-hint-y",
        type=int,
        default=329,
    )

    parser.add_argument(
        "--z-offset",
        type=float,
        default=0.27,
    )

    parser.add_argument(
        "--send",
        action="store_true",
        help="Actually send robot/advancer commands. Shadow tests ignore this.",
    )

    args = parser.parse_args()

    run_hardware_test(
        mode=args.mode,
        max_steps=args.max_steps,
        robot_ip=args.robot_ip,
        pivot_hint=(args.pivot_hint_x, args.pivot_hint_y),
        z_offset=args.z_offset,
        send_commands=bool(args.send),
    )


if __name__ == "__main__":
    main()