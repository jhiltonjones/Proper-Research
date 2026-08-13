from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from beam_direction_magnetisation.quarternions.quarternions_functions import (
    T_to_p_quat_wxyz,
)
from beam_direction_magnetisation.quarternions.rotations import ur_pose6_to_T

from proper_research.control.lab_ready_mpc import (
    build_initial_lumen_from_vision,
    rotate_body_xy,
)
from proper_research.hardware.hardware_model_factory_optimized import (
    build_hardware_model_bundle_optimized,
)
from proper_research.hardware.hardware_runner_optimized import (
    HardwareRunConfig,
    run_hardware_control_optimized,
)
from proper_research.parameters import default_beam_params, default_magnet_params
from proper_research.robot.live_hardware_control import LiveHardwareController
from proper_research.simulation.magnetic_beam.contact import ContactParams
from proper_research.simulation.minimal_energy import rod_section_stiffness
from proper_research.simulation.simulations.controller_factory import (
    ControllerDesignConfig,
)
from proper_research.simulation.simulations.controller_factory_optimized import (
    build_controller_optimized,
)
from proper_research.simulation_controller.persistent_osqp import (
    PersistentOSQPSettings,
)
from proper_research.simulation_controller.sim_mpc_control import (
    make_Kbt_inv_profile,
)


@dataclass(frozen=True)
class HardwareControllerConfig:
    Np: int
    N_sqp: int
    solver_mode: str
    controller_kind: str = "mpc"

    def validate(self) -> None:
        if int(self.Np) <= 0:
            raise ValueError("Np must be positive.")
        if int(self.N_sqp) <= 0:
            raise ValueError("N_sqp must be positive.")
        if self.solver_mode not in {"lti", "ltv_oneshot", "sqp_full"}:
            raise ValueError(
                "solver_mode must be 'lti', 'ltv_oneshot', or 'sqp_full'."
            )


def build_physical_stiffness_profile(beam_params):
    wire = rod_section_stiffness(r=200e-6, E=50e6, nu=0.4)
    tip = rod_section_stiffness(
        r=float(beam_params.r),
        E=float(beam_params.E),
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


def smoke_test_hardware_setup(*, mpc, model_bundle) -> None:
    print("\n" + "=" * 80)
    print("Optimized hardware/model smoke test")
    print("=" * 80)
    p8 = np.asarray(mpc.p, dtype=float).reshape(int(mpc.np))
    y = np.asarray(mpc.forward_tip_fn(p8, commit=False), dtype=float).reshape(-1)
    B = np.asarray(mpc.Jxy_fn(p8), dtype=float)
    print("Controller:", type(mpc).__name__)
    print("Prediction model:", type(model_bundle.prediction_model).__name__)
    print("Jacobian model:", type(model_bundle.jacobian_model).__name__)
    print("Sensitivity workers:", model_bundle.sensitivity_workers)
    print("Forward output:", y)
    print("Discrete controller B shape:", B.shape)
    print("Reference mode:", getattr(mpc, "reference_mode", None))
    print("q_contour:", getattr(mpc, "q_contour", None))
    print("q_lag:", getattr(mpc, "q_lag", None))
    print("Progress mode:", getattr(mpc, "progress_constraint_mode", None))
    print("Forward finite:", bool(np.all(np.isfinite(y))))
    print("Jacobian finite:", bool(np.all(np.isfinite(B))))
    if not hasattr(mpc, "_select_reference_indices"):
        raise RuntimeError(
            "Controller is missing the new ReferenceMixin integration "
            "(_select_reference_indices)."
        )
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(B)):
        raise RuntimeError("Smoke test produced non-finite model output.")
    print("SMOKE TEST PASS")


def run_hardware(args: argparse.Namespace) -> None:
    live_mode = args.mode == "live"
    if live_mode and not args.send:
        raise ValueError("Live mode requires --send.")
    if args.mode != "live":
        args.send = False

    hw = LiveHardwareController(
        robot_ip=args.robot_ip,
        dry_run=False,
        use_advancer=bool(live_mode and args.send),
        advancer_port=args.advancer_port,
        advancer_baud=args.advancer_baud,
        advancer_delay_us=args.advancer_delay_us,
        advancer_min_cmd_mm=args.advancer_min_cmd_mm,
        xyz_min=(0.20, -1.50, -0.30),
        xyz_max=(1.20, +1.50, +1.50),
        max_trans_m=0.01,
        max_rot_rad=0.2,
        z_offset=float(args.z_offset),
        use_moveL_params=False,
        v=0.10,
        a=0.30,
    )
    robot_pose6 = hw.get_robot_pose_once()
    if robot_pose6 is None:
        raise RuntimeError("Could not read the current UR robot pose.")
    robot_pose6 = np.asarray(robot_pose6, dtype=float).reshape(6)

    pivot_point = np.array(
        [
            args.pivot_x,
            args.pivot_y,
            args.pivot_z,
            args.pivot_rx,
            args.pivot_ry,
            args.pivot_rz,
        ],
        dtype=float,
    )
    lumen_C, lumen_R, _ = build_initial_lumen_from_vision(
        pivot_point=pivot_point,
        image_filename=args.image_filename,
        roi_polygon_path=args.roi_polygon_path,
        blue_roi_path=args.blue_roi_path,
        green_roi_path=args.green_roi_path,
        pivot_hint=(args.pivot_hint_x, args.pivot_hint_y),
        show=False,
    )

    mag_params = default_magnet_params()
    beam_params = default_beam_params()
    T_pivot = ur_pose6_to_T(pivot_point)
    p0_ur, q0_ur = T_to_p_quat_wxyz(T_pivot)
    m_body_nominal = np.array(
        [-float(mag_params.mag_epm), 0.0, 0.0], dtype=float
    )
    m_body = rotate_body_xy(m_body_nominal, args.magnet_yaw_cal_deg)
    Kinv_fun = build_physical_stiffness_profile(beam_params)
    contact_params = ContactParams(
        r_beam=args.beam_contact_radius_m,
        k=args.contact_k,
        pen_switch=args.contact_pen_switch_m,
        k_hard=args.contact_k_hard,
        smooth=True,
        smooth_eps=args.contact_smooth_eps_m,
        window=args.contact_window,
    )

    model_bundle = build_hardware_model_bundle_optimized(
        p0_ur=p0_ur,
        q0_ur=q0_ur,
        Kinv_fun=Kinv_fun,
        m_body=m_body,
        lumen_C_robot_m=lumen_C,
        lumen_R_robot_m=lumen_R,
        contact_params=contact_params,
        jacobian_variant=args.jacobian_variant,
        sensitivity_workers=args.workers,
        result_detail="contact",
        store_history=False,
        store_vectors_in_info=False,
        copy_cached_results=False,
    )

    controller_cfg = HardwareControllerConfig(
        Np=args.Np,
        N_sqp=args.N_sqp,
        solver_mode=args.solver_mode,
    )

    # Make the new continuous path reference and contouring objective explicit
    # in the hardware entry point. This avoids silently depending on factory
    # defaults and makes contact/no-contact experiments reproducible.
    design_cfg = ControllerDesignConfig(
        n_out=6,
        w_tracking=(1000.0, 1000.0, 0.0, 0.0, 0.0, 0.0),
        reference_mode=str(args.reference_mode),
        ref_lookahead_m=float(args.ref_lookahead_mm) / 1.0e3,
        ref_stride_m=float(args.ref_stride_mm) / 1.0e3,
        ref_search_backward_m=float(args.ref_search_backward_mm) / 1.0e3,
        ref_search_forward_m=float(args.ref_search_forward_mm) / 1.0e3,
        ref_max_progress_per_step_m=float(args.ref_max_progress_mm) / 1.0e3,
        ref_weight_start=float(args.ref_weight_start),
        ref_weight_end=float(args.ref_weight_end),
        q_contour=float(args.q_contour),
        q_lag=float(args.q_lag),
        contouring_axis_weights=(1.0, 1.0, 1.0),
        progress_constraint_mode=str(args.progress_mode),
        enable_progress_slack=(args.progress_mode == "soft_slack"),
        enable_hard_progress_constraint=(args.progress_mode == "hard"),
        progress_request_m=float(args.progress_request_mm) / 1.0e3,
        progress_reward_weight=float(args.progress_reward_weight),
        progress_slack_quadratic_weight=float(
            args.progress_slack_quadratic_weight
        ),
        progress_slack_linear_weight=float(args.progress_slack_linear_weight),
        progress_slack_max_fraction=float(args.progress_slack_max_fraction),
        enable_hard_tip_tangent_angle=bool(args.enable_mpc_tip_angle_constraint),
        tip_tangent_max_angle_deg=float(args.mpc_tip_angle_deg),
        tip_tangent_activation_clearance_m=(
            float(args.mpc_tip_angle_activation_clearance_mm) / 1.0e3
        ),
    )
    design_cfg.validate()

    qp_settings = PersistentOSQPSettings(
        eps_abs=args.osqp_eps_abs,
        eps_rel=args.osqp_eps_rel,
        max_iter=args.osqp_max_iter,
        polishing=False,
        verbose=False,
    )
    controller_pack = build_controller_optimized(
        start_point=robot_pose6,
        L0=args.L0,
        dt=args.dt,
        plant_model=model_bundle.prediction_model,
        jacobian_model=model_bundle.jacobian_model,
        lumen_C=model_bundle.lumen_C,
        lumen_R=model_bundle.lumen_R,
        run_cfg=controller_cfg,
        design_cfg=design_cfg,
        copy_models=False,
        validate_nonlinear_candidate=bool(
            args.validate_candidate or args.enable_mpc_tip_angle_constraint
        ),
        collect_full_diagnostics=bool(args.full_diagnostics),
        verbose_controller=False,
        qp_settings=qp_settings,
    )
    mpc = controller_pack["mpc"]

    if args.mode == "smoke":
        smoke_test_hardware_setup(mpc=mpc, model_bundle=model_bundle)
        return

    run_config = HardwareRunConfig(
        max_steps=args.max_steps,
        solver_mode=args.solver_mode,
        rollout_steps=1,
        send_commands=bool(args.send),
        image_filename=args.image_filename,
        roi_polygon_path=args.roi_polygon_path,
        manual_boundary_path=args.manual_boundary_path,
        blue_roi_path=args.blue_roi_path,
        green_roi_path=args.green_roi_path,
        pivot_hint=(args.pivot_hint_x, args.pivot_hint_y),
        show_vision=False,
        save_vision_overlays=bool(args.save_overlays),
        save_controller_plots=bool(args.save_plots),
        output_dir=args.output_dir,
        run_name=args.run_name,
        verbose=not args.quiet,
        max_translation_step_mm=args.max_translation_step_mm,
        max_rotation_step_rad=args.max_rotation_step_rad,
        max_insertion_step_mm=args.max_insertion_step_mm,
        hard_max_translation_step_mm=args.hard_max_translation_step_mm,
        hard_max_rotation_step_rad=args.hard_max_rotation_step_rad,
        hard_max_insertion_step_mm=args.hard_max_insertion_step_mm,
        clearance_stop_mm=args.clearance_stop_mm,
        tip_angle_stop_deg=args.tip_angle_stop_deg,
        tip_angle_activation_clearance_mm=args.tip_angle_activation_clearance_mm,
        max_length_correction_per_frame_m=args.max_length_correction_mm / 1.0e3,
        use_output_bias=not args.disable_output_bias,
        collect_model_bias=True,
    )
    result = run_hardware_control_optimized(
        mpc=mpc,
        pivot_point=pivot_point,
        lumen_C_robot_m=model_bundle.lumen_C,
        lumen_R_robot_m=model_bundle.lumen_R,
        config=run_config,
        hw=hw,
        run_metadata={
            "jacobian_variant": args.jacobian_variant,
            "sensitivity_workers": args.workers,
            "controller_Np": args.Np,
            "controller_N_sqp": args.N_sqp,
            "controller_dt_s": args.dt,
            "validate_nonlinear_candidate": bool(
                args.validate_candidate or args.enable_mpc_tip_angle_constraint
            ),
            "controller_design": asdict(design_cfg),
            "contact_parameters": {
                "r_beam": args.beam_contact_radius_m,
                "k": args.contact_k,
                "pen_switch": args.contact_pen_switch_m,
                "k_hard": args.contact_k_hard,
                "smooth_eps": args.contact_smooth_eps_m,
                "window": args.contact_window,
            },
        },
    )
    print("CSV:", result["csv_path"])
    print("Summary:", result["summary_path"])
    print("Stop reason:", result["stop_reason"])


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimized fixed-mode hardware MPC with publication logging."
    )
    parser.add_argument("--mode", choices=("smoke", "shadow", "live"), default="smoke")
    parser.add_argument("--send", action="store_true", help="Send commands in live mode.")
    parser.add_argument(
        "--solver-mode",
        choices=("lti", "ltv_oneshot", "sqp_full"),
        default="lti",
    )
    parser.add_argument(
        "--jacobian-variant", choices=("contact", "no_contact"), default="contact"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--Np", type=int, default=5)
    parser.add_argument("--N-sqp", dest="N_sqp", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--L0", type=float, default=0.044)
    parser.add_argument("--max-steps", type=int, default=100)

    # Continuous path reference and contouring objective. Distances are exposed
    # in millimetres at the CLI and converted to SI units internally.
    parser.add_argument(
        "--reference-mode", choices=("point", "contouring"), default="contouring"
    )
    parser.add_argument("--q-contour", type=float, default=1000.0)
    parser.add_argument("--q-lag", type=float, default=50.0)
    parser.add_argument("--ref-lookahead-mm", type=float, default=0.5)
    parser.add_argument("--ref-stride-mm", type=float, default=0.5)
    parser.add_argument("--ref-search-backward-mm", type=float, default=0.5)
    parser.add_argument("--ref-search-forward-mm", type=float, default=5.0)
    parser.add_argument("--ref-max-progress-mm", type=float, default=1.0)
    parser.add_argument("--ref-weight-start", type=float, default=2.0)
    parser.add_argument("--ref-weight-end", type=float, default=0.5)

    parser.add_argument(
        "--progress-mode",
        choices=("none", "soft_slack", "hard"),
        default="none",
        help=(
            "Terminal progress handling. Use soft_slack for normal safe-progress "
            "control and hard for formal requested-progress feasibility tests."
        ),
    )
    parser.add_argument("--progress-request-mm", type=float, default=0.5)
    parser.add_argument("--progress-reward-weight", type=float, default=0.0)
    parser.add_argument(
        "--progress-slack-quadratic-weight", type=float, default=1000.0
    )
    parser.add_argument("--progress-slack-linear-weight", type=float, default=0.0)
    parser.add_argument("--progress-slack-max-fraction", type=float, default=1.0)

    parser.add_argument(
        "--enable-mpc-tip-angle-constraint",
        action="store_true",
        help=(
            "Enable the MPC's hard tip-tangent safety constraint. Tangent "
            "tracking weights remain zero."
        ),
    )
    parser.add_argument("--mpc-tip-angle-deg", type=float, default=40.0)
    parser.add_argument(
        "--mpc-tip-angle-activation-clearance-mm", type=float, default=1.0
    )

    parser.add_argument("--robot-ip", default="192.168.56.101")
    parser.add_argument("--z-offset", type=float, default=0.27)
    parser.add_argument("--advancer-port", default="/dev/ttyACM0")
    parser.add_argument("--advancer-baud", type=int, default=115200)
    parser.add_argument("--advancer-delay-us", type=int, default=20)
    parser.add_argument("--advancer-min-cmd-mm", type=float, default=0.166)

    parser.add_argument("--pivot-x", type=float, default=0.84813282)
    parser.add_argument("--pivot-y", type=float, default=-0.68127317)
    parser.add_argument("--pivot-z", type=float, default=-0.10000000)
    parser.add_argument("--pivot-rx", type=float, default=float(np.pi))
    parser.add_argument("--pivot-ry", type=float, default=1.0e-3)
    parser.add_argument("--pivot-rz", type=float, default=1.0e-3)
    parser.add_argument("--pivot-hint-x", type=int, default=309)
    parser.add_argument("--pivot-hint-y", type=int, default=330)
    parser.add_argument("--magnet-yaw-cal-deg", type=float, default=-10.0)

    parser.add_argument("--image-filename", default="focused_image.jpg")
    parser.add_argument(
        "--roi-polygon-path", default="/home/jack/Proper-Research/custom_area.json"
    )
    parser.add_argument(
        "--manual-boundary-path",
        default="/home/jack/Proper-Research/manual_vessel_boundaries.json",
    )
    parser.add_argument("--blue-roi-path", default="blue_roi_box.json")
    parser.add_argument("--green-roi-path", default="green_roi_box.json")

    parser.add_argument("--run-name", default="hardware_optimized")
    parser.add_argument("--output-dir", default="hardware_publication_runs")
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--validate-candidate", action="store_true")
    parser.add_argument("--full-diagnostics", action="store_true")
    parser.add_argument("--disable-output-bias", action="store_true")

    parser.add_argument("--max-translation-step-mm", type=float, default=4.0)
    parser.add_argument("--max-rotation-step-rad", type=float, default=0.04)
    parser.add_argument("--max-insertion-step-mm", type=float, default=1.2)
    parser.add_argument("--hard-max-translation-step-mm", type=float, default=6.0)
    parser.add_argument("--hard-max-rotation-step-rad", type=float, default=0.06)
    parser.add_argument("--hard-max-insertion-step-mm", type=float, default=1.8)
    parser.add_argument("--clearance-stop-mm", type=float, default=-0.5)
    parser.add_argument("--tip-angle-stop-deg", type=float, default=40.0)
    parser.add_argument(
        "--tip-angle-activation-clearance-mm", type=float, default=1.0
    )
    parser.add_argument("--max-length-correction-mm", type=float, default=0.3)

    parser.add_argument("--beam-contact-radius-m", type=float, default=0.001)
    parser.add_argument("--contact-k", type=float, default=1.0e5)
    parser.add_argument("--contact-pen-switch-m", type=float, default=5.0e-5)
    parser.add_argument("--contact-k-hard", type=float, default=1.0e10)
    parser.add_argument("--contact-smooth-eps-m", type=float, default=1.0e-5)
    parser.add_argument("--contact-window", type=int, default=3)

    parser.add_argument("--osqp-eps-abs", type=float, default=1.0e-5)
    parser.add_argument("--osqp-eps-rel", type=float, default=1.0e-5)
    parser.add_argument("--osqp-max-iter", type=int, default=100000)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1.")
    if args.q_contour < 0.0 or args.q_lag < 0.0:
        raise ValueError("--q-contour and --q-lag must be non-negative.")
    if not (0.0 <= args.progress_slack_max_fraction <= 1.0):
        raise ValueError("--progress-slack-max-fraction must lie in [0, 1].")
    for name in (
        "ref_lookahead_mm",
        "ref_search_backward_mm",
        "progress_request_mm",
    ):
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    for name in (
        "ref_stride_mm",
        "ref_search_forward_mm",
        "ref_max_progress_mm",
    ):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.ref_weight_start < 0.0 or args.ref_weight_end < 0.0:
        raise ValueError("Reference stage weights must be non-negative.")
    if (
        args.progress_reward_weight < 0.0
        or args.progress_slack_quadratic_weight < 0.0
        or args.progress_slack_linear_weight < 0.0
    ):
        raise ValueError("Progress objective/slack weights must be non-negative.")
    run_hardware(args)


if __name__ == "__main__":
    main()
