from proper_research.hardware.hierarchical_policy import (
    HierarchicalMPCPolicyConfig,
    HierarchicalMPCState,
    choose_hierarchical_mpc_action,
)
from proper_research.vision.detect_blue import image_points_to_base_local,load_manual_vessel_boundaries_with_frame
from proper_research.simulation.simulations.curvature_utils import (
    compute_discrete_path_curvature,
    curvature_ahead_metrics,
)
from proper_research.control.lab_ready_mpc import plot_reference_debug ,plot_reference_debug_simple ,dipole_dir_from_p8,build_measured_p8_from_pose6_and_length,vision_result_to_x_meas_robot, closest_index_in_window_monotone, rebuild_controller_with_new_node_count 
from proper_research.simulation.simulations.logging_utils import (
    compute_centreline_metrics,
)
import os
import csv
import numpy as np
from proper_research.vision.bounds_beam import reconstruct_beam_within_vessel, load_polygon
from proper_research.vision.measure_length import new_capture

MANUAL_VESSEL_BOUNDARY_FILE = "/home/jack/Proper-Research/manual_vessel_boundaries.json"
def update_mpc_lumen_from_camera(mpc, lumen_C_robot_m, lumen_R_robot_m):
    mpc.lumen_C = np.asarray(lumen_C_robot_m, float)
    mpc.lumen_R = np.asarray(lumen_R_robot_m, float)

    for obj in [
        getattr(mpc, "forward_tip_fn", None),
        getattr(mpc, "Jxy_fn", None),
    ]:
        if obj is None:
            continue

        # unwrap wrappers if needed
        fm = getattr(obj, "fwd", obj)

        if hasattr(fm, "set_lumen"):
            fm.set_lumen(mpc.lumen_C, mpc.lumen_R, reset_cache=True)
def configure_hardware_solver(mpc, solver_mode_current: str, N_sqp_current: int):
    """
    Map simulation hierarchy solver names onto the old hardware MPC options.

    Simulation names:
        lti
        ltv_oneshot
        sqp_full

    Hardware controller names:
        model_mode = "lti" or "ltv"
        N_sqp controls repeated SQP-like iterations
    """
    solver_mode_current = str(solver_mode_current)

    if solver_mode_current == "lti":
        mpc.model_mode = "lti"
        mpc.N_sqp = 1

    elif solver_mode_current == "ltv_oneshot":
        mpc.model_mode = "ltv"
        mpc.N_sqp = 1

    elif solver_mode_current == "sqp_full":
        mpc.model_mode = "ltv"
        mpc.N_sqp = int(N_sqp_current)

    else:
        raise ValueError(f"Unknown solver_mode_current: {solver_mode_current}")
def run_control(
    mpc,
    pivot_point,
    image_filename="focused_image.jpg",
    red_roi_path="/home/jack/Proper-Research/custom_area_w_wall.json",
    blue_roi_path="blue_roi_box.json",
    green_roi_path="green_roi_box.json",
    pivot_hint=None,
    max_steps=100,
    show=False,
    send_commands=False,
    hw=None,
    save_plots=True,
    plot_dir="mpc_debug_plots",
    lumen_C_robot_m=None,
    lumen_R_robot_m=None,
    csv_log_path="control_run_log_test_run_testing.csv",

    # Hierarchy options.
    hierarchical_mpc_enabled=False,
    hierarchical_policy_config=None,
    rollout_steps_max=3,
    solver_mode="ltv_oneshot",

    # Dynamic forward-model resolution upgrade
    enable_node_upgrade=True,
    node_upgrade_ref_idx=130,
    initial_N_nodes=12,
    upgraded_N_nodes=24,
):
    manual = load_manual_vessel_boundaries_with_frame(MANUAL_VESSEL_BOUNDARY_FILE)
    history = []
    csv_rows = []
    prev_ref_xyz = None
    prev_pred_xyz = None
    prev_i_ref = None
    L_est = None
    prev_jac_test = None
    mpc_command_buffer = []
    mpc_pred_buffer = []
    mpc_replan_every = 1
    mpc.using_upgraded_nodes = False
    mpc.N_nodes_active = int(initial_N_nodes)
    # ------------------------------------------------------------
    # Hierarchical MPC state
    # ------------------------------------------------------------
    rollout_steps_max = int(rollout_steps_max)

    if hierarchical_policy_config is None:
        hierarchical_policy_config = HierarchicalMPCPolicyConfig(
            enabled=bool(hierarchical_mpc_enabled),

            rollout_min=1,
            rollout_max=rollout_steps_max,
            ltv_rollout_cap=min(2, rollout_steps_max),
            sqp_full_iters=3,

            # Start conservative on hardware.
            curvature_warn_1pm=90.0,
            curvature_bad_1pm=190.0,

            sqp_du_rel_warn=0.5,
            sqp_du_rel_bad=2.0,

            clearance_warn_mm=2.0,
            clearance_bad_mm=1.2,

            beam_cond_warn=5e3,
            beam_cond_bad=1e5,

            mpc_cond_warn=5e7,
            mpc_cond_bad=1e8,

            downgrade_patience=2,
        )

    hierarchy_state = HierarchicalMPCState(
        rollout_steps=1,
        previous_level=0,
        downgrade_counter=0,
    )

    previous_info = {
        "status": "solved",
        "infeasible": False,
    }
    if lumen_C_robot_m is None or lumen_R_robot_m is None:
        raise ValueError("Pass fixed lumen_C_robot_m and lumen_R_robot_m into run_control")

    lumen_C_robot_m = np.asarray(lumen_C_robot_m, dtype=float)
    lumen_R_robot_m = np.asarray(lumen_R_robot_m, dtype=float).reshape(-1)

    if lumen_C_robot_m.ndim != 2 or lumen_C_robot_m.shape[1] < 3:
        raise ValueError(f"lumen_C_robot_m has invalid shape: {lumen_C_robot_m.shape}")

    if lumen_R_robot_m.shape[0] != lumen_C_robot_m.shape[0]:
        raise ValueError(
            f"lumen_R_robot_m length {lumen_R_robot_m.shape[0]} does not match "
            f"lumen_C_robot_m length {lumen_C_robot_m.shape[0]}"
        )
    path_for_curvature = lumen_C_robot_m
    path_curvature = compute_discrete_path_curvature(path_for_curvature)

    for k in range(max_steps):
        print(f"\n================ CONTROL STEP {k} ================")

        new_capture()
        overlay_path = f"debug_outputs_easter/reconstruction_step_{k:04d}.png"
        roi_polygon = load_polygon(red_roi_path)

        vision_result = reconstruct_beam_within_vessel(
            image_filename=image_filename,
            red_roi_polygon=roi_polygon,
            blue_roi_path=blue_roi_path,
            green_roi_path=green_roi_path,
            pivot_hint=pivot_hint,
            show=show,
            save_overlay_path=f"debug_outputs_run_track_dtesting/reconstruction_overlay_step_{k:04d}.png",
            base_px_ref=manual["base_px"],
            ex_ref=manual["ex_img"],
            ey_ref=manual["ey_img"],
        )

        vision_result["base_px_ref"] = manual["base_px"]
        vision_result["ex_ref"] = manual["ex_img"]
        vision_result["ey_ref"] = manual["ey_img"]
        # vision_result["lumen_C_robot_m"] = transform_local_points_to_robot(
        #     vision_result["lumen_C_m"],
        #     pivot_point
        # )
        update_mpc_lumen_from_camera(
            mpc,
            lumen_C_robot_m,
            lumen_R_robot_m,
        )
        # mpc.lumen_C = np.asarray(vision_result["lumen_C_robot_m"], dtype=float)
        # mpc.lumen_R = np.asarray(vision_result["lumen_R_m"], dtype=float)

        # print("[VISION] lumen first point robot =", mpc.lumen_C[0])
        # print("[VISION] lumen last point robot  =", mpc.lumen_C[-1])
        # print("[VISION] lumen radius min/max [mm] =",
        #     1000.0 * np.min(mpc.lumen_R),
        #     1000.0 * np.max(mpc.lumen_R))

        x_meas = vision_result_to_x_meas_robot(
            vision_result,
            pivot_point_pose6=pivot_point,
        )
        x_meas = np.asarray(x_meas, dtype=float).reshape(-1)
        meas_tip = np.asarray(x_meas[:3], dtype=float)
        meas_tan = np.asarray(x_meas[3:6], dtype=float)
        # x_meas = np.asarray(x_meas, dtype=float).reshape(-1)
        # print("same tip x_meas:", x_meas[:3])

        jac_row = {
            "jac_prev_step": np.nan,

            "jac_dx_pred_x": np.nan,
            "jac_dx_pred_y": np.nan,
            "jac_dx_pred_z": np.nan,

            "jac_dx_real_x": np.nan,
            "jac_dx_real_y": np.nan,
            "jac_dx_real_z": np.nan,

            "jac_dx_pred_x_mm": np.nan,
            "jac_dx_pred_y_mm": np.nan,
            "jac_dx_pred_z_mm": np.nan,

            "jac_dx_real_x_mm": np.nan,
            "jac_dx_real_y_mm": np.nan,
            "jac_dx_real_z_mm": np.nan,

            "jac_pred_xy_mm": np.nan,
            "jac_real_xy_mm": np.nan,
            "jac_gain_real_over_pred": np.nan,
            "jac_cos_xy": np.nan,

            "jac_err_x_mm": np.nan,
            "jac_err_y_mm": np.nan,
            "jac_err_z_mm": np.nan,
            "jac_err_xy_mm": np.nan,
            "jac_err_xyz_mm": np.nan,

            "jac_interpretation": "",
        }


        if prev_jac_test is not None:
            x_prev = prev_jac_test["x_meas"]
            u_prev = prev_jac_test["u0"]
            B_prev = prev_jac_test["B_first"]

            dx_real = x_meas[:3] - x_prev[:3]
            dx_pred = B_prev[:3, :] @ u_prev

            pred_xy = np.linalg.norm(dx_pred[:2])
            real_xy = np.linalg.norm(dx_real[:2])

            gain_real_over_pred = real_xy / (pred_xy + 1e-12)

            cos_xy = np.dot(dx_pred[:2], dx_real[:2]) / (
                pred_xy * real_xy + 1e-12
            )
            dx_jac_err = dx_real[:3] - dx_pred[:3]

            if cos_xy > 0.7 and gain_real_over_pred < 0.8:
                jac_interpretation = "direction_ok_jac_too_sensitive"
            elif cos_xy > 0.7 and gain_real_over_pred > 1.2:
                jac_interpretation = "direction_ok_jac_not_sensitive_enough"
            elif cos_xy < 0.0:
                jac_interpretation = "sign_or_frame_problem"
            else:
                jac_interpretation = "mixed_noise_backlash_delay_or_constraints"

            jac_row.update({
                "jac_prev_step": int(prev_jac_test["k"]),

                "jac_dx_pred_x": float(dx_pred[0]),
                "jac_dx_pred_y": float(dx_pred[1]),
                "jac_dx_pred_z": float(dx_pred[2]),

                "jac_dx_real_x": float(dx_real[0]),
                "jac_dx_real_y": float(dx_real[1]),
                "jac_dx_real_z": float(dx_real[2]),

                "jac_dx_pred_x_mm": float(1000.0 * dx_pred[0]),
                "jac_dx_pred_y_mm": float(1000.0 * dx_pred[1]),
                "jac_dx_pred_z_mm": float(1000.0 * dx_pred[2]),

                "jac_dx_real_x_mm": float(1000.0 * dx_real[0]),
                "jac_dx_real_y_mm": float(1000.0 * dx_real[1]),
                "jac_dx_real_z_mm": float(1000.0 * dx_real[2]),

                "jac_pred_xy_mm": float(1000.0 * pred_xy),
                "jac_real_xy_mm": float(1000.0 * real_xy),
                "jac_gain_real_over_pred": float(gain_real_over_pred),
                "jac_cos_xy": float(cos_xy),

                "jac_err_x_mm": float(1000.0 * dx_jac_err[0]),
                "jac_err_y_mm": float(1000.0 * dx_jac_err[1]),
                "jac_err_z_mm": float(1000.0 * dx_jac_err[2]),
                "jac_err_xy_mm": float(1000.0 * np.linalg.norm(dx_jac_err[:2])),
                "jac_err_xyz_mm": float(1000.0 * np.linalg.norm(dx_jac_err[:3])),

                "jac_interpretation": jac_interpretation,
            })
            print("[JAC SENSITIVITY TEST]")
            print("prev step =", prev_jac_test["k"])
            print("dx_pred xyz [mm] =", 1000.0 * dx_pred)
            print("dx_real xyz [mm] =", 1000.0 * dx_real)
            print("pred_xy [mm] =", 1000.0 * pred_xy)
            print("real_xy [mm] =", 1000.0 * real_xy)
            print("gain real/pred =", gain_real_over_pred)
            print("cos direction xy =", cos_xy)

            print("[JAC INTERPRETATION]", jac_interpretation)
        if prev_ref_xyz is not None:
            actual_err_to_prev_ref_xy = np.linalg.norm(x_meas[:2] - prev_ref_xyz[:2])
            pred_err_to_prev_ref_xy = np.linalg.norm(prev_pred_xyz[:2] - prev_ref_xyz[:2])

            print("[ACTUAL VS PREVIOUS REF]")
            print("prev_i_ref =", prev_i_ref)
            print("prev_ref_xyz =", prev_ref_xyz)
            print("current_measured_xyz =", x_meas[:3])
            print("previous_predicted_xyz =", prev_pred_xyz)
            print("actual err to prev ref xy [mm] =", 1000.0 * actual_err_to_prev_ref_xy)
            print("predicted err to prev ref xy [mm] =", 1000.0 * pred_err_to_prev_ref_xy)
            print("actual followed prediction? xy [mm] =",
                1000.0 * np.linalg.norm(x_meas[:2] - prev_pred_xyz[:2]))
        C_live = np.asarray(mpc.lumen_C, float)
        d2_live = np.sum((C_live - x_meas[:3][None, :])**2, axis=1)
        i_live = int(np.argmin(d2_live))
        print("LIVE/fixed lumen nearest dist [mm] =", 1000*np.sqrt(d2_live[i_live]))

        C_mpc = np.asarray(mpc.lumen_C, float)
        d2_mpc = np.sum((C_mpc - x_meas[:3][None, :])**2, axis=1)
        i_mpc = int(np.argmin(d2_mpc))
        print("MPC lumen nearest dist [mm] =", 1000*np.sqrt(d2_mpc[i_mpc]))
        print("pivot robot xyz:", pivot_point[:3])
        print("x_meas robot xyz:", x_meas[:3])
        print("distance from pivot [m]:", np.linalg.norm(x_meas[:3] - pivot_point[:3]))
        print("measured tangent robot:", x_meas[3:6])

        # print("[VISION] x_meas =", x_meas)
        # print("[VISION] tip position norm [m] =", np.linalg.norm(x_meas[:3]))
        # print("[VISION] tangent norm =", np.linalg.norm(x_meas[3:6]))
        # print("[VISION] lumen_C shape =", mpc.lumen_C.shape)
        # print("[VISION] lumen_R shape =", mpc.lumen_R.shape)
        # print("[VISION] closest wall distance [mm] =", vision_result["tip_distance_info_mm"]["closest_distance_mm"])
        # print("[VISION] left beam-wall angle [deg] =", vision_result["tip_wall_angle_info"]["beam_left_wall_tangent_angle_deg"])
        # print("[VISION] right beam-wall angle [deg] =", vision_result["tip_wall_angle_info"]["beam_right_wall_tangent_angle_deg"])
        # print("[DBG] i_ref_last before search =", getattr(mpc, "i_ref_last", None))
        # print("[DBG] risk_window =", getattr(mpc, "risk_window", None))
        # print("[DBG] len(lumen_C) =", len(mpc.lumen_C))
        current_robot_pose6 = None
        if hw is not None:
            current_robot_pose6 = hw.get_robot_pose_once()
        tip_xyz = x_meas[:3]
        i_ref = closest_index_in_window_monotone(
            mpc.lumen_C,
            tip_xyz,
            int(getattr(mpc, "i_ref_last", 70)),
            window=int(getattr(mpc, "risk_window", 200)),
        )
        C = np.asarray(mpc.lumen_C, float)
        d2_all = np.sum((C - tip_xyz[None, :])**2, axis=1)
        i_ref_global = int(np.argmin(d2_all))
        print("[DBG] global closest index =", i_ref_global)
        print("[DBG] global closest dist [mm] =", 1000*np.sqrt(d2_all[i_ref_global]))
        # if k == 0:
        # ------------------------------------------------------------
        # Optional dynamic model-resolution upgrade
        # ------------------------------------------------------------
        if (
            enable_node_upgrade
            and int(i_ref_global) >= int(node_upgrade_ref_idx)
            and getattr(mpc, "using_upgraded_nodes", False) is False
        ):
            if hw is None or current_robot_pose6 is None:
                raise RuntimeError(
                    "Cannot dynamically rebuild MPC without hardware robot pose. "
                    "Pass hw, or disable enable_node_upgrade."
                )

            # Use current MPC length estimate if available.
            current_L_for_rebuild = float(np.asarray(mpc.p, dtype=float).reshape(8,)[7])

            print(
                f"[MODEL] i_ref_global={i_ref_global} >= {node_upgrade_ref_idx}. "
                f"Upgrading forward model from N_nodes={getattr(mpc, 'N_nodes_active', 'unknown')} "
                f"to N_nodes={upgraded_N_nodes}."
            )

            mpc, forward6d_pred_new, p_min_new, p_max_new, u_max_new = (
                rebuild_controller_with_new_node_count(
                    old_mpc=mpc,
                    pivot_point=pivot_point,
                    current_robot_pose6=current_robot_pose6,
                    current_L=current_L_for_rebuild,
                    dt=float(mpc.dt),
                    lumen_C_robot_m=lumen_C_robot_m,
                    lumen_R_robot_m=lumen_R_robot_m,
                    N_nodes_new=upgraded_N_nodes,
                )
            )
            mpc.using_upgraded_nodes = True
            mpc.N_nodes_active = int(upgraded_N_nodes)
            # Clear open-loop plans and Jacobian diagnostics because the model changed.
            mpc_command_buffer.clear()
            mpc_pred_buffer.clear()
            prev_jac_test = None

            # Keep the reference memory at the detected location.
            mpc.i_ref_last = int(i_ref_global)

            print("[MODEL] Upgrade complete. Cleared buffered MPC commands.")
        mpc.i_ref_last = int(i_ref_global)
        # print("[VISION] i_ref =", i_ref)
        # print("[VISION] lumen point at i_ref =", mpc.lumen_C[i_ref])
        print("[VISION] tip-to-reference distance [mm] =",
            1000.0 * np.linalg.norm(x_meas[:3] - mpc.lumen_C[i_ref]))

        mag_pos_current = np.array([np.nan, np.nan, np.nan], dtype=float)
        mag_dir_current = np.array([np.nan, np.nan, np.nan], dtype=float)
        if hw is not None:
            robot_pose6 = current_robot_pose6

            L_model = float(mpc.p[7])
            L_vision_raw = float(vision_result["beam_length_mm"]) / 1000.0

            # Reject impossible vision lengths.
            if not np.isfinite(L_vision_raw) or L_vision_raw < 0.005 or L_vision_raw > 0.08:
                L_est = L_model
                L_source = "model_only_bad_vision"
            else:
                max_correction_per_frame = 0.01  # 0.3 mm
                alpha_L = 1

                L_err = L_vision_raw - L_model
                L_err_clipped = np.clip(
                    L_err,
                    -max_correction_per_frame,
                    +max_correction_per_frame,
                )

                L_est = L_model + alpha_L * L_err_clipped
                L_est = float(np.clip(L_est, 0.01, 0.05))
                L_source = "model_plus_limited_vision"

            p_meas8 = build_measured_p8_from_pose6_and_length(
                robot_pose6,
                L_est,
                z_offset=hw.z_offset,
            )

            mpc.set_measured_params(p_meas8)

            mag_pos_current = np.asarray(p_meas8[:3], dtype=float)
            mag_dir_current = np.asarray(dipole_dir_from_p8(p_meas8), dtype=float)

            print("[MEAS P] robot_pose6 =", robot_pose6)
            print("[MEAS P] L_model  [mm] =", 1e3 * L_model)
            print("[MEAS P] L_vision [mm] =", 1e3 * L_vision_raw)
            print("[MEAS P] L_est    [mm] =", 1e3 * L_est)
            print("[MEAS P] L_source =", L_source)
            print("[MEAS P] p_meas8 =", p_meas8)

        plot_reference_debug_simple(mpc, x_meas, n_ref=10)
        
        # p_now, x_now, info = mpc.step(x_meas=x_meas)


        # ------------------------------------------------------------
        # Current hierarchy/safety input from camera measurement
        # ------------------------------------------------------------
        tip_meas = np.asarray(x_meas[:3], float)
        tan_meas = np.asarray(x_meas[3:6], float) if x_meas.size >= 6 else None

        centreline_metrics_current = compute_centreline_metrics(
            lumen_C=mpc.lumen_C,
            lumen_R=mpc.lumen_R,
            tip_pos=tip_meas,
            tip_tan=tan_meas,
        )

        curvature_metrics_current = curvature_ahead_metrics(
            path=path_for_curvature,
            curvature=path_curvature,
            tip_pos=tip_meas,
            lookahead_points=10,
        )

        info_pre = dict(previous_info)
        info_pre.update(centreline_metrics_current)
        info_pre.update(curvature_metrics_current)

        info_pre["clearance_mm"] = float(info_pre.get("clearance_mm", np.nan))
        info_pre["tip_vessel_angle_deg"] = float(
            info_pre.get("tip_vessel_angle_deg", np.nan)
        )

        # ------------------------------------------------------------
        # MPC planning / hierarchy decision
        # ------------------------------------------------------------
        if len(mpc_command_buffer) == 0:
            if hierarchical_mpc_enabled:
                decision, hierarchy_state = choose_hierarchical_mpc_action(
                    info_pre=info_pre,
                    state=hierarchy_state,
                    cfg=hierarchical_policy_config,
                )

                solver_mode_current = str(decision.solver_mode)
                N_sqp_current = int(decision.N_sqp)
                n_exec_requested = int(decision.rollout_steps)

            else:
                decision = None
                solver_mode_current = str(solver_mode)
                N_sqp_current = int(getattr(mpc, "N_sqp", 1))
                n_exec_requested = int(mpc_replan_every)

            n_exec_requested = int(np.clip(
                n_exec_requested,
                1,
                rollout_steps_max,
            ))

            old_N_sqp = int(getattr(mpc, "N_sqp", N_sqp_current))
            mpc.N_sqp = int(N_sqp_current)

            try:
                p_now, x_now, info = mpc.step(
                    x_meas=x_meas,
                    rollout_steps=1,  # hardware applies rollout through command buffer
                    solver_mode=solver_mode_current,
                )
            finally:
                mpc.N_sqp = old_N_sqp

            if int(info.get("infeasible", 0)):
                U_seq = np.zeros((1, int(mpc.m)), dtype=float)
                X_pred_plan = np.asarray(x_meas, dtype=float).reshape(1, -1)
            else:
                U_seq = np.asarray(info["U_seq"], dtype=float)
                X_pred_plan = np.asarray(info["X_pred"], dtype=float)

            n_exec = min(
                n_exec_requested,
                U_seq.shape[0],
                X_pred_plan.shape[0],
            )

            if n_exec <= 0:
                raise RuntimeError("MPC returned an empty command sequence.")

            mpc_command_buffer = [U_seq[i].copy() for i in range(n_exec)]
            mpc_pred_buffer = [X_pred_plan[i, :3].copy() for i in range(n_exec)]

            if hasattr(mpc, "shift_U_guess"):
                mpc.shift_U_guess(U_seq, n_shift=n_exec)

            info["solver_mode_used"] = str(solver_mode_current)
            info["N_sqp_used"] = int(N_sqp_current)
            info["rollout_steps_used"] = int(n_exec)

            info["hierarchy_input_clearance_mm"] = float(
                info_pre.get("clearance_mm", np.nan)
            )
            info["hierarchy_input_tip_vessel_angle_deg"] = float(
                info_pre.get("tip_vessel_angle_deg", np.nan)
            )
            info["hierarchy_input_curvature_max_ahead_1pm"] = float(
                info_pre.get("curvature_max_ahead_1pm", np.nan)
            )

            if decision is not None:
                info["hierarchy_enabled"] = True
                info["hierarchy_solver_mode"] = str(decision.solver_mode)
                info["hierarchy_rollout_steps"] = int(decision.rollout_steps)
                info["hierarchy_N_sqp"] = int(decision.N_sqp)
                info["hierarchy_confidence"] = float(decision.confidence)
                info["hierarchy_risk"] = float(decision.risk)
                info["hierarchy_level"] = int(decision.level)
                info["hierarchy_reason"] = str(decision.reason)
            else:
                info["hierarchy_enabled"] = False
                info["hierarchy_solver_mode"] = ""
                info["hierarchy_rollout_steps"] = ""
                info["hierarchy_N_sqp"] = ""
                info["hierarchy_confidence"] = np.nan
                info["hierarchy_risk"] = np.nan
                info["hierarchy_level"] = ""
                info["hierarchy_reason"] = ""

            new_plan = True

        else:
            info = None
            decision = None
            new_plan = False

        u_cmd = mpc_command_buffer.pop(0)
        pred_tip = mpc_pred_buffer.pop(0)

        pred_meas_err_xyz = meas_tip - pred_tip
        pred_meas_err_xy = float(np.linalg.norm(pred_meas_err_xyz[:2]))
        if new_plan:
            # mpc.step() already propagated self.p using the first command internally
            p_now = mpc.p.copy()
            x_now = mpc.x.copy()
        else:
            # For buffered commands, manually propagate internal actuator estimate
            p_now = mpc.apply_open_loop_control(u_cmd)
            x_now = x_meas.copy()

        hardware_stop = False
        hardware_stop_reason = ""

        if not np.all(np.isfinite(u_cmd)):
            hardware_stop = True
            hardware_stop_reason = "nonfinite_u_cmd"

        if info is not None and int(info.get("infeasible", 0)):
            hardware_stop = True
            hardware_stop_reason = "mpc_infeasible"

        clearance_stop_mm = 0.8
        clearance_now_mm = float(info_pre.get("clearance_mm", np.nan))

        if np.isfinite(clearance_now_mm) and clearance_now_mm < clearance_stop_mm:
            hardware_stop = True
            hardware_stop_reason = "clearance_stop"

        if hardware_stop:
            print("[HARDWARE STOP]", hardware_stop_reason)

            if hw is not None:
                try:
                    hw.stop()
                except Exception:
                    pass

        if send_commands and not hardware_stop:
            print("[DBG] before hw.send_step")
            hw.send_step(p_now=p_now, u0=u_cmd, dt=mpc.dt)
            print("[DBG] after hw.send_step")
        if info is not None and info.get("B_first", None) is not None:
            prev_jac_test = {
                "k": int(k),
                "x_meas": x_meas.copy(),
                "u0": np.asarray(info["u0"], dtype=float).copy(),
                "B_first": np.asarray(info["B_first"], dtype=float).copy(),
            }
        else:
            prev_jac_test = None
        # print("[RUN_CONTROL INFO CHECK]")
        # print("info one_step_pred_err_xy =", info.get("one_step_pred_err_xy"))
        # print("info one_step_pred_err_xy [mm] =",
        #     1000.0 * float(info.get("one_step_pred_err_xy", np.nan)))
        L_est = float(p_now[7])
        mag_pos_next = np.asarray(p_now[:3], dtype=float)
        mag_dir_next = np.asarray(dipole_dir_from_p8(p_now), dtype=float)
        # print("[INFO DEBUG]")
        # print("info keys =", info.keys())
        # print("mpc_debug type =", type(info.get("mpc_debug")))
        # print("X_pred type =", type(info.get("X_pred")))
        # print("X_aff_last type =", type(info.get("X_aff_last")))
        if info is not None:
            track_dbg = (
                info.get("mpc_debug", {})
                    .get("penalties", {})
                    .get("track", None)

            )
        else:
            track_dbg = None
        err_to_center = np.nan
        if (
            track_dbg is not None
            and "X_ref" in track_dbg
            and info.get("X_pred", None) is not None
            and info.get("X_aff_last", None) is not None
        ):
            X_pred_arr = np.asarray(info["X_pred"], float)
            n_pred = X_pred_arr.shape[1]
            err_to_center = float(info.get("err_to_center", np.nan))
            X_ref_used = np.asarray(track_dbg["X_ref"], float).reshape(-1, n_pred)
            X_aff_stack = np.asarray(info["X_aff_last"], float).reshape(-1, n_pred)

            ref = X_ref_used[0, :3]
            pred = X_pred_arr[0, :3]
            aff = X_aff_stack[0, :3]
            meas = np.asarray(x_meas, float)[:3]

            # print("[Y FLIP CHECK]")
            # print("robot-frame dy meas-ref [mm] =", 1000.0 * (meas[1] - ref[1]))
            # print("plot/image-style dy ref-meas [mm] =", 1000.0 * (ref[1] - meas[1]))
            # print("robot-frame dx meas-ref [mm] =", 1000.0 * (meas[0] - ref[0]))
            # print("pred-ref xy [mm] =", 1000.0 * (pred[:2] - ref[:2]))

            # print("[Y SIGN CHECK]")
            # print("ref_y - aff_y [mm] =", 1000.0 * (ref[1] - aff[1]))
            # print("pred_y - aff_y [mm] =", 1000.0 * (pred[1] - aff[1]))
            # print("u0 vy =", np.asarray(info["u0"], float)[1])

            prev_ref_xyz = ref.copy()
            prev_pred_xyz = pred.copy()
            prev_i_ref = int(np.asarray(track_dbg["idx_ref"]).reshape(-1)[0])

        else:
            print("[Y FLIP CHECK] Missing track debug data")
            print("track_dbg is None:", track_dbg is None)
            print("has X_ref:", track_dbg is not None and "X_ref" in track_dbg)

            if info is not None:
                print("X_pred is None:", info.get("X_pred", None) is None)
                print("X_aff_last is None:", info.get("X_aff_last", None) is None)
            else:
                print("[TRACK DBG] skipped on open-loop buffered step")
    
    

        plot_path = None
        if save_plots:
            plot_path = os.path.join(plot_dir, f"step_{k:04d}_reference_debug.png")

        # plot_reference_debug(
        #     mpc,
        #     x_meas,
        #     info=info,
        #     n_ref=3,
        #     show_boundaries=True,
        #     save_path=plot_path,
        #     mag_pos=mag_pos_current,
        #     mag_dir=mag_dir_current,
        #     mag_pos_next=mag_pos_next,
        #     mag_dir_next=mag_dir_next,
        # )
        if info is not None:
            plot_reference_debug(
                mpc,
                x_meas,
                info=info,
                n_ref=3,
                show_boundaries=True,
                save_path=plot_path,
                mag_pos=mag_pos_current,
                mag_dir=mag_dir_current,
                mag_pos_next=mag_pos_next,
                mag_dir_next=mag_dir_next,
            )
        else:
            print("[PLOT] skipping MPC debug plot on open-loop buffered step")
        meas_tip = np.asarray(x_meas[:3], dtype=float)

        left_tan = np.asarray(
            vision_result["tip_wall_angle_info"]["left_wall_tangent_vec_cartesian"],
            dtype=float
        )

        right_tan = np.asarray(
            vision_result["tip_wall_angle_info"]["right_wall_tangent_vec_cartesian"],
            dtype=float
        )
        if info is not None:
            u0_log = np.asarray(info["u0"], dtype=float).copy()

            print("[MPC] jac cond =", info.get("jac_svd_cond", np.nan))
            print("[MPC] jac rank =", info.get("jac_svd_rank", np.nan))
            print("Previous one-step prediction error xy [mm] =",
                1000.0 * float(info.get("one_step_pred_err_xy", np.nan)))
            print("u0_proposed =", info["u0"])
            print("status =", info.get("status"))
            print("infeasible =", info.get("infeasible"))

            X_pred = np.asarray(info["X_pred"], dtype=float)
            X_nom = np.asarray(info["X_nom"], dtype=float).reshape(-1)

            # pred_tip = np.asarray(X_pred[0, :3], dtype=float)
            nom_tip = np.asarray(X_nom[:3], dtype=float)

            raw_model_err_xy = float(info.get("raw_model_err_xy", np.nan))
            nom_corr_resid_xy = float(info.get("nom_corr_resid_xy", np.nan))
            pred0_current_resid_xy = float(info.get("pred0_current_resid_xy", np.nan))
            nom_ref_err_xy = float(info.get("nom_ref_err_xy", np.nan))
            pred_ref_err_xy = float(info.get("pred_ref_err_xy", np.nan))
            one_step_pred_err_xy = float(info.get("one_step_pred_err_xy", np.nan))
            meas_ref_err_xy = float(info.get("meas_ref_err_xy", np.nan))
            meas_ref_idx = int(info.get("meas_ref_idx", -1))
            meas_ref_xyz = np.asarray(info.get("meas_ref_xyz", np.full(3, np.nan)), dtype=float).reshape(3,)
            model_bias_xyz = np.asarray(info.get("model_bias_xyz", np.full(3, np.nan)), dtype=float).reshape(3,)

        else:
            u0_log = u_cmd.copy()

            print("[MPC] open-loop buffered command")
            print("u_cmd =", u_cmd)

            # pred_tip = np.full(3, np.nan)
            nom_tip = np.full(3, np.nan)

            raw_model_err_xy = np.nan
            nom_corr_resid_xy = np.nan
            pred0_current_resid_xy = np.nan
            nom_ref_err_xy = np.nan
            pred_ref_err_xy = np.nan
            one_step_pred_err_xy = np.nan
            meas_ref_err_xy = np.nan
            meas_ref_idx = -1
            meas_ref_xyz = np.full(3, np.nan)
            model_bias_xyz = np.full(3, np.nan)

        row = {
            "step": int(k),

            "mag_x": float(mag_pos_current[0]),
            "mag_y": float(mag_pos_current[1]),
            "mag_z": float(mag_pos_current[2]),

            "pred_x": float(pred_tip[0]),
            "pred_y": float(pred_tip[1]),
            "pred_z": float(pred_tip[2]),

            "meas_x": float(meas_tip[0]),
            "meas_y": float(meas_tip[1]),
            "meas_z": float(meas_tip[2]),
            "meas_ref_err_xy": meas_ref_err_xy,
            "meas_ref_idx": meas_ref_idx,
            "meas_ref_x": float(meas_ref_xyz[0]),
            "meas_ref_y": float(meas_ref_xyz[1]),
            "meas_ref_z": float(meas_ref_xyz[2]),
            "raw_model_err_xy": raw_model_err_xy,
            "nom_corr_resid_xy": nom_corr_resid_xy,
            "pred0_current_resid_xy": pred0_current_resid_xy,
            "nom_ref_err_xy": nom_ref_err_xy,
            "pred_ref_err_xy": pred_ref_err_xy,
            "one_step_pred_err_xy": one_step_pred_err_xy,
            "pred_meas_err_xy": float(pred_meas_err_xy),
            "pred_meas_err_x": float(pred_meas_err_xyz[0]),
            "pred_meas_err_y": float(pred_meas_err_xyz[1]),
            "pred_meas_err_z": float(pred_meas_err_xyz[2]),
            "model_bias_x": float(model_bias_xyz[0]),
            "model_bias_y": float(model_bias_xyz[1]),
            "model_bias_z": float(model_bias_xyz[2]),

            "i_ref": int(mpc.i_ref_last),
            "N_nodes_active": int(getattr(mpc, "N_nodes_active", -1)),
            "using_upgraded_nodes": bool(getattr(mpc, "using_upgraded_nodes", False)),
            "left_wall_tan_x": float(left_tan[0]),
            "left_wall_tan_y": float(left_tan[1]),
            "right_wall_tan_x": float(right_tan[0]),
            "right_wall_tan_y": float(right_tan[1]),

            "dist_left_center_mm": float(vision_result["tip_distance_info_mm"]["dist_left_center_mm"]),
            "dist_right_center_mm": float(vision_result["tip_distance_info_mm"]["dist_right_center_mm"]),
            "dist_left_edge_mm": float(vision_result["tip_distance_info_mm"]["dist_left_edge_mm"]),
            "dist_right_edge_mm": float(vision_result["tip_distance_info_mm"]["dist_right_edge_mm"]),
            "closest_distance_mm": float(vision_result["tip_distance_info_mm"]["closest_distance_mm"]),

            "beam_left_wall_angle_deg": float(vision_result["tip_wall_angle_info"]["beam_left_wall_tangent_angle_deg"]),
            "beam_right_wall_angle_deg": float(vision_result["tip_wall_angle_info"]["beam_right_wall_tangent_angle_deg"]),
            "err_to_center": err_to_center,
            "hardware_stop": bool(hardware_stop),
            "hardware_stop_reason": str(hardware_stop_reason),
            "send_commands": bool(send_commands),
        }
        row.update(jac_row)
        if info is not None:
            row.update({
                "solver_mode_used": info.get("solver_mode_used", ""),
                "N_sqp_used": info.get("N_sqp_used", ""),
                "rollout_steps_used": info.get("rollout_steps_used", ""),

                "hierarchy_enabled": info.get("hierarchy_enabled", ""),
                "hierarchy_level": info.get("hierarchy_level", ""),
                "hierarchy_confidence": info.get("hierarchy_confidence", ""),
                "hierarchy_risk": info.get("hierarchy_risk", ""),
                "hierarchy_reason": info.get("hierarchy_reason", ""),

                "hierarchy_input_clearance_mm": info.get(
                    "hierarchy_input_clearance_mm",
                    "",
                ),
                "hierarchy_input_tip_vessel_angle_deg": info.get(
                    "hierarchy_input_tip_vessel_angle_deg",
                    "",
                ),
                "hierarchy_input_curvature_max_ahead_1pm": info.get(
                    "hierarchy_input_curvature_max_ahead_1pm",
                    "",
                ),

                "cond_H_beam": info.get("cond_H_beam", ""),
                "cond_H_mpc": info.get("cond_H_mpc", ""),
                "sqp_du_final": info.get("sqp_du_final", ""),
                "sqp_du_rel_final": info.get("sqp_du_rel_final", ""),

                "mpc_eig_cond": info.get("mpc_eig_cond", ""),
                "mpc_weak_channel_name": info.get("mpc_weak_channel_name", ""),
                "mpc_weak_channel_energy": info.get("mpc_weak_channel_energy", ""),
                "mpc_strong_channel_name": info.get("mpc_strong_channel_name", ""),
                "mpc_strong_channel_energy": info.get("mpc_strong_channel_energy", ""),
            })
        else:
            row.update({
                "solver_mode_used": "buffered",
                "N_sqp_used": "",
                "rollout_steps_used": "",
                "hierarchy_enabled": "",
                "hierarchy_level": "",
                "hierarchy_confidence": "",
                "hierarchy_risk": "",
                "hierarchy_reason": "",
                "hierarchy_input_clearance_mm": "",
                "hierarchy_input_tip_vessel_angle_deg": "",
                "hierarchy_input_curvature_max_ahead_1pm": "",
                "cond_H_beam": "",
                "cond_H_mpc": "",
                "sqp_du_final": "",
                "sqp_du_rel_final": "",
                "mpc_eig_cond": "",
                "mpc_weak_channel_name": "",
                "mpc_weak_channel_energy": "",
                "mpc_strong_channel_name": "",
                "mpc_strong_channel_energy": "",
            })
        csv_rows.append(row)
        # write CSV each step so you do not lose data if run stops
        fieldnames = sorted(
            set().union(*(row_i.keys() for row_i in csv_rows))
        )

        with open(csv_log_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(csv_rows)

        history.append({
            "k": int(k),
            "x_meas": x_meas.copy(),
            "x_now": np.asarray(x_now, dtype=float).copy(),
            "p_now": np.asarray(p_now, dtype=float).copy(),
            "u0_proposed": u0_log.copy(),
            "i_ref": int(mpc.i_ref_last),
            "vision_result": vision_result,
            "info": info,
        })
        if info is not None:
            previous_info = dict(info)
        if hardware_stop:
            break
        # if send_commands:
        #     print("[DBG] before hw.send_step")
        #     hw.send_step(p_now=p_now, u0=u0, dt=mpc.dt)
        #     print("[DBG] after hw.send_step")

    print(f"[SAVE] CSV log saved to {csv_log_path}")
    return history