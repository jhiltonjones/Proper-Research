from pathlib import Path
import numpy as np

from proper_research.simulation.simulations.controller_factory import build_controller
from proper_research.simulation.simulations.scenario import make_curvature_jacobian_grid
from proper_research.hardware.hierarchical_policy import (
    HierarchicalMPCPolicyConfig,
)
from proper_research.control.lab_ready_mpc import build_initial_lumen_from_vision
from .hardware_hierarchical_runner import run_control
from proper_research.robot.live_hardware_control import LiveHardwareController
from proper_research.hardware.hardware_model_factory import build_hardware_model_bundle
from proper_research.robot.live_hardware_control import LiveHardwareController
from proper_research.parameters import default_magnet_params, default_beam_params
from beam_direction_magnetisation.magnetism.beam_geometry import Kbt_inv_profile

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
    hw = LiveHardwareController()

    robot_pose6 = hw.get_robot_pose_once()
    pivot_point = robot_pose6.copy()

    lumen_C_robot_m, lumen_R_robot_m, lumen_base_robot_m = build_initial_lumen_from_vision(
        pivot_point=pivot_point,
        image_filename="focused_image.jpg",
        roi_polygon_path="/home/jack/Proper-Research/custom_area.json",
        blue_roi_path="blue_roi_box.json",
        green_roi_path="green_roi_box.json",
        pivot_hint=None,
        show=False,
    )

    mag_params = default_magnet_params()
    beam_params = default_beam_params()

    # Match your simulator convention.
    p0_ur = np.asarray(lumen_base_robot_m, float).reshape(3)

    # Use the same q0_ur convention as simulation.
    # Replace this with your actual base quaternion if you already compute it.
    q0_ur = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    m_body = np.asarray(mag_params.m_body, float).reshape(3)

    Kinv_fun = Kbt_inv_profile(
        beam_params,
        # Use the same arguments you use in simulation.
    )

    contact_params = ContactParams(
        # Use the exact same contact values as your simulator.
        # Example placeholders:
        # k_wall=...,
        # delta=...,
        # ...
    )

    model_bundle = build_hardware_model_bundle(
        p0_ur=p0_ur,
        q0_ur=q0_ur,
        Kinv_fun=Kinv_fun,
        m_body=m_body,
        lumen_C_robot_m=lumen_C_robot_m,
        lumen_R_robot_m=lumen_R_robot_m,
        contact_params=contact_params,
        jacobian_variant="contact",
    )

    run_cfg = make_hardware_controller_config(
        Np=5,
        N_sqp=5,
        max_steps=100,
        rollout_steps=2,
        solver_mode="lti",
    )

    L0 = 0.02
    dt = 0.05

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

    hierarchy_cfg = HierarchicalMPCPolicyConfig(
        enabled=True,

        rollout_min=1,
        rollout_max=2,
        ltv_rollout_cap=2,
        sqp_full_iters=3,

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

    history = run_control(
        mpc=mpc,
        pivot_point=pivot_point,

        image_filename="focused_image.jpg",
        red_roi_path="/home/jack/Proper-Research/custom_area_w_wall.json",
        blue_roi_path="blue_roi_box.json",
        green_roi_path="green_roi_box.json",

        max_steps=100,
        show=False,

        send_commands=False,
        hw=hw,

        save_plots=True,
        plot_dir="hardware_hierarchy_debug_plots",

        lumen_C_robot_m=model_bundle.lumen_C,
        lumen_R_robot_m=model_bundle.lumen_R,

        csv_log_path="hardware_hierarchy_log.csv",

        hierarchical_mpc_enabled=True,
        hierarchical_policy_config=hierarchy_cfg,
        rollout_steps_max=2,
        solver_mode="lti",

        enable_node_upgrade=True,
        node_upgrade_ref_idx=130,
        initial_N_nodes=10,
        upgraded_N_nodes=24,
    )

    print(f"Finished hardware run with {len(history)} steps.")


if __name__ == "__main__":
    main()