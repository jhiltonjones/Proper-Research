import json
import traceback
from pathlib import Path

import numpy as np

from proper_research.simulation.simulations.scenario import (
    ExperimentConfig,
    make_curvature_jacobian_grid, LumenConfig, LumenBend
)
from proper_research.simulation.simulations.controller_factory import ControllerDesignConfig
from proper_research.simulation.simulations.model_factory import build_model_bundle
from proper_research.simulation.simulations.controller_factory import build_controller
from proper_research.simulation.simulations.runner import run_simulation
from proper_research.simulation.simulations.logging_utils import setup_output_dirs
from proper_research.simulation.simulations.initial_conditions import make_initial_poses
from proper_research.hardware.evidence_based_hierarchical_policy import (
    HierarchicalMPCPolicyConfig,
)
from proper_research.simulation.simulations.rollout_diagnostics import (
    reference_spacing_report,
    print_reference_spacing_report,
)
def save_experiment_metadata(
    *,
    exp_cfg: ExperimentConfig,
    out_root: Path,
    dt: float,
    L0: float,
    p0_ur,
    p0,
    p_min,
    p_max,
    u_max,
):
    """
    Save one experiment's full metadata.

    This is deliberately separate from setup_output_dirs() so that the
    experiment config and controller/initial-condition metadata are both saved.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    meta = exp_cfg.to_dict()

    meta.update(
        {
            "dt": float(dt),
            "L0": float(L0),
            "p0_ur": np.asarray(p0_ur, float).reshape(-1).tolist(),
            "p0": np.asarray(p0, float).reshape(-1).tolist(),
            "p_min": np.asarray(p_min, float).reshape(-1).tolist(),
            "p_max": np.asarray(p_max, float).reshape(-1).tolist(),
            "u_max": np.asarray(u_max, float).reshape(-1).tolist(),
        }
    )

    with open(out_root / "experiment_config.json", "w") as f:
        json.dump(meta, f, indent=2)


def run_experiment(
    exp_cfg: ExperimentConfig,
    design_cfg: ControllerDesignConfig,
    *,
    save_plots: bool = True,
    hierarchical_mpc_enabled: bool = False,
    hierarchical_policy_config=None,
):
    """
    Run one experiment.

    The model decomposition is:

        plant_model:
            actual simulation model used for rollout/application

        jacobian_model:
            model used internally by MPC to compute local Jacobians

    For your main study:
        plant_model    = contact-aware truth
        jacobian_model = contact or no_contact
    """
    exp_cfg.validate()

    out_root = exp_cfg.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[RUN] {exp_cfg.run_name}")

    pivot_point, start_point, L0, dt = make_initial_poses()

    bundle = build_model_bundle(
        pivot_point=pivot_point,
        L0=L0,
        lumen_cfg=exp_cfg.lumen,
        plant_contact=exp_cfg.model.plant_contact,
    )
    # report = reference_spacing_report(
    #     bundle.lumen_C,
    #     ref_stride_pts=design_cfg.ref_stride_pts,
    #     ref_lookahead_pts=design_cfg.ref_lookahead_pts,
    #     Np=exp_cfg.controller.Np,
    # )

    # print_reference_spacing_report(report)
    plant_model = bundle.models["plant"]
    jacobian_model = bundle.models[exp_cfg.model.jacobian_variant]

    controller_pack = build_controller(
        start_point=start_point,
        L0=L0,
        dt=dt,
        plant_model=plant_model,
        jacobian_model=jacobian_model,
        lumen_C=bundle.lumen_C,
        lumen_R=bundle.lumen_R,
        run_cfg=exp_cfg.controller,
        design_cfg=design_cfg,
    )
    J_fn = controller_pack["J_fn"]
    owned_jacobian_model = J_fn.model

    contact_cfg = owned_jacobian_model.contact_cfg

    actual = {
        "variant": str(exp_cfg.model.jacobian_variant),
        "model_type": type(owned_jacobian_model).__name__,
        "contact_forward_enabled": bool(contact_cfg.enabled),
        "contact_in_jacobian": bool(contact_cfg.use_in_jacobian),
        "lumen_attached": owned_jacobian_model.lumen_query is not None,
        "copied_model_id": id(owned_jacobian_model),
        "bundle_model_id": id(jacobian_model),
    }


    controller = controller_pack["controller"]
    p0 = controller_pack["p0"]
    p_min = controller_pack["p_min"]
    p_max = controller_pack["p_max"]
    u_max = controller_pack["u_max"]
    forward6d = controller_pack["forward6d_plant"]
    is_mpc = (
        exp_cfg.controller.controller_kind == "mpc"
    )

    if not is_mpc and hierarchical_mpc_enabled:
        raise ValueError(
            "hierarchical_mpc_enabled cannot currently be used "
            "with the inverse-Jacobian controller. The hierarchy "
            "depends on MPC/SQP diagnostics."
        )

    if (
        not is_mpc
        and exp_cfg.controller.adaptive_rollout_enabled
    ):
        raise ValueError(
            "adaptive_rollout_enabled cannot currently be used "
            "with the inverse-Jacobian baseline. Use fixed rollout "
            "lengths for the comparison study."
        )
    frames_dir, log_csv_path = setup_output_dirs(
        out_root=out_root,
        lumen_C=bundle.lumen_C,
        lumen_R=bundle.lumen_R,
        mpc=controller,  # Existing parameter name retained.
        u_max=u_max,
        p_min=p_min,
        p_max=p_max,
        save_plots=save_plots,
    )

    if log_csv_path.exists():
        log_csv_path.unlink()

    save_experiment_metadata(
        exp_cfg=exp_cfg,
        out_root=out_root,
        dt=dt,
        L0=L0,
        p0_ur=bundle.p0_ur,
        p0=p0,
        p_min=p_min,
        p_max=p_max,
        u_max=u_max,
    )

    runtime_solver_mode = (
        exp_cfg.controller.solver_mode
        if is_mpc
        else "lti"
    )
    # controller.ref_lookahead_m = (
    #     design_cfg.ref_lookahead_pts
    # )

    # controller.ref_stride_m = (
    #     design_cfg.ref_stride_pts
    # )
    controller.jacobian_diag_input_scale = np.asarray(
        design_cfg.trust_radius,
        float,
    ).copy()

    stats = run_simulation(
        mpc=controller,  # Existing runner parameter name.
        forward6d=forward6d,
        p0_ur=bundle.p0_ur,
        p0=p0,
        lumen_C=bundle.lumen_C,
        lumen_R=bundle.lumen_R,
        lumen_path=bundle.lumen_path,
        s_path=bundle.s_path,
        frames_dir=frames_dir,
        log_csv_path=log_csv_path,
        max_steps=exp_cfg.controller.max_steps,
        rollout_steps=exp_cfg.controller.rollout_steps,
        solver_mode=runtime_solver_mode,
        save_plots=save_plots,

        diagnostic_plots_enabled=True,
        diagnostic_plot_stride=1,
        diagnostic_plot_dpi=160,
        adaptive_rollout_enabled=(
            exp_cfg.controller.adaptive_rollout_enabled
            if is_mpc
            else False
        ),
        hierarchical_mpc_enabled=(
            hierarchical_mpc_enabled
            if is_mpc
            else False
        ),
        hierarchical_policy_config=(
            hierarchical_policy_config
            if is_mpc
            else None
        ),
        enable_tip_tangent_stop=True,
        tip_tangent_stop_deg=40.0,

        # None means inherit the controller value.
        tip_tangent_stop_clearance_m=None,

        # None means inherit mpc.tip_radius_m.
        tip_radius_m=None,

        tip_tangent_invalid_is_stop=True,
    )

    summary = summarize_stats(
        exp_cfg=exp_cfg,
        stats=stats,
        log_csv_path=log_csv_path,
    )

    with open(out_root / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "config": exp_cfg,
        "stats": stats,
        "summary": summary,
        "out_root": out_root,
        "log_csv_path": log_csv_path,
    }


def summarize_stats(
    *,
    exp_cfg: ExperimentConfig,
    stats: dict,
    log_csv_path: Path,
) -> dict:
    """
    Lightweight summary from in-memory stats.

    More detailed analysis can be done later from log.csv.
    """
    status_hist = stats.get("status", [])
    infeasible_hist = stats.get("infeasible", [])
    i_ref_hist = stats.get("i_ref", [])
    tip_hist = stats.get("tip", [])
    sqp_iters = stats.get("sqp_iters", [])
    jac_cond = stats.get("jac_cond", [])

    n_steps = len(status_hist)

    if n_steps > 0:
        final_status = str(status_hist[-1])
        final_infeasible = int(infeasible_hist[-1])
        final_i_ref = int(i_ref_hist[-1])
    else:
        final_status = "empty"
        final_infeasible = -1
        final_i_ref = -1

    if len(tip_hist) > 0:
        final_tip = np.asarray(tip_hist[-1], float).reshape(-1).tolist()
    else:
        final_tip = []
    controller_kind = str(
        exp_cfg.controller.controller_kind
    )

    inverse_mode = (
        str(exp_cfg.controller.inverse_sequence_mode)
        if controller_kind == "inverse_jacobian"
        else ""
    )
    return {
        "run_name": exp_cfg.run_name,
        "log_csv_path": str(log_csv_path),

        # "bend_angle_deg": float(
        #     exp_cfg.lumen.bend_angles_deg
        # ),
        "plant_contact": bool(
            exp_cfg.model.plant_contact
        ),
        "jacobian_variant": str(
            exp_cfg.model.jacobian_variant
        ),

        "controller_kind": controller_kind,
        "inverse_sequence_mode": inverse_mode,
        "inverse_kp": (
            float(exp_cfg.controller.inverse_kp)
            if controller_kind == "inverse_jacobian"
            else np.nan
        ),

        "solver_mode": (
            str(exp_cfg.controller.solver_mode)
            if controller_kind == "mpc"
            else ""
        ),

        "rollout_steps": int(
            exp_cfg.controller.rollout_steps
        ),
        "Np": int(exp_cfg.controller.Np),

        "N_sqp": (
            int(exp_cfg.controller.N_sqp)
            if controller_kind == "mpc"
            else 0
        ),

        "n_steps": int(n_steps),
        "final_status": final_status,
        "final_infeasible": int(final_infeasible),
        "final_i_ref": int(final_i_ref),
        "final_tip": final_tip,

        "num_infeasible_steps": int(np.sum(np.asarray(infeasible_hist, int)))
        if len(infeasible_hist)
        else 0,

        "mean_sqp_iters": (
            float(np.nanmean(sqp_iters))
            if controller_kind == "mpc" and len(sqp_iters)
            else np.nan
        ),

        "mean_jac_cond": float(np.nanmean(jac_cond))
        if len(jac_cond)
        else np.nan,

        "max_jac_cond": float(np.nanmax(jac_cond))
        if len(jac_cond)
        else np.nan,
    }


def write_grid_summary(results, run_root: Path):
    """
    Save one JSON and one CSV-style summary of all completed/failed runs.

    This avoids needing pandas as a hard dependency.
    """
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    rows = []

    for result in results:
        rows.append(result)

    with open(run_root / "grid_summary.json", "w") as f:
        json.dump(rows, f, indent=2)

    # Also write a simple CSV.
    if rows:
        import csv

        all_keys = sorted({key for row in rows for key in row.keys()})

        with open(run_root / "grid_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(rows)


from collections.abc import Sequence
from pathlib import Path


def run_experiment_grid(
    *,
    run_root: Path = Path("run_controller_comparison"),

    # Complete lumen geometries, including single or multiple bends.
    lumen_configs: Sequence[LumenConfig] | None = None,

    jacobian_variants=("contact",),
    controller_kinds=("mpc",),
    solver_modes=("sqp_full",),
    inverse_sequence_modes=("rollout_ltv",),
    inverse_kp_values=(1,),
    inverse_desired_step_max_m=4.0e-3,
    inverse_extra_damping=0.0,

    rollout_steps_values=(1,),
    Np: int = 5,
    N_sqp: int = 5,
    max_steps: int = 100,
    plant_contact: bool = True,
    save_plots: bool = False,
    adaptive_rollout_enabled: bool = False,
    hierarchical_mpc_enabled: bool = False,
    hierarchical_policy_config=None,
    stop_on_failure: bool = False,
):
    """
    Run experiments for every supplied lumen configuration and
    controller/model combination.
    """
    run_root = Path(run_root)

    # Supply a default geometry when none is passed.
    if lumen_configs is None:
        lumen_configs = (
            make_double_bend_lumen_config(
                first_angle_deg=-90.0,
                second_angle_deg=90.0,
            ),
        )
    else:
        lumen_configs = tuple(lumen_configs)

    if not lumen_configs:
        raise ValueError(
            "lumen_configs must contain at least one LumenConfig."
        )

    for lumen_cfg in lumen_configs:
        if not isinstance(lumen_cfg, LumenConfig):
            raise TypeError(
                "Every item in lumen_configs must be a LumenConfig; "
                f"got {type(lumen_cfg).__name__}."
            )

        lumen_cfg.validate()

    experiments = make_curvature_jacobian_grid(
        run_root=run_root,

        # Replace bend_angles_deg with this.
        lumen_configs=lumen_configs,

        jacobian_variants=jacobian_variants,
        controller_kinds=controller_kinds,
        solver_modes=solver_modes,
        inverse_sequence_modes=inverse_sequence_modes,
        inverse_kp_values=inverse_kp_values,
        inverse_desired_step_max_m=(
            inverse_desired_step_max_m
        ),
        inverse_extra_damping=(
            inverse_extra_damping
        ),
        rollout_steps_values=rollout_steps_values,
        Np=Np,
        N_sqp=N_sqp,
        max_steps=max_steps,
        plant_contact=plant_contact,
        adaptive_rollout_enabled=(
            adaptive_rollout_enabled
        ),
    )

    print(
        f"[GRID] Number of experiments: "
        f"{len(experiments)}"
    )
    print(f"[GRID] Output root: {run_root}")
    print(
        f"[GRID] Hierarchical MPC: "
        f"{hierarchical_mpc_enabled}"
    )
    print(
        f"[GRID] Adaptive rollout: "
        f"{adaptive_rollout_enabled}"
    )

    summary_rows = []

    for i, exp_cfg in enumerate(
        experiments,
        start=1,
    ):
        print("\n" + "=" * 100)
        print(
            f"[GRID] {i}/{len(experiments)}: "
            f"{exp_cfg.run_name}"
        )
        print("=" * 100)

        design_cfg = ControllerDesignConfig()

        try:
            result = run_experiment(
                exp_cfg,
                design_cfg,
                save_plots=save_plots,
                hierarchical_mpc_enabled=(
                    hierarchical_mpc_enabled
                ),
                hierarchical_policy_config=(
                    hierarchical_policy_config
                ),
            )

            row = dict(result["summary"])
            row["python_success"] = True
            row["error"] = ""
            row["hierarchical_mpc_enabled"] = bool(
                hierarchical_mpc_enabled
            )
            row["adaptive_rollout_enabled"] = bool(
                adaptive_rollout_enabled
            )
            row["hierarchy_policy"] = (
                type(
                    hierarchical_policy_config
                ).__name__
                if hierarchical_policy_config is not None
                else ""
            )

            summary_rows.append(row)

        except Exception as exc:
            print(
                f"[FAILED] {exp_cfg.run_name}: {exc}"
            )
            traceback.print_exc()

            row = exp_cfg.to_dict()
            row.update(
                {
                    "python_success": False,
                    "error": repr(exc),
                    "hierarchical_mpc_enabled": bool(
                        hierarchical_mpc_enabled
                    ),
                    "adaptive_rollout_enabled": bool(
                        adaptive_rollout_enabled
                    ),
                }
            )

            summary_rows.append(row)

            fail_dir = exp_cfg.out_root
            fail_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            with open(
                fail_dir / "failure.txt",
                "w",
            ) as stream:
                stream.write(
                    traceback.format_exc()
                )

            if stop_on_failure:
                raise

    write_grid_summary(
        summary_rows,
        run_root,
    )

    print("\n" + "=" * 100)
    print("[GRID] Finished")
    print(
        "[GRID] Summary written to: "
        f"{run_root / 'grid_summary.csv'}"
    )
    print("=" * 100)

    return summary_rows

hierarchy_cfg_evidence_lti_low = HierarchicalMPCPolicyConfig(
    enabled=True,

    rollout_min=1,
    rollout_max=5,

    use_lti_for_low_risk=True,
    lti_requires_previous_accepted=True,
    missing_beam_risk_bin="high",

    beam_risk_low_max=8.8665,
    beam_risk_medium_max=8.9859,

    low_risk_rollout=5,
    medium_risk_rollout=3,
    high_risk_rollout=1,

    sqp_cap_low=7,
    sqp_cap_medium=4,
    sqp_cap_high=3,
    sqp_cap_high_extension=5,

    low_abs_tol=2.4,
    low_rel_guard=2.0,
    medium_abs_tol=2.4,
    medium_rel_guard=2.0,
    high_rel_tol=0.60,
    high_rel_strict_tol=0.30,
    high_abs_guard=7.5,

    clearance_stop_mm=0.8,
    clearance_bad_mm=1.0,
    clearance_warn_mm=1.5,

    mpc_cond_extreme=1e8,
    downgrade_patience=2,
)
def make_double_bend_lumen_config(
    first_angle_deg: float = 30.0,
    second_angle_deg: float = -70.0,
) -> LumenConfig:
    return LumenConfig(
        length=0.04,
        n_pts=240,
        n_ref_pts=100,
        radius=0.004,
        ds_target=1e-3,
        bends=(
            LumenBend(
                bend_axis=(0.0, 0.0, 1.0),
                bend_angle_rad=np.deg2rad(
                    first_angle_deg
                ),
                bend_start=0.0,
                bend_end=0.01,
            ),
            LumenBend(
                bend_axis=(0.0, 0.0, 1.0),
                bend_angle_rad=np.deg2rad(
                    second_angle_deg
                ),
                bend_start=0.015,
                bend_end=0.025,
            ),
        ),
    )
if __name__ == "__main__":
    double_bend_lumen = make_double_bend_lumen_config(
        first_angle_deg=0.0,
        second_angle_deg=-110.0,
    )

    run_experiment_grid(
        run_root=Path("experiment3_sim_3"),

        lumen_configs=(
            double_bend_lumen,
        ),

        jacobian_variants=(
             "contact", "no_contact"
        ),
        inverse_sequence_modes=("held",),
        controller_kinds=( "mpc", ),
        solver_modes=( "sqp_full" ,),
        rollout_steps_values=(10,),

        Np=15,
        N_sqp=50,
        max_steps=40,

        plant_contact=True,
        save_plots=True,
        adaptive_rollout_enabled=False,
        hierarchical_mpc_enabled=False,
        stop_on_failure=False,
    )
