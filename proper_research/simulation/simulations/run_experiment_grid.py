import json
import traceback
from pathlib import Path

import numpy as np

from proper_research.simulation.simulations.scenario import (
    ExperimentConfig,
    make_curvature_jacobian_grid,
)

from proper_research.simulation.simulations.model_factory import build_model_bundle
from proper_research.simulation.simulations.controller_factory import build_controller
from proper_research.simulation.simulations.runner import run_simulation
from proper_research.simulation.simulations.logging_utils import setup_output_dirs
from proper_research.simulation.simulations.initial_conditions import make_initial_poses
from proper_research.hardware.hierarchical_policy import (
    HierarchicalMPCPolicyConfig,
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
    )

    mpc = controller_pack["mpc"]
    p0 = controller_pack["p0"]
    p_min = controller_pack["p_min"]
    p_max = controller_pack["p_max"]
    u_max = controller_pack["u_max"]
    forward6d = controller_pack["forward6d_plant"]

    frames_dir, log_csv_path = setup_output_dirs(
        out_root=out_root,
        lumen_C=bundle.lumen_C,
        lumen_R=bundle.lumen_R,
        mpc=mpc,
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

    stats = run_simulation(
        mpc=mpc,
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
        solver_mode=exp_cfg.controller.solver_mode,
        save_plots=save_plots,
        adaptive_rollout_enabled=exp_cfg.controller.adaptive_rollout_enabled,
        hierarchical_mpc_enabled=hierarchical_mpc_enabled,
        hierarchical_policy_config=hierarchical_policy_config,
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

    return {
        "run_name": exp_cfg.run_name,
        "log_csv_path": str(log_csv_path),

        "bend_angle_deg": float(exp_cfg.lumen.bend_angle_deg),
        "plant_contact": bool(exp_cfg.model.plant_contact),
        "jacobian_variant": str(exp_cfg.model.jacobian_variant),
        "solver_mode": str(exp_cfg.controller.solver_mode),
        "rollout_steps": int(exp_cfg.controller.rollout_steps),
        "Np": int(exp_cfg.controller.Np),
        "N_sqp": int(exp_cfg.controller.N_sqp),

        "n_steps": int(n_steps),
        "final_status": final_status,
        "final_infeasible": int(final_infeasible),
        "final_i_ref": int(final_i_ref),
        "final_tip": final_tip,

        "num_infeasible_steps": int(np.sum(np.asarray(infeasible_hist, int)))
        if len(infeasible_hist)
        else 0,

        "mean_sqp_iters": float(np.nanmean(sqp_iters))
        if len(sqp_iters)
        else np.nan,

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


def run_experiment_grid(
    *,
    run_root: Path = Path("run_hierarchy_simple"),
    bend_angles_deg=(-30.0, -60.0, -90.0),
    jacobian_variants=("contact",),
    solver_modes=("lti",),
    rollout_steps_values=(5,),
    Np: int = 5,
    N_sqp: int = 5,
    max_steps: int = 100,
    plant_contact: bool = True,
    save_plots: bool = False,
    adaptive_rollout_enabled: bool = False,
    hierarchical_mpc_enabled: bool = True,
    hierarchical_policy_config=None,
    stop_on_failure: bool = False,
):
    """
    Run experiment grid.

    For hierarchical MPC:
        - solver_modes should usually be ("lti",)
        - rollout_steps_values should usually be (rollout_max,)
        - the hierarchy switches solver_mode_used internally
        - the hierarchy switches rollout_steps_used internally
    """
    run_root = Path(run_root)

    experiments = make_curvature_jacobian_grid(
        run_root=run_root,
        bend_angles_deg=bend_angles_deg,
        jacobian_variants=jacobian_variants,
        solver_modes=solver_modes,
        rollout_steps_values=rollout_steps_values,
        Np=Np,
        N_sqp=N_sqp,
        max_steps=max_steps,
        plant_contact=plant_contact,
        adaptive_rollout_enabled=adaptive_rollout_enabled,
    )

    print(f"[GRID] Number of experiments: {len(experiments)}")
    print(f"[GRID] Output root: {run_root}")
    print(f"[GRID] Hierarchical MPC: {hierarchical_mpc_enabled}")
    print(f"[GRID] Adaptive rollout: {adaptive_rollout_enabled}")

    summary_rows = []

    for i, exp_cfg in enumerate(experiments, start=1):
        print("\n" + "=" * 100)
        print(f"[GRID] {i}/{len(experiments)}: {exp_cfg.run_name}")
        print("=" * 100)

        try:
            result = run_experiment(
                exp_cfg,
                save_plots=save_plots,
                hierarchical_mpc_enabled=hierarchical_mpc_enabled,
                hierarchical_policy_config=hierarchical_policy_config,
            )

            row = dict(result["summary"])
            row["python_success"] = True
            row["error"] = ""
            row["hierarchical_mpc_enabled"] = bool(hierarchical_mpc_enabled)
            row["adaptive_rollout_enabled"] = bool(adaptive_rollout_enabled)
            row["hierarchy_policy"] = type(hierarchical_policy_config).__name__ if hierarchical_policy_config is not None else ""
            summary_rows.append(row)

        except Exception as e:
            print(f"[FAILED] {exp_cfg.run_name}: {e}")
            traceback.print_exc()

            row = exp_cfg.to_dict()
            row.update(
                {
                    "python_success": False,
                    "error": repr(e),
                    "hierarchical_mpc_enabled": bool(hierarchical_mpc_enabled),
                    "adaptive_rollout_enabled": bool(adaptive_rollout_enabled),
                }
            )

            summary_rows.append(row)

            fail_dir = exp_cfg.out_root
            fail_dir.mkdir(parents=True, exist_ok=True)

            with open(fail_dir / "failure.txt", "w") as f:
                f.write(traceback.format_exc())

            if stop_on_failure:
                raise

    write_grid_summary(summary_rows, run_root)

    print("\n" + "=" * 100)
    print("[GRID] Finished")
    print(f"[GRID] Summary written to: {run_root / 'grid_summary.csv'}")
    print("=" * 100)

    return summary_rows


if __name__ == "__main__":

    run_experiment_grid(
        run_root=Path(f"run_figure4_Nr_eval"),
        bend_angles_deg=(-30.0, -60.0, -90.0),
        jacobian_variants=("no_contact", "contact",),
        solver_modes=("lti", "ltv_oneshot", "sqp_full",),
        rollout_steps_values=(3,5),
        Np=5,
        N_sqp=5,
        max_steps=80,
        plant_contact=True,
        save_plots=False,
        adaptive_rollout_enabled=False,
        hierarchical_mpc_enabled=False,
        hierarchical_policy_config=None,
        stop_on_failure=False,
    )