from pathlib import Path

from proper_research.simulation.simulations.controller_factory import (
    ControllerDesignConfig,
)
from proper_research.simulation.simulations.run_experiment_grid import (
    make_double_bend_lumen_config,
)
from proper_research.simulation.simulations.scenario import (
    make_curvature_jacobian_grid,
)


def make_experiment():
    """
    Example factory used by run_optimized_simulation.py.

    Change solver_modes to one of:
        ("lti",)
        ("ltv_oneshot",)
        ("sqp_full",)

    Run each mode as a separate experiment for the research comparison.
    """
    lumen = make_double_bend_lumen_config(
        first_angle_deg=0.0,
        second_angle_deg=-6.0,
    )

    experiments = make_curvature_jacobian_grid(
        run_root=Path("optimized_controller_run"),
        lumen_configs=(lumen,),
        jacobian_variants=("contact",),
        controller_kinds=("mpc",),
        solver_modes=("sqp_full",),
        inverse_sequence_modes=("held",),
        inverse_kp_values=(1.0,),
        inverse_desired_step_max_m=4.0e-3,
        inverse_extra_damping=0.0,
        rollout_steps_values=(10,),
        Np=15,
        N_sqp=50,
        max_steps=100,
        plant_contact=True,
        adaptive_rollout_enabled=False,
    )
    if len(experiments) != 1:
        raise RuntimeError(
            f"Expected one experiment, generated {len(experiments)}."
        )
    return experiments[0], ControllerDesignConfig()
