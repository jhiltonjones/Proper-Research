# Migration notes

## Implemented from the active original workflow

| Original responsibility | New location |
|---|---|
| Physical and solver constants | `cosserat_clean/config.py` |
| Local/private research imports | `cosserat_clean/external.py` |
| Quaternion perturbations and source dipole rotations | `cosserat_clean/quaternions.py` |
| Section stiffness and wire/tip transition | `cosserat_clean/mechanics.py` |
| Point-dipole field and derivative formulas | `cosserat_clean/magnetics.py` |
| Cosserat RHS, boundary conditions, BVP model | `cosserat_clean/rod.py` |
| Analytic state/source-pose Jacobians | `cosserat_clean/jacobians.py` |
| Seven-column shooting sensitivity | `cosserat_clean/sensitivity.py` |
| Bending, net wrench, energy, SVD metrics | `cosserat_clean/analysis.py` |
| Nominal setup and placement/orientation sweep | `cosserat_clean/experiment.py` |
| Comparison plots | `cosserat_clean/plotting.py` |

## Intentional changes

1. The original active placement key `side_90` actually used a -30-degree rotation. It is now named `side_30`.
2. The nominal calculation and print block appeared multiple times. It now runs once.
3. `CosseratForwardModel.forward` called an undefined function. Tip angles are now calculated by `tip_bending_angles_signed`.
4. Energy used a global `beam_params`. Gravity density is now an explicit argument.
5. Model factories depended on globals such as `p0_ur`, `q0_ur`, `nodes`, and `Kinv_fun`. They now use a `SimulationContext`.
6. The straight initial guess now begins at the configured base pose and follows its actual tangent, rather than beginning at the world origin.
7. Solver failures are checked explicitly and sweep failures are recorded in the output CSV instead of silently contaminating maps.
8. The NPZ export now includes field components, energy, Jacobian norms, and all SVD metrics.

## Dormant research experiments not copied verbatim

The original file contained several inactive or alternative code paths: three separate BVP sensitivity implementations, cached interpolation sensitivity, lumen/contact helpers, finite-difference benchmarking, hysteresis experiments that were commented out, and unused plotting imports. Those were not copied verbatim into the clean runtime because they were not part of the active execution path and several relied on hidden notebook state.

The project retains the active nonlinear solve, analytic Jacobians, seven-column shooting sensitivity, nominal analysis, and placement/orientation sweep. Additional legacy experiments should be reintroduced as separate scripts only after defining their required inputs and expected outputs.

## Validation performed

- Every Python file compiles.
- Seven private-package-independent unit tests pass.
- A controlled interface smoke test passed for:
  - nominal BVP execution,
  - shooting sensitivity,
  - a 27-point quick sweep,
  - CSV and NPZ output,
  - all standard plot files.

The controlled smoke test used stand-ins only to verify control flow and array interfaces. It is not a physical-model validation. Real numerical validation requires the original `beam_direction_magnetisation` and `proper_research` packages.
