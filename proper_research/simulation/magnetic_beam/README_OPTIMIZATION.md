# Magnetic-beam forward/Jacobian optimisation bundle

This bundle keeps the controller research formulations unchanged. It does **not**
merge, switch, or adapt LTI, LTV one-shot, or full-SQP modes. It reduces repeated
work inside each nonlinear forward solve and each implicit-Jacobian evaluation.

## Important validation note

The original `sensitivity.py` file was not among the uploaded files. Therefore,
`sensitivity_optimized.py` was reconstructed against the interface used by
`forward_model.py` and the implicit-equilibrium relation

```text
g(u, theta) = 0

du/dtheta = -H^{-1} G_theta

J_tip = P_u du/dtheta + P_theta
```

Do not replace the legacy sensitivity implementation until the supplied benchmark
passes on your full repository and representative recorded trajectories. The
legacy implementation in your repository remains the numerical reference.

## Files

- `energy_optimized.py`: vectorised elastic energy and energy evaluation from an
  already integrated state.
- `solver_optimized.py`: optimised fixed-length and continuation solves, including
  complete timing and accurate warm-direct/fallback accounting.
- `sensitivity_optimized.py`: optimised implicit sensitivity with batched RHS
  solves, nominal-data reuse, reduced sensitivity-state allocation, optional
  finite-difference column parallelism, and Hessian reuse.
- `forward_model_optimized.py`: drop-in subclass using the optimised solver and
  sensitivity implementation.
- `compare_forward_jacobian_before_after.py`: paired legacy-versus-optimised
  speed and numerical-agreement benchmark.
- `run_iridis_benchmark.slurm`: example AMD-partition benchmark launcher.

## Optimisations made

### Forward solve

1. Batched inversion of all segment compliance matrices.
2. Vectorised elastic-energy calculation.
3. Reuse of the explicit initial objective evaluation when L-BFGS-B requests the
   same initial vector.
4. Exact one-entry caches for repeated objective and gradient callbacks.
5. Final beam state is integrated once; final energy is evaluated from that state
   rather than integrating the beam a second time.
6. Intermediate continuation stages omit full diagnostic arrays.
7. Optional removal of continuation history and large vectors from `info`.
8. Cached magnetisation-profile factories for repeated insertion lengths.
9. Accurate accounting of failed warm-direct work plus continuation work.

### Implicit Jacobian

1. Reuses nominal arclength/stiffness data for the six source-pose columns of
   `G_theta`; only insertion-length perturbations rebuild them.
2. Computes `d p_tip / d u` from the cached nominal quaternion trajectory without
   rebuilding the full forward state or full 3-D sensitivity histories.
3. Solves all seven implicit-sensitivity right-hand sides simultaneously.
4. Preserves the existing nearby-pose Hessian-reuse mechanism.
5. Optional parallel finite-difference columns via `--workers` or
   `MAGBEAM_SENSITIVITY_WORKERS`.
6. No Hessian eigendecomposition in the optimised online diagnostics.

## Installation

Keep your original files. Copy the four implementation modules into the existing
magnetic-beam package:

```bash
cp energy_optimized.py \
   solver_optimized.py \
   sensitivity_optimized.py \
   forward_model_optimized.py \
   /path/to/your/repo/proper_research/simulation/magnetic_beam/
```

Copy the comparison script to the repository root:

```bash
cp compare_forward_jacobian_before_after.py /path/to/your/repo/
```

No existing source file has to be overwritten.

## Run the before/after comparison

From the repository root, in the same Python environment used by the project:

```bash
python compare_forward_jacobian_before_after.py \
  --variant contact \
  --jacobian-mode fast \
  --num-poses 20 \
  --repeats 5 \
  --workers 1 \
  --result-detail contact \
  --strict \
  --output-dir benchmark_contact_fast_serial
```

Repeat for the no-contact research model:

```bash
python compare_forward_jacobian_before_after.py \
  --variant no_contact \
  --jacobian-mode fast \
  --num-poses 20 \
  --repeats 5 \
  --workers 1 \
  --strict \
  --output-dir benchmark_no_contact_fast_serial
```

Then test CPU-column parallelism separately:

```bash
python compare_forward_jacobian_before_after.py \
  --variant contact \
  --jacobian-mode fast \
  --num-poses 20 \
  --repeats 5 \
  --workers 4 \
  --strict \
  --output-dir benchmark_contact_fast_workers4
```

Start with one worker. More workers are beneficial only if the gradient kernels
release enough of the Python GIL and the node is not oversubscribed.

## Use a recorded trajectory

A recorded sequence is preferable to the built-in small perturbation trajectory.
The script accepts:

- `.npy`: array of shape `(N, 7)`;
- `.npz`: key `p7`, `poses`, or `p7_sequence`;
- `.csv`: columns `x,y,z,rx,ry,rz,L`.

Example:

```bash
python compare_forward_jacobian_before_after.py \
  --poses recorded_controller_poses.npy \
  --variant contact \
  --jacobian-mode fast \
  --repeats 5 \
  --workers 1 \
  --strict \
  --output-dir benchmark_recorded_contact
```

## Outputs

Each output directory contains:

- `poses_used.npy`;
- `per_call_results.csv`;
- `summary.json`;
- `timing_comparison.png`;
- `accuracy_comparison.png`.

The summary reports median, P95, and P99 forward/Jacobian times, speed-up, solve
success, tip difference, Jacobian difference, and the software/thread environment.

The default strict thresholds are:

```text
tip error <= 1e-8 m
relative Jacobian error <= 5e-3
all solves successful
```

These are initial engineering checks, not universal research tolerances. Set them
from your model-verification requirements using `--tip-atol` and
`--jacobian-rtol`.

## Integrate the optimised model after validation

Change only the forward-model import in the model factory:

```python
from proper_research.simulation.magnetic_beam.forward_model_optimized import (
    MagneticBeamForwardModelOptimized as MagneticBeamForwardModel,
)
```

Do not change the controller's LTI/LTV/SQP selection. The same controller modes
will call the faster model implementation.

For a diagnostic-equivalent run, construct the model with `result_detail="full"`,
`store_history=True`, and `store_vectors_in_info=True`. For controller timing,
`result_detail="contact"`, `store_history=False`, and
`store_vectors_in_info=False` retain contact/safety information while avoiding
large diagnostic arrays.

## Fair timing rules

1. Run legacy and optimised code in the same job and environment.
2. Set BLAS thread counts explicitly.
3. Exclude the first cold call from steady-state statistics.
4. Disable plotting and unrelated per-step file output.
5. Compare numerical agreement before accepting speed-up.
6. Report median, P95, P99, and maximum/deadline misses—not only the mean.
