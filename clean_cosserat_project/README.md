# Clean Cosserat Magnetic-Beam Project

This project is a clean replacement for the original single-file research script. It separates configuration, rod mechanics, magnetic-field calculations, analytic Jacobians, sensitivity solves, experiment orchestration, plotting, and command-line entry points.

## What was fixed

- Removed duplicated imports, functions, calculations, and print blocks.
- Replaced hidden module-level globals with explicit dataclass configuration.
- Replaced the undefined `tip_bending_angles_from_tangent` call with one signed-angle implementation.
- Made gravity an explicit input to the energy calculation instead of reading a global `beam_params` variable.
- Corrected magnetisation-profile calls so they consistently receive arclength arrays and the magnetic moment parameter.
- Added solver-success checks and clearer exceptions.
- Renamed the misleading `side_90` placement to `side_30` because the original code actually rotated it by 30 degrees.
- Saved all calculated maps, including magnetic-field components and total energy.
- Added dependency checks, unit tests, type hints, and runnable scripts.

## Required dependencies

Public Python packages:

```bash
python -m pip install -r requirements.txt
```

The original code also depends on two local/private packages that are not bundled here:

- `beam_direction_magnetisation`
- `proper_research`

Install those packages in the same Python environment before running the physical simulation. The dependency checker reports exactly what is missing:

```bash
python check_install.py
```

## Run a nominal solve

```bash
python run_nominal.py
```

Useful options:

```bash
python run_nominal.py --nodes 120 --output results
python run_nominal.py --skip-sensitivity
```

## Run the orientation/placement sweep

A quick smoke-test sweep:

```bash
python run_orientation_sweep.py --quick --no-plots
```

The full sweep matching the active section of the original script:

```bash
python run_orientation_sweep.py
```

The full sweep evaluates three placements over 101 dipole rotations and can be computationally expensive.

## Run tests

The internal tests do not require the two private packages:

```bash
python -m unittest discover -s tests -v
```

## Project layout

- `run_nominal.py` — nominal forward solve and optional sensitivity calculation.
- `run_orientation_sweep.py` — placement-versus-dipole-angle comparison.
- `check_install.py` — dependency and import preflight.
- `cosserat_clean/config.py` — explicit simulation configuration.
- `cosserat_clean/external.py` — isolated access to local/private dependencies.
- `cosserat_clean/quaternions.py` — quaternion perturbation utilities.
- `cosserat_clean/mechanics.py` — beam stiffness and insertion-length logic.
- `cosserat_clean/magnetics.py` — dipole field and analytic derivatives.
- `cosserat_clean/rod.py` — Cosserat rod equations and forward model.
- `cosserat_clean/jacobians.py` — analytic state and control Jacobians.
- `cosserat_clean/sensitivity.py` — shooting sensitivity solvers.
- `cosserat_clean/analysis.py` — bending, force/torque, SVD metrics, and energy.
- `cosserat_clean/experiment.py` — nominal and sweep workflows.
- `cosserat_clean/plotting.py` — result plots.

## Important compatibility note

The project has been syntax-checked and its private-package-independent tests pass. A complete physical simulation could not be executed in the build environment because the two local packages listed above were unavailable. The wrapper in `cosserat_clean/external.py` produces an actionable error rather than failing with a long import traceback.
