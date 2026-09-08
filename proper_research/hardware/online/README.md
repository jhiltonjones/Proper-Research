# Online (streaming) hardware stack

Real-time replacement for the stop-and-go `run_hardware_control_optimized` loop.
Single process, multi-threaded, latest-value message passing. The controller and
Jacobian code are reused unchanged.

```
CameraSource ──StateEstimate──▶ OnlineMPCRunner ──ControlCommand──▶ RobotSink ──speedL/speedJ──▶ UR
   (held-open cam,                  │  solve(estimate, dt)           (1 RTDE session,
    grab + reconstruct              │                                 stale-cmd watchdog)
    threads, /dev/shm handoff,      └──u0[6] rate──▶ AdvancerSink ─────▶ Arduino
    optional joints_getter)                          (1 serial conn,
                                                       fire-and-forget steps,
                                                       async DONE reader)
```

## Components

| Component | Owns | Thread(s) | Publishes / consumes |
|---|---|---|---|
| `CameraSource` | `cv2.VideoCapture` | grab (~250 Hz drain) + reconstruct | publishes `StateEstimate` to `state_slot` |
| `RobotSink` | one `URRTDERobot` RTDE session | stream loop at `control_frequency_hz` | consumes `ControlCommand` from `command_slot` |
| `AdvancerSink` | one `serial.Serial` | writer (~50 Hz) + reader (blocking `readline`) | consumes rate via `submit_rate`; publishes `AdvancerFeedback` |
| `OnlineMPCRunner` | the three above | control loop at `dt` + supervisor at 20 Hz | calls injected `solve(estimate, dt) -> u0` |
| `controller_adapters` | one offline controller | (runs inside the control loop, no thread of its own) | implements `solve(estimate, dt) -> SolveResult` |

`LatestSlot` is the single-slot latest-value buffer: no back-pressure, every
write carries a `time.monotonic()` stamp, consumers reject stale data.

## Shadow mode

Every sink has `dry_run` (default **True**). In dry-run the full pipeline runs -
vision, integration, clamping, watchdogs, feedback, logging - but no RTDE or
serial writes happen. This is the old `send_commands=False`. Flip each config's
`dry_run=False` to go live.

## Wiring to the existing MPC

`OnlineMPCRunner` takes a `solve` callback so it never hard-codes a controller
signature. Two families of controller can fill that seam, and they command the
arm differently:

### Magnet-pose-space (`lab_ready_mpc`) — Cartesian twist, `RobotSink(control_mode="cartesian")` (default)

```python
def solve(estimate, dt):
    mpc.set_output_bias(estimate.x_meas[:mpc.n] - mpc.forward_tip_fn(p_meas, commit=False)[:mpc.n])
    p_cmd, x_model, info = mpc.step(x_meas=estimate.x_meas[:mpc.n])
    return SolveResult(u0=info["u0"], p_commanded=p_cmd, infeasible=info.get("infeasible", False))
```
(`lab_ready_mpc`'s `MPCController.step` takes only `x_meas`, plus optional
`solver_mode`/`u_prev`/etc — check the class at construction time, not
`rollout_steps=`, which belongs to a different, older signature.)

### Joint-space offline controllers — six joint velocities, `RobotSink(control_mode="joint")`

`proper_research/controllers/inverse_jacobian_controller.py`
(`naive_inverse_jacobian`) and the two condensed-QP rungs in
`proper_research/controllers/mpc_variants.py` built on
`simulate_time_parameterized_beam_output_mpc.BeamOutputTrackingMPC`
(`mpc_lti`, `mpc_ltv_offline`) all measure and command
`z = [q1..q6, insertion]` — not a Cartesian twist. `controller_adapters.py` is
the seam for these three:

```python
from proper_research.controllers import beam_jacobian_providers, mpc_variants
from proper_research.hardware.online import (
    CameraConfig, CameraSource, RobotSink, RobotSinkConfig,
    AdvancerSink, AdvancerSinkConfig, OnlineMPCConfig, OnlineMPCRunner,
    OfflineControllerConfig, build_offline_solver, JsonlStepLogger,
)
from proper_research.simulation.simulations import (
    simulate_time_parameterized_configuration_mpc as base_module,
    simulate_time_parameterized_beam_output_mpc as beam_module,
)

# 1. The timed L3 reference this run tracks.
reference = base_module.load_configuration_reference(
    "results/.../time_parameterized_configuration_path"
)

# 2. A Jacobian provider from the SAME model class validated in
#    compare_controllers.py — from_model_bundle() if the bundle carries a
#    `.models` dict (simulation-style build_planning_context), otherwise
#    beam_jacobian_providers.build_diagnostic_adapter(beam_model=..., ...)
#    wrapped by hand for a HardwareModelBundle from hardware_model_factory.py.
jacobian_provider = beam_jacobian_providers.from_model_bundle(
    bundle=bundle, controller_pack=controller_pack, contact=False,
)
mpc_config = base_module.make_default_mpc_config(
    reference=reference, controller_pack=controller_pack,
)
beam_config = beam_module.BeamOutputMPCConfig()

# 3. Build the controller and wrap it as the runner's solve callback.
solver = build_offline_solver(
    "mpc_ltv_offline",   # or "naive_inverse_jacobian" / "mpc_lti"
    reference=reference, jacobian_provider=jacobian_provider,
    mpc_config=mpc_config, beam_config=beam_config,
    adapter_config=OfflineControllerConfig(progress_mode="wallclock"),
)

# 4. CameraSource needs joint angles alongside vision (robot_joints_getter),
#    not just the TCP pose; RobotSink must stream speedJ, not speedL.
robot = URRTDERobot(ROBOT_IP)
robot_sink = RobotSink(robot, RobotSinkConfig(control_mode="joint", dry_run=True))
camera = CameraSource(
    CameraConfig(), pivot_point_pose6=PIVOT,
    robot_joints_getter=robot.get_joints,
    insertion_length_getter=lambda: advancer_insertion_m(advancer),
)
advancer = AdvancerSink(AdvancerSinkConfig(dry_run=True))

logger = JsonlStepLogger("results/online_run/run.jsonl")
runner = OnlineMPCRunner(
    camera=camera, robot=robot_sink, advancer=advancer,
    solve=solver, config=OnlineMPCConfig(dt=reference.sample_period_s),
    on_step=logger,
)
try:
    reason = runner.run()
finally:
    logger.close()
```

`build_offline_solver` and `OfflineJointControllerAdapter` are in
`controller_adapters.py`; `python -m proper_research.hardware.online.controller_adapters`
runs its self-test (drives all three controllers through fake `StateEstimate`s
with no camera/robot/serial connection — verifies the seam's shapes and
progress tracking, not the controller maths, which `compare_controllers.py
--self-test` already covers).

## Safety

- `RobotSink`: command older than `max_command_age_s` → `speedStop` + hold.
  Also sets the RTDE-side watchdog (`set_watchdog`).
- `OnlineMPCRunner` supervisor: stale vision, dead worker thread, non-finite
  `u0`, or solver exception → `abort()` → `emergency_stop` on the robot.
- `AdvancerSink`: waits for the async `DONE` before the next increment, but
  gives up after `in_flight_timeout_s` so a dropped reply cannot deadlock it.

## New `URRTDERobot` methods

`speed_l`, `speed_j`, `speed_stop`, `servo_l`, `servo_stop`, `get_tcp_speed`,
`init_period`, `wait_period`, `set_watchdog`.
