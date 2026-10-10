"""Online (streaming) hardware stack for real-time MPC.

Three long-lived components, each owning one hardware resource for the whole run
and exchanging latest-value messages through :class:`LatestSlot`:

* :class:`CameraSource` - held-open camera + background beam reconstruction,
  publishes :class:`StateEstimate`.
* :class:`RobotSink` - one RTDE session, streams ``speedL(u0[:6])``, staleness
  watchdog -> ``speedStop``. Consumes :class:`ControlCommand`.
* :class:`AdvancerSink` - one serial connection, fire-and-forget incremental
  steps from the integrated insertion rate, async step-count feedback. Publishes
  :class:`AdvancerFeedback`.

The MPC / Jacobian code is reused unchanged: ``CameraSource`` calls the existing
``reconstruct_beam_within_vessel`` + ``vision_result_to_x_meas_robot`` pipeline,
and the orchestrator feeds the resulting ``x_meas`` to the existing controller
and turns its ``u0`` into a :class:`ControlCommand`.
"""

from .messages import (
    AdvancerFeedback,
    ControlCommand,
    HeartbeatLoop,
    LatestSlot,
    StateEstimate,
    now_monotonic,
)
from .camera_source import CameraConfig, CameraSource
from .robot_sink import RobotSink, RobotSinkConfig
from .advancer_sink import AdvancerSink, AdvancerSinkConfig
from .online_mpc_runner import OnlineMPCConfig, OnlineMPCRunner, SolveResult
from .controller_adapters import (
    JsonlStepLogger,
    OfflineControllerConfig,
    OfflineJointControllerAdapter,
    build_offline_solver,
)

__all__ = [
    "now_monotonic",
    "LatestSlot",
    "HeartbeatLoop",
    "StateEstimate",
    "ControlCommand",
    "AdvancerFeedback",
    "CameraSource",
    "CameraConfig",
    "RobotSink",
    "RobotSinkConfig",
    "AdvancerSink",
    "AdvancerSinkConfig",
    "OnlineMPCRunner",
    "OnlineMPCConfig",
    "SolveResult",
    "OfflineJointControllerAdapter",
    "OfflineControllerConfig",
    "build_offline_solver",
    "JsonlStepLogger",
]
