CORRECTION (2026-11-20, after root-causing the segfault): this script was
labeled "hardware-blocked" in its own docstring, but it was NOT -- that
label was wrong. close_loop_path_follow.py:1409 does a LOCAL
"from proper_research.hardware.ur_rtde_robot import URRTDERobot" inside
main()'s own body, re-executed every call, which silently bypassed this
script's pf.URRTDERobot module-attribute patch. The real URRTDERobot
class was constructed every run, and its real rtde_control.
RTDEControlInterface(...) genuinely attempted a socket connection to
192.168.56.101 -- this was a REAL (failed) hardware connection attempt
from an unintentionally-unmocked test process, not a hardware-blocked
harness exercising mocks. The RTDEControlInterface error text below and
the eventual segfault are native-library behavior under that genuine,
repeated, rapid connection-retry load -- not an OSQP, multiprocessing, or
mock-logic defect. The corrected harness
(proper_research/hardware/online/e2e_smoke/_mocked_hardware_harness.py)
patches proper_research.hardware.ur_rtde_robot.URRTDERobot directly (the
source module the local import re-reads) and is genuinely hardware-
blocked, confirmed by three passing scenarios with zero RTDE connection
attempts in their logs.

Exact command that produced the segfault:
  cd /home/jack/Proper-Research && timeout 30 python3 -u /home/jack/.claude/jobs/043c0b8b/tmp/e2e_main_loop_smoke_test.py normal > e2e_log.txt 2>&1

Observed: "Segmentation fault" reported by bash job-control (exit via signal),
preceded in the log by RTDEControlInterface error text ("Could not receive
data from robot...", "Bad file descriptor", "Robot is disconnected,
reconnecting...", "std::bad_alloc") despite pf.URRTDERobot and
pf.RobotJointStream both confirmed (via direct identity check,
pf.RobotJointStream printed as the mock lambda) to be the mock classes at
the point main() is called. MockReader's own __init__ print never appeared
in the log, even with python3 -u and no pipe -- meaning RobotJointStream
was very likely never actually reached before the crash (not a stdout
buffering artifact, confirmed by the -u/no-pipe rerun).

Source manifest at the time of this failing run (canonical checkout):
578a898f01e67498b6133c9f0b9b9ccddbf690dcdca7c378c47349276d22a5bc  proper_research/hardware/online/close_loop_path_follow.py
b85d2b6054025fc575cb8b6decb0e4da83322649a7976e0e5a862729cfb2b85b  proper_research/hardware/online/camera_source.py
a3651db68c6f1c0799e640f74a3c5a3da74b23f0676707943ceffcf08ef9a754  proper_research/hardware/online/messages.py
07abe8ae5071a832a8ff5062e37d2a036da1f73ebb4e37ee27b7074b064dbca0  proper_research/hardware/online/state_stream.py
51a0ed232b9ebb5494d27513510323ded5e78390e3f9eddab2f20226e2116268  proper_research/hardware/ur_rtde_robot.py
6989d9d8807f609f52c86c8c5093dd964acc163faec71b95a497902a09c29d44  proper_research/controllers/mpc_delay_aware/worker_process.py
9391b43c5821d426da25984e74b529dbd652bde0653a8711c3b4c4e03da30364  proper_research/controllers/mpc_delay_aware/process_isolated_adapter.py
f2017ab58264e74655034998082e94986515adc0ed12383174e7e5f39b8cdb0d  proper_research/hardware/online/rectangle_stage_a/run_mpc_delay_aware_insertion_anchor.py
