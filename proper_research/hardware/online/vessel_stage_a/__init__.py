"""Vessel-navigation live-hardware runners for the model-necessity study
(2026-09-23): open-loop baseline + the two frozen closed-loop LTV+d
conditions (contact-aware vs no-contact Jacobian, exact Q_N=0, R=700R0).

Mirrors `rectangle_stage_a`'s execution-layer-C architecture exactly
(same servoJ streaming, same process-isolated MPC worker, same
preflight/health-check conventions) with three vessel-specific
additions: (1) the beam base/magnet/vision reference pose is raised
+30mm in z to match the physically raised rig -- see `common.py`;
(2) the Jacobian schedule can be built from the real digitized vessel
lumen with an explicit contact/no-contact toggle, instead of the
generic free-space plan; (3) the worker supports an exact Q_N=0
controller (`ExactQNZeroTaskNullspaceDelayAwareMPC`), not merely
gamma=0 (which leaves the nullspace term fully active -- see
`paper/model_necessity_study.tex`, Sec. mn-exp4, for why that
distinction was load-bearing for this study's live comparison).
"""
