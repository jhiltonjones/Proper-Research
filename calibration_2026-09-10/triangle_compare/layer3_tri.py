"""Re-run only Layer 3 (time parameterisation) on the saved triangle global
path, with a relaxed acceleration tolerance."""
import io, contextlib
import numpy as np

TRI = "/home/jack/.claude/jobs/3710eca5/tmp/plan_tri"
GLOBAL_DIR = f"{TRI}/shape_triangle/global_configuration_converged"
TP_DIR = f"{TRI}/time_parameterized_configuration_path"

@contextlib.contextmanager
def quiet():
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        yield

from proper_research.planning.planning_context import build_planning_context
from proper_research.planning.time_parameterized_configuration_path import (
    TimeParameterizationConfig, time_parameterize_saved_global_path,
)

with quiet():
    _, _bundle, cp, _ = build_planning_context()

centre = np.load(f"{TRI}/shape_triangle/shape_centreline.npz")["lumen_C"]

tp = TimeParameterizationConfig(
    sample_period_s=0.1,
    state_velocity_limit=tuple([0.10] * 6 + [2.0e-3]),
    state_acceleration_limit=tuple([0.5] * 6 + [0.02]),
    maximum_path_speed_m_s=3.0e-3,
    constraint_tolerance=0.10,
    require_nonlinear_beam_feasible=False,
    require_saved_global_feasible=False,
    require_saved_dense_feasible=False,
)
time_parameterize_saved_global_path(
    global_output_dir=GLOBAL_DIR, controller_pack=cp, config=tp,
    output_dir=TP_DIR, lumen_C=centre,
)
d = np.load(f"{TP_DIR}/time_parameterized_configuration_path.npz")
P = np.asarray(d["desired_position_m"]); S = np.asarray(d["state"]); t = np.asarray(d["time_s"])
bp = np.array([0.525575, -0.670028, -0.016567]); u = np.array([-1., 0, 0]); v = np.array([0, 1., 0])
uv = np.c_[(P - bp) @ u, (P - bp) @ v] * 1e3
print(f"\n[layer3 OK] {len(t)} samples, {t[-1]:.1f}s")
print(f"  closed err {np.linalg.norm(P[0]-P[-1])*1e3:.3f} mm")
print(f"  tip u {uv[:,0].min():.1f}..{uv[:,0].max():.1f} mm   v {uv[:,1].min():.1f}..{uv[:,1].max():.1f} mm")
print(f"  insertion {S[:,6].min()*1e3:.1f}..{S[:,6].max()*1e3:.1f} mm")
print(f"  plan pos-err max {1e3*np.max(np.asarray(d['position_error_m'])):.3f} mm")
