from dataclasses import dataclass
from scipy.linalg import solve_discrete_are
import numpy as np
@dataclass
class PIDState:
    e_init: float = 0.0
    e_prev: float = 0.0
    
def pid_step(error: float, state: PIDState, dt:float, Kp:float, Ki:float, Kd:float):
    e_init_new = state.e_init+error*dt
    e_dot = (error - state.e_prev)/dt
    u = Kp * error + Ki*e_init_new + Kd*e_dot
    new_state = PIDState(e_init=e_init_new, e_prev=error)
    return u, new_state

def damped_inverse_scalar(J:float, cmd:float, damping:float):
    JJt = J * J
    gain = cmd/ (JJt + damping**2)
    return J*gain 
def dare_stabilising_k(A, B, Q, R):
    P = solve_discrete_are(A,B,Q,R)
    K = -np.linalg.solve(R + B.T @ P @ B, B.T, P @ A)