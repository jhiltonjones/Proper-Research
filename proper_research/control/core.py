from dataclasses import dataclass

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
