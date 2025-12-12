"""
Visual servoing test harness (2D) with online Jacobians.

State/control interpretation (matches your description):
- phi   : magnet rotation around the beam base (rotates bending plane in image)
- alpha : magnet's own axis rotation (dipole direction in x-y plane)
- L     : beam/advancer length (in meters)

We simulate:
q = [L, phi, alpha]
y = tip pixel position u = [u, v]

Controller:
- minimize pixel error e = u_tip - u_target
- use online Jacobian J = du/dq
- initialize J via finite differences, then update with Broyden rank-1

NOTE: The plant mapping alpha -> theta is intentionally simple.
You should swap it with something closer to your system if desired.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# -----------------------------
# Helpers
# -----------------------------

def rot2(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s],
                     [s,  c]], dtype=float)

def wrap_angle(a: float) -> float:
    # Wrap to (-pi, pi]
    return (a + np.pi) % (2*np.pi) - np.pi

def constant_curvature_tip(L: float, theta: float) -> np.ndarray:
    """
    Tip position (x,y) in base frame for constant-curvature arc.
    - If theta ~ 0 => straight: x = L, y = 0
    """
    if abs(theta) < 1e-6:
        return np.array([L, 0.0], dtype=float)

    x = (L / theta) * np.sin(theta)
    y = (L / theta) * (1.0 - np.cos(theta))
    return np.array([x, y], dtype=float)


# -----------------------------
# Camera model (2D -> pixels)
# -----------------------------

@dataclass
class Camera2D:
    px_per_m: float = 5000.0   # scale: pixels per meter
    center_px: np.ndarray = np.array([640.0, 360.0], dtype=float)
    y_flip: bool = True        # image v increases downward

    def project(self, p_m: np.ndarray) -> np.ndarray:
        """
        Project 2D point in meters to pixels.
        """
        p = p_m.copy()
        if self.y_flip:
            p[1] *= -1.0
        return self.center_px + self.px_per_m * p


# -----------------------------
# Plant simulation
# -----------------------------

@dataclass
class BeamPlantSim:
    """
    Simplified plant:
    - alpha affects bending angle theta via nonlinear "magnetic response"
    - phi rotates the entire geometry about the base
    - L sets arc length

    Adds:
    - first-order lag on theta to emulate dynamics
    - measurement noise in pixels
    """
    cam: Camera2D
    dt: float = 0.05
    theta_tau: float = 0.25          # time constant (s)
    theta_max: float = np.deg2rad(55)
    alpha_gain: float = 1.6          # affects alpha->theta curve steepness
    alpha_bias: float = 0.0          # center bias (rad)
    meas_noise_px: float = 0.8       # pixel std dev
    rng: np.random.Generator = np.random.default_rng(1)

    # internal state
    theta: float = 0.0

    def alpha_to_theta_target(self, alpha: float) -> float:
        """
        Nonlinear mapping alpha -> target theta.
        This is a stand-in for your true magnet/beam relationship.
        """
        # Smooth saturation: theta_target in [-theta_max, +theta_max]
        x = self.alpha_gain * (alpha - self.alpha_bias)
        theta_tgt = self.theta_max * np.tanh(x)
        return theta_tgt

    def step(self, q: np.ndarray) -> np.ndarray:
        """
        q = [L, phi, alpha]
        returns measured tip pixels u = [u, v]
        """
        L, phi, alpha = float(q[0]), float(q[1]), float(q[2])

        # theta dynamics (first-order lag)
        theta_tgt = self.alpha_to_theta_target(alpha)
        a = np.exp(-self.dt / max(1e-6, self.theta_tau))
        self.theta = a * self.theta + (1.0 - a) * theta_tgt

        # geometry: constant-curvature arc in base frame, rotate by phi about base
        tip_base = constant_curvature_tip(L, self.theta)       # meters
        tip_world = rot2(phi) @ tip_base

        # project to pixels and add measurement noise
        u = self.cam.project(tip_world)
        u += self.rng.normal(0.0, self.meas_noise_px, size=(2,))
        return u


# -----------------------------
# Online Jacobian estimation
# -----------------------------

@dataclass
class OnlineJacobian:
    """
    Maintain J = du/dq (2x3 here).
    - init via finite differences around q0
    - update via Broyden rank-1 after each move
    """
    J: np.ndarray | None = None

    def init_finite_difference(
        self,
        plant: BeamPlantSim,
        q0: np.ndarray,
        deltas: np.ndarray,
        settle_steps: int = 8
    ) -> np.ndarray:
        """
        Estimate J columns by nudging each q_i.
        deltas: size (3,) step sizes for [L, phi, alpha]
        """
        q0 = q0.astype(float).copy()

        # settle at q0 then measure baseline
        for _ in range(settle_steps):
            u0 = plant.step(q0)
        u0 = plant.step(q0)

        J = np.zeros((2, len(q0)), dtype=float)

        for i in range(len(q0)):
            dq = np.zeros_like(q0)
            dq[i] = deltas[i]
            q1 = q0 + dq

            # settle at q1 then measure
            for _ in range(settle_steps):
                _ = plant.step(q1)
            u1 = plant.step(q1)

            J[:, i] = (u1 - u0) / deltas[i]

        self.J = J
        return J

    def broyden_update(self, dq: np.ndarray, du: np.ndarray) -> None:
        """
        J_{k+1} = J_k + ((du - J_k dq) dq^T) / (dq^T dq)
        """
        if self.J is None:
            raise RuntimeError("Jacobian not initialized")

        dq = dq.reshape(-1, 1)    # (3,1)
        du = du.reshape(-1, 1)    # (2,1)

        denom = float(dq.T @ dq)
        if denom < 1e-12:
            return

        resid = du - self.J @ dq
        self.J = self.J + (resid @ dq.T) / denom


# -----------------------------
# Visual servo controller
# -----------------------------

@dataclass
class VisualServoController:
    lam: float = 0.25      # gain
    mu: float = 5e-3       # damping
    max_step: np.ndarray = np.array([0.002, np.deg2rad(3.0), np.deg2rad(5.0)], dtype=float)
    q_min: np.ndarray = np.array([0.010, -np.pi, -np.pi], dtype=float)
    q_max: np.ndarray = np.array([0.050,  np.pi,  np.pi], dtype=float)

    def compute_step(self, J: np.ndarray, e: np.ndarray) -> np.ndarray:
        """
        Solve damped least squares: dq = -lam * (J^T J + mu I)^-1 J^T e
        e is (2,), pixels: e = u_tip - u_target
        """
        JTJ = J.T @ J
        A = JTJ + self.mu * np.eye(J.shape[1])
        b = J.T @ e
        dq = -self.lam * np.linalg.solve(A, b)

        # step limits
        dq = np.clip(dq, -self.max_step, self.max_step)
        return dq

    def clip_q(self, q: np.ndarray) -> np.ndarray:
        q2 = q.copy()
        q2[1] = wrap_angle(q2[1])
        q2[2] = wrap_angle(q2[2])
        q2 = np.minimum(np.maximum(q2, self.q_min), self.q_max)
        return q2


# -----------------------------
# Demo run
# -----------------------------

def run_demo():
    cam = Camera2D(px_per_m=5500.0, center_px=np.array([640.0, 360.0]))
    plant = BeamPlantSim(cam=cam, dt=0.05, theta_tau=0.20, meas_noise_px=0.6)
    jac = OnlineJacobian()
    ctrl = VisualServoController(
        lam=0.35,
        mu=2e-2,
        max_step=np.array([0.0015, np.deg2rad(2.0), np.deg2rad(4.0)]),
        q_min=np.array([0.012, -np.pi, -np.pi]),
        q_max=np.array([0.045,  np.pi,  np.pi]),
    )

    # Initial command/state
    q = np.array([0.020, np.deg2rad(15.0), np.deg2rad(5.0)], dtype=float)  # [L, phi, alpha]

    # Target tip in pixels (simulate a user click)
    u_target = np.array([820.0, 260.0], dtype=float)

    # Initialize Jacobian around q with finite differences
    deltas = np.array([0.001, np.deg2rad(2.0), np.deg2rad(3.0)], dtype=float)
    J0 = jac.init_finite_difference(plant, q, deltas=deltas, settle_steps=10)
    print("Initial J (finite-diff):\n", J0)

    # Iterate closed-loop
    max_iters = 40
    tol_px = 3.0

    # Measure initial
    for _ in range(10):
        u = plant.step(q)
    u = plant.step(q)

    for k in range(max_iters):
        e = u - u_target
        err = float(np.linalg.norm(e))

        print(f"iter {k:02d} | q=[L={q[0]*1000:5.1f}mm, phi={np.rad2deg(q[1]):6.1f}deg, alpha={np.rad2deg(q[2]):6.1f}deg]"
              f" | tip={u} | err={err:6.2f}px")

        if err < tol_px:
            print("Converged.")
            break

        # compute control step
        dq = ctrl.compute_step(jac.J, e)

        # apply step and clip
        q_next = ctrl.clip_q(q + dq)

        # step plant to new measurement (with settle)
        for _ in range(8):
            _ = plant.step(q_next)
        u_next = plant.step(q_next)

        # update Jacobian with Broyden using observed changes
        jac.broyden_update(dq=(q_next - q), du=(u_next - u))

        # advance
        q, u = q_next, u_next

    print("Final q:", q)
    print("Final tip:", u)
    print("Target:", u_target)


if __name__ == "__main__":
    run_demo()
