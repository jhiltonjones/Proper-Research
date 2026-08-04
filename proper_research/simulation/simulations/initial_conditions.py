import numpy as np

from proper_research.robot.transformations import get_point


def make_initial_poses() -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Build the fixed initial robot/lumen experiment pose.

    Returns
    -------
    pivot_point:
        UR-style pose6 for the pivot/reference point:
        [x, y, z, rx, ry, rz]

    start_point:
        UR-style pose6 for the initial magnet pose:
        [x, y, z, rx, ry, rz]

    L_cmd:
        Initial insertion length.

    dt:
        Controller timestep.
    """
    L_cmd = 0.013

    pivot_point = np.array(
        [
            0.7981328220229531,
            -0.70992731669220016,
            -0.1,
            np.pi,
            0.001,
            0.001,
        ],
        dtype=float,
    )

    base_point = np.array(
        [
            pivot_point[0] - (L_cmd + 0.17),
            pivot_point[1],
            -0.1,
            np.pi,
            0.001,
            0.001,
        ],
        dtype=float,
    )

    start_point = np.asarray(
        get_point(0, 0, base_point, pivot_point),
        dtype=float,
    )

    dt = 0.01

    print(f"[INIT] start_point={start_point}")
    print(f"[INIT] pivot_point={pivot_point}")
    print(f"[INIT] L0={L_cmd:.4f}, dt={dt:.4f}")

    return pivot_point, start_point, L_cmd, dt