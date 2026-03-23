from proper_research.advancer_unit.advancer_unit_cmd import AdvancerUnit
from proper_research.control.core import pid_step, PIDState
from proper_research.parameters import default_visual_pid_params
import numpy as np
from proper_research.vision.vision_w_arco import measure_lengths_mm  # adjust to your real module


def measure_total_length_mm_new_vision(
    *, image_filename="focused_image.jpg", use_roi=True, show=False, cam_index=0
):
    res = measure_lengths_mm(
        image_filename=image_filename,
        use_roi=use_roi,
        show=show,
        cam_index=cam_index,
    )
    return {"total_length_mm": float(res["total_length_mm"]), "raw": res}


def advancer_go(length_des_mm):
    ady = AdvancerUnit(port="/dev/ttyACM0")

    pid_params = default_visual_pid_params()
    pid_params.Kp = 0.2

    pid_state = PIDState()

    tol_mm = 2.0
    max_iter = 3
    u_max_mm = 5.0
    u_min_mm = -5.0
    deadband_mm = 0.5
    dt = pid_params.dt

    try:
        for k in range(max_iter):

            # ---- SAME LOGIC: first iteration "big move" ----
            if k == 0:
                result = measure_total_length_mm_new_vision(
                    image_filename="focused_image.jpg",
                    use_roi=True,
                    show=False,
                    cam_index=0,
                )
                length_curr = result["total_length_mm"]
                e = length_des_mm - length_curr
                print(f"Error is {e}")
                if e<10:
                    if e > 0:
                        print(f"  Moving FORWARD by {e:.2f} mm")
                        ady.forward(e)
                    else:
                        print(f"  Moving BACKWARD by {abs(e):.2f} mm")
                        ady.backward(abs(e))

            # ---- SAME LOGIC: re-measure + PID refine ----
            result = measure_total_length_mm_new_vision(
                image_filename="focused_image.jpg",
                use_roi=True,
                show=False,
                cam_index=0,
            )
            length_curr = result["total_length_mm"]
            e = length_des_mm - length_curr

            print(f"\n[Iter {k}]")
            print(f"  Measured length: {length_curr:.2f} mm")
            print(f"  Desired length:  {length_des_mm:.2f} mm")
            print(f"  Error:           {e:.2f} mm")

            if abs(e) <= tol_mm:
                print("  Beam control complete (within tolerance).")
                break

            u, pid_state = pid_step(e, pid_state, dt, pid_params.Kp,
                                    pid_params.Ki, pid_params.Kd)

            u = max(min(u, u_max_mm), u_min_mm)

            print(f"  Raw PID output:  {u:.2f} mm")

            if abs(u) < deadband_mm:
                print("  Command within deadband; not moving.")
                continue

            if u > 0:
                print(f"  Moving FORWARD by {u:.2f} mm")
                ady.forward(u)
            else:
                print(f"  Moving BACKWARD by {abs(u):.2f} mm")
                ady.backward(abs(u))

        final_result = measure_total_length_mm_new_vision(
            image_filename="focused_image.jpg",
            use_roi=True,
            show=False,
            cam_index=0,
        )
        print("\nFinal length: {:.2f} mm".format(final_result["total_length_mm"]))

    finally:
        ady.shutdown()


if __name__ == '__main__':
    advancer_go(length_des_mm=45)
