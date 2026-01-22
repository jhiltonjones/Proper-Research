from proper_research.advancer_unit.advancer_unit_cmd import AdvancerUnit
from proper_research.control.core import pid_step, PIDState
from proper_research.parameters import default_visual_pid_params
from proper_research.vision.measure_length import measure_beam_and_tip_lengths_mm_with_checkerboard
import numpy as np

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


            if k ==0:
                result = measure_beam_and_tip_lengths_mm_with_checkerboard(
                image_filename="focused_image.jpg",
                use_roi=True,
                show=False,
                )
                length_curr = result["total_length_mm"]
                e = length_des_mm - length_curr
                if e > 0:
                    print(f"  Moving FORWARD by {e:.2f} mm")
                    ady.forward(e)
                else:
                    print(f"  Moving BACKWARD by {abs(e):.2f} mm")
                    ady.backward(abs(e))
            result = measure_beam_and_tip_lengths_mm_with_checkerboard(
                image_filename="focused_image.jpg",
                use_roi=True,
                show=False,
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

        final_result = measure_beam_and_tip_lengths_mm_with_checkerboard(
            image_filename="focused_image.jpg",
            use_roi=True,
            show=False,
        )
        print("\nFinal length: {:.2f} mm".format(final_result["total_length_mm"]))

    finally:
        ady.shutdown()
if __name__ == '__main__':
    advancer_go(length_des_mm=50)