import time
import traceback
from datetime import datetime

from robot_class import URRtde


ROBOT_IP = "192.168.56.101"

NUM_CYCLES = 10
CONNECTED_TEST_DURATION_S = 3.0
READ_INTERVAL_S = 0.25
BETWEEN_CYCLES_S = 2.0


def timestamp():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def main():
    successes = 0
    failures = 0

    print("=" * 70)
    print("External Control connect/disconnect stress test")
    print("=" * 70)
    print(f"Robot IP: {ROBOT_IP}")
    print(f"Cycles:   {NUM_CYCLES}")
    print()
    print("NO ROBOT MOTION WILL BE COMMANDED.")
    print()
    print(
        "For each cycle, Python will wait for the External Control "
        "program to be started from the teach pendant."
    )
    print()

    for cycle in range(1, NUM_CYCLES + 1):

        print("\n" + "=" * 70)
        print(f"[{timestamp()}] CYCLE {cycle}/{NUM_CYCLES}")
        print("=" * 70)

        robot = None
        cycle_ok = True

        try:
            print(f"[{timestamp()}] Creating URRtde...")
            print(
                ">>> Make sure the External Control program is loaded.\n"
                ">>> Press PLAY on the teach pendant when Python starts waiting."
            )

            t0 = time.perf_counter()

            robot = URRtde(
                ROBOT_IP,
                external_control=True,
            )

            connect_time = time.perf_counter() - t0

            print(
                f"[{timestamp()}] CONNECTED "
                f"(constructor took {connect_time:.3f} s)"
            )

            # Basic connection state
            print(
                f"[{timestamp()}] is_connected(): "
                f"{robot.is_connected()}"
            )

            # Read robot state once
            joints = robot.get_joints()
            pose = robot.get_pose()

            print(f"[{timestamp()}] Initial joints:")
            print(joints)

            print(f"[{timestamp()}] Initial TCP pose:")
            print(pose)

            try:
                print(
                    f"[{timestamp()}] Robot mode: "
                    f"{robot.robot_mode()}"
                )
            except Exception as exc:
                print(f"Robot mode read warning: {exc}")

            try:
                print(
                    f"[{timestamp()}] Safety mode: "
                    f"{robot.safety_mode()}"
                )
            except Exception as exc:
                print(f"Safety mode read warning: {exc}")

            # Keep the connection alive and repeatedly read data.
            print(
                f"[{timestamp()}] Testing reads for "
                f"{CONNECTED_TEST_DURATION_S:.1f} s..."
            )

            test_start = time.perf_counter()
            read_count = 0

            while (
                time.perf_counter() - test_start
                < CONNECTED_TEST_DURATION_S
            ):
                if not robot.is_connected():
                    raise RuntimeError(
                        "RTDE connection dropped during connected test"
                    )

                q = robot.get_joints()

                if len(q) != 6:
                    raise RuntimeError(
                        f"Unexpected joint vector length: {len(q)}"
                    )

                read_count += 1
                time.sleep(READ_INTERVAL_S)

            print(
                f"[{timestamp()}] Completed {read_count} "
                "successful joint reads"
            )

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received.")
            raise

        except Exception as exc:
            cycle_ok = False
            failures += 1

            print()
            print("!!! CYCLE FAILED !!!")
            print(f"[{timestamp()}] {type(exc).__name__}: {exc}")
            traceback.print_exc()

        finally:
            if robot is not None:
                try:
                    print(
                        f"[{timestamp()}] Shutting down RTDE session..."
                    )

                    t0 = time.perf_counter()
                    robot.shutdown()
                    shutdown_time = time.perf_counter() - t0

                    print(
                        f"[{timestamp()}] Shutdown completed "
                        f"in {shutdown_time:.3f} s"
                    )

                except Exception as exc:
                    cycle_ok = False

                    print(
                        f"[{timestamp()}] Shutdown error: {exc}"
                    )

        if cycle_ok:
            successes += 1
            print(f"[{timestamp()}] CYCLE {cycle}: PASS")
        else:
            print(f"[{timestamp()}] CYCLE {cycle}: FAIL")

        if cycle != NUM_CYCLES:
            print()
            print(
                "The External Control program has now been stopped."
            )
            print(
                "For the next cycle, leave/reload the External Control "
                "program ready on the pendant."
            )
            print(
                f"Waiting {BETWEEN_CYCLES_S:.1f} s before reconnect..."
            )

            time.sleep(BETWEEN_CYCLES_S)

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(f"Successful cycles: {successes}/{NUM_CYCLES}")
    print(f"Failed cycles:     {failures}/{NUM_CYCLES}")

    if failures == 0:
        print("RESULT: all External Control sessions completed cleanly.")
    else:
        print(
            "RESULT: one or more sessions failed. "
            "Inspect the cycle immediately before the failure."
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest stopped by user.")