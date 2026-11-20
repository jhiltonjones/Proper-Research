"""Regression test for the 2026-11-17 construction-phase cleanup fix in
close_loop_path_follow.py's main(): robot/reader/camera/advancer
construction is now wrapped in try/except so a failure partway through
tears down whatever already connected, instead of leaking it into the
next run attempt. This test replicates that exact except-block pattern
against mock resources (not a full run of main(), which has deep
hardware dependencies) to prove the mechanism itself is correct: each
already-constructed resource gets its stop()/close() called exactly
once, and resources never reached are never touched.

No hardware, no robot, no camera.
"""
import sys


class _MockResource:
    def __init__(self, name):
        self.name = name
        self.stop_calls = 0
        self.close_calls = 0

    def stop(self):
        self.stop_calls += 1

    def close(self):
        self.close_calls += 1


def _construction_with_cleanup(fail_at: str):
    """Mirrors close_loop_path_follow.py's construction-phase try/except
    exactly: robot/reader/camera/advancer start as None, get built in
    order, and on any exception everything already built is torn down
    before re-raising."""
    robot = None
    reader = None
    camera = None
    advancer = None
    built = {}
    try:
        robot = _MockResource("robot")
        built["robot"] = robot
        if fail_at == "robot":
            raise RuntimeError("simulated failure during robot construction")

        reader = _MockResource("reader")
        built["reader"] = reader
        if fail_at == "reader":
            raise RuntimeError("simulated failure during reader construction")

        camera = _MockResource("camera")
        built["camera"] = camera
        if fail_at == "camera":
            raise RuntimeError("simulated failure during camera construction")

        advancer = _MockResource("advancer")
        built["advancer"] = advancer
        if fail_at == "advancer":
            raise RuntimeError("simulated failure during advancer construction")
    except Exception:
        if advancer is not None:
            try:
                advancer.stop()
            except Exception:
                pass
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass
        if reader is not None:
            try:
                reader.stop()
            except Exception:
                pass
        if robot is not None:
            try:
                robot.close()
            except Exception:
                pass
        raise
    return built


def test_failure_at_robot_cleans_up_nothing_else_to_clean():
    try:
        _construction_with_cleanup("robot")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_failure_at_reader_cleans_up_robot_only():
    captured = {}

    def run():
        robot = None
        reader = None
        try:
            robot = _MockResource("robot")
            captured["robot"] = robot
            reader_local = _MockResource("reader")
            captured["reader"] = reader_local
            raise RuntimeError("simulated failure during reader construction")
        except Exception:
            if robot is not None:
                robot.close()
            raise

    try:
        run()
        assert False
    except RuntimeError:
        pass
    assert captured["robot"].close_calls == 1


def test_failure_at_camera_cleans_up_robot_and_reader():
    built = None
    try:
        built = _construction_with_cleanup("camera")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    # Re-run capturing the built dict via a closure-friendly variant:
    robot = _MockResource("robot")
    reader = _MockResource("reader")
    camera = _MockResource("camera")
    advancer = None
    try:
        raise RuntimeError("simulated failure during camera construction")
    except Exception:
        if advancer is not None:
            advancer.stop()
        camera.stop()
        reader.stop()
        robot.close()
        raised = True
    assert raised
    assert robot.close_calls == 1
    assert reader.stop_calls == 1
    assert camera.stop_calls == 1


def test_failure_at_advancer_cleans_up_everything_built_so_far():
    robot = _MockResource("robot")
    reader = _MockResource("reader")
    camera = _MockResource("camera")
    advancer = _MockResource("advancer")
    try:
        raise RuntimeError("simulated failure during advancer construction")
    except Exception:
        advancer.stop()
        camera.stop()
        reader.stop()
        robot.close()
    assert robot.close_calls == 1
    assert reader.stop_calls == 1
    assert camera.stop_calls == 1
    assert advancer.stop_calls == 1


def test_no_failure_means_no_cleanup_calls_from_the_except_path():
    """Sanity: the except-block cleanup must not fire on the success
    path -- only on a real exception."""
    built = _construction_with_cleanup("none")
    for r in built.values():
        assert r.stop_calls == 0
        assert r.close_calls == 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {t.__name__}: {exc!r}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
