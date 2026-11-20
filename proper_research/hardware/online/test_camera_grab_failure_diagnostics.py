"""2026-11-20: the baseline-A launch failed with "no fresh tip+joints
within warmup_timeout_s (vis_fail=0, '')" despite preflight's own,
separate camera session having just grabbed 20/20 valid frames moments
earlier. Root cause: CameraSource._grab_body() returned silently on
every cap.grab()/cap.retrieve() failure, touching no counter and no
last_error, so camera.healthy stayed True (thread alive, no exception)
with zero frames ever landing -- and the warmup-timeout error only
reported the (irrelevant, since the vision mapper is never reached
without a frame) mapper.detection_failures/mapper.last_error fields.

This test exercises the real CameraSource._grab_body() directly against
a fake cv2.VideoCapture-shaped object whose grab()/retrieve() fail, with
no real camera device -- confirms the new grab_failures counter and
last_error field are populated, which is what the corrected warmup-
timeout error message (close_loop_path_follow.py) now surfaces.
"""
from proper_research.hardware.online.camera_source import CameraSource


class _FakeCapAlwaysFailsGrab:
    def grab(self):
        return False


class _FakeCapRetrieveFails:
    def grab(self):
        return True

    def retrieve(self):
        return False, None


def _bare_camera_source():
    """A CameraSource instance with no real start() call -- only
    _grab_body() and the counters/last_error it touches are exercised."""
    return CameraSource.__new__(CameraSource)


def test_grab_body_records_failure_when_cap_is_none():
    cam = _bare_camera_source()
    cam._cap = None
    cam.frames_grabbed = 0
    cam.grab_failures = 0
    cam.last_error = ""
    cam._grab_body(0.0)
    assert cam.grab_failures == 1
    assert cam.frames_grabbed == 0
    assert "cap is None" in cam.last_error


def test_grab_body_records_failure_when_grab_returns_false():
    cam = _bare_camera_source()
    cam._cap = _FakeCapAlwaysFailsGrab()
    cam.frames_grabbed = 0
    cam.grab_failures = 0
    cam.last_error = ""
    cam._grab_body(0.0)
    assert cam.grab_failures == 1
    assert cam.frames_grabbed == 0
    assert "grab() returned False" in cam.last_error


def test_grab_body_records_failure_when_retrieve_fails():
    cam = _bare_camera_source()
    cam._cap = _FakeCapRetrieveFails()
    cam.frames_grabbed = 0
    cam.grab_failures = 0
    cam.last_error = ""
    cam._grab_body(0.0)
    assert cam.grab_failures == 1
    assert cam.frames_grabbed == 0
    assert "retrieve() failed" in cam.last_error


def test_warmup_timeout_error_message_cites_camera_diagnostics_not_just_mapper():
    """Source-grounds the actual fix: the error raised in
    close_loop_path_follow.py's warmup loop must reference the camera's
    own frames_grabbed/grab_failures/last_error, not only the mapper's."""
    import inspect
    import proper_research.hardware.online.close_loop_path_follow as pf
    src = inspect.getsource(pf)
    anchor = "no fresh tip+joints within warmup_timeout_s"
    idx = src.index(anchor)
    snippet = src[idx:idx + 400]
    assert "camera.frames_grabbed" in snippet
    assert "camera.grab_failures" in snippet
    assert "camera.last_error" in snippet
