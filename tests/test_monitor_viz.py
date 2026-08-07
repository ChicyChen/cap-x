"""Annotated tool-output images for the CaP-X dashboard.

CaP-X already overlays SAM2/SAM3 masks and Molmo points before logging them. Two
failure-prone tools did not: detection logged the RAW rgb (boxes never drawn) and
Contact GraspNet logged no image at all. Those blind spots are what made the
TiPToP-side failures so slow to diagnose.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

from capx.monitor import viz


def _rgb(h=120, w=160):
    return np.full((h, w, 3), 90, np.uint8)


def _png(s):
    return isinstance(s, str) and base64.b64decode(s)[:4] == b"\x89PNG"


def _K():
    return np.array([[100.0, 0, 80], [0, 100.0, 60], [0, 0, 1]])


class TestDetectionBoxes:
    def test_draws_pixel_boxes_with_scores(self):
        dets = [
            {"box": [20, 30, 60, 70], "label": "orange_bottle", "score": 0.91},
            {"box": [80, 40, 120, 90], "label": "bin", "score": 0.42},
        ]
        assert _png(viz.detection_boxes(_rgb(), dets))

    def test_empty_detections_still_render(self):
        """An empty result is itself the diagnosis; it must be visible."""
        assert _png(viz.detection_boxes(_rgb(), []))

    def test_malformed_box_is_skipped(self):
        assert _png(viz.detection_boxes(_rgb(), [{"box": [1, 2], "label": "bad"}]))

    def test_missing_score_is_tolerated(self):
        assert _png(viz.detection_boxes(_rgb(), [{"box": [1, 2, 3, 4], "label": "x"}]))


class TestGraspCandidates:
    def _poses(self, n=5, z=0.8):
        T = np.tile(np.eye(4), (n, 1, 1))
        T[:, 0, 3] = np.linspace(-0.05, 0.05, n)
        T[:, 2, 3] = z
        return T

    def test_projects_camera_frame_poses(self):
        assert _png(viz.grasp_candidates(_rgb(), self._poses(), _K()))

    def test_marks_the_chosen_grasp(self):
        sc = np.array([0.1, 0.9, 0.3, 0.5, 0.2])
        out = viz.grasp_candidates(_rgb(), self._poses(), _K(), sc, chosen_index=1)
        assert _png(out)

    def test_single_pose_is_accepted(self):
        T = np.eye(4)
        T[2, 3] = 0.8
        assert _png(viz.grasp_candidates(_rgb(), T, _K()))

    def test_empty_returns_none(self):
        assert viz.grasp_candidates(_rgb(), np.zeros((0, 4, 4)), _K()) is None

    def test_behind_camera_returns_none(self):
        T = self._poses(z=-0.8)
        assert viz.grasp_candidates(_rgb(), T, _K()) is None

    def test_world_frame_poses_need_extrinsics(self):
        T = self._poses(z=0.0)
        T[:, 2, 3] = 0.05                      # world z, not camera z
        cfw = np.eye(4)
        cfw[2, 3] = 1.0                        # push in front of the camera
        assert _png(viz.grasp_candidates(_rgb(), T, _K(), cam_from_world=cfw))

    def test_mismatched_scores_are_ignored_not_fatal(self):
        out = viz.grasp_candidates(_rgb(), self._poses(5), _K(), np.array([0.5]))
        assert _png(out)


class TestDepthHeatmap:
    def test_reports_valid_fraction(self):
        d = np.full((60, 80), np.nan, np.float32)
        d[10:50, 10:70] = 0.9
        assert _png(viz.depth_heatmap(d))

    def test_all_invalid_is_still_rendered(self):
        assert _png(viz.depth_heatmap(np.zeros((20, 20), np.float32)))

    def test_squeezes_trailing_channel(self):
        assert _png(viz.depth_heatmap(np.full((20, 20, 1), 0.8, np.float32)))


class TestNeverRaises:
    """A rendering bug must not affect the robot."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda: viz.detection_boxes(None, [{"box": [1, 2, 3, 4]}]),
            lambda: viz.grasp_candidates(None, "garbage", None),
            lambda: viz.depth_heatmap("garbage"),
            lambda: viz.detection_boxes(_rgb(), "garbage"),
        ],
    )
    def test_bad_input_returns_none(self, call):
        assert call() is None


class TestInstrumentationTargetsTheRealApi:
    """A wrong class name makes every wrapper silently no-op — which is exactly
    what happened on the first attempt (control_reduced vs
    control_reduced_skill_library)."""

    def test_the_real_methods_get_wrapped(self):
        from capx.integrations.franka.control_reduced_skill_library import (
            FrankaControlApiReducedSkillLibrary as Api,
        )
        from capx.monitor import instrument

        instrument._installed = False
        instrument.instrument_tools()
        assert Api.plan_grasp.__name__ == "wrapped"
        assert Api.detect_object_owlvit.__name__ == "wrapped"

    def test_install_is_idempotent(self):
        from capx.integrations.franka.control_reduced_skill_library import (
            FrankaControlApiReducedSkillLibrary as Api,
        )
        from capx.monitor import instrument

        instrument._installed = False
        instrument.instrument_tools()
        first = Api.plan_grasp
        instrument.instrument_tools()
        assert Api.plan_grasp is first, "must not double-wrap"


class TestZeroResultsAreLoud:
    def test_no_detections_is_an_error_row(self):
        from capx.monitor.events import BUS, EventKind
        from capx.monitor import hooks as h

        BUS._history.clear()
        h.tool_image("Detection boxes", "0 detection(s) — NOTHING DETECTED", None, is_error=True)
        assert BUS._history[-1].kind == EventKind.ERROR

    def test_grasps_found_is_a_tool_row(self):
        from capx.monitor.events import BUS, EventKind
        from capx.monitor import hooks as h

        BUS._history.clear()
        h.tool_image("Contact GraspNet", "31 grasp candidate(s)", None)
        assert BUS._history[-1].kind == EventKind.TOOL


class TestAnnotatedResultsReachTheDashboard:
    """Operator report: "the images shown are not annotated at all".

    CaP-X tools log TWICE:
        _log_step(tool, "Running ...", images=RAW_RGB)      <- before
        _log_step_update(text=result, images=ANNOTATED)      <- after
    Molmo overlays its point, SAM2/SAM3 overlay their masks -- but only in the
    UPDATE. The hook wrapped _log_step alone, so every dashboard image was the
    un-annotated input frame.
    """

    def _api(self):
        import numpy as np

        from capx.integrations.base_api import ApiBase
        from capx.monitor import hooks as h

        h._installed = False
        h.install_tool_hook()

        class Api(ApiBase):
            def __init__(self):
                self._webui_enabled = False

            def functions(self):
                return {}

        return Api()

    def test_log_step_update_is_published(self):
        import numpy as np

        from capx.monitor.events import BUS, EventKind

        BUS._history.clear()
        api = self._api()
        api._log_step("Molmo Point Prompt", "Querying …", images=np.zeros((32, 32, 3), np.uint8))
        api._log_step_update(text="Result: (640, 320)", images=np.ones((32, 32, 3), np.uint8))
        tools = [e for e in BUS._history if e.kind == EventKind.TOOL]
        assert len(tools) == 2, f"expected pre-run + annotated, got {len(tools)}"
        assert tools[-1].data.get("annotated") is True
        assert tools[-1].images, "the annotated image must be attached"

    def test_pre_run_frame_is_labelled_as_input(self):
        import numpy as np

        from capx.monitor.events import BUS

        BUS._history.clear()
        api = self._api()
        api._log_step("SAM3 Text Segmentation", "Running …", images=np.zeros((16, 16, 3), np.uint8))
        e = BUS._history[-1]
        assert e.data.get("annotated") is False
        assert "[input frame]" in e.text, "must not be mistaken for the result"

    def test_update_without_an_image_publishes_nothing_extra(self):
        """A failed tool ('No point found.') has no annotated image."""
        from capx.monitor.events import BUS, EventKind

        BUS._history.clear()
        api = self._api()
        api._log_step_update(text="No point found.")
        assert not [e for e in BUS._history if e.kind == EventKind.TOOL]

    def test_annotated_event_carries_the_tool_name(self):
        """Headless runs have no CaP-X history, so the name must be remembered
        from the preceding _log_step -- otherwise every annotated row is
        labelled 'tool result' and you cannot tell which tool it came from."""
        import numpy as np

        from capx.monitor.events import BUS

        BUS._history.clear()
        api = self._api()
        api._log_step("SAM3 Text Segmentation", "Running …", images=np.zeros((8, 8, 3), np.uint8))
        api._log_step_update(text="1 mask", images=np.ones((8, 8, 3), np.uint8))
        assert BUS._history[-1].data["tool"] == "SAM3 Text Segmentation"

    def test_update_still_reaches_capx_own_logger(self):
        """Wrapping must not break CaP-X's own history."""
        import inspect

        from capx.monitor import hooks as h

        src = inspect.getsource(h.install_tool_hook)
        assert "original_update(self" in src
