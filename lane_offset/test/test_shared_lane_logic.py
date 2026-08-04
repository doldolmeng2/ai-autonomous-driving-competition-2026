"""Regression checks for the independent mission copy of timed lane control."""

from types import SimpleNamespace

from lane_offset.mission_lane_offset_node import MissionLaneOffsetNode
from lane_offset.timed_lane_offset_node import TimedLaneOffsetNggNode


def test_mission_is_independent_from_timed_lane_node():
    assert not issubclass(MissionLaneOffsetNode, TimedLaneOffsetNggNode)


def test_recognition_and_steering_methods_are_copied():
    copied_methods = (
        'segment_colors',
        'warp_to_bev',
        'make_white_mask',
        'find_center_line',
        'track_center_dashed_line',
        'measure_near_x',
        'map_line_x_to_offset',
        'publish_offset',
    )
    for method_name in copied_methods:
        assert method_name in MissionLaneOffsetNode.__dict__
        assert method_name in TimedLaneOffsetNggNode.__dict__
        assert (
            getattr(MissionLaneOffsetNode, method_name)
            is not getattr(TimedLaneOffsetNggNode, method_name)
        )


def test_initial_steering_mapping_matches_timed_lane_node():
    settings = SimpleNamespace(offset_error_limit_px=130, lane_offset_limit=45)
    for measured_x in (350, 475, 540, 605, 730):
        timed = TimedLaneOffsetNggNode.map_line_x_to_offset(
            settings, measured_x, 540
        )
        mission = MissionLaneOffsetNode.map_line_x_to_offset(
            settings, measured_x, 540
        )
        assert mission == timed
