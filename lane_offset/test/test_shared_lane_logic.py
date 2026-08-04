"""Regression checks for the independent mission copy of timed lane control."""

from types import SimpleNamespace

import numpy as np

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


def test_mission_light_gray_mask_ignores_right_half():
    node = object.__new__(MissionLaneOffsetNode)
    node.color_classes = [
        {'name': 'light_gray', 'color_bgr': (211, 211, 211)},
        {'name': 'green', 'color_bgr': (0, 255, 0)},
    ]
    segmented = np.full((2, 8, 3), 211, dtype=np.uint8)

    mask = node.make_class_color_mask(segmented, 'light_gray')

    assert np.all(mask[:, :4] == 255)
    assert np.all(mask[:, 4:] == 0)


def test_mission_light_gray_mask_ignores_pixels_right_of_green():
    node = object.__new__(MissionLaneOffsetNode)
    node.green_min_pixels = 3
    node.color_classes = [
        {'name': 'light_gray', 'color_bgr': (211, 211, 211)},
        {'name': 'green', 'color_bgr': (0, 255, 0)},
    ]
    segmented = np.full((4, 12, 3), 211, dtype=np.uint8)
    segmented[:, 2] = (0, 255, 0)
    # 오른쪽의 단일 초록 노이즈는 유효한 뭉탱이/cutoff로 취급하지 않는다.
    segmented[0, 5] = (0, 255, 0)

    mask = node.make_class_color_mask(segmented, 'light_gray')

    assert np.all(mask[:, :2] == 255)
    assert np.all(mask[:, 2:] == 0)


def test_mission_light_gray_requires_minimum_count_and_ratio():
    node = object.__new__(MissionLaneOffsetNode)
    node.green_min_pixels = 10
    node.light_gray_min_pixels = 10
    node.light_gray_min_ratio = 0.08

    assert not node.boundary_color_evidence_valid('left', 9, 100)
    assert not node.boundary_color_evidence_valid('left', 10, 200)
    assert node.boundary_color_evidence_valid('left', 10, 125)


def test_mission_green_keeps_absolute_pixel_threshold_only():
    node = object.__new__(MissionLaneOffsetNode)
    node.green_min_pixels = 10
    node.light_gray_min_pixels = 10
    node.light_gray_min_ratio = 0.08

    assert not node.boundary_color_evidence_valid('right', 9, 1)
    assert node.boundary_color_evidence_valid('right', 10, 10000)
