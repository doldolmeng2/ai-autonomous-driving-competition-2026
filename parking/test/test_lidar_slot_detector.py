import math
import numpy as np

from parking.lidar_slot_detector import (
    LidarDetectorConfig,
    LidarSlotDetector,
)
from parking.models import SlotSide


def vertical_segment(x, y_start, y_end, count=40):
    y = np.linspace(y_start, y_end, count)
    return np.column_stack((np.full_like(y, x), y))


def horizontal_surface(x_start, x_end, y, count=40):
    x = np.linspace(x_start, x_end, count)
    return np.column_stack((x, np.full_like(x, y)))


def test_detect_left_perpendicular_gap():
    first = horizontal_surface(-1.5, -0.50, 0.75)
    second = horizontal_surface(0.45, 1.45, 0.75)
    first_boundary = vertical_segment(-0.50, 0.55, 1.45)
    second_boundary = vertical_segment(0.45, 0.55, 1.45)
    points = np.vstack(
        (first, second, first_boundary, second_boundary)
    )

    detector = LidarSlotDetector(
        LidarDetectorConfig(
            x_min=-2.0,
            x_max=2.0,
            lateral_min=0.3,
            lateral_max=1.8,
            expected_slot_width=0.95,
            minimum_slot_width=0.8,
            maximum_slot_width=1.1,
            bin_size=0.05,
            minimum_points_per_bin=1,
            minimum_vehicle_segment_length=0.20,
            minimum_confidence=0.20,
        )
    )
    estimate = detector.detect(points, stamp=1.0)

    assert estimate is not None
    assert estimate.side == SlotSide.LEFT
    assert abs(estimate.width - 0.95) < 0.12
    assert estimate.confidence > 0.2
    assert math.sin(estimate.inward_yaw) > 0.0
