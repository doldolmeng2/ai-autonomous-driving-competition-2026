"""YYM LiDAR controller for reverse perpendicular parking on the right.

Scan convention used by this vehicle:
    rear=0 deg, right=+90 deg, front=+/-180 deg, left=-90 deg.

The controller holds still for a configurable startup delay, then approaches
at an explicit 0-degree steering target and, when the first parked vehicle is
confirmed, performs a calibrated left turn. LiDAR
then confirms that both parked vehicles are visible and performs one-second
midpoint-angle corrections while reversing. Once either parked vehicle enters
the 1 m ring, the controller stops for five seconds. Precision reverse starts
after that stop. A selected slot-2 or slot-3 high-camera template detects the
top horizontal white line and its calibrated longitudinal branch/corner. Their
two angle errors and the corner's horizontal position have priority over LiDAR
steering. Otherwise the existing red/green line or parked-vehicle tilt
correction is used. After two initial 0.5-second corrections the controller
stops and centers steering, then reverses continuously; white-line guidance
still takes priority if the lines first appear during that motion.
Parking completes when either original parked vehicle, not a later pillar or
unit, disappears below the rear-mounted LiDAR's horizontal x=0 line.
After a two-second parked hold, the controller executes the timed exit
sequence and finishes stopped with centered steering.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import os
import signal
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Int16MultiArray


class ParkingState(str, Enum):
    WAIT_FOR_SCAN = 'WAIT_FOR_SCAN'
    START_DELAY = 'START_DELAY'
    APPROACH_FIRST_CAR = 'APPROACH_FIRST_CAR'
    SET_LEFT_STEER = 'SET_LEFT_STEER'
    TURN_LEFT_TIMED = 'TURN_LEFT_TIMED'
    RECOGNITION_COMPLETE = 'RECOGNITION_COMPLETE'
    SETTLE_AND_ACQUIRE_GAP = 'SETTLE_AND_ACQUIRE_GAP'
    REVERSE_CENTER = 'REVERSE_CENTER'
    PARKED = 'PARKED'
    EXIT_FORWARD = 'EXIT_FORWARD'
    EXIT_SET_RIGHT_STEER = 'EXIT_SET_RIGHT_STEER'
    EXIT_RIGHT_TURN = 'EXIT_RIGHT_TURN'
    EXIT_CENTER_STEER = 'EXIT_CENTER_STEER'
    EXIT_FINAL_FORWARD = 'EXIT_FINAL_FORWARD'
    EXIT_COMPLETE = 'EXIT_COMPLETE'
    PARKING_FAILED = 'PARKING_FAILED'
    EMERGENCY_STOP = 'EMERGENCY_STOP'


class ParkingMode(str, Enum):
    RECOGNITION = 'RECOGNITION'
    PARKING = 'PARKING'
    EXIT = 'EXIT'


@dataclass
class VehicleCluster:
    points: np.ndarray
    center: np.ndarray
    axis_angle: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass
class ParkingPair:
    lower: VehicleCluster
    upper: VehicleCluster
    reference_point: np.ndarray
    gap_center_y: float
    gap_width: float
    left_clearance: float
    right_clearance: float


@dataclass
class LidarObservation:
    scan_valid: bool
    points: np.ndarray
    vehicles: list[VehicleCluster]
    right_vehicles: list[VehicleCluster]
    pair: Optional[ParkingPair]
    rear_min_distance: Optional[float]
    pair_is_fallback: bool = False


@dataclass
class ParkingLineObservation:
    valid: bool
    stamp: float
    horizontal_line: Optional[tuple[float, float, float, float]] = None
    guide_line: Optional[tuple[float, float, float, float]] = None
    anchor_point: Optional[tuple[float, float]] = None
    horizontal_angle_error_deg: float = 0.0
    guide_angle_error_deg: float = 0.0
    anchor_x_error: float = 0.0
    anchor_y_error: float = 0.0
    steering_deg: int = 0
    confidence: float = 0.0


class ParkingNodeYym(Node):
    """LiDAR-feedback parking into random right-side slot 2 or 3."""

    def __init__(self) -> None:
        super().__init__('parking_node_yym')

        slot_from_environment = os.environ.get('PARKING_SLOT_NUMBER', '2')
        parking_slot_default = (
            int(slot_from_environment)
            if slot_from_environment in ('2', '3')
            else 2
        )

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('motor_topic', '/motor_control')
        self.declare_parameter('high_camera_topic', '/camera/high/image_raw')
        self.declare_parameter('control_hz', 20.0)
        self.declare_parameter('scan_timeout_sec', 0.5)
        self.declare_parameter('scan_quality_min_points', 10)
        self.declare_parameter('invalid_scan_confirm_frames', 5)
        self.declare_parameter('startup_delay_sec', 5.0)
        # start_mode=recognition runs the normal complete sequence.
        # start_mode=parking skips recognition for debugging from an already
        # stationary pose after the timed left turn. recognition_only latches
        # a stop as soon as that recognition turn is complete.
        self.declare_parameter('start_mode', 'recognition')
        self.declare_parameter('recognition_only', False)

        self.declare_parameter('debug_view', True)
        self.declare_parameter('debug_window_name', 'parking_yym_debug')
        self.declare_parameter('debug_hz', 20.0)
        self.declare_parameter('debug_max_range_m', 4.0)
        self.declare_parameter('debug_image_size', 820)

        # Only the rear and side field is relevant after the left entry turn.
        self.declare_parameter('valid_sector_max_abs_deg', 125.0)
        self.declare_parameter('parking_min_range_m', 0.15)
        self.declare_parameter('cluster_max_range_m', 4.0)
        self.declare_parameter('cluster_neighbor_distance_m', 0.20)
        self.declare_parameter('cluster_min_points', 7)
        self.declare_parameter('obstacle_min_extent_m', 0.22)
        self.declare_parameter('right_detection_margin_m', 0.12)
        # Before the 1 m/five-second latch, only clusters on or below the
        # green horizontal x=0 line can represent the two parked vehicles.
        self.declare_parameter('pre_final_vehicle_max_x_m', 0.0)

        self.declare_parameter('first_car_confirm_frames', 3)
        # Recognition only: distant wall/noise clusters must not trigger the
        # timed left turn. Parking-mode clustering keeps its full 4 m range.
        self.declare_parameter('first_car_max_distance_m', 2.0)
        self.declare_parameter('gap_confirm_frames', 3)
        self.declare_parameter('gap_min_width_m', 0.48)
        self.declare_parameter('gap_max_width_m', 1.40)
        self.declare_parameter('gap_track_max_center_m', 0.85)

        # Recognition-mode test speed requested for the real vehicle.
        self.declare_parameter('approach_speed', 110)
        self.declare_parameter('turn_speed', 110)
        self.declare_parameter('reverse_speed', -110)
        self.declare_parameter('left_max_steer_deg', -45)
        # Legacy +/-90-degree parameters remain available only for visual
        # diagnostics. They do not control the simplified parking sequence.
        self.declare_parameter('lidar_side_gate_half_width_deg', 15.0)
        self.declare_parameter('lidar_side_gate_min_points', 3)
        self.declare_parameter('lidar_side_gate_confirm_frames', 3)
        self.declare_parameter('lidar_side_far_distance_m', 2.0)
        self.declare_parameter('lidar_side_far_half_width_deg', 5.0)
        self.declare_parameter('lidar_side_far_confirm_frames', 3)
        self.declare_parameter('lidar_side_far_stop_sec', 4.0)
        self.declare_parameter('enable_exit_sequence', False)
        self.declare_parameter('exit_wait_after_park_sec', 2.0)
        self.declare_parameter('exit_forward_duration_sec', 3.0)
        self.declare_parameter('exit_right_turn_duration_sec', 8.0)
        self.declare_parameter('exit_final_forward_duration_sec', 10.0)
        self.declare_parameter('exit_right_steer_deg', 45)
        self.declare_parameter('exit_speed', 110)
        # Real-vehicle testing showed the raw geometric angle was too weak.
        self.declare_parameter('reverse_steer_multiplier', 10.0)
        # Use gentler corrections after the 1 m trigger and five-second stop.
        self.declare_parameter('final_reverse_steer_multiplier', 5.0)
        self.declare_parameter('final_line_alignment_tolerance_deg', 5.0)
        self.declare_parameter('final_correction_duration_sec', 0.5)
        self.declare_parameter('final_correction_segment_count', 2)
        # Precision reverse begins only after the five-second final stop.
        # A valid pair of white parking lines overrides LiDAR guidance.
        self.declare_parameter('precision_camera_enabled', True)
        self.declare_parameter('parking_slot_number', parking_slot_default)
        self.declare_parameter('precision_camera_timeout_sec', 0.5)
        self.declare_parameter('precision_camera_confirm_frames', 1)
        self.declare_parameter('precision_white_value_min', 170)
        self.declare_parameter('precision_white_saturation_max', 80)
        self.declare_parameter('precision_camera_roi_top_ratio', 0.00)
        self.declare_parameter('precision_camera_roi_bottom_ratio', 0.20)
        self.declare_parameter('precision_camera_center_x_ratio', 0.50)
        self.declare_parameter('precision_horizontal_max_angle_deg', 12.0)
        self.declare_parameter('precision_horizontal_min_length_ratio', 0.30)
        self.declare_parameter('precision_guide_min_height_ratio', 0.025)
        self.declare_parameter('precision_guide_max_angle_deg', 55.0)
        self.declare_parameter('precision_slot2_target_x_ratio', 0.500)
        self.declare_parameter('precision_slot2_target_y_ratio', 0.052)
        self.declare_parameter('precision_slot2_guide_angle_deg', 0.0)
        self.declare_parameter('precision_slot3_target_x_ratio', 0.690)
        self.declare_parameter('precision_slot3_target_y_ratio', 0.052)
        self.declare_parameter('precision_slot3_guide_angle_deg', -33.0)
        self.declare_parameter('precision_horizontal_target_angle_deg', 0.0)
        self.declare_parameter('precision_horizontal_angle_gain', 0.8)
        self.declare_parameter('precision_guide_angle_gain', 0.8)
        self.declare_parameter('precision_anchor_x_gain_deg', 40.0)
        self.declare_parameter('precision_camera_steering_sign', 1.0)
        self.declare_parameter('steer_settle_sec', 0.6)
        # Recognition test: after steering settles at -45 degrees, drive for
        # this calibrated duration with maximum left steering, then stop.
        self.declare_parameter('left_turn_duration_sec', 7.0)
        # Keep publishing stop briefly before shutting down the launch.
        self.declare_parameter('recognition_shutdown_delay_sec', 0.5)
        self.declare_parameter('approach_timeout_sec', 30.0)
        self.declare_parameter('gap_acquire_timeout_sec', 4.0)
        # Zero disables the elapsed-time stop. Parking depth is determined by
        # the rear-half LiDAR point condition below, not a guessed duration.
        self.declare_parameter('reverse_timeout_sec', 0.0)
        # Reverse for one second, then stop and recompute the LiDAR-based
        # center/steering target before the next segment.
        self.declare_parameter('reverse_segment_duration_sec', 1.0)
        self.declare_parameter('reverse_measure_stop_sec', 0.4)
        self.declare_parameter('vehicle_pair_track_max_jump_m', 1.25)
        # Retained for compatibility with older launch parameter files and
        # the unused single-vehicle fallback helper. The tighter tracker
        # prevents a pillar/next unit from replacing an original vehicle.
        self.declare_parameter('final_center_gain', 1.0)
        self.declare_parameter('final_alignment_gain', 1.0)
        self.declare_parameter('final_vehicle_track_max_jump_m', 0.25)
        # Green horizontal line is x=0 at the rear-mounted LiDAR. Stop with
        # centered steering on the first scan with no valid points below it,
        # then confirm while stationary to reject a one-frame dropout.
        self.declare_parameter('rear_half_stop_margin_m', 0.0)
        self.declare_parameter('rear_half_empty_confirm_frames', 3)
        # The second debug range ring is 1.0 m. Once any detected vehicle
        # contributes a point inside it, stop for five seconds at 0 deg and
        # then reverse straight.
        self.declare_parameter('straight_reverse_radius_m', 1.0)
        self.declare_parameter('straight_reverse_stop_sec', 5.0)
        # With only one parked vehicle visible, its PCA line replaces the
        # unavailable two-vehicle midpoint. A vertical debug line is 0 deg.
        self.declare_parameter('single_vehicle_angle_deadband_deg', 2.0)

        self.declare_parameter('rear_hard_stop_angle_deg', 11.0)
        self.declare_parameter('rear_hard_stop_distance_m', 0.18)
        self.declare_parameter('vehicle_width_m', 0.38)
        self.declare_parameter('minimum_side_clearance_m', 0.05)
        # LiDAR is mounted at the vehicle rear end in this vehicle.
        self.declare_parameter('lidar_to_rear_bumper_m', 0.0)

        self.scan_topic = str(self._value('scan_topic'))
        self.motor_topic = str(self._value('motor_topic'))
        self.high_camera_topic = str(self._value('high_camera_topic'))
        self.control_hz = max(1.0, float(self._value('control_hz')))
        self.scan_timeout = max(0.05, float(self._value('scan_timeout_sec')))
        self.scan_quality_min_points = int(self._value('scan_quality_min_points'))
        self.invalid_scan_confirm_frames = int(
            self._value('invalid_scan_confirm_frames')
        )
        self.startup_delay = max(
            0.0, float(self._value('startup_delay_sec'))
        )
        requested_start_mode = str(self._value('start_mode')).strip().lower()
        if requested_start_mode not in ('recognition', 'parking'):
            self.get_logger().warn(
                f'Unknown start_mode={requested_start_mode!r}; '
                'using recognition'
            )
            requested_start_mode = 'recognition'
        self.start_mode = (
            ParkingMode.PARKING
            if requested_start_mode == 'parking'
            else ParkingMode.RECOGNITION
        )
        self.recognition_only = bool(self._value('recognition_only'))

        self.debug_view = bool(self._value('debug_view'))
        self.debug_window_name = str(self._value('debug_window_name'))
        self.debug_hz = max(1.0, float(self._value('debug_hz')))
        self.debug_max_range = max(0.5, float(self._value('debug_max_range_m')))
        self.debug_image_size = max(500, int(self._value('debug_image_size')))

        self.valid_sector_max_abs = math.radians(
            float(self._value('valid_sector_max_abs_deg'))
        )
        self.parking_min_range = max(
            0.0, float(self._value('parking_min_range_m'))
        )
        self.cluster_max_range = max(
            self.parking_min_range,
            float(self._value('cluster_max_range_m')),
        )
        self.cluster_neighbor_distance = max(
            0.02, float(self._value('cluster_neighbor_distance_m'))
        )
        self.cluster_min_points = max(2, int(self._value('cluster_min_points')))
        self.obstacle_min_extent = max(
            0.05, float(self._value('obstacle_min_extent_m'))
        )
        self.right_detection_margin = max(
            0.0, float(self._value('right_detection_margin_m'))
        )
        self.pre_final_vehicle_max_x = float(
            self._value('pre_final_vehicle_max_x_m')
        )

        self.first_car_confirm_frames = max(
            1, int(self._value('first_car_confirm_frames'))
        )
        self.first_car_max_distance = max(
            self.parking_min_range,
            float(self._value('first_car_max_distance_m')),
        )
        self.gap_confirm_frames = max(1, int(self._value('gap_confirm_frames')))
        self.gap_min_width = float(self._value('gap_min_width_m'))
        self.gap_max_width = float(self._value('gap_max_width_m'))
        self.gap_track_max_center = float(
            self._value('gap_track_max_center_m')
        )

        self.approach_speed = int(self._value('approach_speed'))
        self.turn_speed = int(self._value('turn_speed'))
        self.reverse_speed = -abs(int(self._value('reverse_speed')))
        self.left_max_steer = -abs(int(self._value('left_max_steer_deg')))
        self.lidar_side_gate_half_width = math.radians(max(
            1.0, float(self._value('lidar_side_gate_half_width_deg'))
        ))
        self.lidar_side_gate_min_points = max(
            1, int(self._value('lidar_side_gate_min_points'))
        )
        self.lidar_side_gate_confirm_frames = max(
            1, int(self._value('lidar_side_gate_confirm_frames'))
        )
        self.lidar_side_far_distance = max(
            0.1, float(self._value('lidar_side_far_distance_m'))
        )
        self.lidar_side_far_half_width = math.radians(max(
            1.0, float(self._value('lidar_side_far_half_width_deg'))
        ))
        self.lidar_side_far_confirm_frames = max(
            1, int(self._value('lidar_side_far_confirm_frames'))
        )
        self.lidar_side_far_stop = max(
            0.0, float(self._value('lidar_side_far_stop_sec'))
        )
        self.enable_exit_sequence = bool(
            self._value('enable_exit_sequence')
        )
        self.exit_wait_after_park = max(
            0.0, float(self._value('exit_wait_after_park_sec'))
        )
        self.exit_forward_duration = max(
            0.0, float(self._value('exit_forward_duration_sec'))
        )
        self.exit_right_turn_duration = max(
            0.0, float(self._value('exit_right_turn_duration_sec'))
        )
        self.exit_final_forward_duration = max(
            0.0, float(self._value('exit_final_forward_duration_sec'))
        )
        self.exit_right_steer = abs(
            int(self._value('exit_right_steer_deg'))
        )
        self.exit_speed = abs(int(self._value('exit_speed')))
        self.reverse_steer_multiplier = max(
            0.0, float(self._value('reverse_steer_multiplier'))
        )
        self.final_reverse_steer_multiplier = max(
            0.0, float(self._value('final_reverse_steer_multiplier'))
        )
        self.final_line_alignment_tolerance = max(
            0.0,
            float(self._value('final_line_alignment_tolerance_deg')),
        )
        self.final_correction_duration = max(
            0.1, float(self._value('final_correction_duration_sec'))
        )
        self.final_correction_segment_count = max(
            1, int(self._value('final_correction_segment_count'))
        )
        self.precision_camera_enabled = bool(
            self._value('precision_camera_enabled')
        )
        self.parking_slot_number = int(self._value('parking_slot_number'))
        if self.parking_slot_number not in (2, 3):
            self.get_logger().warn(
                f'parking_slot_number={self.parking_slot_number} is invalid; '
                'using slot 2'
            )
            self.parking_slot_number = 2
        self.precision_camera_timeout = max(
            0.05, float(self._value('precision_camera_timeout_sec'))
        )
        self.precision_camera_confirm_frames = max(
            1, int(self._value('precision_camera_confirm_frames'))
        )
        self.precision_white_value_min = int(np.clip(
            self._value('precision_white_value_min'), 0, 255
        ))
        self.precision_white_saturation_max = int(np.clip(
            self._value('precision_white_saturation_max'), 0, 255
        ))
        self.precision_camera_roi_top = float(np.clip(
            self._value('precision_camera_roi_top_ratio'), 0.0, 0.90
        ))
        self.precision_camera_roi_bottom = float(np.clip(
            self._value('precision_camera_roi_bottom_ratio'),
            self.precision_camera_roi_top + 0.05,
            1.0,
        ))
        self.precision_camera_center_x = float(np.clip(
            self._value('precision_camera_center_x_ratio'), 0.20, 0.80
        ))
        self.precision_horizontal_max_angle = max(
            1.0, float(self._value('precision_horizontal_max_angle_deg'))
        )
        self.precision_horizontal_min_length = float(np.clip(
            self._value('precision_horizontal_min_length_ratio'), 0.10, 0.90
        ))
        self.precision_guide_min_height = float(np.clip(
            self._value('precision_guide_min_height_ratio'), 0.01, 0.30
        ))
        self.precision_guide_max_angle = max(
            5.0, float(self._value('precision_guide_max_angle_deg'))
        )
        if self.parking_slot_number == 2:
            self.precision_target_x = float(
                self._value('precision_slot2_target_x_ratio')
            )
            self.precision_target_y = float(
                self._value('precision_slot2_target_y_ratio')
            )
            self.precision_target_guide_angle = float(
                self._value('precision_slot2_guide_angle_deg')
            )
        else:
            self.precision_target_x = float(
                self._value('precision_slot3_target_x_ratio')
            )
            self.precision_target_y = float(
                self._value('precision_slot3_target_y_ratio')
            )
            self.precision_target_guide_angle = float(
                self._value('precision_slot3_guide_angle_deg')
            )
        self.precision_target_x = float(np.clip(
            self.precision_target_x, 0.05, 0.95
        ))
        self.precision_target_y = float(np.clip(
            self.precision_target_y, 0.0, 0.50
        ))
        self.precision_horizontal_target_angle = float(
            self._value('precision_horizontal_target_angle_deg')
        )
        self.precision_horizontal_angle_gain = float(
            self._value('precision_horizontal_angle_gain')
        )
        self.precision_guide_angle_gain = float(
            self._value('precision_guide_angle_gain')
        )
        self.precision_anchor_x_gain_deg = float(
            self._value('precision_anchor_x_gain_deg')
        )
        self.precision_camera_steering_sign = float(
            self._value('precision_camera_steering_sign')
        )
        self.steer_settle_sec = float(self._value('steer_settle_sec'))
        self.left_turn_duration_sec = max(
            0.1, float(self._value('left_turn_duration_sec'))
        )
        self.recognition_shutdown_delay_sec = max(
            0.1, float(self._value('recognition_shutdown_delay_sec'))
        )
        self.approach_timeout = float(self._value('approach_timeout_sec'))
        self.gap_acquire_timeout = float(
            self._value('gap_acquire_timeout_sec')
        )
        self.reverse_timeout = max(
            0.0, float(self._value('reverse_timeout_sec'))
        )
        self.reverse_segment_duration = max(
            0.1, float(self._value('reverse_segment_duration_sec'))
        )
        self.reverse_measure_stop = max(
            0.1, float(self._value('reverse_measure_stop_sec'))
        )
        self.vehicle_pair_track_max_jump = max(
            0.1, float(self._value('vehicle_pair_track_max_jump_m'))
        )
        self.final_center_gain = max(
            0.0, float(self._value('final_center_gain'))
        )
        self.final_alignment_gain = max(
            0.0, float(self._value('final_alignment_gain'))
        )
        self.final_vehicle_track_max_jump = max(
            0.05, float(self._value('final_vehicle_track_max_jump_m'))
        )
        self.rear_half_stop_margin = max(
            0.0, float(self._value('rear_half_stop_margin_m'))
        )
        self.rear_half_empty_confirm_frames = max(
            1, int(self._value('rear_half_empty_confirm_frames'))
        )
        self.straight_reverse_radius = max(
            0.1, float(self._value('straight_reverse_radius_m'))
        )
        self.straight_reverse_stop = max(
            0.0, float(self._value('straight_reverse_stop_sec'))
        )
        self.single_vehicle_angle_deadband = max(
            0.0, float(self._value('single_vehicle_angle_deadband_deg'))
        )

        self.rear_hard_stop_angle = math.radians(
            float(self._value('rear_hard_stop_angle_deg'))
        )
        self.rear_hard_stop_distance = float(
            self._value('rear_hard_stop_distance_m')
        )
        self.vehicle_width = float(self._value('vehicle_width_m'))
        self.minimum_side_clearance = float(
            self._value('minimum_side_clearance_m')
        )
        self.lidar_to_rear_bumper = float(
            self._value('lidar_to_rear_bumper_m')
        )

        now = time.monotonic()
        self.startup_delay_deadline = now + self.startup_delay
        self.mode = self.start_mode
        self.state = ParkingState.WAIT_FOR_SCAN
        self.state_started_at = now
        self.last_scan_at: Optional[float] = None
        self.lidar_side_left_m: Optional[float] = None
        self.lidar_side_right_m: Optional[float] = None
        self.lidar_raw_left_m: Optional[float] = None
        self.lidar_raw_right_m: Optional[float] = None
        self.lidar_side_far_frames = 0
        self.lidar_side_gate_frames = 0
        self.lidar_side_gate_seen = False
        self.last_pair_at: Optional[float] = None
        self.last_command = (0, 0)
        self.failure_reason = ''
        self.shutdown_started = False
        self.reverse_segment_started_at: Optional[float] = None
        self.reverse_phase_started_at: Optional[float] = None
        self.reverse_phase = 'IDLE'
        self.reverse_segment_steer = 0
        self.reverse_segment_index = 0
        self.final_correction_count = 0
        self.reverse_segment_drive_duration = self.reverse_segment_duration
        self.lower_vehicle_track_center: Optional[np.ndarray] = None
        self.upper_vehicle_track_center: Optional[np.ndarray] = None
        self.rear_half_lidar_point_count = 0
        self.rear_half_empty_frames = 0
        self.rear_half_points_seen = False
        self.reference_lower_below_count = 0
        self.reference_upper_below_count = 0
        self.reference_lower_missing_frames = 0
        self.reference_upper_missing_frames = 0
        self.reference_lower_gone = False
        self.reference_upper_gone = False
        self.final_completion_tracking_started = False
        self.current_reference_lower: Optional[VehicleCluster] = None
        self.current_reference_upper: Optional[VehicleCluster] = None
        self.final_target_half_gap = self.gap_min_width / 2.0
        self.straight_reverse_latched = False
        self.straight_reverse_started = False
        self.straight_reverse_trigger_distance = math.inf
        self.invalid_scan_count = 0
        self.first_car_frames = 0
        self.gap_frames = 0
        self.latest_scan: Optional[LaserScan] = None
        self.observation = self._empty_observation()
        self.parking_line_observation = ParkingLineObservation(False, 0.0)
        self.precision_camera_valid_frames = 0
        self.precision_camera_debug_image: Optional[np.ndarray] = None

        self.motor_publisher = self.create_publisher(
            Int16MultiArray, self.motor_topic, 10
        )
        self.create_subscription(
            LaserScan, self.scan_topic, self.scan_callback, 10
        )
        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(
            Image,
            self.high_camera_topic,
            self.high_camera_callback,
            camera_qos,
        )
        self.create_timer(1.0 / self.control_hz, self.control_tick)
        if self.debug_view:
            self.create_timer(1.0 / self.debug_hz, self.draw_debug)

        self.get_logger().info(
            f'parking_node_yym: startup stop {self.startup_delay:.1f}s '
            '-> straight approach -> first-car trigger -> '
            f'first-car range<={self.first_car_max_distance:.2f}m -> '
            f'{self.left_turn_duration_sec:.2f}s max-left timed turn '
            f'-> pre-final vehicle x<={self.pre_final_vehicle_max_x:.2f}m '
            'rear-half filter '
            '-> strict gap pair or two-cluster fallback midpoint '
            '-> repeated 1s midpoint-angle reverse corrections '
            f'-> {self.straight_reverse_radius:.2f}m ring stop for '
            f'{self.straight_reverse_stop:.1f}s '
            f'-> {self.final_correction_segment_count} x '
            f'{self.final_correction_duration:.1f}s '
            f'conditional corrections (LiDAR line tolerance +/-'
            f'{self.final_line_alignment_tolerance:.1f}deg) '
            f'with slot-{self.parking_slot_number} high-camera white-line '
            f'template priority on '
            f'{self.high_camera_topic} '
            '-> mandatory stop + steer=0 settle '
            '-> steer=0 continuous reverse '
            '-> either original vehicle clears line -> PARKED '
            f'-> exit_enabled={self.enable_exit_sequence} '
            f'-> wait {self.exit_wait_after_park:.1f}s '
            f'-> forward {self.exit_forward_duration:.1f}s '
            f'-> max-right {self.exit_right_turn_duration:.1f}s '
            f'-> forward {self.exit_final_forward_duration:.1f}s '
            '-> EXIT_COMPLETE; '
            f'start_mode={self.start_mode.value}, '
            f'recognition_only={self.recognition_only}'
        )

    def _value(self, name: str):
        return self.get_parameter(name).value

    @staticmethod
    def _empty_observation() -> LidarObservation:
        return LidarObservation(
            False, np.empty((0, 2)), [], [], None, None
        )

    def scan_callback(self, msg: LaserScan) -> None:
        now = time.monotonic()
        self.latest_scan = msg
        self.last_scan_at = now
        observation = self.observe(msg)
        if not observation.scan_valid:
            self.invalid_scan_count += 1
            return

        self.invalid_scan_count = 0
        if (
            self.state == ParkingState.SETTLE_AND_ACQUIRE_GAP
            and observation.pair is None
        ):
            fallback_pair = self.fallback_visible_pair(
                observation.vehicles
            )
            if fallback_pair is not None:
                observation.pair = fallback_pair
                observation.pair_is_fallback = True
        if self.state == ParkingState.REVERSE_CENTER:
            tracked_pair = self.track_parking_pair(
                observation.vehicles,
                max_jump=(
                    self.final_vehicle_track_max_jump
                    if self.straight_reverse_latched
                    else self.vehicle_pair_track_max_jump
                ),
            )
            if tracked_pair is None and not self.straight_reverse_latched:
                tracked_pair = self.fallback_visible_pair(
                    observation.vehicles
                )
            if tracked_pair is not None:
                observation.pair = tracked_pair
        self.observation = observation
        if observation.pair is not None:
            self.last_pair_at = now
        if self.state == ParkingState.REVERSE_CENTER:
            self.update_rear_half_stop_observation(observation)
            self.update_straight_reverse_condition(observation.vehicles)
            if self.final_completion_tracking_started:
                self.update_reference_vehicle_completion(
                    observation.vehicles,
                    observation.pair,
                )
        if self.state == ParkingState.APPROACH_FIRST_CAR:
            close_right_vehicles = [
                vehicle
                for vehicle in observation.right_vehicles
                if self.vehicle_nearest_distance(vehicle)
                <= self.first_car_max_distance
            ]
            self.first_car_frames = (
                self.first_car_frames + 1
                if close_right_vehicles else 0
            )
        else:
            self.first_car_frames = 0

        if self.state == ParkingState.SETTLE_AND_ACQUIRE_GAP:
            self.gap_frames = (
                self.gap_frames + 1 if observation.pair is not None else 0
            )
        else:
            self.gap_frames = 0

    def high_camera_callback(self, msg: Image) -> None:
        """Detect both white slot lines only during precision reverse."""
        now = time.monotonic()
        if not self.precision_camera_enabled:
            return
        try:
            frame = self.camera_image_to_bgr(msg)
        except (ValueError, cv2.error) as error:
            self.get_logger().warning(
                f'High-camera conversion failed: {error}',
                throttle_duration_sec=2.0,
            )
            self.parking_line_observation = ParkingLineObservation(False, now)
            self.precision_camera_valid_frames = 0
            return

        # Merely subscribing to the camera must not alter any behavior before
        # the existing five-second stop has completed.
        if (
            self.state != ParkingState.REVERSE_CENTER
            or not self.straight_reverse_started
        ):
            self.parking_line_observation = ParkingLineObservation(False, now)
            self.precision_camera_valid_frames = 0
            if self.debug_view:
                self.precision_camera_debug_image = frame.copy()
            return

        observation, debug_image = self.detect_parking_lines(frame, now)
        if observation.valid:
            self.precision_camera_valid_frames += 1
        else:
            self.precision_camera_valid_frames = 0
        observation.valid = (
            observation.valid
            and self.precision_camera_valid_frames
            >= self.precision_camera_confirm_frames
        )
        self.parking_line_observation = observation
        if self.debug_view:
            self.precision_camera_debug_image = debug_image

    @staticmethod
    def camera_image_to_bgr(msg: Image) -> np.ndarray:
        """Convert the encodings published by sensor_topic to BGR."""
        data = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding in ('yuv422_yuy2', 'yuyv', 'yuyv422'):
            yuyv = data.reshape((msg.height, msg.width, 2))
            return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
        if msg.encoding in ('bgr8', '8UC3'):
            return data.reshape((msg.height, msg.width, 3)).copy()
        if msg.encoding == 'rgb8':
            rgb = data.reshape((msg.height, msg.width, 3))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if msg.encoding in ('mono8', '8UC1'):
            mono = data.reshape((msg.height, msg.width))
            return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
        raise ValueError(f'unsupported encoding {msg.encoding!r}')

    def detect_parking_lines(
        self,
        frame: np.ndarray,
        stamp: float,
    ) -> tuple[ParkingLineObservation, np.ndarray]:
        """Visual-servo toward the selected slot's calibrated line template."""
        height, width = frame.shape[:2]
        debug_image = frame.copy()
        if height < 40 or width < 40:
            return ParkingLineObservation(False, stamp), debug_image

        roi_top = int(round(height * self.precision_camera_roi_top))
        roi_bottom = int(round(height * self.precision_camera_roi_bottom))
        roi_bottom = max(roi_top + 10, min(height, roi_bottom))
        hsv = cv2.cvtColor(frame[roi_top:roi_bottom], cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(
            hsv,
            (0, 0, self.precision_white_value_min),
            (179, self.precision_white_saturation_max, 255),
        )
        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)),
        )
        white_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )

        full_mask = np.zeros((height, width), dtype=np.uint8)
        full_mask[roi_top:roi_bottom] = white_mask

        edges = cv2.Canny(full_mask, 50, 150)
        segments = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            25,
            minLineLength=max(
                20, int(round(width * self.precision_horizontal_min_length))
            ),
            maxLineGap=max(10, int(round(width * 0.04))),
        )
        horizontal_candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        if segments is not None:
            for x1, y1, x2, y2 in segments[:, 0]:
                dx = float(x2 - x1)
                dy = float(y2 - y1)
                length = math.hypot(dx, dy)
                angle = math.degrees(math.atan2(dy, dx))
                if angle > 90.0:
                    angle -= 180.0
                elif angle < -90.0:
                    angle += 180.0
                if abs(angle) <= self.precision_horizontal_max_angle:
                    target_y = height * self.precision_target_y
                    center_y = 0.5 * (y1 + y2)
                    score = length - 0.20 * abs(center_y - target_y)
                    horizontal_candidates.append(
                        (score, (int(x1), int(y1), int(x2), int(y2)))
                    )
        if not horizontal_candidates:
            return self.finish_parking_line_debug(
                ParkingLineObservation(False, stamp),
                debug_image,
                full_mask,
                roi_top,
                roi_bottom,
            )

        _, seed_line = max(horizontal_candidates, key=lambda item: item[0])
        sx1, sy1, sx2, sy2 = seed_line
        seed_dx = max(float(sx2 - sx1), 1.0)
        seed_slope = float(sy2 - sy1) / seed_dx
        seed_intercept = float(sy1) - seed_slope * float(sx1)
        mask_y, mask_x = np.nonzero(full_mask)
        horizontal_band = np.abs(
            mask_y - (seed_slope * mask_x + seed_intercept)
        ) <= max(4.0, height * 0.012)
        horizontal_x = mask_x[horizontal_band]
        horizontal_y = mask_y[horizontal_band]
        if len(horizontal_x) < 20:
            return self.finish_parking_line_debug(
                ParkingLineObservation(False, stamp),
                debug_image,
                full_mask,
                roi_top,
                roi_bottom,
            )
        horizontal_fit = cv2.fitLine(
            np.column_stack((horizontal_x, horizontal_y)).astype(np.float32),
            cv2.DIST_HUBER,
            0,
            0.01,
            0.01,
        ).reshape(-1)
        hvx, hvy, hx0, hy0 = (float(value) for value in horizontal_fit)
        if abs(hvx) < 0.20:
            return self.finish_parking_line_debug(
                ParkingLineObservation(False, stamp),
                debug_image,
                full_mask,
                roi_top,
                roi_bottom,
            )
        if hvx < 0.0:
            hvx, hvy = -hvx, -hvy
        horizontal_slope = hvy / hvx
        horizontal_intercept = hy0 - horizontal_slope * hx0
        horizontal_line = (
            0.0,
            horizontal_intercept,
            float(width - 1),
            horizontal_slope * float(width - 1) + horizontal_intercept,
        )
        horizontal_angle = math.degrees(math.atan2(hvy, hvx))

        column_indices = np.arange(width, dtype=np.float32)[None, :]
        row_indices = np.arange(height, dtype=np.float32)[:, None]
        horizontal_y_map = (
            horizontal_slope * column_indices + horizontal_intercept
        )
        # Removing the horizontal strip separates each short longitudinal
        # branch into its own connected component, including the slot-3 corner.
        guide_mask = np.where(
            row_indices < horizontal_y_map - max(3.0, height * 0.012),
            full_mask,
            0,
        ).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            guide_mask, connectivity=8
        )
        guide_candidates: list[
            tuple[
                float,
                float,
                tuple[float, float],
                tuple[float, float, float, float],
                float,
            ]
        ] = []
        minimum_area = max(8, int(round(width * height * 0.00003)))
        minimum_height = max(5.0, height * self.precision_guide_min_height)
        target_x_px = width * self.precision_target_x
        for label in range(1, count):
            _, _, component_width, component_height, area = (
                int(value) for value in stats[label]
            )
            if (
                area < minimum_area
                or component_height < minimum_height
                or component_height < 0.45 * max(component_width, 1)
            ):
                continue
            ys, xs = np.nonzero(labels == label)
            unique_rows = np.unique(ys)
            if len(unique_rows) < minimum_height:
                continue
            row_centers = np.array([
                float(np.median(xs[ys == row])) for row in unique_rows
            ])
            guide_dx_dy, guide_intercept = np.polyfit(
                unique_rows.astype(np.float64),
                row_centers,
                1,
            )
            guide_dx_dy = float(guide_dx_dy)
            guide_intercept = float(guide_intercept)
            # The guide vector points upward, hence its image x component is
            # the negative of dx/dy measured with increasing image y.
            guide_angle = math.degrees(math.atan2(-guide_dx_dy, 1.0))
            if abs(guide_angle) > self.precision_guide_max_angle:
                continue
            denominator = 1.0 - horizontal_slope * guide_dx_dy
            if abs(denominator) < 0.05:
                continue
            anchor_x = (
                guide_dx_dy * horizontal_intercept + guide_intercept
            ) / denominator
            anchor_y = (
                horizontal_slope * anchor_x + horizontal_intercept
            )
            if not (
                -0.05 * width <= anchor_x <= 1.05 * width
                and roi_top - 5 <= anchor_y <= roi_bottom + 5
            ):
                continue
            line_top_y = float(max(roi_top, int(np.min(ys))))
            line_top_x = guide_dx_dy * line_top_y + guide_intercept
            score = (
                component_height / max(height, 1)
                + 0.25 * area / max(component_width * component_height, 1)
                - 0.80 * abs(anchor_x - target_x_px) / max(width, 1)
            )
            guide_candidates.append((
                score,
                guide_angle,
                (anchor_x, anchor_y),
                (anchor_x, anchor_y, line_top_x, line_top_y),
                component_height / max(height, 1),
            ))
        horizontal_observation = ParkingLineObservation(
            False,
            stamp,
            horizontal_line=horizontal_line,
        )
        if not guide_candidates:
            return self.finish_parking_line_debug(
                horizontal_observation,
                debug_image,
                full_mask,
                roi_top,
                roi_bottom,
            )

        (
            _,
            guide_angle,
            anchor_point,
            guide_line,
            guide_height_ratio,
        ) = max(guide_candidates, key=lambda item: item[0])
        anchor_x, anchor_y = anchor_point
        horizontal_error = (
            horizontal_angle - self.precision_horizontal_target_angle
        )
        guide_error = guide_angle - self.precision_target_guide_angle
        anchor_x_error = anchor_x / max(width, 1) - self.precision_target_x
        anchor_y_error = anchor_y / max(height, 1) - self.precision_target_y
        calculated_steer = self.precision_camera_steering_sign * (
            self.precision_horizontal_angle_gain * horizontal_error
            + self.precision_guide_angle_gain * guide_error
            + self.precision_anchor_x_gain_deg * anchor_x_error
        )
        steering = int(round(np.clip(calculated_steer, -45.0, 45.0)))
        horizontal_span = (
            float(np.max(horizontal_x) - np.min(horizontal_x))
            / max(width, 1)
        )
        confidence = float(np.clip(
            min(
                horizontal_span,
                guide_height_ratio / max(
                    2.0 * self.precision_guide_min_height, 0.01
                ),
            ),
            0.0,
            1.0,
        ))
        observation = ParkingLineObservation(
            True,
            stamp,
            horizontal_line=horizontal_line,
            guide_line=guide_line,
            anchor_point=anchor_point,
            horizontal_angle_error_deg=horizontal_error,
            guide_angle_error_deg=guide_error,
            anchor_x_error=anchor_x_error,
            anchor_y_error=anchor_y_error,
            steering_deg=steering,
            confidence=confidence,
        )
        return self.finish_parking_line_debug(
            observation,
            debug_image,
            full_mask,
            roi_top,
            roi_bottom,
        )

    def finish_parking_line_debug(
        self,
        observation: ParkingLineObservation,
        image: np.ndarray,
        mask: np.ndarray,
        roi_top: int,
        roi_bottom: int,
    ) -> tuple[ParkingLineObservation, np.ndarray]:
        """Draw exactly what the precision-reverse camera controller uses."""
        mask_overlay = np.zeros_like(image)
        mask_overlay[:, :, 1] = mask
        image = cv2.addWeighted(image, 1.0, mask_overlay, 0.25, 0.0)
        height, width = image.shape[:2]
        center_x = int(round(width * self.precision_camera_center_x))
        cv2.line(image, (center_x, 0), (center_x, height - 1),
                 (0, 0, 255), 2)
        cv2.rectangle(image, (0, roi_top), (width - 1, roi_bottom - 1),
                      (100, 100, 100), 1)
        target_point = (
            int(round(width * self.precision_target_x)),
            int(round(height * self.precision_target_y)),
        )
        cv2.drawMarker(
            image,
            target_point,
            (255, 0, 255),
            cv2.MARKER_CROSS,
            22,
            2,
        )
        if observation.horizontal_line is not None:
            line = tuple(
                int(round(value)) for value in observation.horizontal_line
            )
            cv2.line(image, (line[0], line[1]), (line[2], line[3]),
                     (0, 255, 255), 3)
        if observation.guide_line is not None:
            line = tuple(int(round(value)) for value in observation.guide_line)
            cv2.line(image, (line[0], line[1]), (line[2], line[3]),
                     (255, 100, 0), 4)
        if observation.anchor_point is not None:
            anchor = tuple(
                int(round(value)) for value in observation.anchor_point
            )
            cv2.circle(image, anchor, 7, (0, 255, 0), -1)
        status = 'CAMERA PRIORITY' if observation.valid else 'LiDAR fallback'
        color = (0, 255, 0) if observation.valid else (0, 200, 255)
        cv2.putText(
            image,
            status,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            color,
            2,
            cv2.LINE_AA,
        )
        if observation.valid:
            cv2.putText(
                image,
                f'slot={self.parking_slot_number} '
                f'h={observation.horizontal_angle_error_deg:+.1f}deg '
                f'v={observation.guide_angle_error_deg:+.1f}deg '
                f'x={observation.anchor_x_error:+.3f} '
                f'y={observation.anchor_y_error:+.3f} '
                f'steer={observation.steering_deg:+d}',
                (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return observation, image

    def active_parking_line_observation(
        self,
        now: Optional[float] = None,
    ) -> Optional[ParkingLineObservation]:
        """Return only a confirmed, fresh two-line camera measurement."""
        observation = self.parking_line_observation
        current_time = time.monotonic() if now is None else now
        if (
            not self.precision_camera_enabled
            or not observation.valid
            or current_time - observation.stamp > self.precision_camera_timeout
        ):
            return None
        return observation

    def lidar_side_vehicle_distances(
        self, vehicles: list[VehicleCluster]
    ) -> Optional[tuple[float, float]]:
        """Return -90/+90 distances from two distinct vehicle clusters."""
        left_candidates: list[tuple[int, float]] = []
        right_candidates: list[tuple[int, float]] = []
        half_width = self.lidar_side_gate_half_width

        for index, vehicle in enumerate(vehicles):
            if len(vehicle.points) == 0:
                continue
            scan_angles = np.arctan2(
                -vehicle.points[:, 1],
                -vehicle.points[:, 0],
            )
            left_count = int(np.count_nonzero(
                np.abs(scan_angles + math.pi / 2.0) <= half_width
            ))
            right_count = int(np.count_nonzero(
                np.abs(scan_angles - math.pi / 2.0) <= half_width
            ))
            if left_count >= self.lidar_side_gate_min_points:
                left_mask = (
                    np.abs(scan_angles + math.pi / 2.0) <= half_width
                )
                left_candidates.append((
                    index,
                    float(np.median(np.linalg.norm(
                        vehicle.points[left_mask], axis=1
                    ))),
                ))
            if right_count >= self.lidar_side_gate_min_points:
                right_mask = (
                    np.abs(scan_angles - math.pi / 2.0) <= half_width
                )
                right_candidates.append((
                    index,
                    float(np.median(np.linalg.norm(
                        vehicle.points[right_mask], axis=1
                    ))),
                ))

        distinct_pairs = [
            (left_distance + right_distance, left_distance, right_distance)
            for left_index, left_distance in left_candidates
            for right_index, right_distance in right_candidates
            if left_index != right_index
        ]
        if not distinct_pairs:
            return None
        _, left_distance, right_distance = min(distinct_pairs)
        return left_distance, right_distance

    def lidar_raw_side_distances(
        self, msg: LaserScan
    ) -> tuple[Optional[float], Optional[float]]:
        """Return raw median ranges near scan -90 and +90 degrees."""
        left_ranges: list[float] = []
        right_ranges: list[float] = []
        for index, raw_distance in enumerate(msg.ranges):
            distance = float(raw_distance)
            if (
                not math.isfinite(distance)
                or distance < msg.range_min
                or distance > msg.range_max
            ):
                continue
            angle = msg.angle_min + index * msg.angle_increment
            angle = math.atan2(math.sin(angle), math.cos(angle))
            if (
                abs(angle + math.pi / 2.0)
                <= self.lidar_side_far_half_width
            ):
                left_ranges.append(distance)
            if (
                abs(angle - math.pi / 2.0)
                <= self.lidar_side_far_half_width
            ):
                right_ranges.append(distance)
        return (
            float(np.median(left_ranges)) if left_ranges else None,
            float(np.median(right_ranges)) if right_ranges else None,
        )

    def held_steering_command(self) -> int:
        """Reuse the last steering target already sent through /motor_control."""
        return int(self.last_command[0])

    def update_rear_half_stop_observation(
        self, observation: LidarObservation
    ) -> None:
        """Track all valid LiDAR points below the green horizontal x=0 line."""
        self.rear_half_lidar_point_count = int(np.count_nonzero(
            observation.points[:, 0] < -self.rear_half_stop_margin
        ))
        if self.rear_half_lidar_point_count > 0:
            self.rear_half_points_seen = True
            self.rear_half_empty_frames = 0
        elif (
            self.rear_half_points_seen
            or self.straight_reverse_started
        ):
            self.rear_half_empty_frames += 1

    def match_reference_vehicle(
        self,
        vehicles: list[VehicleCluster],
        target_center: Optional[np.ndarray],
        excluded: Optional[VehicleCluster] = None,
    ) -> Optional[VehicleCluster]:
        """Match one original parked vehicle without adopting a distant unit."""
        if target_center is None:
            return None
        candidates = [
            (
                float(np.linalg.norm(vehicle.center - target_center)),
                vehicle,
            )
            for vehicle in vehicles
            if vehicle is not excluded
        ]
        if not candidates:
            return None
        distance, vehicle = min(candidates, key=lambda item: item[0])
        if distance > self.final_vehicle_track_max_jump:
            return None
        return vehicle

    def update_reference_vehicle_completion(
        self,
        vehicles: list[VehicleCluster],
        pair: Optional[ParkingPair],
    ) -> None:
        """Latch each original vehicle gone once its below-line points vanish."""
        if pair is not None:
            lower_vehicle = pair.lower
            upper_vehicle = pair.upper
        else:
            lower_vehicle = self.match_reference_vehicle(
                vehicles, self.lower_vehicle_track_center
            )
            upper_vehicle = self.match_reference_vehicle(
                vehicles,
                self.upper_vehicle_track_center,
                excluded=lower_vehicle,
            )

        self.current_reference_lower = (
            None if self.reference_lower_gone else lower_vehicle
        )
        self.current_reference_upper = (
            None if self.reference_upper_gone else upper_vehicle
        )
        if lower_vehicle is not None:
            self.lower_vehicle_track_center = lower_vehicle.center.copy()
        if upper_vehicle is not None:
            self.upper_vehicle_track_center = upper_vehicle.center.copy()

        self.reference_lower_below_count = (
            0
            if lower_vehicle is None
            else int(np.count_nonzero(
                lower_vehicle.points[:, 0] < -self.rear_half_stop_margin
            ))
        )
        self.reference_upper_below_count = (
            0
            if upper_vehicle is None
            else int(np.count_nonzero(
                upper_vehicle.points[:, 0] < -self.rear_half_stop_margin
            ))
        )

        if not self.reference_lower_gone:
            if self.reference_lower_below_count > 0:
                self.reference_lower_missing_frames = 0
            else:
                self.reference_lower_missing_frames += 1
                if (
                    self.reference_lower_missing_frames
                    >= self.rear_half_empty_confirm_frames
                ):
                    self.reference_lower_gone = True
                    self.get_logger().info(
                        'Reference lower/right vehicle cleared the green '
                        'horizontal line; later objects on that side ignored'
                    )

        if not self.reference_upper_gone:
            if self.reference_upper_below_count > 0:
                self.reference_upper_missing_frames = 0
            else:
                self.reference_upper_missing_frames += 1
                if (
                    self.reference_upper_missing_frames
                    >= self.rear_half_empty_confirm_frames
                ):
                    self.reference_upper_gone = True
                    self.get_logger().info(
                        'Reference upper/left vehicle cleared the green '
                        'horizontal line; later objects on that side ignored'
                    )
        if self.reference_lower_gone:
            self.current_reference_lower = None
        if self.reference_upper_gone:
            self.current_reference_upper = None

    def update_straight_reverse_condition(
        self, vehicles: list[VehicleCluster]
    ) -> None:
        """Latch when any obstacle-vehicle point enters the second ring."""
        if (
            self.straight_reverse_latched
            or self.reverse_phase not in (
                'STEER_SETTLE',
                'DRIVE',
                'MEASURE_STOP',
            )
        ):
            return
        if not vehicles:
            self.straight_reverse_trigger_distance = math.inf
            return

        self.straight_reverse_trigger_distance = min(
            self.vehicle_nearest_distance(vehicle)
            for vehicle in vehicles
        )
        if (
            self.straight_reverse_trigger_distance
            <= self.straight_reverse_radius
        ):
            self.straight_reverse_latched = True
            self.straight_reverse_started = False
            self.final_correction_count = 0
            if self.observation.pair is not None:
                self.final_target_half_gap = max(
                    self.vehicle_width / 2.0,
                    self.observation.pair.gap_width / 2.0,
                )
            self.reverse_phase = 'FINAL_STOP'
            self.reverse_phase_started_at = time.monotonic()
            self.get_logger().info(
                'Final correction latched: a parked-vehicle point entered '
                f'the {self.straight_reverse_radius:.2f}m ring; stopping for '
                f'{self.straight_reverse_stop:.1f}s with steer=0 before '
                'final two-vehicle corrections.'
            )

    def observe(self, msg: LaserScan) -> LidarObservation:
        points: list[tuple[float, float]] = []
        rear_distances: list[float] = []
        scan_point_count = 0

        for index, raw_distance in enumerate(msg.ranges):
            distance = float(raw_distance)
            if not math.isfinite(distance):
                continue
            if distance < msg.range_min or distance > msg.range_max:
                continue
            scan_point_count += 1
            if distance < self.parking_min_range:
                continue

            angle = msg.angle_min + index * msg.angle_increment
            angle = math.atan2(math.sin(angle), math.cos(angle))
            if (
                abs(angle) <= self.rear_hard_stop_angle
                and distance <= self.cluster_max_range
            ):
                rear_distances.append(distance)
            if (
                abs(angle) > self.valid_sector_max_abs
                or distance > self.cluster_max_range
            ):
                continue

            # Vehicle coordinates: +x forward, +y left.
            points.append(
                (-distance * math.cos(angle), -distance * math.sin(angle))
            )

        point_array = (
            np.asarray(points, dtype=np.float64)
            if points else np.empty((0, 2), dtype=np.float64)
        )
        vehicle_point_array = point_array
        if (
            self.mode == ParkingMode.PARKING
            and not self.straight_reverse_latched
        ):
            vehicle_point_array = point_array[
                point_array[:, 0] <= self.pre_final_vehicle_max_x
            ]
        vehicles = [
            self.make_vehicle_cluster(component)
            for component in self.connected_components(vehicle_point_array)
        ]
        vehicles = [
            vehicle for vehicle in vehicles
            if max(
                vehicle.x_max - vehicle.x_min,
                vehicle.y_max - vehicle.y_min,
            ) >= self.obstacle_min_extent
        ]
        vehicles.sort(key=lambda item: len(item.points), reverse=True)
        right_vehicles = [
            item for item in vehicles
            if item.center[1] < -self.right_detection_margin
        ]
        pair = self.select_parking_pair(vehicles)
        return LidarObservation(
            scan_valid=scan_point_count >= self.scan_quality_min_points,
            points=point_array,
            vehicles=vehicles,
            right_vehicles=right_vehicles,
            pair=pair,
            rear_min_distance=(
                min(rear_distances) if rear_distances else None
            ),
        )

    def connected_components(self, points: np.ndarray) -> list[np.ndarray]:
        """Region-grow nearby LiDAR returns into physical obstacle bundles."""
        if len(points) == 0:
            return []
        unassigned = set(range(len(points)))
        components: list[np.ndarray] = []
        while unassigned:
            seed = unassigned.pop()
            component = [seed]
            pending = [seed]
            while pending:
                current = pending.pop()
                candidates = np.fromiter(unassigned, dtype=np.intp)
                if len(candidates) == 0:
                    continue
                distances = np.linalg.norm(
                    points[candidates] - points[current], axis=1
                )
                nearby = candidates[
                    distances <= self.cluster_neighbor_distance
                ]
                for neighbor in nearby:
                    neighbor_index = int(neighbor)
                    unassigned.remove(neighbor_index)
                    component.append(neighbor_index)
                    pending.append(neighbor_index)
            if len(component) >= self.cluster_min_points:
                components.append(points[component])
        return components

    @staticmethod
    def make_vehicle_cluster(points: np.ndarray) -> VehicleCluster:
        center = np.median(points, axis=0)
        centered = points - center
        covariance = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        if axis[0] < 0.0:
            axis = -axis
        axis_angle = math.atan2(float(axis[1]), float(axis[0]))
        return VehicleCluster(
            points=points,
            center=center,
            axis_angle=axis_angle,
            x_min=float(np.quantile(points[:, 0], 0.05)),
            x_max=float(np.quantile(points[:, 0], 0.95)),
            y_min=float(np.quantile(points[:, 1], 0.05)),
            y_max=float(np.quantile(points[:, 1], 0.95)),
        )

    def select_parking_pair(
        self, vehicles: list[VehicleCluster]
    ) -> Optional[ParkingPair]:
        """Choose the adjacent vehicles whose free gap is nearest the ego."""
        if len(vehicles) < 2:
            return None

        ordered = sorted(vehicles, key=lambda item: item.center[1])
        candidates: list[tuple[float, ParkingPair]] = []
        for lower, upper in zip(ordered, ordered[1:]):
            pair = self.build_parking_pair(
                lower, upper, require_valid_gap=True
            )
            if pair is None:
                continue
            score = abs(pair.gap_center_y) + 0.15 * (
                np.linalg.norm(lower.center)
                + np.linalg.norm(upper.center)
            )
            candidates.append((float(score), pair))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def build_parking_pair(
        self,
        first: VehicleCluster,
        second: VehicleCluster,
        *,
        require_valid_gap: bool,
    ) -> Optional[ParkingPair]:
        lower, upper = sorted(
            (first, second), key=lambda item: item.center[1]
        )
        gap_low = lower.y_max
        gap_high = upper.y_min
        gap_width = gap_high - gap_low
        gap_center = 0.5 * (gap_low + gap_high)
        if require_valid_gap and not (
            self.gap_min_width <= gap_width <= self.gap_max_width
            and abs(gap_center) <= self.gap_track_max_center
        ):
            return None

        return ParkingPair(
            lower=lower,
            upper=upper,
            # The red reference point in the debug view: midpoint of the two
            # obstacle representative points (median cluster centers).
            reference_point=0.5 * (lower.center + upper.center),
            gap_center_y=gap_center,
            gap_width=gap_width,
            left_clearance=gap_high - self.vehicle_width / 2.0,
            right_clearance=-self.vehicle_width / 2.0 - gap_low,
        )

    def track_parking_pair(
        self,
        vehicles: list[VehicleCluster],
        *,
        max_jump: Optional[float] = None,
    ) -> Optional[ParkingPair]:
        """Keep both original vehicles even when the strict gap test fails."""
        if (
            len(vehicles) < 2
            or self.lower_vehicle_track_center is None
            or self.upper_vehicle_track_center is None
        ):
            return None

        allowed_jump = (
            self.vehicle_pair_track_max_jump
            if max_jump is None
            else max_jump
        )
        best = None
        for lower_index, lower in enumerate(vehicles):
            for upper_index, upper in enumerate(vehicles):
                if lower_index == upper_index:
                    continue
                lower_jump = float(np.linalg.norm(
                    lower.center - self.lower_vehicle_track_center
                ))
                upper_jump = float(np.linalg.norm(
                    upper.center - self.upper_vehicle_track_center
                ))
                if (
                    lower_jump > allowed_jump
                    or upper_jump > allowed_jump
                ):
                    continue
                cost = lower_jump + upper_jump
                if best is None or cost < best[0]:
                    best = (cost, lower, upper)
        if best is None:
            return None

        pair = self.build_parking_pair(
            best[1], best[2], require_valid_gap=False
        )
        if self.final_completion_tracking_started:
            pair_text = (
                f'original L/R below='
                f'{self.reference_upper_below_count}/'
                f'{self.reference_lower_below_count} '
                f'missing={self.reference_upper_missing_frames}/'
                f'{self.reference_lower_missing_frames} '
                f'gone={int(self.reference_upper_gone)}/'
                f'{int(self.reference_lower_gone)}'
            )
        elif pair is None:
            return None
        self.lower_vehicle_track_center = pair.lower.center.copy()
        self.upper_vehicle_track_center = pair.upper.center.copy()
        return pair

    def fallback_visible_pair(
        self, vehicles: list[VehicleCluster]
    ) -> Optional[ParkingPair]:
        """Use the two visible clusters around the rear centerline.

        This keeps steering updates alive when perspective makes the original
        strict gap-width or inter-frame tracking test fail even though both
        parked vehicles are still plainly visible.
        """
        if len(vehicles) < 2:
            return None

        candidates: list[tuple[float, ParkingPair]] = []
        for first_index, first in enumerate(vehicles):
            for second in vehicles[first_index + 1:]:
                pair = self.build_parking_pair(
                    first, second, require_valid_gap=False
                )
                if pair is None:
                    continue
                straddles_center = (
                    pair.lower.center[1] <= 0.0
                    <= pair.upper.center[1]
                )
                score = (
                    (0.0 if straddles_center else 10.0)
                    + abs(float(pair.reference_point[1]))
                    + 0.05 * (
                        float(np.linalg.norm(pair.lower.center))
                        + float(np.linalg.norm(pair.upper.center))
                    )
                )
                candidates.append((score, pair))
        if not candidates:
            return None

        pair = min(candidates, key=lambda item: item[0])[1]
        self.lower_vehicle_track_center = pair.lower.center.copy()
        self.upper_vehicle_track_center = pair.upper.center.copy()
        return pair

    def transition(self, next_state: ParkingState, now: float) -> None:
        if next_state == self.state:
            return
        self.get_logger().info(
            f'YYM parking: {self.state.value} -> {next_state.value}'
        )
        self.state = next_state
        self.state_started_at = now
        if next_state == ParkingState.SETTLE_AND_ACQUIRE_GAP:
            self.gap_frames = 0
        elif next_state == ParkingState.REVERSE_CENTER:
            self.last_pair_at = now
            pair = self.observation.pair
            self.lower_vehicle_track_center = (
                pair.lower.center.copy() if pair is not None else None
            )
            self.upper_vehicle_track_center = (
                pair.upper.center.copy() if pair is not None else None
            )
            self.rear_half_lidar_point_count = int(np.count_nonzero(
                self.observation.points[:, 0]
                < -self.rear_half_stop_margin
            ))
            self.rear_half_points_seen = (
                self.rear_half_lidar_point_count > 0
            )
            self.rear_half_empty_frames = 0
            self.reference_lower_below_count = 0
            self.reference_upper_below_count = 0
            self.reference_lower_missing_frames = 0
            self.reference_upper_missing_frames = 0
            self.reference_lower_gone = False
            self.reference_upper_gone = False
            self.final_completion_tracking_started = False
            self.current_reference_lower = (
                pair.lower if pair is not None else None
            )
            self.current_reference_upper = (
                pair.upper if pair is not None else None
            )
            self.final_target_half_gap = (
                max(self.vehicle_width / 2.0, pair.gap_width / 2.0)
                if pair is not None
                else self.gap_min_width / 2.0
            )
            self.straight_reverse_latched = False
            self.straight_reverse_started = False
            self.final_correction_count = 0
            self.straight_reverse_trigger_distance = math.inf
            self.lidar_side_gate_frames = 0
            self.lidar_side_left_m = None
            self.lidar_side_right_m = None
            self.lidar_raw_left_m = None
            self.lidar_raw_right_m = None
            self.lidar_side_far_frames = 0
            self.lidar_side_gate_seen = False
            self.reverse_phase = 'STEER_SETTLE'
            self.reverse_phase_started_at = now
            self.reverse_segment_started_at = None
            self.reverse_segment_index = 1
            self.reverse_segment_drive_duration = self.reverse_segment_duration
            self.reverse_segment_steer = (
                self.reverse_steering(pair)
                if pair is not None
                else 0
            )
            self.get_logger().info(
                f'LiDAR reverse segment 1: steer='
                f'{self.reverse_segment_steer}deg for '
                f'{self.reverse_segment_drive_duration:.1f}s; '
                'waiting for vehicles at both +/-90deg side gates'
            )

    def enter_parking_mode(self, now: float) -> None:
        if self.mode != ParkingMode.PARKING:
            self.get_logger().info(
                'YYM mode: RECOGNITION -> PARKING'
            )
        self.mode = ParkingMode.PARKING
        self.transition(ParkingState.SETTLE_AND_ACQUIRE_GAP, now)

    def control_tick(self) -> None:
        now = time.monotonic()
        if self.state in (
            ParkingState.EXIT_COMPLETE,
            ParkingState.PARKING_FAILED,
            ParkingState.EMERGENCY_STOP,
        ):
            # Terminal states are latched. Avoid repeatedly logging the same
            # LiDAR timeout/failure at the 20 Hz control rate.
            self.reverse_phase = f'{self.state.value}_HOLD'
            self.publish_control(0, 0)
            return
        if self.last_scan_at is None:
            # Do not send a 0-deg target while waiting to start. With no input,
            # drive_control keeps both PWM outputs at zero, so the stationary
            # wheels remain at their current angle.
            return
        if now - self.last_scan_at > self.scan_timeout:
            self.get_logger().error('LiDAR timeout: emergency stop')
            self.transition(ParkingState.EMERGENCY_STOP, now)
            self.publish_control(0, 0)
            return
        if self.invalid_scan_count > 0:
            if self.state == ParkingState.WAIT_FOR_SCAN:
                # Same startup behavior for invalid initial scans: leave the
                # stationary steering untouched until a valid scan arrives.
                return
            if self.invalid_scan_count >= self.invalid_scan_confirm_frames:
                self.get_logger().error('Invalid LiDAR stream: emergency stop')
                self.transition(ParkingState.EMERGENCY_STOP, now)
            self.publish_control(0, 0)
            return

        if self.state == ParkingState.WAIT_FOR_SCAN:
            self.transition(ParkingState.START_DELAY, now)
            self.publish_control(0, 0)
            return

        if self.state == ParkingState.START_DELAY:
            remaining = max(0.0, self.startup_delay_deadline - now)
            self.reverse_phase = f'START_DELAY_{remaining:.1f}s'
            self.publish_control(0, 0)
            if remaining <= 0.0:
                if self.mode == ParkingMode.PARKING:
                    self.transition(
                        ParkingState.SETTLE_AND_ACQUIRE_GAP, now
                    )
                else:
                    self.transition(ParkingState.APPROACH_FIRST_CAR, now)
                self.get_logger().info(
                    'Startup delay complete: beginning driving sequence'
                )
            return

        elapsed = now - self.state_started_at
        if self.state == ParkingState.APPROACH_FIRST_CAR:
            if elapsed >= self.approach_timeout:
                self.fail('first parked vehicle detection timeout', now)
                return
            if self.first_car_frames >= self.first_car_confirm_frames:
                self.transition(ParkingState.SET_LEFT_STEER, now)
                self.publish_control(self.left_max_steer, 0)
                return
            # Approach always requests an explicit straight wheel angle.
            self.publish_control(0, self.approach_speed)
            return

        if self.state == ParkingState.SET_LEFT_STEER:
            if elapsed >= self.steer_settle_sec:
                self.transition(ParkingState.TURN_LEFT_TIMED, now)
                self.publish_control(
                    self.left_max_steer, self.turn_speed
                )
                return
            self.publish_control(self.left_max_steer, 0)
            return

        if self.state == ParkingState.TURN_LEFT_TIMED:
            if elapsed >= self.left_turn_duration_sec:
                if self.recognition_only:
                    self.transition(
                        ParkingState.RECOGNITION_COMPLETE, now
                    )
                    self.get_logger().info(
                        'Recognition complete: timed left turn finished; '
                        'stopping before automatic shutdown'
                    )
                else:
                    self.enter_parking_mode(now)
                self.publish_control(0, 0)
                return
            self.publish_control(self.left_max_steer, self.turn_speed)
            return

        if self.state == ParkingState.RECOGNITION_COMPLETE:
            self.publish_control(0, 0)
            if elapsed >= self.recognition_shutdown_delay_sec:
                self.shutdown_program()
            return

        if self.state == ParkingState.SETTLE_AND_ACQUIRE_GAP:
            if elapsed >= self.gap_acquire_timeout:
                self.fail('two-vehicle parking gap acquisition timeout', now)
                return
            if (
                elapsed >= self.steer_settle_sec
                and self.gap_frames >= self.gap_confirm_frames
            ):
                pair = self.observation.pair
                if pair is None:
                    self.publish_control(0, 0)
                    return
                required_width = (
                    self.vehicle_width + 2.0 * self.minimum_side_clearance
                )
                if (
                    not self.observation.pair_is_fallback
                    and pair.gap_width < required_width
                ):
                    self.fail(
                        f'gap too narrow: {pair.gap_width:.2f} m '
                        f'< {required_width:.2f} m',
                        now,
                    )
                    return
                if self.observation.pair_is_fallback:
                    self.get_logger().warning(
                        'Starting reverse with two-cluster fallback midpoint; '
                        'strict gap pair remains preferred when available'
                    )
                self.transition(ParkingState.REVERSE_CENTER, now)
            self.publish_control(0, 0)
            return

        if self.state == ParkingState.REVERSE_CENTER:
            if (
                self.reverse_timeout > 0.0
                and elapsed >= self.reverse_timeout
            ):
                self.fail('LiDAR reverse/exit timeout', now)
                return

            lidar_reverse_phases = (
                'STEER_SETTLE',
                'DRIVE',
                'MEASURE_STOP',
            )
            if (
                self.reverse_phase in lidar_reverse_phases
                and self.observation.rear_min_distance is not None
                and self.observation.rear_min_distance
                <= self.rear_hard_stop_distance
            ):
                self.fail('rear obstacle inside hard-stop distance', now)
                return

            phase_elapsed = (
                0.0
                if self.reverse_phase_started_at is None
                else now - self.reverse_phase_started_at
            )
            if self.reverse_phase == 'FINAL_STOP':
                self.publish_control(0, 0)
                if phase_elapsed >= self.straight_reverse_stop:
                    self.reverse_phase = 'FINAL_MEASURE_STOP'
                    self.reverse_phase_started_at = now
                    self.straight_reverse_started = True
                    self.reverse_segment_index = 0
                    self.final_correction_count = 0
                    self.final_completion_tracking_started = True
                    self.reference_lower_missing_frames = 0
                    self.reference_upper_missing_frames = 0
                    self.reference_lower_gone = False
                    self.reference_upper_gone = False
                    self.get_logger().info(
                        'Five-second stop complete: starting '
                        'precision reverse; high-camera two-line guidance '
                        'has priority, with LiDAR fallback. Initial fallback '
                        f'{self.final_correction_segment_count} x '
                        f'{self.final_correction_duration:.1f}s corrections '
                        'using line angle when misaligned or average parked-'
                        'vehicle tilt when aligned'
                    )
                return

            final_reverse_phases = (
                'FINAL_STEER_SETTLE',
                'FINAL_DRIVE',
                'FINAL_MEASURE_STOP',
                'FINAL_ZERO_STEER_SETTLE',
                'FINAL_STRAIGHT_DRIVE',
            )
            if self.reverse_phase in final_reverse_phases:
                if (
                    self.reference_lower_gone
                    or self.reference_upper_gone
                ):
                    self.transition(ParkingState.PARKED, now)
                    self.reverse_phase = 'PARKED_HOLD'
                    self.publish_control(0, 0)
                    self.get_logger().info(
                        'PARKED: an original parked vehicle cleared the green '
                        'horizontal line; later pillars/units ignored'
                    )
                    return

                camera_lines = self.active_parking_line_observation(now)
                if self.reverse_phase == 'FINAL_STRAIGHT_DRIVE':
                    # White lines may first enter the high-camera view after
                    # the two initial correction segments. Use them as soon
                    # as they do; stale/invalid camera data falls back to the
                    # existing zero-steer continuous reverse.
                    camera_steer = (
                        camera_lines.steering_deg
                        if camera_lines is not None
                        else 0
                    )
                    self.publish_control(camera_steer, self.reverse_speed)
                    return

                if self.reverse_phase == 'FINAL_ZERO_STEER_SETTLE':
                    self.publish_control(0, 0)
                    if phase_elapsed >= self.steer_settle_sec:
                        self.reverse_phase = 'FINAL_STRAIGHT_DRIVE'
                        self.reverse_phase_started_at = now
                        self.publish_control(0, self.reverse_speed)
                        self.get_logger().info(
                            'Steering centered at zero while stopped: '
                            'starting straight reverse'
                        )
                    return

                if camera_lines is not None:
                    calculated_steer = camera_lines.steering_deg
                    target_description = (
                        f'HIGH CAMERA PRIORITY: slot '
                        f'{self.parking_slot_number} template, '
                        f'horizontal='
                        f'{camera_lines.horizontal_angle_error_deg:+.1f}deg, '
                        f'guide={camera_lines.guide_angle_error_deg:+.1f}deg, '
                        f'anchor-x={camera_lines.anchor_x_error:+.3f}, '
                        f'anchor-y={camera_lines.anchor_y_error:+.3f}, '
                        f'confidence={camera_lines.confidence:.2f}'
                    )
                else:
                    pair = self.observation.pair
                    if (
                        pair is None
                        and self.current_reference_lower is not None
                        and self.current_reference_upper is not None
                    ):
                        pair = self.build_parking_pair(
                            self.current_reference_lower,
                            self.current_reference_upper,
                            require_valid_gap=False,
                        )
                if camera_lines is None and pair is not None:
                    red_line_angle = self.reverse_reference_angle(pair)
                    lines_aligned = (
                        abs(red_line_angle)
                        <= self.final_line_alignment_tolerance
                    )
                    if lines_aligned:
                        calculated_steer = (
                            self.final_initial_alignment_steering(pair)
                        )
                        target_description = (
                            f'lines aligned ({red_line_angle:+.1f}deg): '
                            'average vehicle tilt='
                            f'{self.final_average_alignment_angle(pair):+.1f}'
                            f'deg x{self.final_reverse_steer_multiplier:.1f}'
                        )
                    else:
                        calculated_steer = self.reverse_steering(pair)
                        target_description = (
                            f'lines misaligned ({red_line_angle:+.1f}deg): '
                            f'line angle x{self.reverse_steer_multiplier:.1f}'
                        )
                    target_description += (
                        f', axes='
                        f'{math.degrees(pair.lower.axis_angle):+.0f}/'
                        f'{math.degrees(pair.upper.axis_angle):+.0f}deg'
                    )
                elif camera_lines is None:
                    self.reverse_phase = 'FINAL_MEASURE_STOP'
                    self.reverse_phase_started_at = now
                    self.publish_control(self.held_steering_command(), 0)
                    return

                if self.reverse_phase == 'FINAL_STEER_SETTLE':
                    self.publish_control(self.reverse_segment_steer, 0)
                    if phase_elapsed >= self.steer_settle_sec:
                        self.reverse_phase = 'FINAL_DRIVE'
                        self.reverse_segment_started_at = now
                        self.get_logger().info(
                            f'Final reverse segment '
                            f'{self.reverse_segment_index}: '
                            f'steer={self.reverse_segment_steer}deg for '
                            f'{self.final_correction_duration:.1f}s'
                        )
                    return

                if self.reverse_phase == 'FINAL_DRIVE':
                    if (
                        self.reverse_segment_started_at is not None
                        and now - self.reverse_segment_started_at
                        >= self.final_correction_duration
                    ):
                        if (
                            self.final_correction_count
                            >= self.final_correction_segment_count
                        ):
                            self.reverse_phase = 'FINAL_ZERO_STEER_SETTLE'
                            self.reverse_phase_started_at = now
                            self.publish_control(0, 0)
                            self.get_logger().info(
                                f'{self.final_correction_segment_count} tilt '
                                'corrections complete: mandatory stop and '
                                'steer=0 centering before straight reverse'
                            )
                            return
                        self.reverse_phase = 'FINAL_MEASURE_STOP'
                        self.reverse_phase_started_at = now
                        self.publish_control(
                            self.held_steering_command(), 0
                        )
                        return
                    self.publish_control(
                        self.reverse_segment_steer,
                        self.reverse_speed,
                    )
                    return

                self.publish_control(self.held_steering_command(), 0)
                if phase_elapsed < self.reverse_measure_stop:
                    return

                self.reverse_segment_steer = calculated_steer
                self.final_correction_count += 1
                self.reverse_segment_index = self.final_correction_count
                self.reverse_phase = 'FINAL_STEER_SETTLE'
                self.reverse_phase_started_at = now
                self.get_logger().info(
                    f'Final reverse segment '
                    f'{self.final_correction_count}: '
                    f'{target_description}, '
                    f'steer={self.reverse_segment_steer}deg'
                )
                return

            if self.reverse_phase in lidar_reverse_phases:
                pair = self.observation.pair
                if pair is None:
                    self.reverse_phase = 'MEASURE_STOP'
                    self.reverse_phase_started_at = now
                    self.publish_control(self.held_steering_command(), 0)
                    return

                if self.reverse_phase == 'STEER_SETTLE':
                    self.publish_control(self.reverse_segment_steer, 0)
                    if (
                        self.reverse_phase_started_at is not None
                        and now - self.reverse_phase_started_at
                        >= self.steer_settle_sec
                    ):
                        self.reverse_segment_drive_duration = (
                            self.reverse_segment_duration
                        )
                        self.reverse_phase = 'DRIVE'
                        self.reverse_segment_started_at = now
                        self.get_logger().info(
                            f'LiDAR reverse segment '
                            f'{self.reverse_segment_index}: '
                            f'steer={self.reverse_segment_steer}deg for '
                            f'{self.reverse_segment_drive_duration:.1f}s'
                        )
                    return

                if self.reverse_phase == 'DRIVE':
                    if (
                        self.reverse_segment_started_at is not None
                        and now - self.reverse_segment_started_at
                        >= self.reverse_segment_drive_duration
                    ):
                        self.reverse_phase = 'MEASURE_STOP'
                        self.reverse_phase_started_at = now
                        self.publish_control(
                            self.held_steering_command(), 0
                        )
                        return
                    self.publish_control(
                        self.reverse_segment_steer,
                        self.reverse_speed,
                    )
                    return

                self.publish_control(self.held_steering_command(), 0)
                if (
                    self.reverse_phase_started_at is None
                    or now - self.reverse_phase_started_at
                    < self.reverse_measure_stop
                ):
                    return

                self.reverse_segment_index += 1
                self.reverse_segment_steer = self.reverse_steering(pair)
                target_description = (
                    f'reference=({pair.reference_point[0]:+.2f},'
                    f'{pair.reference_point[1]:+.2f})m'
                )
                self.reverse_phase = 'STEER_SETTLE'
                self.reverse_phase_started_at = now
                self.get_logger().info(
                    f'LiDAR reverse segment '
                    f'{self.reverse_segment_index}: '
                    f'{target_description}, '
                    f'steer={self.reverse_segment_steer}deg'
                )
                return

            self.reverse_phase = 'MEASURE_STOP'
            self.reverse_phase_started_at = now
            self.lidar_side_far_frames = 0
            self.publish_control(0, 0)
            return

        if self.state == ParkingState.PARKED:
            self.reverse_phase = (
                'PARKED_WAIT_EXIT'
                if self.enable_exit_sequence
                else 'PARKED_HOLD'
            )
            self.publish_control(0, 0)
            if (
                self.enable_exit_sequence
                and elapsed >= self.exit_wait_after_park
            ):
                self.mode = ParkingMode.EXIT
                self.transition(ParkingState.EXIT_FORWARD, now)
                self.reverse_phase = 'EXIT_FORWARD'
                self.publish_control(0, self.exit_speed)
                self.get_logger().info(
                    f'EXIT: starting {self.exit_forward_duration:.1f}s '
                    'straight drive'
                )
            return

        if self.state == ParkingState.EXIT_FORWARD:
            self.reverse_phase = 'EXIT_FORWARD'
            if elapsed >= self.exit_forward_duration:
                self.transition(ParkingState.EXIT_SET_RIGHT_STEER, now)
                self.reverse_phase = 'EXIT_SET_RIGHT_STEER'
                self.publish_control(self.exit_right_steer, 0)
                self.get_logger().info(
                    'EXIT: first straight complete; stopped to set maximum '
                    'right steering'
                )
                return
            self.publish_control(0, self.exit_speed)
            return

        if self.state == ParkingState.EXIT_SET_RIGHT_STEER:
            self.reverse_phase = 'EXIT_SET_RIGHT_STEER'
            if elapsed >= self.steer_settle_sec:
                self.transition(ParkingState.EXIT_RIGHT_TURN, now)
                self.reverse_phase = 'EXIT_RIGHT_TURN'
                self.publish_control(
                    self.exit_right_steer, self.exit_speed
                )
                self.get_logger().info(
                    f'EXIT: starting {self.exit_right_turn_duration:.1f}s '
                    'maximum-right turn'
                )
                return
            self.publish_control(self.exit_right_steer, 0)
            return

        if self.state == ParkingState.EXIT_RIGHT_TURN:
            self.reverse_phase = 'EXIT_RIGHT_TURN'
            if elapsed >= self.exit_right_turn_duration:
                self.transition(ParkingState.EXIT_CENTER_STEER, now)
                self.reverse_phase = 'EXIT_CENTER_STEER'
                self.publish_control(0, 0)
                self.get_logger().info(
                    'EXIT: right turn complete; stopped to center steering'
                )
                return
            self.publish_control(self.exit_right_steer, self.exit_speed)
            return

        if self.state == ParkingState.EXIT_CENTER_STEER:
            self.reverse_phase = 'EXIT_CENTER_STEER'
            if elapsed >= self.steer_settle_sec:
                self.transition(ParkingState.EXIT_FINAL_FORWARD, now)
                self.reverse_phase = 'EXIT_FINAL_FORWARD'
                self.publish_control(0, self.exit_speed)
                self.get_logger().info(
                    f'EXIT: starting final '
                    f'{self.exit_final_forward_duration:.1f}s straight drive'
                )
                return
            self.publish_control(0, 0)
            return

        if self.state == ParkingState.EXIT_FINAL_FORWARD:
            self.reverse_phase = 'EXIT_FINAL_FORWARD'
            if elapsed >= self.exit_final_forward_duration:
                self.transition(ParkingState.EXIT_COMPLETE, now)
                self.reverse_phase = 'EXIT_COMPLETE_HOLD'
                self.publish_control(0, 0)
                self.get_logger().info(
                    'EXIT COMPLETE: vehicle stopped with steering centered'
                )
                return
            self.publish_control(0, self.exit_speed)
            return

        # Other safe states hold their current angle to avoid hunting.
        self.publish_control(self.held_steering_command(), 0)

    def reverse_steering(self, pair: ParkingPair) -> int:
        """Return the red angle from the rear baseline to the reference point."""
        return self.scaled_reverse_steering(
            self.reverse_reference_angle(pair)
        )

    @staticmethod
    def reverse_reference_angle(pair: ParkingPair) -> float:
        """Return the unscaled red/green-line angular error in degrees."""
        reference_x = float(pair.reference_point[0])
        reference_y = float(pair.reference_point[1])
        # Rear baseline is -x. In the vehicle's LiDAR convention, a point to
        # the right has positive bearing and therefore commands right steer.
        rear_distance = max(0.05, -reference_x)
        return math.degrees(
            math.atan2(-reference_y, rear_distance)
        )

    @staticmethod
    def final_average_alignment_angle(pair: ParkingPair) -> float:
        """Return the mean tilt of the two parked vehicles in degrees."""
        return math.degrees(
            0.5 * (
                float(pair.lower.axis_angle)
                + float(pair.upper.axis_angle)
            )
        )

    def final_initial_alignment_steering(
        self, pair: ParkingPair
    ) -> int:
        """Scale only the two parked vehicles' mean tilt."""
        return self.scaled_reverse_steering(
            self.final_average_alignment_angle(pair),
            multiplier=self.final_reverse_steer_multiplier,
        )

    def final_conditional_steering(self, pair: ParkingPair) -> int:
        """Choose line correction unless red and green are already aligned."""
        if (
            abs(self.reverse_reference_angle(pair))
            <= self.final_line_alignment_tolerance
        ):
            return self.final_initial_alignment_steering(pair)
        return self.reverse_steering(pair)

    def final_single_reference_steering(
        self,
        vehicle: VehicleCluster,
        *,
        is_lower: bool,
    ) -> int:
        """Continue after one original vehicle clears, without using a pillar."""
        inferred_gap_center = (
            float(vehicle.y_max) + self.final_target_half_gap
            if is_lower
            else float(vehicle.y_min) - self.final_target_half_gap
        )
        reference_depth = max(0.05, -float(vehicle.center[0]))
        center_angle = math.degrees(math.atan2(
            -inferred_gap_center,
            reference_depth,
        ))
        alignment_angle = math.degrees(float(vehicle.axis_angle))
        calculated_angle = (
            self.final_center_gain * center_angle
            + self.final_alignment_gain * alignment_angle
        )
        return self.scaled_reverse_steering(
            calculated_angle,
            multiplier=self.final_reverse_steer_multiplier,
        )

    def single_vehicle_steering(self, vehicle: VehicleCluster) -> int:
        """Parallelize the ego with the only visible parked-vehicle line."""
        axis_angle = math.degrees(vehicle.axis_angle)
        if abs(axis_angle) <= self.single_vehicle_angle_deadband:
            return 0
        # In this LiDAR/debug convention +axis tilts toward the right while
        # reversing, so it directly produces a positive (right) steer.
        return self.scaled_reverse_steering(axis_angle)

    def scaled_reverse_steering(
        self,
        calculated_angle: float,
        *,
        multiplier: Optional[float] = None,
    ) -> int:
        """Scale a LiDAR reverse angle and clamp it to the steering range."""
        steer_multiplier = (
            self.reverse_steer_multiplier
            if multiplier is None
            else multiplier
        )
        scaled_angle = calculated_angle * steer_multiplier
        return int(round(max(-45.0, min(45.0, scaled_angle))))

    @staticmethod
    def vehicle_nearest_distance(vehicle: VehicleCluster) -> float:
        """Distance to the nearest LiDAR point belonging to one vehicle."""
        if len(vehicle.points) == 0:
            return math.inf
        return float(np.min(np.linalg.norm(vehicle.points, axis=1)))

    def fail(self, reason: str, now: float) -> None:
        self.failure_reason = reason
        self.get_logger().error(f'Parking failed: {reason}')
        self.transition(ParkingState.PARKING_FAILED, now)
        self.publish_control(self.held_steering_command(), 0)

    def shutdown_program(self) -> None:
        """Stop this node and its ros2 launch parent after recognition."""
        if self.shutdown_started:
            return
        self.shutdown_started = True
        self.publish_control(0, 0)
        self.get_logger().info(
            'Recognition-only test complete; shutting down parking launch'
        )

        # rclpy.shutdown() stops only this process. When this node was started
        # through ``ros2 launch``, also notify that exact parent so its sensor
        # and motor processes are cleaned up. Never signal an unrelated parent.
        parent_pid = os.getppid()
        try:
            with open(
                f'/proc/{parent_pid}/cmdline',
                'rb',
            ) as command_file:
                parent_command = command_file.read().replace(b'\x00', b' ')
            if b'ros2' in parent_command and b'launch' in parent_command:
                os.kill(parent_pid, signal.SIGINT)
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            pass

        if rclpy.ok():
            rclpy.shutdown()

    def publish_control(self, steer: int, speed: int) -> None:
        steer = int(max(-45, min(45, steer)))
        speed = int(max(-130, min(130, speed)))
        self.last_command = (steer, speed)
        message = Int16MultiArray()
        message.data = [steer, speed]
        self.motor_publisher.publish(message)

    def draw_debug(self) -> None:
        size = self.debug_image_size
        center = size // 2
        scale = size * 0.42 / self.debug_max_range
        image = np.zeros((size, size, 3), dtype=np.uint8)

        for radius_m in np.arange(0.5, self.debug_max_range + 0.01, 0.5):
            radius = int(radius_m * scale)
            cv2.circle(image, (center, center), radius, (0, 60, 0), 1)
        cv2.line(image, (center, 20), (center, size - 20), (0, 70, 0), 1)
        cv2.line(image, (20, center), (size - 20, center), (0, 70, 0), 1)

        def pixel(point: np.ndarray | tuple[float, float]):
            x_forward, y_left = float(point[0]), float(point[1])
            return (
                int(center - y_left * scale),
                int(center - x_forward * scale),
            )

        for point in self.observation.points:
            cv2.circle(image, pixel(point), 1, (100, 100, 100), -1)

        colors = [
            (0, 220, 255), (255, 150, 0), (180, 80, 255), (60, 200, 60)
        ]
        for index, vehicle in enumerate(self.observation.vehicles):
            color = colors[index % len(colors)]
            for point in vehicle.points:
                cv2.circle(image, pixel(point), 2, color, -1)
            location = pixel(vehicle.center)
            cv2.putText(
                image,
                f'V{index + 1} {math.degrees(vehicle.axis_angle):+.0f}deg',
                (location[0] + 5, location[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA,
            )

        pair = self.observation.pair
        if pair is not None:
            target_y = pair.gap_center_y
            line_start = pixel((-self.debug_max_range, target_y))
            line_end = pixel((self.debug_max_range, target_y))
            cv2.line(image, line_start, line_end, (0, 255, 0), 2)
            reference_pixel = pixel(pair.reference_point)
            rear_baseline_end = pixel(
                (-min(1.8, self.debug_max_range), 0.0)
            )
            cv2.line(
                image,
                (center, center),
                rear_baseline_end,
                (0, 0, 255),
                3,
            )
            cv2.line(
                image,
                (center, center),
                reference_pixel,
                (0, 0, 255),
                3,
            )
            cv2.circle(image, reference_pixel, 9, (0, 0, 255), -1)

        # Vehicle marker and LiDAR-to-rear-bumper reference.
        cv2.circle(image, (center, center), 7, (255, 255, 255), -1)
        bumper_left = pixel((-self.lidar_to_rear_bumper, -self.vehicle_width / 2))
        bumper_right = pixel((-self.lidar_to_rear_bumper, self.vehicle_width / 2))
        cv2.line(image, bumper_left, bumper_right, (255, 255, 255), 3)

        cv2.rectangle(image, (8, 8), (size - 8, 132), (55, 55, 55), -1)
        if self.state in (
            ParkingState.PARKED,
            ParkingState.EXIT_COMPLETE,
        ):
            state_text = self.state.value
            state_color = (0, 255, 0)
        elif self.state in (
            ParkingState.PARKING_FAILED,
            ParkingState.EMERGENCY_STOP,
        ):
            state_text = 'FAILED'
            state_color = (0, 0, 255)
        else:
            state_text = self.state.value
            state_color = (0, 255, 255)
        cv2.putText(
            image,
            f'MODE: {self.mode.value} | STATE: {state_text}',
            (18, 36),
            cv2.FONT_HERSHEY_SIMPLEX, 0.70, state_color, 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f'cmd steer/speed={self.last_command[0]}/{self.last_command[1]} '
            f'phase={self.reverse_phase} via=/motor_control',
            (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        if pair is None:
            if len(self.observation.vehicles) == 1:
                pair_text = (
                    'one vehicle visible: waiting for two-vehicle midpoint '
                    f'| segment={self.reverse_segment_index} '
                    f'rear-lidar={self.rear_half_lidar_point_count} '
                    f'empty={self.rear_half_empty_frames}/'
                    f'{self.rear_half_empty_confirm_frames}'
                )
            else:
                pair_text = (
                    'gap: not acquired '
                    f'| vehicles={len(self.observation.vehicles)} '
                    f'segment={self.reverse_segment_index} '
                    f'rear-lidar={self.rear_half_lidar_point_count} '
                    f'empty={self.rear_half_empty_frames}/'
                    f'{self.rear_half_empty_confirm_frames}'
                )
        else:
            camera_lines = self.active_parking_line_observation()
            displayed_steer = (
                (
                    camera_lines.steering_deg
                    if camera_lines is not None
                    else (
                        0
                        if self.reverse_phase in (
                            'FINAL_ZERO_STEER_SETTLE',
                            'FINAL_STRAIGHT_DRIVE',
                        )
                        else self.final_conditional_steering(pair)
                    )
                )
                if self.reverse_phase.startswith('FINAL_')
                else self.reverse_steering(pair)
            )
            pair_text = (
                f'pair='
                f'{"FALLBACK" if self.observation.pair_is_fallback else "STRICT"} '
                f'seg={self.reverse_segment_index} '
                f'gap={pair.gap_width:.2f}m center='
                f'{pair.gap_center_y:+.2f}m L/R='
                f'{pair.left_clearance:.2f}/'
                f'{pair.right_clearance:.2f}m '
                f'ref-angle={displayed_steer:+d}deg '
                f'rear-lidar={self.rear_half_lidar_point_count} '
                f'empty={self.rear_half_empty_frames}/'
                f'{self.rear_half_empty_confirm_frames}'
            )
        cv2.putText(
            image, pair_text, (18, 96),
            cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1,
            cv2.LINE_AA,
        )
        if self.failure_reason:
            cv2.putText(
                image,
                f'failure: {self.failure_reason}',
                (18, 121),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (80, 80, 255),
                1,
                cv2.LINE_AA,
            )
        elif self.state == ParkingState.REVERSE_CENTER:
            if self.reverse_phase in (
                'STEER_SETTLE',
                'DRIVE',
                'MEASURE_STOP',
            ):
                guidance_text = (
                    f'1s midpoint correction | nearest vehicle='
                    f'{self.straight_reverse_trigger_distance:.2f}m '
                    f'gate={self.straight_reverse_radius:.2f}m'
                )
            elif self.reverse_phase == 'FINAL_STOP':
                remaining = max(
                    0.0,
                    self.straight_reverse_stop
                    - (
                        0.0
                        if self.reverse_phase_started_at is None
                        else time.monotonic()
                        - self.reverse_phase_started_at
                    ),
                )
                guidance_text = (
                    f'1m LATCHED | STOP + STEER 0 | '
                    f'{remaining:.1f}s remaining'
                )
            elif self.reverse_phase in (
                'FINAL_STEER_SETTLE',
                'FINAL_DRIVE',
                'FINAL_MEASURE_STOP',
                'FINAL_ZERO_STEER_SETTLE',
                'FINAL_STRAIGHT_DRIVE',
                'PARKED_HOLD',
            ):
                if self.reverse_phase == 'FINAL_ZERO_STEER_SETTLE':
                    guidance_text = (
                        'MANDATORY STOP | centering steer=0 before reverse'
                    )
                elif self.reverse_phase == 'FINAL_STRAIGHT_DRIVE':
                    camera_lines = self.active_parking_line_observation()
                    if camera_lines is not None:
                        guidance_text = (
                            'PRECISION: CAMERA PRIORITY | '
                            f'H={camera_lines.horizontal_angle_error_deg:+.1f} '
                            f'V={camera_lines.guide_angle_error_deg:+.1f} '
                            f'X={camera_lines.anchor_x_error:+.3f} '
                            f'steer={camera_lines.steering_deg:+d}'
                        )
                    else:
                        guidance_text = (
                            'PRECISION: LiDAR/steer=0 fallback | '
                            f'original gone L/R='
                            f'{int(self.reference_upper_gone)}/'
                            f'{int(self.reference_lower_gone)}'
                        )
                else:
                    guidance_text = (
                        f'{self.final_correction_duration:.1f}s tilt '
                        f'correction {self.final_correction_count}/'
                        f'{self.final_correction_segment_count} | '
                        f'original gone L/R='
                        f'{int(self.reference_upper_gone)}/'
                        f'{int(self.reference_lower_gone)}'
                    )
            else:
                guidance_text = f'phase={self.reverse_phase}'
            cv2.putText(
                image,
                guidance_text,
                (18, 121),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (120, 255, 120),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            image,
            'REAR 0 | RIGHT +90 | FRONT +/-180 | LEFT -90',
            (18, size - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            (0, 220, 220), 1, cv2.LINE_AA,
        )
        cv2.imshow(self.debug_window_name, image)
        if self.precision_camera_debug_image is not None:
            cv2.imshow(
                f'{self.debug_window_name}_high_camera',
                self.precision_camera_debug_image,
            )
        cv2.waitKey(1)

    def destroy_node(self):
        try:
            self.publish_control(0, 0)
            if self.debug_view:
                cv2.destroyWindow(self.debug_window_name)
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ParkingNodeYym()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
