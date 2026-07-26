"""YYM LiDAR controller for reverse perpendicular parking on the right.

Scan convention used by this vehicle:
    rear=0 deg, right=+90 deg, front=+/-180 deg, left=-90 deg.

The controller approaches straight and, when the first parked vehicle is
confirmed, performs a calibrated 90-degree left turn relative to its starting
heading.  Parked-vehicle orientation is deliberately not used during this
recognition phase.  It then reverses between the two parked vehicles. During
reverse it continuously targets the middle of the free gap. Parking ends when
no valid LiDAR points remain below the LiDAR's horizontal x=0 line.
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
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int16MultiArray


class ParkingState(str, Enum):
    WAIT_FOR_SCAN = 'WAIT_FOR_SCAN'
    APPROACH_FIRST_CAR = 'APPROACH_FIRST_CAR'
    SET_LEFT_STEER = 'SET_LEFT_STEER'
    TURN_LEFT_TIMED = 'TURN_LEFT_TIMED'
    RECOGNITION_COMPLETE = 'RECOGNITION_COMPLETE'
    SETTLE_AND_ACQUIRE_GAP = 'SETTLE_AND_ACQUIRE_GAP'
    REVERSE_CENTER = 'REVERSE_CENTER'
    PARKED = 'PARKED'
    PARKING_FAILED = 'PARKING_FAILED'
    EMERGENCY_STOP = 'EMERGENCY_STOP'


class ParkingMode(str, Enum):
    RECOGNITION = 'RECOGNITION'
    PARKING = 'PARKING'


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


class ParkingNodeYym(Node):
    """LiDAR-feedback parking into random right-side slot 2 or 3."""

    def __init__(self) -> None:
        super().__init__('parking_node_yym')

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('motor_topic', '/motor_control')
        self.declare_parameter('control_hz', 20.0)
        self.declare_parameter('scan_timeout_sec', 0.5)
        self.declare_parameter('scan_quality_min_points', 10)
        self.declare_parameter('invalid_scan_confirm_frames', 5)
        # start_mode=recognition runs the normal complete sequence.
        # start_mode=parking skips recognition for debugging from an already
        # stationary pose after the timed left turn. recognition_only latches
        # a stop as soon as that recognition turn is complete.
        self.declare_parameter('start_mode', 'recognition')
        self.declare_parameter('recognition_only', False)

        self.declare_parameter('debug_view', True)
        self.declare_parameter('debug_window_name', 'parking_yym_debug')
        self.declare_parameter('debug_hz', 20.0)
        self.declare_parameter('debug_max_range_m', 3.0)
        self.declare_parameter('debug_image_size', 820)

        # Only the rear and side field is relevant after the left entry turn.
        self.declare_parameter('valid_sector_max_abs_deg', 125.0)
        self.declare_parameter('parking_min_range_m', 0.15)
        self.declare_parameter('cluster_max_range_m', 3.0)
        self.declare_parameter('cluster_neighbor_distance_m', 0.20)
        self.declare_parameter('cluster_min_points', 7)
        self.declare_parameter('obstacle_min_extent_m', 0.22)
        self.declare_parameter('right_detection_margin_m', 0.12)

        self.declare_parameter('first_car_confirm_frames', 3)
        self.declare_parameter('gap_confirm_frames', 3)
        self.declare_parameter('gap_min_width_m', 0.48)
        self.declare_parameter('gap_max_width_m', 1.40)
        self.declare_parameter('gap_track_max_center_m', 0.85)

        # Recognition-mode test speed requested for the real vehicle.
        self.declare_parameter('approach_speed', 110)
        self.declare_parameter('turn_speed', 110)
        self.declare_parameter('reverse_speed', -110)
        self.declare_parameter('left_max_steer_deg', -45)
        # No parking-specific steering cap: calculated reverse steering may
        # use the vehicle's full physical command range of +/-45 degrees.
        # Multiply every LiDAR-calculated reverse angle by this gain before
        # publishing it through /motor_control.
        self.declare_parameter('reverse_steer_multiplier', 3)
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
        self.declare_parameter('reverse_segment_duration_sec', 2.0)
        self.declare_parameter('reverse_measure_stop_sec', 0.4)
        self.declare_parameter('vehicle_pair_track_max_jump_m', 1.25)
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
        self.control_hz = max(1.0, float(self._value('control_hz')))
        self.scan_timeout = max(0.05, float(self._value('scan_timeout_sec')))
        self.scan_quality_min_points = int(self._value('scan_quality_min_points'))
        self.invalid_scan_confirm_frames = int(
            self._value('invalid_scan_confirm_frames')
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

        self.first_car_confirm_frames = max(
            1, int(self._value('first_car_confirm_frames'))
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
        self.reverse_steer_multiplier = max(
            0.0, float(self._value('reverse_steer_multiplier'))
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
        self.mode = self.start_mode
        self.state = ParkingState.WAIT_FOR_SCAN
        self.state_started_at = now
        self.last_scan_at: Optional[float] = None
        self.last_pair_at: Optional[float] = None
        self.last_command = (0, 0)
        self.failure_reason = ''
        self.shutdown_started = False
        self.reverse_segment_started_at: Optional[float] = None
        self.reverse_phase_started_at: Optional[float] = None
        self.reverse_phase = 'IDLE'
        self.reverse_segment_steer = 0
        self.reverse_segment_index = 0
        self.reverse_segment_drive_duration = self.reverse_segment_duration
        self.lower_vehicle_track_center: Optional[np.ndarray] = None
        self.upper_vehicle_track_center: Optional[np.ndarray] = None
        self.rear_half_lidar_point_count = 0
        self.rear_half_empty_frames = 0
        self.rear_half_points_seen = False
        self.straight_reverse_latched = False
        self.straight_reverse_started = False
        self.straight_reverse_trigger_distance = math.inf
        self.invalid_scan_count = 0
        self.first_car_frames = 0
        self.gap_frames = 0
        self.latest_scan: Optional[LaserScan] = None
        self.observation = self._empty_observation()

        self.motor_publisher = self.create_publisher(
            Int16MultiArray, self.motor_topic, 10
        )
        self.create_subscription(
            LaserScan, self.scan_topic, self.scan_callback, 10
        )
        self.create_timer(1.0 / self.control_hz, self.control_tick)
        if self.debug_view:
            self.create_timer(1.0 / self.debug_hz, self.draw_debug)

        self.get_logger().info(
            'parking_node_yym: straight approach -> first-car trigger -> '
            f'{self.left_turn_duration_sec:.2f}s max-left timed turn '
            '-> centered reverse between two right-slot vehicles; '
            f'reverse_steer_multiplier={self.reverse_steer_multiplier:.2f}x, '
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
        if self.state == ParkingState.REVERSE_CENTER:
            tracked_pair = self.track_parking_pair(observation.vehicles)
            if tracked_pair is None:
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

        if self.state == ParkingState.APPROACH_FIRST_CAR:
            self.first_car_frames = (
                self.first_car_frames + 1
                if observation.right_vehicles else 0
            )
        else:
            self.first_car_frames = 0

        if self.state == ParkingState.SETTLE_AND_ACQUIRE_GAP:
            self.gap_frames = (
                self.gap_frames + 1 if observation.pair is not None else 0
            )
        else:
            self.gap_frames = 0

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

    def update_straight_reverse_condition(
        self, vehicles: list[VehicleCluster]
    ) -> None:
        """Latch when any obstacle-vehicle point enters the second ring."""
        if self.straight_reverse_latched:
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
            self.reverse_phase = 'STRAIGHT_CENTER'
            self.reverse_phase_started_at = time.monotonic()
            self.get_logger().info(
                'Straight reverse locked: an obstacle point entered '
                f'the {self.straight_reverse_radius:.2f}m ring; stopping for '
                f'{self.straight_reverse_stop:.1f}s'
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
        vehicles = [
            self.make_vehicle_cluster(component)
            for component in self.connected_components(point_array)
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
        self, vehicles: list[VehicleCluster]
    ) -> Optional[ParkingPair]:
        """Keep both original vehicles even when the strict gap test fails."""
        if (
            len(vehicles) < 2
            or self.lower_vehicle_track_center is None
            or self.upper_vehicle_track_center is None
        ):
            return None

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
                    lower_jump > self.vehicle_pair_track_max_jump
                    or upper_jump > self.vehicle_pair_track_max_jump
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
        if pair is None:
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
            self.straight_reverse_latched = False
            self.straight_reverse_started = False
            self.straight_reverse_trigger_distance = math.inf
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
                f'Reverse segment 1 target: steer='
                f'{self.reverse_segment_steer}deg; '
                f'settling before '
                f'{self.reverse_segment_drive_duration:.1f}s drive'
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
        if self.state == ParkingState.PARKED:
            # PARKED is a final latch at zero steering and zero speed.
            self.reverse_phase = 'PARKED_HOLD'
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
            if self.mode == ParkingMode.PARKING:
                self.transition(
                    ParkingState.SETTLE_AND_ACQUIRE_GAP, now
                )
            else:
                self.transition(ParkingState.APPROACH_FIRST_CAR, now)

        elapsed = now - self.state_started_at
        if self.state == ParkingState.APPROACH_FIRST_CAR:
            if elapsed >= self.approach_timeout:
                self.fail('first parked vehicle detection timeout', now)
                return
            if self.first_car_frames >= self.first_car_confirm_frames:
                self.transition(ParkingState.SET_LEFT_STEER, now)
                self.publish_control(self.left_max_steer, 0)
                return
            self.publish_control(
                self.held_steering_command(),
                self.approach_speed,
            )
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
                if pair.gap_width < required_width:
                    self.fail(
                        f'gap too narrow: {pair.gap_width:.2f} m '
                        f'< {required_width:.2f} m',
                        now,
                    )
                    return
                self.transition(ParkingState.REVERSE_CENTER, now)
            self.publish_control(0, 0)
            return

        if self.state == ParkingState.REVERSE_CENTER:
            if (
                self.reverse_timeout > 0.0
                and elapsed >= self.reverse_timeout
            ):
                self.fail('segmented reverse timeout', now)
                return
            if (
                self.observation.rear_min_distance is not None
                and self.observation.rear_min_distance
                <= self.rear_hard_stop_distance
            ):
                self.fail('rear obstacle inside hard-stop distance', now)
                return

            if (
                self.straight_reverse_latched
                and not self.straight_reverse_started
            ):
                # The trigger owns steering from this point onward. Send one
                # stable target only; no pair/single-vehicle corrections run.
                self.reverse_phase = 'STRAIGHT_CENTER'
                self.publish_control(0, 0)
                if (
                    self.reverse_phase_started_at is not None
                    and now - self.reverse_phase_started_at
                    >= self.straight_reverse_stop
                ):
                    self.straight_reverse_started = True
                    self.reverse_phase = 'STRAIGHT_DRIVE'
                    self.reverse_phase_started_at = now
                    self.rear_half_points_seen = (
                        self.rear_half_lidar_point_count > 0
                    )
                    self.rear_half_empty_frames = 0
                return

            if self.rear_half_empty_frames > 0:
                # Stop on the very first empty scan. Only latch PARKED after
                # several consecutive empty scans while already stationary.
                self.publish_control(0, 0)
                if self.reverse_phase != 'MEASURE_STOP':
                    self.reverse_phase = 'MEASURE_STOP'
                    self.reverse_phase_started_at = now
                if (
                    self.rear_half_empty_frames
                    >= self.rear_half_empty_confirm_frames
                ):
                    self.transition(ParkingState.PARKED, now)
                    self.publish_control(0, 0)
                    self.get_logger().info(
                        'Parking complete: no valid LiDAR points remain '
                        'below the LiDAR x=0 line'
                    )
                return

            if self.straight_reverse_latched:
                self.reverse_phase = 'STRAIGHT_DRIVE'
                self.publish_control(0, self.reverse_speed)
                return

            pair = self.observation.pair
            single_vehicle = (
                self.observation.vehicles[0]
                if pair is None and len(self.observation.vehicles) == 1
                else None
            )
            if pair is None and single_vehicle is None:
                # No usable vehicle geometry is available. Wait stopped for
                # either a two-vehicle pair or one vehicle line to reappear.
                if self.reverse_phase != 'MEASURE_STOP':
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
                        f'Reverse segment {self.reverse_segment_index} '
                        f'driving with steer='
                        f'{self.reverse_segment_steer}deg for '
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

            if self.reverse_phase == 'MEASURE_STOP':
                self.publish_control(self.held_steering_command(), 0)
                if (
                    self.reverse_phase_started_at is None
                    or now - self.reverse_phase_started_at
                    < self.reverse_measure_stop
                ):
                    return

                self.reverse_segment_started_at = now
                self.reverse_segment_index += 1
                if pair is not None:
                    self.reverse_segment_steer = (
                        self.reverse_steering(pair)
                    )
                    target_description = (
                        f'reference=({pair.reference_point[0]:+.2f},'
                        f'{pair.reference_point[1]:+.2f})m'
                    )
                else:
                    self.reverse_segment_steer = (
                        self.single_vehicle_steering(single_vehicle)
                    )
                    target_description = (
                        'single-axis='
                        f'{math.degrees(single_vehicle.axis_angle):+.1f}deg'
                    )
                self.reverse_phase = 'STEER_SETTLE'
                self.reverse_phase_started_at = now
                self.get_logger().info(
                    f'Reverse segment {self.reverse_segment_index} target: '
                    f'{target_description}, '
                    f'steer={self.reverse_segment_steer}deg'
                )
                return

            self.reverse_phase = 'MEASURE_STOP'
            self.reverse_phase_started_at = now
            self.publish_control(self.held_steering_command(), 0)
            return

        if self.state == ParkingState.PARKED:
            # Finish with the front wheels centered as requested.
            self.publish_control(0, 0)
        else:
            # Other safe states hold their current angle to avoid hunting.
            self.publish_control(self.held_steering_command(), 0)

    def reverse_steering(self, pair: ParkingPair) -> int:
        """Return the red angle from the rear baseline to the reference point."""
        reference_x = float(pair.reference_point[0])
        reference_y = float(pair.reference_point[1])
        # Rear baseline is -x. In the vehicle's LiDAR convention, a point to
        # the right has positive bearing and therefore commands right steer.
        rear_distance = max(0.05, -reference_x)
        reference_angle = math.degrees(
            math.atan2(-reference_y, rear_distance)
        )
        return self.scaled_reverse_steering(reference_angle)

    def single_vehicle_steering(self, vehicle: VehicleCluster) -> int:
        """Parallelize the ego with the only visible parked-vehicle line."""
        axis_angle = math.degrees(vehicle.axis_angle)
        if abs(axis_angle) <= self.single_vehicle_angle_deadband:
            return 0
        # In this LiDAR/debug convention +axis tilts toward the right while
        # reversing, so it directly produces a positive (right) steer.
        return self.scaled_reverse_steering(axis_angle)

    def scaled_reverse_steering(self, calculated_angle: float) -> int:
        """Scale a LiDAR reverse angle and clamp it to the steering range."""
        scaled_angle = calculated_angle * self.reverse_steer_multiplier
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
        straight_gate_radius = int(self.straight_reverse_radius * scale)
        straight_gate_color = (
            (0, 220, 0) if self.straight_reverse_latched else (0, 100, 0)
        )
        cv2.circle(
            image,
            (center, center),
            straight_gate_radius,
            straight_gate_color,
            2,
        )
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
        cv2.putText(
            image,
            f'MODE: {self.mode.value} | STATE: {self.state.value}',
            (18, 36),
            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2,
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
                single_vehicle = self.observation.vehicles[0]
                pair_text = (
                    f'single-axis='
                    f'{math.degrees(single_vehicle.axis_angle):+.1f}deg '
                    f'steer='
                    f'{self.single_vehicle_steering(single_vehicle):+d}deg '
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
            pair_text = (
                f'seg={self.reverse_segment_index} '
                f'gap={pair.gap_width:.2f}m center='
                f'{pair.gap_center_y:+.2f}m L/R='
                f'{pair.left_clearance:.2f}/'
                f'{pair.right_clearance:.2f}m '
                f'ref-angle={self.reverse_steering(pair):+d}deg '
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
            radius_text = (
                f'{self.straight_reverse_trigger_distance:.2f}m'
                if math.isfinite(self.straight_reverse_trigger_distance)
                else '--'
            )
            straight_state = (
                'LOCKED'
                if self.straight_reverse_latched
                else 'WAIT'
            )
            cv2.putText(
                image,
                f'straight={straight_state} nearest={radius_text} '
                f'gate={self.straight_reverse_radius:.2f}m '
                f'step={self.reverse_segment_drive_duration:.1f}s '
                f'gain={self.reverse_steer_multiplier:.2f}x',
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
