"""LiDAR-only parking controller: /scan -> /motor_control.

Scan convention:
    0 deg = vehicle front, + = vehicle left, - = vehicle right.
Only rear quadrants 3 and 4 are used (|angle| >= 90 deg); points in front
quadrants 1 and 2 are discarded before any parking decision is made.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int16MultiArray


class ParkingState(str, Enum):
    SEARCH_TWO_CARS = 'SEARCH_TWO_CARS'
    PASS_SECOND_CAR = 'PASS_SECOND_CAR'
    SET_REVERSE_STEER = 'SET_REVERSE_STEER'
    REVERSE_HARD_RIGHT = 'REVERSE_HARD_RIGHT'
    REVERSE_BALANCE = 'REVERSE_BALANCE'
    PARK_STOP = 'PARK_STOP'
    FORWARD_BALANCE = 'FORWARD_BALANCE'
    EXIT_RIGHT_TURN = 'EXIT_RIGHT_TURN'
    EXIT_FORWARD = 'EXIT_FORWARD'
    DONE = 'DONE'
    EMERGENCY_STOP = 'EMERGENCY_STOP'


@dataclass
class RearObservation:
    """Clusters and nearest left/right parked-car distances in rear quadrants."""

    clusters: list[np.ndarray]
    left_distance: Optional[float]
    right_distance: Optional[float]

    @property
    def two_car_bundles(self) -> bool:
        return len(self.clusters) >= 2

    @property
    def both_sides_visible(self) -> bool:
        return self.left_distance is not None and self.right_distance is not None


class ParkingNodeOsy(Node):
    """Hard-coded T-parking sequence with rear-quadrant LiDAR fine alignment."""

    def __init__(self) -> None:
        super().__init__('parking_node_osy')

        # This node deliberately has only one input and one output.
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('motor_topic', '/motor_control')
        self.declare_parameter('control_hz', 20.0)
        self.declare_parameter('scan_timeout_sec', 0.5)

        # Use only rear quadrants 3 and 4: +90..+180 and -180..-90 deg.
        self.declare_parameter('rear_sector_min_abs_deg', 90.0)
        self.declare_parameter('cluster_max_range_m', 2.0)
        self.declare_parameter('cluster_join_distance_m', 0.18)
        self.declare_parameter('cluster_min_points', 5)
        self.declare_parameter('two_car_confirm_frames', 4)
        self.declare_parameter('bundle_missing_frames', 4)

        # Motion signs are hardware dependent.  Per request, entry/exit turn
        # defaults are right turn; tune these values on the vehicle.
        self.declare_parameter('forward_speed', 22)
        self.declare_parameter('reverse_speed', -18)
        self.declare_parameter('balance_reverse_speed', -11)
        self.declare_parameter('right_turn_steer', -45)
        self.declare_parameter('steer_settle_sec', 0.60)
        self.declare_parameter('pass_second_car_sec', 0.70)
        self.declare_parameter('reverse_seek_timeout_sec', 5.0)
        self.declare_parameter('park_stop_sec', 1.0)
        self.declare_parameter('forward_seek_timeout_sec', 4.0)
        self.declare_parameter('exit_right_turn_sec', 1.20)
        self.declare_parameter('exit_forward_sec', 2.0)

        # Fine alignment: target left/right distance equality.  The sign is
        # exposed because reverse steering direction differs between vehicles.
        self.declare_parameter('balance_steer_kp', 55.0)
        self.declare_parameter('balance_steer_sign', 1.0)
        self.declare_parameter('balance_max_steer', 20)

        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.motor_topic = str(self.get_parameter('motor_topic').value)
        self.control_hz = max(1.0, float(self.get_parameter('control_hz').value))
        self.scan_timeout_sec = float(self.get_parameter('scan_timeout_sec').value)
        self.rear_sector_min_abs = math.radians(float(
            self.get_parameter('rear_sector_min_abs_deg').value
        ))
        self.cluster_max_range = float(self.get_parameter('cluster_max_range_m').value)
        self.cluster_join_distance = float(
            self.get_parameter('cluster_join_distance_m').value
        )
        self.cluster_min_points = int(self.get_parameter('cluster_min_points').value)
        self.two_car_confirm_frames = int(
            self.get_parameter('two_car_confirm_frames').value
        )
        self.bundle_missing_frames = int(
            self.get_parameter('bundle_missing_frames').value
        )
        self.forward_speed = int(self.get_parameter('forward_speed').value)
        self.reverse_speed = int(self.get_parameter('reverse_speed').value)
        self.balance_reverse_speed = int(
            self.get_parameter('balance_reverse_speed').value
        )
        self.right_turn_steer = int(self.get_parameter('right_turn_steer').value)
        self.steer_settle_sec = float(self.get_parameter('steer_settle_sec').value)
        self.pass_second_car_sec = float(
            self.get_parameter('pass_second_car_sec').value
        )
        self.reverse_seek_timeout_sec = float(
            self.get_parameter('reverse_seek_timeout_sec').value
        )
        self.park_stop_sec = float(self.get_parameter('park_stop_sec').value)
        self.forward_seek_timeout_sec = float(
            self.get_parameter('forward_seek_timeout_sec').value
        )
        self.exit_right_turn_sec = float(
            self.get_parameter('exit_right_turn_sec').value
        )
        self.exit_forward_sec = float(self.get_parameter('exit_forward_sec').value)
        self.balance_steer_kp = float(self.get_parameter('balance_steer_kp').value)
        self.balance_steer_sign = float(self.get_parameter('balance_steer_sign').value)
        self.balance_max_steer = int(self.get_parameter('balance_max_steer').value)

        self.state = ParkingState.SEARCH_TWO_CARS
        self.state_started_at = time.monotonic()
        self.last_scan_at: Optional[float] = None
        self.observation = RearObservation([], None, None)
        self.two_car_frames = 0
        self.missing_bundle_frames = 0
        self.forward_bundles_seen = False
        self.last_balance_steer = 0

        self.motor_pub = self.create_publisher(Int16MultiArray, self.motor_topic, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)
        self.create_timer(1.0 / self.control_hz, self.control_tick)
        self.get_logger().info(
            'parking_node_osy: LiDAR-only /scan -> /motor_control; '
            'using rear quadrants 3/4 only (|angle| >= 90 deg)'
        )

    def scan_callback(self, msg: LaserScan) -> None:
        self.observation = self.observe_rear_car_bundles(msg)
        self.last_scan_at = time.monotonic()

    def observe_rear_car_bundles(self, msg: LaserScan) -> RearObservation:
        """Cluster only rear-quadrant returns, discarding front quadrants 1/2."""
        ordered_points: list[tuple[float, float]] = []
        for index, distance in enumerate(msg.ranges):
            if not math.isfinite(distance):
                continue
            if distance < msg.range_min or distance > min(msg.range_max, self.cluster_max_range):
                continue
            angle = msg.angle_min + index * msg.angle_increment
            angle = math.atan2(math.sin(angle), math.cos(angle))
            if abs(angle) < self.rear_sector_min_abs:
                continue  # discard quadrants 1 and 2 completely
            ordered_points.append((distance * math.cos(angle), distance * math.sin(angle)))

        clusters: list[np.ndarray] = []
        current: list[tuple[float, float]] = []
        previous: Optional[np.ndarray] = None
        for point_tuple in ordered_points:
            point = np.asarray(point_tuple, dtype=np.float64)
            if previous is not None and np.linalg.norm(point - previous) > self.cluster_join_distance:
                if len(current) >= self.cluster_min_points:
                    clusters.append(np.asarray(current, dtype=np.float64))
                current = []
            current.append(point_tuple)
            previous = point
        if len(current) >= self.cluster_min_points:
            clusters.append(np.asarray(current, dtype=np.float64))

        # A vehicle bundle is assigned to its side by its centroid.  The
        # nearest bundle on each side drives fine left/right distance control.
        left = [cluster for cluster in clusters if float(cluster[:, 1].mean()) > 0.0]
        right = [cluster for cluster in clusters if float(cluster[:, 1].mean()) < 0.0]
        left_distance = self.nearest_lateral_distance(left, side=1.0)
        right_distance = self.nearest_lateral_distance(right, side=-1.0)
        return RearObservation(clusters, left_distance, right_distance)

    @staticmethod
    def nearest_lateral_distance(clusters: list[np.ndarray], side: float) -> Optional[float]:
        if not clusters:
            return None
        distances = [float(np.median(cluster[:, 1] * side)) for cluster in clusters]
        return min(distance for distance in distances if distance > 0.0)

    def transition(self, next_state: ParkingState, now: float) -> None:
        if self.state != next_state:
            self.get_logger().info(f'Parking: {self.state.value} -> {next_state.value}')
            self.state = next_state
            self.state_started_at = now
            self.missing_bundle_frames = 0

    def balance_steer(self) -> int:
        """Equalise left/right parked-car distance in rear quadrants 3/4."""
        left = self.observation.left_distance
        right = self.observation.right_distance
        if left is None or right is None:
            return self.last_balance_steer
        # Positive error means the left vehicle is farther than the right one.
        error = left - right
        steer = int(np.clip(
            self.balance_steer_sign * self.balance_steer_kp * error,
            -self.balance_max_steer,
            self.balance_max_steer,
        ))
        self.last_balance_steer = steer
        return steer

    def control_tick(self) -> None:
        now = time.monotonic()
        if self.last_scan_at is None or now - self.last_scan_at > self.scan_timeout_sec:
            self.transition(ParkingState.EMERGENCY_STOP, now)
            self.publish(0, 0)
            return

        elapsed = now - self.state_started_at
        two_bundles = self.observation.two_car_bundles
        both_sides = self.observation.both_sides_visible

        if self.state == ParkingState.SEARCH_TWO_CARS:
            self.two_car_frames = self.two_car_frames + 1 if two_bundles else 0
            if self.two_car_frames >= self.two_car_confirm_frames:
                self.transition(ParkingState.PASS_SECOND_CAR, now)
            self.publish(0, self.forward_speed)
            return

        if self.state == ParkingState.PASS_SECOND_CAR:
            # Second parked-car bundle is confirmed: move forward a little,
            # then lock maximum right steering before the hard-coded reverse.
            if elapsed >= self.pass_second_car_sec:
                self.transition(ParkingState.SET_REVERSE_STEER, now)
            self.publish(0, self.forward_speed)
            return

        if self.state == ParkingState.SET_REVERSE_STEER:
            if elapsed >= self.steer_settle_sec:
                self.transition(ParkingState.REVERSE_HARD_RIGHT, now)
            self.publish(self.right_turn_steer, 0)
            return

        if self.state == ParkingState.REVERSE_HARD_RIGHT:
            # Hard-coded right reverse until both parked-car bundles enter
            # rear quadrants 3 and 4, then begin LiDAR fine alignment.
            if both_sides:
                self.transition(ParkingState.REVERSE_BALANCE, now)
            elif elapsed >= self.reverse_seek_timeout_sec:
                self.transition(ParkingState.PARK_STOP, now)
                self.publish(0, 0)
                return
            self.publish(self.right_turn_steer, self.reverse_speed)
            return

        if self.state == ParkingState.REVERSE_BALANCE:
            if both_sides:
                self.missing_bundle_frames = 0
                self.publish(self.balance_steer(), self.balance_reverse_speed)
            else:
                self.missing_bundle_frames += 1
                # Both vehicle bundles disappearing means the car has reached
                # the parking depth requested in the mission sequence.
                if self.missing_bundle_frames >= self.bundle_missing_frames:
                    self.transition(ParkingState.PARK_STOP, now)
                    self.publish(0, 0)
                else:
                    self.publish(self.last_balance_steer, self.balance_reverse_speed)
            return

        if self.state == ParkingState.PARK_STOP:
            if elapsed >= self.park_stop_sec:
                self.forward_bundles_seen = False
                self.transition(ParkingState.FORWARD_BALANCE, now)
            self.publish(0, 0)
            return

        if self.state == ParkingState.FORWARD_BALANCE:
            # Drive forward while the two rear bundles are visible, preserving
            # equal clearance.  After seeing them, their disappearance starts
            # the hard-coded right exit turn.
            if both_sides:
                self.forward_bundles_seen = True
                self.missing_bundle_frames = 0
                self.publish(self.balance_steer(), self.forward_speed)
                return
            self.missing_bundle_frames += 1
            should_exit = (
                self.forward_bundles_seen
                and self.missing_bundle_frames >= self.bundle_missing_frames
            ) or elapsed >= self.forward_seek_timeout_sec
            if should_exit:
                self.transition(ParkingState.EXIT_RIGHT_TURN, now)
                self.publish(self.right_turn_steer, self.forward_speed)
            else:
                self.publish(self.last_balance_steer, self.forward_speed)
            return

        if self.state == ParkingState.EXIT_RIGHT_TURN:
            if elapsed >= self.exit_right_turn_sec:
                self.transition(ParkingState.EXIT_FORWARD, now)
                self.publish(0, self.forward_speed)
            else:
                self.publish(self.right_turn_steer, self.forward_speed)
            return

        if self.state == ParkingState.EXIT_FORWARD:
            if elapsed >= self.exit_forward_sec:
                self.transition(ParkingState.DONE, now)
                self.publish(0, 0)
            else:
                self.publish(0, self.forward_speed)
            return

        self.publish(0, 0)  # DONE or EMERGENCY_STOP

    def publish(self, steer: int, speed: int) -> None:
        message = Int16MultiArray()
        message.data = [int(np.clip(steer, -45, 45)), int(speed)]
        self.motor_pub.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ParkingNodeOsy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish(0, 0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
