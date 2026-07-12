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

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int16MultiArray


class ParkingState(str, Enum):
    WAIT_FOR_SCAN = 'WAIT_FOR_SCAN'
    WAIT_Q4_CLEAR_INITIAL = 'WAIT_Q4_CLEAR_INITIAL'
    Q4_IGNORE_DELAY = 'Q4_IGNORE_DELAY'
    WAIT_CAR1_ENTRY = 'WAIT_CAR1_ENTRY'
    WAIT_Q4_CLEAR_AFTER_CAR1 = 'WAIT_Q4_CLEAR_AFTER_CAR1'
    WAIT_CAR2_ENTRY = 'WAIT_CAR2_ENTRY'
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
    q4_clusters: list[np.ndarray]
    left_distance: Optional[float]
    right_distance: Optional[float]

    @property
    def two_car_bundles(self) -> bool:
        return len(self.clusters) >= 2

    @property
    def both_sides_visible(self) -> bool:
        return self.left_distance is not None and self.right_distance is not None

    @property
    def q4_visible(self) -> bool:
        return bool(self.q4_clusters)


class ParkingNodeOsy(Node):
    """Hard-coded T-parking sequence with rear-quadrant LiDAR fine alignment."""

    def __init__(self) -> None:
        super().__init__('parking_node_osy')

        # This node deliberately has only one input and one output.
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('motor_topic', '/motor_control')
        self.declare_parameter('control_hz', 20.0)
        self.declare_parameter('scan_timeout_sec', 0.5)
        self.declare_parameter('debug_view', True)
        self.declare_parameter('debug_window_name', 'parking_lidar_sequence_debug')

        # Use only rear quadrants 3 and 4: +90..+180 and -180..-90 deg.
        self.declare_parameter('rear_sector_min_abs_deg', 90.0)
        self.declare_parameter('cluster_max_range_m', 2.0)
        self.declare_parameter('cluster_join_distance_m', 0.18)
        self.declare_parameter('cluster_min_points', 5)
        self.declare_parameter('q4_empty_confirm_frames', 3)
        self.declare_parameter('q4_entry_confirm_frames', 3)
        self.declare_parameter('q4_ignore_after_clear_sec', 1.0)
        self.declare_parameter('bundle_missing_frames', 4)

        # Motion signs are hardware dependent.  Per request, entry/exit turn
        # defaults are right turn; tune these values on the vehicle.
        self.declare_parameter('forward_speed', 110)
        self.declare_parameter('reverse_speed', -110)
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
        self.declare_parameter('balance_max_steer', 45)

        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.motor_topic = str(self.get_parameter('motor_topic').value)
        self.control_hz = max(1.0, float(self.get_parameter('control_hz').value))
        self.scan_timeout_sec = float(self.get_parameter('scan_timeout_sec').value)
        self.debug_view = bool(self.get_parameter('debug_view').value)
        self.debug_window_name = str(self.get_parameter('debug_window_name').value)
        self.rear_sector_min_abs = math.radians(float(
            self.get_parameter('rear_sector_min_abs_deg').value
        ))
        self.cluster_max_range = float(self.get_parameter('cluster_max_range_m').value)
        self.cluster_join_distance = float(
            self.get_parameter('cluster_join_distance_m').value
        )
        self.cluster_min_points = int(self.get_parameter('cluster_min_points').value)
        self.q4_empty_confirm_frames = int(
            self.get_parameter('q4_empty_confirm_frames').value
        )
        self.q4_entry_confirm_frames = int(
            self.get_parameter('q4_entry_confirm_frames').value
        )
        self.q4_ignore_after_clear_sec = float(
            self.get_parameter('q4_ignore_after_clear_sec').value
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

        # LiDAR 드라이버가 기동되어 첫 /scan을 내보낼 때까지는 emergency가 아닌
        # 안전 정지 대기 상태로 유지한다.
        self.state = ParkingState.WAIT_FOR_SCAN
        self.state_started_at = time.monotonic()
        self.last_scan_at: Optional[float] = None
        self.observation = RearObservation([], [], None, None)
        self.q4_empty_frames = 0
        self.q4_entry_frames = 0
        self.missing_bundle_frames = 0
        self.forward_bundles_seen = False
        self.last_balance_steer = 0
        self.last_command = (0, 0)

        self.motor_pub = self.create_publisher(Int16MultiArray, self.motor_topic, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)
        self.create_timer(1.0 / self.control_hz, self.control_tick)
        if self.debug_view:
            self.create_timer(0.1, self.draw_debug)
        self.get_logger().info(
            'parking_node_osy: LiDAR-only /scan -> /motor_control; '
            'using rear quadrants 3/4 only; car1/car2 are counted by Q4 entries'
        )

    def scan_callback(self, msg: LaserScan) -> None:
        self.observation = self.observe_rear_car_bundles(msg)
        self.last_scan_at = time.monotonic()

    def observe_rear_car_bundles(self, msg: LaserScan) -> RearObservation:
        """Cluster only rear-quadrant returns, discarding front quadrants 1/2."""
        ordered_points: list[tuple[float, float]] = []
        q4_ordered_points: list[tuple[float, float]] = []
        for index, distance in enumerate(msg.ranges):
            if not math.isfinite(distance):
                continue
            if distance < msg.range_min or distance > min(msg.range_max, self.cluster_max_range):
                continue
            angle = msg.angle_min + index * msg.angle_increment
            angle = math.atan2(math.sin(angle), math.cos(angle))
            if abs(angle) < self.rear_sector_min_abs:
                continue  # discard quadrants 1 and 2 completely
            point = (distance * math.cos(angle), distance * math.sin(angle))
            ordered_points.append(point)
            # Q4 = rear-right sector: -180 .. -90 deg.  Only this sector is
            # used to count the sequential car1/car2 entry events.
            if angle <= -self.rear_sector_min_abs:
                q4_ordered_points.append(point)

        clusters = self.cluster_ordered_points(ordered_points)
        q4_clusters = self.cluster_ordered_points(q4_ordered_points)

        # A vehicle bundle is assigned to its side by its centroid.  The
        # nearest bundle on each side drives fine left/right distance control.
        left = [cluster for cluster in clusters if float(cluster[:, 1].mean()) > 0.0]
        right = [cluster for cluster in clusters if float(cluster[:, 1].mean()) < 0.0]
        left_distance = self.nearest_lateral_distance(left, side=1.0)
        right_distance = self.nearest_lateral_distance(right, side=-1.0)
        return RearObservation(clusters, q4_clusters, left_distance, right_distance)

    def cluster_ordered_points(self, ordered_points: list[tuple[float, float]]) -> list[np.ndarray]:
        """Split consecutive scan returns into parked-car point bundles."""
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
        return clusters

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
        if self.last_scan_at is None:
            self.publish(0, 0)
            return
        if self.state == ParkingState.WAIT_FOR_SCAN:
            self.transition(ParkingState.WAIT_Q4_CLEAR_INITIAL, now)
        if now - self.last_scan_at > self.scan_timeout_sec:
            self.transition(ParkingState.EMERGENCY_STOP, now)
            self.publish(0, 0)
            return

        elapsed = now - self.state_started_at
        both_sides = self.observation.both_sides_visible
        q4_visible = self.observation.q4_visible

        if self.state == ParkingState.WAIT_Q4_CLEAR_INITIAL:
            # Ignore every Q3/Q4 object visible at start.  Only after Q4 is
            # truly empty do we begin counting car bundles.
            self.q4_empty_frames = self.q4_empty_frames + 1 if not q4_visible else 0
            if self.q4_empty_frames >= self.q4_empty_confirm_frames:
                self.q4_entry_frames = 0
                self.transition(ParkingState.Q4_IGNORE_DELAY, now)
            self.publish(0, self.forward_speed)
            return

        if self.state == ParkingState.Q4_IGNORE_DELAY:
            # Initial Q4 bundles are deliberately ignored for one second.
            # Only a bundle entering after this delay can become car #1.
            if elapsed >= self.q4_ignore_after_clear_sec:
                self.q4_entry_frames = 0
                self.transition(ParkingState.WAIT_CAR1_ENTRY, now)
            self.publish(0, self.forward_speed)
            return

        if self.state == ParkingState.WAIT_CAR1_ENTRY:
            self.q4_entry_frames = self.q4_entry_frames + 1 if q4_visible else 0
            if self.q4_entry_frames >= self.q4_entry_confirm_frames:
                self.get_logger().info('Q4 car bundle #1 detected')
                self.q4_empty_frames = 0
                self.transition(ParkingState.WAIT_Q4_CLEAR_AFTER_CAR1, now)
            self.publish(0, self.forward_speed)
            return

        if self.state == ParkingState.WAIT_Q4_CLEAR_AFTER_CAR1:
            self.q4_empty_frames = self.q4_empty_frames + 1 if not q4_visible else 0
            if self.q4_empty_frames >= self.q4_empty_confirm_frames:
                self.q4_entry_frames = 0
                self.transition(ParkingState.WAIT_CAR2_ENTRY, now)
            self.publish(0, self.forward_speed)
            return

        if self.state == ParkingState.WAIT_CAR2_ENTRY:
            self.q4_entry_frames = self.q4_entry_frames + 1 if q4_visible else 0
            if self.q4_entry_frames >= self.q4_entry_confirm_frames:
                self.get_logger().info('Q4 car bundle #2 detected; starting parking entry')
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
        self.last_command = (int(np.clip(steer, -45, 45)), int(speed))
        message = Int16MultiArray()
        message.data = list(self.last_command)
        self.motor_pub.publish(message)

    def destroy_node(self):
        """Always send a final stop command on Ctrl+C or launch shutdown."""
        try:
            self.publish(0, 0)
        except Exception:
            # ROS publisher/context may already be torn down during shutdown.
            pass
        if self.debug_view:
            cv2.destroyAllWindows()
        return super().destroy_node()

    def draw_debug(self) -> None:
        """Visualise the active parking sequence and rear-quadrant car bundles."""
        size = 760
        center = size // 2
        scale = (size * 0.40) / max(self.cluster_max_range, 0.1)
        image = np.zeros((size, size, 3), dtype=np.uint8)

        # Rear half only: quadrants 3 and 4 used by the controller.
        radius = int(self.cluster_max_range * scale)
        cv2.ellipse(image, (center, center), (radius, radius), 0, 90, 270,
                    (0, 120, 120), 1)
        cv2.circle(image, (center, center), 8, (0, 255, 0), -1)
        cv2.putText(image, 'CAR', (center + 12, center + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(image, 'REAR Q3/Q4 ONLY', (center - 120, center + radius - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1, cv2.LINE_AA)

        colors = [(0, 255, 0), (0, 180, 255), (255, 100, 0), (255, 0, 255)]
        for cluster_index, cluster in enumerate(self.observation.clusters):
            color = colors[cluster_index % len(colors)]
            for x_forward, y_left in cluster:
                x = int(center - y_left * scale)
                y = int(center - x_forward * scale)
                cv2.circle(image, (x, y), 2, color, -1)

        steps = [
            ParkingState.WAIT_FOR_SCAN,
            ParkingState.WAIT_Q4_CLEAR_INITIAL, ParkingState.Q4_IGNORE_DELAY,
            ParkingState.WAIT_CAR1_ENTRY,
            ParkingState.WAIT_Q4_CLEAR_AFTER_CAR1, ParkingState.WAIT_CAR2_ENTRY,
            ParkingState.PASS_SECOND_CAR,
            ParkingState.SET_REVERSE_STEER, ParkingState.REVERSE_HARD_RIGHT,
            ParkingState.REVERSE_BALANCE, ParkingState.PARK_STOP,
            ParkingState.FORWARD_BALANCE, ParkingState.EXIT_RIGHT_TURN,
            ParkingState.EXIT_FORWARD, ParkingState.DONE,
        ]
        cv2.rectangle(image, (8, 8), (size - 8, 260), (50, 50, 50), 1)
        cv2.putText(image, f'STATE: {self.state.value}', (18, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        left = '-' if self.observation.left_distance is None else f'{self.observation.left_distance:.2f}'
        right = '-' if self.observation.right_distance is None else f'{self.observation.right_distance:.2f}'
        cv2.putText(image, f'bundles={len(self.observation.clusters)} q4={len(self.observation.q4_clusters)} left={left}m right={right}m',
                    (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, f'cmd: steer={self.last_command[0]}, speed={self.last_command[1]}',
                    (18, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        for index, step in enumerate(steps):
            col, row = index % 2, index // 2
            color = (0, 255, 0) if step == self.state else (165, 165, 165)
            prefix = '>> ' if step == self.state else '   '
            cv2.putText(image, prefix + step.value, (18 + col * 365, 114 + row * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        cv2.imshow(self.debug_window_name, image)
        cv2.waitKey(1)


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
