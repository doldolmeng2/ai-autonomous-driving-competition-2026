"""LiDAR-only perpendicular parking node.

Physical scan convention used by this node:
    0 deg       : vehicle rear
    +/-180 deg  : vehicle front
    positive    : clockwise

The rear LiDAR is occluded in scan quadrants 1 and 2, therefore only the
[-180, 0] degree sector is used.  This implementation intentionally has only
one input (/scan) and one output (/motor_control).
"""

from __future__ import annotations

from collections import deque
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
    SEARCH = 'SEARCH'
    APPROACH = 'APPROACH'
    STEER_IN = 'STEER_IN'
    REVERSE_ARC = 'REVERSE_ARC'
    COUNTER_STEER = 'COUNTER_STEER'
    REVERSE_ALIGN = 'REVERSE_ALIGN'
    HOLD = 'HOLD'
    EXIT_STRAIGHT = 'EXIT_STRAIGHT'
    EXIT_TURN = 'EXIT_TURN'
    DONE = 'DONE'
    EMERGENCY_STOP = 'EMERGENCY_STOP'


@dataclass
class SlotCandidate:
    start_x: float
    end_x: float
    side_distance: float

    @property
    def center_x(self) -> float:
        return 0.5 * (self.start_x + self.end_x)


class ParkingNodeOsy(Node):
    """Single-node LiDAR-only T-parking controller."""

    def __init__(self) -> None:
        super().__init__('parking_node_osy')

        # Topics: PDF-defined /scan input and /motor_control output only.
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('motor_topic', '/motor_control')
        self.declare_parameter('control_hz', 20.0)

        # Sensor convention and usable rear-LiDAR sector.
        self.declare_parameter('usable_angle_min_deg', -180.0)
        self.declare_parameter('usable_angle_max_deg', 0.0)
        self.declare_parameter('slot_side', 'right')  # right = y < 0
        self.declare_parameter('scan_timeout_sec', 0.5)

        # Gap detector: parked car -> empty gap -> parked car.
        self.declare_parameter('slot_x_min_m', -1.6)
        self.declare_parameter('slot_x_max_m', 3.0)
        self.declare_parameter('slot_side_min_m', 0.25)
        self.declare_parameter('slot_side_max_m', 2.2)
        self.declare_parameter('slot_bin_size_m', 0.08)
        self.declare_parameter('slot_min_points_per_bin', 2)
        self.declare_parameter('slot_min_vehicle_length_m', 0.24)
        self.declare_parameter('slot_min_length_m', 0.65)
        self.declare_parameter('slot_max_length_m', 1.35)
        self.declare_parameter('slot_confirm_frames', 4)

        # Motion and geometry. Signs must be verified on the real vehicle.
        self.declare_parameter('forward_speed', 22)
        self.declare_parameter('reverse_speed', -18)
        self.declare_parameter('align_reverse_speed', -11)
        self.declare_parameter('entry_steer', -45)
        self.declare_parameter('exit_steer', 45)
        self.declare_parameter('approach_pass_x_m', -0.20)
        self.declare_parameter('steer_settle_sec', 0.60)
        self.declare_parameter('reverse_arc_sec', 2.40)
        self.declare_parameter('counter_steer_sec', 0.60)
        self.declare_parameter('align_timeout_sec', 5.0)
        self.declare_parameter('rear_stop_distance_m', 0.18)
        self.declare_parameter('side_hard_stop_distance_m', 0.16)
        self.declare_parameter('side_target_distance_m', 0.42)
        self.declare_parameter('side_steer_kp', 70.0)
        self.declare_parameter('hold_sec', 2.0)
        self.declare_parameter('exit_straight_sec', 1.2)
        self.declare_parameter('exit_turn_sec', 1.2)

        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.motor_topic = str(self.get_parameter('motor_topic').value)
        self.control_hz = max(1.0, float(self.get_parameter('control_hz').value))
        self.usable_min = math.radians(float(
            self.get_parameter('usable_angle_min_deg').value
        ))
        self.usable_max = math.radians(float(
            self.get_parameter('usable_angle_max_deg').value
        ))
        self.slot_side_sign = -1.0 if str(
            self.get_parameter('slot_side').value
        ).lower() == 'right' else 1.0
        self.scan_timeout_sec = float(self.get_parameter('scan_timeout_sec').value)

        self.slot_x_min = float(self.get_parameter('slot_x_min_m').value)
        self.slot_x_max = float(self.get_parameter('slot_x_max_m').value)
        self.slot_side_min = float(self.get_parameter('slot_side_min_m').value)
        self.slot_side_max = float(self.get_parameter('slot_side_max_m').value)
        self.slot_bin_size = float(self.get_parameter('slot_bin_size_m').value)
        self.slot_min_points = int(self.get_parameter('slot_min_points_per_bin').value)
        self.slot_min_vehicle_length = float(
            self.get_parameter('slot_min_vehicle_length_m').value
        )
        self.slot_min_length = float(self.get_parameter('slot_min_length_m').value)
        self.slot_max_length = float(self.get_parameter('slot_max_length_m').value)
        self.slot_confirm_frames = int(self.get_parameter('slot_confirm_frames').value)

        self.forward_speed = int(self.get_parameter('forward_speed').value)
        self.reverse_speed = int(self.get_parameter('reverse_speed').value)
        self.align_reverse_speed = int(
            self.get_parameter('align_reverse_speed').value
        )
        self.entry_steer = int(self.get_parameter('entry_steer').value)
        self.exit_steer = int(self.get_parameter('exit_steer').value)
        self.approach_pass_x = float(self.get_parameter('approach_pass_x_m').value)
        self.steer_settle_sec = float(self.get_parameter('steer_settle_sec').value)
        self.reverse_arc_sec = float(self.get_parameter('reverse_arc_sec').value)
        self.counter_steer_sec = float(
            self.get_parameter('counter_steer_sec').value
        )
        self.align_timeout_sec = float(self.get_parameter('align_timeout_sec').value)
        self.rear_stop_distance = float(
            self.get_parameter('rear_stop_distance_m').value
        )
        self.side_hard_stop_distance = float(
            self.get_parameter('side_hard_stop_distance_m').value
        )
        self.side_target_distance = float(
            self.get_parameter('side_target_distance_m').value
        )
        self.side_steer_kp = float(self.get_parameter('side_steer_kp').value)
        self.hold_sec = float(self.get_parameter('hold_sec').value)
        self.exit_straight_sec = float(
            self.get_parameter('exit_straight_sec').value
        )
        self.exit_turn_sec = float(self.get_parameter('exit_turn_sec').value)

        self.state = ParkingState.SEARCH
        self.state_started_at = time.monotonic()
        self.latest_points = np.empty((0, 2), dtype=np.float64)
        self.last_scan_at: Optional[float] = None
        self.slot_history: deque[SlotCandidate] = deque(maxlen=self.slot_confirm_frames)
        self.locked_slot: Optional[SlotCandidate] = None

        self.motor_pub = self.create_publisher(Int16MultiArray, self.motor_topic, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)
        self.create_timer(1.0 / self.control_hz, self.control_tick)
        self.get_logger().info(
            'parking_node_osy ready: /scan -> /motor_control, '
            f'usable angle={math.degrees(self.usable_min):.0f}..'
            f'{math.degrees(self.usable_max):.0f} deg, '
            f'slot_side={"right" if self.slot_side_sign < 0 else "left"}'
        )

    def scan_callback(self, msg: LaserScan) -> None:
        self.latest_points = self.scan_to_vehicle_points(msg)
        self.last_scan_at = time.monotonic()

    def scan_to_vehicle_points(self, msg: LaserScan) -> np.ndarray:
        """Convert the specified rear-LiDAR convention to vehicle x-forward/y-left."""
        points = []
        for index, distance in enumerate(msg.ranges):
            if not math.isfinite(distance):
                continue
            if distance < msg.range_min or distance > msg.range_max:
                continue
            angle = msg.angle_min + index * msg.angle_increment
            angle = math.atan2(math.sin(angle), math.cos(angle))
            if not self.usable_min <= angle <= self.usable_max:
                continue  # discard quadrants 1 and 2
            # 0 deg points rear; positive angles rotate clockwise.
            x_forward = -distance * math.cos(angle)
            y_left = distance * math.sin(angle)
            points.append((x_forward, y_left))
        return np.asarray(points, dtype=np.float64) if points else np.empty((0, 2))

    def detect_slot(self) -> Optional[SlotCandidate]:
        points = self.latest_points
        if points.shape[0] < 10:
            return None
        side_distance = points[:, 1] * self.slot_side_sign
        mask = (
            (points[:, 0] >= self.slot_x_min)
            & (points[:, 0] <= self.slot_x_max)
            & (side_distance >= self.slot_side_min)
            & (side_distance <= self.slot_side_max)
        )
        side_points = points[mask]
        if side_points.shape[0] < 10:
            return None

        edges = np.arange(
            self.slot_x_min, self.slot_x_max + self.slot_bin_size, self.slot_bin_size
        )
        indices = np.digitize(side_points[:, 0], edges) - 1
        indices = indices[(indices >= 0) & (indices < len(edges) - 1)]
        occupied = np.bincount(indices, minlength=len(edges) - 1) >= self.slot_min_points
        runs = self.true_runs(occupied)
        min_car_bins = max(1, math.ceil(self.slot_min_vehicle_length / self.slot_bin_size))
        runs = [run for run in runs if run[1] - run[0] + 1 >= min_car_bins]

        candidates = []
        for first, second in zip(runs, runs[1:]):
            start_x = edges[first[1] + 1]
            end_x = edges[second[0]]
            length = end_x - start_x
            if self.slot_min_length <= length <= self.slot_max_length:
                candidates.append(SlotCandidate(
                    start_x, end_x, float(np.median(side_distance[mask]))
                ))
        return min(candidates, key=lambda slot: abs(slot.center_x)) if candidates else None

    @staticmethod
    def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
        runs, start = [], None
        for index, value in enumerate(mask):
            if value and start is None:
                start = index
            elif not value and start is not None:
                runs.append((start, index - 1))
                start = None
        if start is not None:
            runs.append((start, len(mask) - 1))
        return runs

    def stable_slot(self, candidate: Optional[SlotCandidate]) -> Optional[SlotCandidate]:
        if candidate is None:
            self.slot_history.clear()
            return None
        self.slot_history.append(candidate)
        if len(self.slot_history) < self.slot_confirm_frames:
            return None
        centers = np.array([slot.center_x for slot in self.slot_history])
        if float(np.std(centers)) > 0.18:
            return None
        starts = np.array([slot.start_x for slot in self.slot_history])
        ends = np.array([slot.end_x for slot in self.slot_history])
        distances = np.array([slot.side_distance for slot in self.slot_history])
        return SlotCandidate(
            float(np.median(starts)), float(np.median(ends)), float(np.median(distances))
        )

    def rear_clearance(self) -> float:
        """Closest object around the rear 0-degree direction."""
        if self.latest_points.size == 0:
            return math.inf
        rear = self.latest_points[(self.latest_points[:, 0] < 0.0) & (np.abs(self.latest_points[:, 1]) < 0.35)]
        return float(np.min(np.linalg.norm(rear, axis=1))) if rear.size else math.inf

    def side_clearance(self) -> float:
        if self.latest_points.size == 0:
            return math.inf
        signed = self.latest_points[:, 1] * self.slot_side_sign
        side = self.latest_points[(signed > 0.12) & (signed < 1.2)]
        return float(np.min(np.abs(side[:, 1]))) if side.size else math.inf

    def transition(self, state: ParkingState, now: float) -> None:
        if state != self.state:
            self.get_logger().info(f'Parking: {self.state.value} -> {state.value}')
            self.state = state
            self.state_started_at = now

    def control_tick(self) -> None:
        now = time.monotonic()
        if self.last_scan_at is None or now - self.last_scan_at > self.scan_timeout_sec:
            self.transition(ParkingState.EMERGENCY_STOP, now)
            self.publish(0, 0)
            return

        candidate = self.detect_slot()
        stable = self.stable_slot(candidate) if self.state == ParkingState.SEARCH else None
        elapsed = now - self.state_started_at
        rear = self.rear_clearance()
        side = self.side_clearance()

        if self.state == ParkingState.SEARCH:
            if stable is not None:
                self.locked_slot = stable
                self.transition(ParkingState.APPROACH, now)
            self.publish(0, self.forward_speed)
            return

        if self.state == ParkingState.APPROACH:
            # Once the slot center passes the rear axle, stop and set entry steer.
            if candidate is not None:
                self.locked_slot = candidate
            if self.locked_slot is not None and self.locked_slot.center_x <= self.approach_pass_x:
                self.transition(ParkingState.STEER_IN, now)
                self.publish(self.entry_steer, 0)
            else:
                self.publish(0, self.forward_speed)
            return

        if self.state == ParkingState.STEER_IN:
            if elapsed >= self.steer_settle_sec:
                self.transition(ParkingState.REVERSE_ARC, now)
            self.publish(self.entry_steer, 0)
            return

        if self.state == ParkingState.REVERSE_ARC:
            if rear <= self.rear_stop_distance or side <= self.side_hard_stop_distance:
                self.transition(ParkingState.HOLD, now)
                self.publish(0, 0)
            elif elapsed >= self.reverse_arc_sec:
                self.transition(ParkingState.COUNTER_STEER, now)
                self.publish(-self.entry_steer, 0)
            else:
                self.publish(self.entry_steer, self.reverse_speed)
            return

        if self.state == ParkingState.COUNTER_STEER:
            if elapsed >= self.counter_steer_sec:
                self.transition(ParkingState.REVERSE_ALIGN, now)
            self.publish(-self.entry_steer, 0)
            return

        if self.state == ParkingState.REVERSE_ALIGN:
            if rear <= self.rear_stop_distance or side <= self.side_hard_stop_distance:
                self.transition(ParkingState.HOLD, now)
                self.publish(0, 0)
                return
            if elapsed >= self.align_timeout_sec:
                self.transition(ParkingState.HOLD, now)
                self.publish(0, 0)
                return
            # Lateral clearance feedback: steer away from the adjacent parked car.
            error = self.side_target_distance - side
            steer = int(np.clip(-self.slot_side_sign * self.side_steer_kp * error, -20, 20))
            self.publish(steer, self.align_reverse_speed)
            return

        if self.state == ParkingState.HOLD:
            if elapsed >= self.hold_sec:
                self.transition(ParkingState.EXIT_STRAIGHT, now)
            self.publish(0, 0)
            return

        if self.state == ParkingState.EXIT_STRAIGHT:
            if elapsed >= self.exit_straight_sec:
                self.transition(ParkingState.EXIT_TURN, now)
            self.publish(0, self.forward_speed)
            return

        if self.state == ParkingState.EXIT_TURN:
            if elapsed >= self.exit_turn_sec:
                self.transition(ParkingState.DONE, now)
                self.publish(0, 0)
            else:
                self.publish(self.exit_steer, self.forward_speed)
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
