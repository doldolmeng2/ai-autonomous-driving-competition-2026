"""Minimal LiDAR parking-node skeleton for YYM.

The node follows the same ROS boundary as ``parking_node_osy``:
``/scan`` in and ``/motor_control`` out.  It deliberately publishes only a
safe stop until the YYM parking state machine is implemented.
"""

from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int16MultiArray


class ParkingNodeYym(Node):
    """Safe starting point for an independent LiDAR parking implementation."""

    def __init__(self) -> None:
        super().__init__('parking_node_yym')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('motor_topic', '/motor_control')
        self.declare_parameter('control_hz', 20.0)
        self.declare_parameter('scan_timeout_sec', 0.5)
        # Kept for compatibility with the existing parking launch file.
        self.declare_parameter('debug_view', False)

        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.motor_topic = str(self.get_parameter('motor_topic').value)
        self.control_hz = max(
            1.0,
            float(self.get_parameter('control_hz').value),
        )
        self.scan_timeout_sec = max(
            0.0,
            float(self.get_parameter('scan_timeout_sec').value),
        )
        self.last_scan_at = None
        self.latest_scan = None

        self.motor_publisher = self.create_publisher(
            Int16MultiArray,
            self.motor_topic,
            10,
        )
        self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            10,
        )
        self.create_timer(1.0 / self.control_hz, self.control_tick)
        self.get_logger().info(
            f'parking_node_yym skeleton: {self.scan_topic} -> '
            f'{self.motor_topic}; output is locked to stop until implemented'
        )

    def scan_callback(self, msg: LaserScan) -> None:
        """Store the newest scan; add YYM LiDAR detection here."""
        self.latest_scan = msg
        self.last_scan_at = time.monotonic()

        # TODO(yym): convert rear-zero scan bearings into parking points.
        # Convention: rear=0, right=+90, front=+/-180, left=-90 degrees.

    def control_tick(self) -> None:
        """Run the future parking FSM; the skeleton always commands a stop."""
        if self.last_scan_at is None:
            self.get_logger().warn(
                'Waiting for /scan; keeping vehicle stopped',
                throttle_duration_sec=2.0,
            )
        elif time.monotonic() - self.last_scan_at > self.scan_timeout_sec:
            self.get_logger().warn(
                'LiDAR scan timeout; keeping vehicle stopped',
                throttle_duration_sec=2.0,
            )

        # TODO(yym): replace with state-machine steering/speed output.
        self.publish_control(0, 0)

    def publish_control(self, steer: int, speed: int) -> None:
        message = Int16MultiArray()
        message.data = [int(steer), int(speed)]
        self.motor_publisher.publish(message)

    def destroy_node(self):
        try:
            self.publish_control(0, 0)
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
