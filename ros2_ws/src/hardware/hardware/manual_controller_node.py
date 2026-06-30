import os
import struct
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy


JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80
JS_EVENT_FORMAT = 'IhBB'
JS_EVENT_SIZE = struct.calcsize(JS_EVENT_FORMAT)
AXIS_MAX = 32767.0


class ManualControllerNode(Node):
    """Read a Linux joystick device and publish its state as sensor_msgs/Joy."""

    def __init__(self):
        super().__init__('manual_controller_node')

        self.declare_parameter('device_path', '/dev/input/js0')
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('deadzone', 0.05)
        self.declare_parameter('topic_name', '/manual_controller/joy')

        self.device_path = (
            self.get_parameter('device_path').get_parameter_value().string_value
        )
        publish_rate = (
            self.get_parameter('publish_rate').get_parameter_value().double_value
        )
        self.deadzone = self.get_parameter('deadzone').get_parameter_value().double_value
        topic_name = self.get_parameter('topic_name').get_parameter_value().string_value

        self.publisher = self.create_publisher(Joy, topic_name, 10)
        self.device_fd = None
        self.axes = []
        self.buttons = []
        self.last_error_log_time = 0.0
        self.last_reconnect_attempt_time = 0.0

        timer_period = 1.0 / publish_rate if publish_rate > 0.0 else 1.0 / 30.0
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(
            f'Publishing joystick input from {self.device_path} to {topic_name}'
        )

    def destroy_node(self):
        self.close_device()
        super().destroy_node()

    def timer_callback(self):
        if self.device_fd is None:
            self.try_open_device()
            return

        try:
            self.read_pending_events()
        except OSError as exc:
            self.log_error_throttled(f'Joystick disconnected or unreadable: {exc}')
            self.close_device()
            return

        self.publish_joy()

    def try_open_device(self):
        now = time.monotonic()
        if now - self.last_reconnect_attempt_time < 1.0:
            return

        self.last_reconnect_attempt_time = now
        try:
            self.device_fd = os.open(self.device_path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as exc:
            self.log_error_throttled(
                f'Waiting for joystick device {self.device_path}: {exc}'
            )
            return

        self.axes = []
        self.buttons = []
        self.get_logger().info(f'Joystick device connected: {self.device_path}')

    def close_device(self):
        if self.device_fd is not None:
            try:
                os.close(self.device_fd)
            except OSError:
                pass
            self.device_fd = None

    def read_pending_events(self):
        while True:
            try:
                data = os.read(self.device_fd, JS_EVENT_SIZE)
            except BlockingIOError:
                break

            if not data:
                raise OSError('device returned no data')
            if len(data) != JS_EVENT_SIZE:
                continue

            _, value, event_type, number = struct.unpack(JS_EVENT_FORMAT, data)
            event_type = event_type & ~JS_EVENT_INIT

            if event_type == JS_EVENT_AXIS:
                self.ensure_axis(number)
                axis_value = max(-1.0, min(1.0, value / AXIS_MAX))
                if abs(axis_value) < self.deadzone:
                    axis_value = 0.0
                self.axes[number] = axis_value
            elif event_type == JS_EVENT_BUTTON:
                self.ensure_button(number)
                self.buttons[number] = 1 if value else 0

    def publish_joy(self):
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'manual_controller'
        msg.axes = list(self.axes)
        msg.buttons = list(self.buttons)
        self.publisher.publish(msg)

    def ensure_axis(self, index):
        while len(self.axes) <= index:
            self.axes.append(0.0)

    def ensure_button(self, index):
        while len(self.buttons) <= index:
            self.buttons.append(0)

    def log_error_throttled(self, message):
        now = time.monotonic()
        if now - self.last_error_log_time >= 5.0:
            self.get_logger().error(message)
            self.last_error_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = ManualControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
