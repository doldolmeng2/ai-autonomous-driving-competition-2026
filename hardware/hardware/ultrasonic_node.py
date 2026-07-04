import math
import time
from glob import glob

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range
from std_msgs.msg import Float32MultiArray

from hardware.serial_port import open_serial


class UltrasonicNode(Node):
    """Read ultrasonic distances from serial and publish Range topics."""

    def __init__(self):
        super().__init__('ultrasonic_node')
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('arduino_boot_delay', 2.0)
        self.declare_parameter('debug_log_raw_line', False)
        self.declare_parameter('sensor_names', ['1', '2', '3', '4', '5', '6'])
        self.declare_parameter('frame_prefix', 'ultrasonic_')
        self.declare_parameter('field_of_view', 0.26)
        self.declare_parameter('min_range', 0.02)
        self.declare_parameter('max_range', 4.0)
        self.declare_parameter('publish_timeout', 0.5)

        self.port = self.get_parameter('port').value
        self.baudrate = int(self.get_parameter('baudrate').value)
        self.arduino_boot_delay = float(
            self.get_parameter('arduino_boot_delay').value
        )
        self.debug_log_raw_line = bool(self.get_parameter('debug_log_raw_line').value)
        self.sensor_names = list(self.get_parameter('sensor_names').value)
        self.frame_prefix = self.get_parameter('frame_prefix').value
        self.field_of_view = float(self.get_parameter('field_of_view').value)
        self.min_range = float(self.get_parameter('min_range').value)
        self.max_range = float(self.get_parameter('max_range').value)
        self.publish_timeout = float(self.get_parameter('publish_timeout').value)

        self.range_publishers = {
            name: self.create_publisher(Range, f'/ultrasonic/range_{name}', 10)
            for name in self.sensor_names
        }
        self.array_publisher = self.create_publisher(
            Float32MultiArray,
            '/ultrasonic/ranges',
            10,
        )
        self.serial = None
        self.active_port = ''
        self.last_open_attempt = 0.0
        self.last_publish = 0.0
        self.last_debug_log = 0.0
        self.timer = self.create_timer(0.01, self.poll)

    def destroy_node(self):
        self.close()
        super().destroy_node()

    def poll(self):
        if self.serial is None and not self.open_serial():
            return

        try:
            line = self.serial.readline().decode(errors='ignore').strip()
        except Exception as exc:
            self.get_logger().warn(
                f'Ultrasonic serial read failed, reopening {self.port}: {exc}',
                throttle_duration_sec=2.0,
            )
            self.close()
            return

        if not line:
            return

        self.log_raw_line(line)
        values = self.parse_line(line)
        if values:
            self.publish(values)

    def open_serial(self):
        now = time.monotonic()
        if now - self.last_open_attempt < 1.0:
            return False
        self.last_open_attempt = now

        candidates = self.serial_candidates()
        if not candidates:
            self.get_logger().warn(
                'Waiting for ultrasonic serial device',
                throttle_duration_sec=5.0,
            )
            return False

        last_error = None
        for port in candidates:
            try:
                self.serial = open_serial(port, self.baudrate, timeout=0.02)
                self.active_port = port
                self.get_logger().info(f'Ultrasonic serial connected on {port}')
                self.wait_for_arduino_boot()
                return True
            except Exception as exc:
                last_error = exc
                self.close()

        self.get_logger().warn(
            f'Waiting for ultrasonic serial device {self.port}: {last_error}',
            throttle_duration_sec=5.0,
        )
        return False

    def serial_candidates(self):
        if self.port != 'auto':
            return [self.port]
        return sorted(glob('/dev/ttyACM*'))

    def wait_for_arduino_boot(self):
        if self.arduino_boot_delay > 0.0:
            time.sleep(self.arduino_boot_delay)

        try:
            if hasattr(self.serial, 'reset_input_buffer'):
                self.serial.reset_input_buffer()
        except Exception as exc:
            self.get_logger().warn(
                f'Ultrasonic serial buffer reset failed: {exc}',
                throttle_duration_sec=5.0,
            )

    def close(self):
        if self.serial is not None:
            try:
                self.serial.close()
            finally:
                self.serial = None
                self.active_port = ''

    def log_raw_line(self, line):
        if not self.debug_log_raw_line:
            return

        now = time.monotonic()
        if now - self.last_debug_log < 1.0:
            return
        self.last_debug_log = now
        self.get_logger().info(f'Ultrasonic raw: {line}')

    def parse_line(self, line):
        try:
            if ':' in line:
                return self.parse_named_line(line)
            return self.parse_csv_line(line)
        except ValueError:
            self.get_logger().warn(
                f'Ignoring malformed ultrasonic line: {line}',
                throttle_duration_sec=2.0,
            )
            return {}

    def parse_named_line(self, line):
        values = {}
        for item in line.replace(';', ',').split(','):
            if ':' not in item:
                continue
            key, value = item.split(':', 1)
            key = key.strip()
            if key in self.sensor_names:
                values[key] = self.normalize_distance(float(value))
        return values

    def parse_csv_line(self, line):
        raw_values = [part.strip() for part in line.split(',') if part.strip()]
        values = {}
        for name, value in zip(self.sensor_names, raw_values):
            values[name] = self.normalize_distance(float(value))
        return values

    def normalize_distance(self, value):
        if value > 20.0:
            value = value / 100.0
        if not math.isfinite(value):
            return math.inf
        return value

    def publish(self, values):
        stamp = self.get_clock().now().to_msg()
        ordered = []
        for name in self.sensor_names:
            distance = values.get(name, math.inf)
            ordered.append(float(distance))
            if name not in values:
                continue

            msg = Range()
            msg.header.stamp = stamp
            msg.header.frame_id = self.frame_prefix + name
            msg.radiation_type = Range.ULTRASOUND
            msg.field_of_view = self.field_of_view
            msg.min_range = self.min_range
            msg.max_range = self.max_range
            msg.range = float(distance)
            self.range_publishers[name].publish(msg)

        array_msg = Float32MultiArray()
        array_msg.data = ordered
        self.array_publisher.publish(array_msg)
        self.last_publish = time.monotonic()


def main(args=None):
    rclpy.init(args=args)
    node = UltrasonicNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
