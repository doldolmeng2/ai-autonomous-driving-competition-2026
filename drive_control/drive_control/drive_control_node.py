import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

from hardware.serial_port import open_serial


class DriveControlNode(Node):
    """Convert manual controller input to safe Arduino motor commands."""

    def __init__(self):
        super().__init__('drive_control_node')

        self.declare_parameter('joy_topic', '/manual_controller/joy')
        self.declare_parameter('serial_port', 'auto')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('command_rate', 20.0)
        self.declare_parameter('command_resend_interval', 0.1)
        self.declare_parameter('joy_timeout', 0.5)
        self.declare_parameter('arduino_boot_delay', 2.0)
        self.declare_parameter('enable_arduino_debug_log', True)
        self.declare_parameter('enable_tx_debug_log', False)
        self.declare_parameter('steer_axis', 3)
        self.declare_parameter('drive_axis', 1)
        self.declare_parameter('invert_steer_axis', False)
        self.declare_parameter('invert_drive_axis', True)
        self.declare_parameter('deadzone', 0.2)
        self.declare_parameter('max_drive_pwm', 30)
        self.declare_parameter('steer_pwm', 40)
        self.declare_parameter('steer_pulse_duration', 1.0)

        self.joy_topic = self.get_parameter('joy_topic').value
        self.serial_port_param = self.get_parameter('serial_port').value
        self.baudrate = int(self.get_parameter('baudrate').value)
        command_rate = float(self.get_parameter('command_rate').value)
        self.command_resend_interval = float(
            self.get_parameter('command_resend_interval').value
        )
        self.joy_timeout = float(self.get_parameter('joy_timeout').value)
        self.arduino_boot_delay = float(
            self.get_parameter('arduino_boot_delay').value
        )
        self.enable_arduino_debug_log = bool(
            self.get_parameter('enable_arduino_debug_log').value
        )
        self.enable_tx_debug_log = bool(self.get_parameter('enable_tx_debug_log').value)
        self.steer_axis = int(self.get_parameter('steer_axis').value)
        self.drive_axis = int(self.get_parameter('drive_axis').value)
        self.invert_steer_axis = bool(self.get_parameter('invert_steer_axis').value)
        self.invert_drive_axis = bool(self.get_parameter('invert_drive_axis').value)
        self.deadzone = float(self.get_parameter('deadzone').value)
        self.max_drive_pwm = self.clamp_pwm(
            int(self.get_parameter('max_drive_pwm').value)
        )
        self.steer_pwm = self.clamp_pwm(int(self.get_parameter('steer_pwm').value))
        self.steer_pulse_duration = float(
            self.get_parameter('steer_pulse_duration').value
        )

        self.serial = None
        self.active_serial_port = ''
        self.last_serial_attempt_time = 0.0
        self.last_error_log_time = 0.0
        self.last_command = None
        self.last_command_sent_time = 0.0
        self.last_joy_time = None
        self.drive_pwm = 0
        self.requested_steer_direction = 0
        self.active_steer_direction = 0
        self.steer_pulse_until = None

        self.create_subscription(Joy, self.joy_topic, self.joy_callback, 10)
        period = 1.0 / command_rate if command_rate > 0.0 else 1.0 / 20.0
        self.timer = self.create_timer(period, self.timer_callback)

        self.get_logger().info(
            f'Subscribing {self.joy_topic}, sending Arduino commands at '
            f'{self.baudrate} baud with drive <= {self.max_drive_pwm}, '
            f'steer pulse {self.steer_pwm} for {self.steer_pulse_duration:.1f}s, '
            f'resend interval {self.command_resend_interval:.2f}s, '
            f'Arduino debug log: {self.enable_arduino_debug_log}, '
            f'TX debug log: {self.enable_tx_debug_log}'
        )

    def joy_callback(self, msg):
        self.last_joy_time = self.get_clock().now()

        steer_value = self.read_axis(msg, self.steer_axis)
        drive_value = self.read_axis(msg, self.drive_axis)
        if self.invert_steer_axis:
            steer_value = -steer_value
        if self.invert_drive_axis:
            drive_value = -drive_value

        self.drive_pwm = self.axis_to_pwm(drive_value, self.max_drive_pwm)
        self.update_steer_request(self.axis_direction(steer_value))

    def timer_callback(self):
        if self.serial is None and not self.open_serial():
            return

        steer_pwm = self.current_steer_pwm()
        drive_pwm = self.drive_pwm
        if self.is_joy_stale():
            steer_pwm = 0
            drive_pwm = 0
            self.requested_steer_direction = 0
            self.active_steer_direction = 0
            self.steer_pulse_until = None

        self.write_command(steer_pwm, drive_pwm)
        self.read_arduino_debug()

    def read_axis(self, msg, index):
        if index < 0 or len(msg.axes) <= index:
            return 0.0
        return max(-1.0, min(1.0, float(msg.axes[index])))

    def axis_direction(self, value):
        if abs(value) < self.deadzone:
            return 0
        return 1 if value > 0.0 else -1

    def axis_to_pwm(self, value, max_pwm):
        if abs(value) < self.deadzone:
            return 0
        pwm = int(round(abs(value) * max_pwm))
        pwm = max(1, min(max_pwm, pwm))
        return pwm if value > 0.0 else -pwm

    def update_steer_request(self, direction):
        now = self.get_clock().now()
        if direction == 0:
            self.requested_steer_direction = 0
            self.active_steer_direction = 0
            self.steer_pulse_until = None
            return

        if direction != self.requested_steer_direction:
            self.requested_steer_direction = direction
            self.active_steer_direction = direction
            self.steer_pulse_until = now.nanoseconds + int(
                self.steer_pulse_duration * 1_000_000_000
            )

    def current_steer_pwm(self):
        if self.active_steer_direction == 0 or self.steer_pulse_until is None:
            return 0

        now_ns = self.get_clock().now().nanoseconds
        if now_ns >= self.steer_pulse_until:
            self.active_steer_direction = 0
            self.steer_pulse_until = None
            return 0

        return self.active_steer_direction * self.steer_pwm

    def is_joy_stale(self):
        if self.last_joy_time is None:
            return True
        age = (self.get_clock().now() - self.last_joy_time).nanoseconds / 1e9
        return age > self.joy_timeout

    def open_serial(self):
        import time

        now = time.monotonic()
        if now - self.last_serial_attempt_time < 1.0:
            return False
        self.last_serial_attempt_time = now

        candidates = self.serial_candidates()
        if not candidates:
            self.log_error_throttled('Waiting for Arduino serial device')
            return False

        last_error = None
        for port in candidates:
            try:
                self.serial = open_serial(
                    port,
                    self.baudrate,
                    timeout=0.02,
                    write_timeout=0.2,
                )
            except Exception as exc:
                last_error = exc
                continue

            self.active_serial_port = port
            self.last_command = None
            self.last_command_sent_time = 0.0
            self.get_logger().info(f'Arduino serial connected on {port}')
            self.wait_for_arduino_boot()
            self.write_command(0, 0)
            return True

        self.log_error_throttled(f'Waiting for Arduino serial device: {last_error}')
        return False

    def serial_candidates(self):
        import glob

        if self.serial_port_param != 'auto':
            return [self.serial_port_param]
        return sorted(glob.glob('/dev/ttyACM*')) + sorted(glob.glob('/dev/ttyUSB*'))

    def write_command(self, steer_pwm, drive_pwm):
        import time

        steer_pwm = self.clamp_signed_pwm(steer_pwm)
        drive_pwm = self.clamp_signed_pwm(drive_pwm)
        command = (steer_pwm, drive_pwm)
        now = time.monotonic()
        if (
            command == self.last_command
            and now - self.last_command_sent_time < self.command_resend_interval
        ):
            return

        line = f'{steer_pwm} {drive_pwm}\n'.encode()
        try:
            self.serial.write(line)
            self.serial.flush()
        except Exception as exc:
            self.log_error_throttled(
                f'Arduino serial write failed on {self.active_serial_port}: {exc}'
            )
            self.close_serial()
            return

        self.last_command = command
        self.last_command_sent_time = now
        if self.enable_tx_debug_log:
            self.get_logger().info(f'TX Arduino: {steer_pwm} {drive_pwm}')

    def read_arduino_debug(self):
        if not self.enable_arduino_debug_log or self.serial is None:
            return

        try:
            waiting = getattr(self.serial, 'in_waiting', None)
            if waiting is None:
                line = self.serial.readline().decode(errors='ignore').strip()
                if line:
                    self.get_logger().info(f'Arduino: {line}')
                return

            while self.serial.in_waiting > 0:
                line = self.serial.readline().decode(errors='ignore').strip()
                if line:
                    self.get_logger().info(f'Arduino: {line}')
        except Exception as exc:
            self.log_error_throttled(f'Arduino serial read failed: {exc}')
            self.close_serial()

    def wait_for_arduino_boot(self):
        import time

        if self.arduino_boot_delay > 0.0:
            time.sleep(self.arduino_boot_delay)

        try:
            if hasattr(self.serial, 'reset_input_buffer'):
                self.serial.reset_input_buffer()
        except Exception as exc:
            self.log_error_throttled(f'Arduino serial buffer reset failed: {exc}')

    def close_serial(self):
        if self.serial is not None:
            try:
                self.serial.close()
            except Exception:
                pass
            self.serial = None
            self.active_serial_port = ''
            self.last_command = None
            self.last_command_sent_time = 0.0

    def destroy_node(self):
        if self.serial is not None:
            self.write_command(0, 0)
        self.close_serial()
        super().destroy_node()

    def clamp_pwm(self, value):
        return max(0, min(255, int(value)))

    def clamp_signed_pwm(self, value):
        return max(-255, min(255, int(value)))

    def log_error_throttled(self, message):
        import time

        now = time.monotonic()
        if now - self.last_error_log_time >= 5.0:
            self.get_logger().error(message)
            self.last_error_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = DriveControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
