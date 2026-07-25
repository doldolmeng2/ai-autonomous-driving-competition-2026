"""Convert target steering/speed into Arduino-ready motor PWM commands.

This node never opens a serial port. sensor_topic's Arduino communication node
exclusively owns the device and consumes /arduino/motor_command.
"""

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray


MOTOR_CONTROL_TOPIC = '/motor_control'
ARDUINO_COMMAND_TOPIC = '/arduino/motor_command'
COMMAND_RATE = 20.0
INPUT_TIMEOUT = 0.5

MAX_DRIVE_PWM = 140
STEER_PWM = 150
STEER_MAX_ANGLE_DEG = 45.0
STEER_CENTER_TIME = 0.45
STEER_ANGLE_TOLERANCE_DEG = 1.0


class DriveControlNode(Node):
    """Translate /motor_control targets into steering and drive PWM."""

    def __init__(self):
        super().__init__('drive_control_node')
        self.declare_parameter('motor_control_topic', MOTOR_CONTROL_TOPIC)
        self.declare_parameter('arduino_command_topic', ARDUINO_COMMAND_TOPIC)
        self.declare_parameter('command_rate_hz', COMMAND_RATE)
        self.declare_parameter('input_timeout_sec', INPUT_TIMEOUT)
        self.declare_parameter('max_drive_pwm', MAX_DRIVE_PWM)
        self.declare_parameter('steer_pwm', STEER_PWM)
        self.declare_parameter('steer_max_angle_deg', STEER_MAX_ANGLE_DEG)
        self.declare_parameter('steer_center_time', STEER_CENTER_TIME)
        self.declare_parameter(
            'steer_angle_tolerance_deg',
            STEER_ANGLE_TOLERANCE_DEG,
        )

        self.motor_control_topic = str(
            self.get_parameter('motor_control_topic').value
        )
        self.arduino_command_topic = str(
            self.get_parameter('arduino_command_topic').value
        )
        self.command_rate_hz = max(
            1.0, float(self.get_parameter('command_rate_hz').value)
        )
        self.input_timeout = max(
            0.05, float(self.get_parameter('input_timeout_sec').value)
        )
        self.max_drive_pwm = max(
            0, min(255, int(self.get_parameter('max_drive_pwm').value))
        )
        self.steer_pwm = max(
            0, min(255, abs(int(self.get_parameter('steer_pwm').value)))
        )
        self.steer_max_angle_deg = max(
            1.0, abs(float(self.get_parameter('steer_max_angle_deg').value))
        )
        self.steer_center_time = max(
            0.01, float(self.get_parameter('steer_center_time').value)
        )
        self.steer_angle_tolerance_deg = max(
            0.0,
            float(self.get_parameter('steer_angle_tolerance_deg').value),
        )
        self.steer_speed_deg_per_sec = (
            self.steer_max_angle_deg / self.steer_center_time
        )

        self.last_input_time = None
        self.drive_pwm = 0
        self.target_steer_angle_deg = 0.0
        self.steer_angle_deg = 0.0
        self.last_steer_update_time = None
        self.steer_motion_direction = 0
        self.steer_motion_end_time = None
        self.steer_motion_target_angle_deg = 0.0
        self.steer_plan_dirty = False

        self.command_publisher = self.create_publisher(
            Int16MultiArray,
            self.arduino_command_topic,
            10,
        )
        self.create_subscription(
            Int16MultiArray,
            self.motor_control_topic,
            self.motor_control_callback,
            10,
        )
        self.create_timer(1.0 / self.command_rate_hz, self.timer_callback)
        self.get_logger().info(
            f'{self.motor_control_topic} target -> '
            f'{self.arduino_command_topic} PWM; no direct serial access'
        )

    def motor_control_callback(self, msg):
        self.last_input_time = self.get_clock().now()
        steer = int(msg.data[0]) if len(msg.data) > 0 else 0
        speed = int(msg.data[1]) if len(msg.data) > 1 else 0
        self.drive_pwm = self.limit_drive_pwm(speed)

        target = self.clamp_steer_angle(steer)
        if target != self.target_steer_angle_deg:
            self.target_steer_angle_deg = target
            self.steer_plan_dirty = True

    def timer_callback(self):
        now = self.get_clock().now()
        self.update_steer_position(now)

        if self.is_input_stale(now):
            self.stop_steer_motion()
            steer_pwm, drive_pwm = 0, 0
        else:
            if self.steer_plan_dirty:
                self.start_steer_motion(now)
            steer_pwm = self.steer_motion_direction * self.steer_pwm
            drive_pwm = self.drive_pwm
        self.publish_command(steer_pwm, drive_pwm)

    def update_steer_position(self, now):
        if self.last_steer_update_time is None:
            self.last_steer_update_time = now
            return
        if self.steer_motion_direction == 0 or self.steer_motion_end_time is None:
            self.last_steer_update_time = now
            return

        move_until = min(now, self.steer_motion_end_time)
        dt = (move_until - self.last_steer_update_time).nanoseconds / 1e9
        self.last_steer_update_time = now
        if dt > 0.0:
            self.steer_angle_deg = self.clamp_steer_angle(
                self.steer_angle_deg
                + self.steer_motion_direction
                * self.steer_speed_deg_per_sec
                * dt
            )
        if now >= self.steer_motion_end_time:
            self.steer_angle_deg = self.steer_motion_target_angle_deg
            self.steer_motion_direction = 0
            self.steer_motion_end_time = None

    def start_steer_motion(self, now):
        error = self.target_steer_angle_deg - self.steer_angle_deg
        self.steer_plan_dirty = False
        if abs(error) <= self.steer_angle_tolerance_deg:
            self.steer_angle_deg = self.target_steer_angle_deg
            self.stop_steer_motion()
            return

        duration = abs(error) / self.steer_speed_deg_per_sec
        self.steer_motion_direction = 1 if error > 0.0 else -1
        self.steer_motion_target_angle_deg = self.target_steer_angle_deg
        self.steer_motion_end_time = now + Duration(seconds=duration)
        self.last_steer_update_time = now

    def stop_steer_motion(self):
        self.steer_motion_direction = 0
        self.steer_motion_end_time = None
        self.target_steer_angle_deg = self.steer_angle_deg
        self.steer_motion_target_angle_deg = self.steer_angle_deg
        self.steer_plan_dirty = False

    def is_input_stale(self, now=None):
        if self.last_input_time is None:
            return True
        if now is None:
            now = self.get_clock().now()
        age = (now - self.last_input_time).nanoseconds / 1e9
        return age > self.input_timeout

    def publish_command(self, steer_pwm, drive_pwm):
        message = Int16MultiArray()
        message.data = [
            max(-255, min(255, int(steer_pwm))),
            self.limit_drive_pwm(drive_pwm),
        ]
        self.command_publisher.publish(message)

    def limit_drive_pwm(self, value):
        return max(-self.max_drive_pwm, min(self.max_drive_pwm, int(value)))

    def clamp_steer_angle(self, value):
        return max(
            -self.steer_max_angle_deg,
            min(self.steer_max_angle_deg, float(value)),
        )

    def destroy_node(self):
        try:
            self.publish_command(0, 0)
        except Exception:
            pass
        return super().destroy_node()


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
