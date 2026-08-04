"""Convert target steering/speed into Arduino-ready motor PWM commands.

This node never opens a serial port. sensor_topic's Arduino communication node
exclusively owns the device and consumes /arduino/motor_command.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16, Int16MultiArray

MOTOR_CONTROL_TOPIC = '/motor_control'
ARDUINO_COMMAND_TOPIC = '/arduino/motor_command'
STEERING_RAW_TOPIC = '/arduino/steering_raw'
COMMAND_RATE = 20.0
INPUT_TIMEOUT = 0.5
STEERING_FEEDBACK_TIMEOUT = 0.5
MAX_DRIVE_PWM = 230
# [소프트 스타트] 구동 출력이 '초당' 최대 몇 PWM까지 올라갈 수 있는지(가속 속도).
# 정지 상태에서 큰 값이 한 번에 나가 트랙이 밀리는 것을 막기 위한 값.
# 크면 빨리 붙고(램프 약함), 작으면 천천히 붙는다(밀림 방지 강함).
# 0→최대(230)까지 걸리는 시간 ≈ MAX_DRIVE_PWM / DRIVE_ACCEL_PWM_PER_SEC 초.
DRIVE_ACCEL_PWM_PER_SEC = 300.0
# [소프트 스타트] 감속/정지 시 '초당' 최대 몇 PWM까지 내릴 수 있는지(감속 속도).
# 스로틀을 놓아 목표가 0이 되거나 목표가 작아질 때 급정지하지 않고 서서히 세운다.
# 방향 전환(전진↔후진)도 이 속도로 0까지 내려간 뒤 반대로 가속한다.
# 최대(230)→0까지 걸리는 시간 ≈ MAX_DRIVE_PWM / DRIVE_DECEL_PWM_PER_SEC 초.
DRIVE_DECEL_PWM_PER_SEC = 400.0
STEER_PWM = 150
STEER_MAX_ANGLE_DEG = 45.0
STEER_ANGLE_TOLERANCE_DEG = 1.0
STEER_RAW_LEFT = 530
STEER_RAW_CENTER = 455
STEER_RAW_RIGHT = 380
# Time-trial lane driving PID.
TIMED_STEER_PID_KP = 10.0
TIMED_STEER_PID_KI = 0.0
TIMED_STEER_PID_KD = 0.5
# Mission lane driving PID. Tune these independently from the timed profile.
MISSION_STEER_PID_KP = 20.0
MISSION_STEER_PID_KI = 0.0
MISSION_STEER_PID_KD = 0.5
STEER_PID_INTEGRAL_LIMIT_PWM = 30.0
STEER_PID_DERIVATIVE_FILTER_ALPHA = 0.25
STEER_MIN_PWM = 40
# The parking controller publishes an already-calculated steering target.  Its
# low-gain proportional loop avoids amplifying a small target update into a
# large steering-motor correction while retaining raw-angle feedback.
PARKING_STEER_PID_KP = 4.0
PARKING_STEER_PID_KI = 0.0
PARKING_STEER_PID_KD = 0.0
STEER_PID_PROFILE = 'timed'
STEER_PID_PROFILES = {
    'timed': (
        TIMED_STEER_PID_KP,
        TIMED_STEER_PID_KI,
        TIMED_STEER_PID_KD,
    ),
    'mission': (
        MISSION_STEER_PID_KP,
        MISSION_STEER_PID_KI,
        MISSION_STEER_PID_KD,
    ),
    'parking': (
        PARKING_STEER_PID_KP,
        PARKING_STEER_PID_KI,
        PARKING_STEER_PID_KD,
    ),
}


def steer_pid_gains(profile):
    """Return the configured gains for a named driving profile."""
    normalized = str(profile).strip().lower()
    try:
        return normalized, STEER_PID_PROFILES[normalized]
    except KeyError as error:
        raise ValueError(
            f'Unknown steer_pid_profile {profile!r}; expected one of '
            f'{sorted(STEER_PID_PROFILES)}'
        ) from error


class DriveControlNode(Node):
    """Translate /motor_control targets into steering and drive PWM."""

    def __init__(self):
        super().__init__('drive_control_node')
        self.declare_parameter('motor_control_topic', MOTOR_CONTROL_TOPIC)
        self.declare_parameter('arduino_command_topic', ARDUINO_COMMAND_TOPIC)
        self.declare_parameter('steering_raw_topic', STEERING_RAW_TOPIC)
        self.declare_parameter('command_rate_hz', COMMAND_RATE)
        self.declare_parameter('input_timeout_sec', INPUT_TIMEOUT)
        self.declare_parameter(
            'steering_feedback_timeout_sec',
            STEERING_FEEDBACK_TIMEOUT,
        )
        self.declare_parameter('max_drive_pwm', MAX_DRIVE_PWM)
        # [소프트 스타트] 가속/감속 속도 파라미터 선언.
        self.declare_parameter('drive_accel_pwm_per_sec', DRIVE_ACCEL_PWM_PER_SEC)
        self.declare_parameter('drive_decel_pwm_per_sec', DRIVE_DECEL_PWM_PER_SEC)
        self.declare_parameter('steer_pwm', STEER_PWM)
        self.declare_parameter('steer_max_angle_deg', STEER_MAX_ANGLE_DEG)
        self.declare_parameter(
            'steer_angle_tolerance_deg',
            STEER_ANGLE_TOLERANCE_DEG,
        )
        self.declare_parameter('steer_raw_left', STEER_RAW_LEFT)
        self.declare_parameter('steer_raw_center', STEER_RAW_CENTER)
        self.declare_parameter('steer_raw_right', STEER_RAW_RIGHT)
        self.declare_parameter('steer_pid_profile', STEER_PID_PROFILE)
        self.steer_pid_profile, profile_gains = steer_pid_gains(
            self.get_parameter('steer_pid_profile').value
        )
        self.declare_parameter('steer_pid_kp', profile_gains[0])
        self.declare_parameter('steer_pid_ki', profile_gains[1])
        self.declare_parameter('steer_pid_kd', profile_gains[2])
        self.declare_parameter(
            'steer_pid_integral_limit_pwm',
            STEER_PID_INTEGRAL_LIMIT_PWM,
        )
        self.declare_parameter(
            'steer_pid_derivative_filter_alpha',
            STEER_PID_DERIVATIVE_FILTER_ALPHA,
        )
        self.declare_parameter('steer_min_pwm', STEER_MIN_PWM)
        self.motor_control_topic = str(
            self.get_parameter('motor_control_topic').value
        )
        self.arduino_command_topic = str(
            self.get_parameter('arduino_command_topic').value
        )
        self.steering_raw_topic = str(
            self.get_parameter('steering_raw_topic').value
        )
        self.command_rate_hz = max(
            1.0, float(self.get_parameter('command_rate_hz').value)
        )
        self.input_timeout = max(
            0.05, float(self.get_parameter('input_timeout_sec').value)
        )
        self.steering_feedback_timeout = max(
            0.05,
            float(
                self.get_parameter(
                    'steering_feedback_timeout_sec'
                ).value
            ),
        )
        self.max_drive_pwm = max(
            0, min(255, int(self.get_parameter('max_drive_pwm').value))
        )
        # [소프트 스타트] 가속/감속 속도(PWM/초) 읽기. 음수가 들어와도 0으로 막아 안전 처리.
        self.drive_accel_pwm_per_sec = max(
            0.0, float(self.get_parameter('drive_accel_pwm_per_sec').value)
        )
        self.drive_decel_pwm_per_sec = max(
            0.0, float(self.get_parameter('drive_decel_pwm_per_sec').value)
        )
        self.steer_pwm = max(
            0, min(255, abs(int(self.get_parameter('steer_pwm').value)))
        )
        self.steer_max_angle_deg = max(
            1.0, abs(float(self.get_parameter('steer_max_angle_deg').value))
        )
        self.steer_angle_tolerance_deg = max(
            0.0,
            float(self.get_parameter('steer_angle_tolerance_deg').value),
        )
        self.steer_raw_left = int(
            self.get_parameter('steer_raw_left').value
        )
        self.steer_raw_center = int(
            self.get_parameter('steer_raw_center').value
        )
        self.steer_raw_right = int(
            self.get_parameter('steer_raw_right').value
        )
        self.steer_pid_kp = max(
            0.0, float(self.get_parameter('steer_pid_kp').value)
        )
        self.steer_pid_ki = max(
            0.0, float(self.get_parameter('steer_pid_ki').value)
        )
        self.steer_pid_kd = max(
            0.0, float(self.get_parameter('steer_pid_kd').value)
        )
        self.steer_pid_integral_limit_pwm = max(
            0.0,
            float(
                self.get_parameter(
                    'steer_pid_integral_limit_pwm'
                ).value
            ),
        )
        self.steer_pid_derivative_filter_alpha = max(
            0.0,
            min(
                1.0,
                float(
                    self.get_parameter(
                        'steer_pid_derivative_filter_alpha'
                    ).value
                ),
            ),
        )
        self.steer_min_pwm = max(
            0,
            min(
                self.steer_pwm,
                abs(int(self.get_parameter('steer_min_pwm').value)),
            ),
        )
        if not (
            self.steer_raw_left
            > self.steer_raw_center
            > self.steer_raw_right
        ):
            self.get_logger().warn(
                'Invalid steering raw calibration; using 530/455/380'
            )
            self.steer_raw_left = STEER_RAW_LEFT
            self.steer_raw_center = STEER_RAW_CENTER
            self.steer_raw_right = STEER_RAW_RIGHT
        self.last_input_time = None
        self.last_steering_feedback_time = None
        self.drive_pwm = 0
        # [소프트 스타트] 지금 실제로 내보내는 구동 값. 목표(drive_pwm)로 곧장 점프하지
        # 않고 매 주기마다 이 값을 조금씩 목표 쪽으로 움직여 서서히 가·감속한다.
        self.current_drive_pwm = 0
        self.target_steer_angle_deg = 0.0
        self.steer_angle_deg = 0.0
        self.steer_raw_value = None
        self.steer_angle_velocity_deg_per_sec = 0.0
        self.pid_integral_error = 0.0
        self.last_pid_update_time = None
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
        self.create_subscription(
            Int16,
            self.steering_raw_topic,
            self.steering_feedback_callback,
            10,
        )
        self.create_timer(1.0 / self.command_rate_hz, self.timer_callback)
        self.get_logger().info(
            f'{self.motor_control_topic} target -> '
            f'{self.arduino_command_topic} PWM using '
            f'{self.steering_raw_topic} closed-loop feedback; '
            f'raw left/center/right='
            f'{self.steer_raw_left}/{self.steer_raw_center}/'
            f'{self.steer_raw_right}; PID profile={self.steer_pid_profile} gains='
            f'{self.steer_pid_kp:.2f}/{self.steer_pid_ki:.2f}/'
            f'{self.steer_pid_kd:.2f}; '
            f'drive ramp accel/decel='
            f'{self.drive_accel_pwm_per_sec:.0f}/'
            f'{self.drive_decel_pwm_per_sec:.0f} pwm/s'
        )

    def motor_control_callback(self, msg):
        self.last_input_time = self.get_clock().now()
        steer = int(msg.data[0]) if len(msg.data) > 0 else 0
        speed = int(msg.data[1]) if len(msg.data) > 1 else 0
        # 목표 구동값만 갱신한다. 실제 출력 가·감속 램프는 timer_callback에서 처리.
        self.drive_pwm = self.limit_drive_pwm(speed)
        previous_error = (
            self.target_steer_angle_deg - self.steer_angle_deg
        )
        self.target_steer_angle_deg = self.clamp_steer_angle(steer)
        next_error = self.target_steer_angle_deg - self.steer_angle_deg
        if previous_error * next_error < 0.0:
            self.pid_integral_error = 0.0

    def steering_feedback_callback(self, msg):
        raw_value = int(msg.data)
        if not 0 <= raw_value <= 1023:
            return
        now = self.get_clock().now()
        next_angle = self.raw_to_steer_angle(raw_value)
        if self.last_steering_feedback_time is not None:
            feedback_dt = (
                now - self.last_steering_feedback_time
            ).nanoseconds / 1e9
            if (
                feedback_dt > 0.0
                and feedback_dt
                <= self.steering_feedback_timeout * 2.0
            ):
                raw_velocity = (
                    next_angle - self.steer_angle_deg
                ) / feedback_dt
                alpha = self.steer_pid_derivative_filter_alpha
                self.steer_angle_velocity_deg_per_sec = (
                    alpha * raw_velocity
                    + (1.0 - alpha)
                    * self.steer_angle_velocity_deg_per_sec
                )
            else:
                self.steer_angle_velocity_deg_per_sec = 0.0
        else:
            self.steer_angle_velocity_deg_per_sec = 0.0
        self.steer_raw_value = raw_value
        self.steer_angle_deg = next_angle
        self.last_steering_feedback_time = now

    def timer_callback(self):
        now = self.get_clock().now()
        if self.is_input_stale(now):
            self.reset_pid_controller()
            # [소프트 스타트] 컨트롤러 신호 끊김 = 비상 상황이므로 감속 없이 즉시 정지.
            self.current_drive_pwm = 0
            steer_pwm, drive_pwm = 0, 0
        elif self.is_steering_feedback_stale(now):
            self.reset_pid_controller()
            self.get_logger().warn(
                'Steering A1 feedback missing/stale; stopping vehicle',
                throttle_duration_sec=1.0,
            )
            # [소프트 스타트] 조향 피드백 끊김도 비상 상황이므로 즉시 정지.
            self.current_drive_pwm = 0
            steer_pwm, drive_pwm = 0, 0
        else:
            steer_pwm = self.calculate_steer_pwm(self.pid_dt(now))
            # [소프트 스타트] 목표 구동값으로 바로 보내지 않고 가·감속 램프를 거친 값을 전송.
            drive_pwm = self.ramp_drive_output(self.drive_pwm)
        self.publish_command(steer_pwm, drive_pwm)

    def ramp_drive_output(self, target):
        # [소프트 스타트] 구동 출력을 목표값으로 곧장 보내지 않고, 매 틱 제한된 양만큼만
        # 목표 쪽으로 움직인다. 값이 커지는 '가속'은 accel_step, 값이 작아지는 '감속·정지'는
        # decel_step 으로 각각 속도를 제한해 급출발과 급정지를 모두 부드럽게 만든다.
        # (컨트롤러 신호가 끊긴 '비상 정지'는 timer_callback에서 즉시 0으로 처리한다.)
        target = self.limit_drive_pwm(target)
        current = self.current_drive_pwm
        accel_step = max(
            1, int(round(self.drive_accel_pwm_per_sec / self.command_rate_hz))
        )
        decel_step = max(
            1, int(round(self.drive_decel_pwm_per_sec / self.command_rate_hz))
        )

        if current == 0:
            # 정지 상태에서 출발 → 목표 방향으로 accel_step 만큼 가속.
            if target > 0:
                current = min(target, accel_step)
            elif target < 0:
                current = max(target, -accel_step)
        elif (current > 0) == (target > 0) and target != 0:
            # 같은 방향(전진끼리 또는 후진끼리).
            if abs(target) >= abs(current):
                # 목표가 더 세다 → 가속(accel_step).
                if current > 0:
                    current = min(target, current + accel_step)
                else:
                    current = max(target, current - accel_step)
            else:
                # 목표가 더 약하다 → 감속(decel_step).
                if current > 0:
                    current = max(target, current - decel_step)
                else:
                    current = min(target, current + decel_step)
        else:
            # 목표가 0(정지)이거나 반대 방향(역전) → decel_step 으로 0까지 감속.
            # 0에 도달하면 다음 틱부터 위의 '정지에서 출발' 분기로 반대 방향 가속.
            if current > 0:
                current = max(0, current - decel_step)
            else:
                current = min(0, current + decel_step)

        self.current_drive_pwm = current
        return current

    def pid_dt(self, now):
        nominal_dt = 1.0 / self.command_rate_hz
        if self.last_pid_update_time is None:
            dt = nominal_dt
        else:
            dt = (now - self.last_pid_update_time).nanoseconds / 1e9
            dt = max(0.001, min(dt, nominal_dt * 2.0))
        self.last_pid_update_time = now
        return dt

    def calculate_steer_pwm(self, dt=None):
        if dt is None:
            dt = 1.0 / self.command_rate_hz
        dt = max(0.001, float(dt))
        error = self.target_steer_angle_deg - self.steer_angle_deg
        if abs(error) <= self.steer_angle_tolerance_deg:
            self.pid_integral_error = 0.0
            return 0
        candidate_integral = self.pid_integral_error + error * dt
        if self.steer_pid_ki > 0.0:
            integral_limit = (
                self.steer_pid_integral_limit_pwm / self.steer_pid_ki
            )
            candidate_integral = max(
                -integral_limit,
                min(integral_limit, candidate_integral),
            )
        else:
            candidate_integral = 0.0
        proportional = self.steer_pid_kp * error
        derivative = (
            -self.steer_pid_kd
            * self.steer_angle_velocity_deg_per_sec
        )
        candidate_output = (
            proportional
            + self.steer_pid_ki * candidate_integral
            + derivative
        )
        # 출력 포화 방향으로 적분이 더 쌓이는 경우에는 이번 적분을 버린다.
        if (
            abs(candidate_output) > self.steer_pwm
            and candidate_output * error > 0.0
        ):
            output = (
                proportional
                + self.steer_pid_ki * self.pid_integral_error
                + derivative
            )
        else:
            self.pid_integral_error = candidate_integral
            output = candidate_output
        output = max(-self.steer_pwm, min(self.steer_pwm, output))
        if 0.0 < abs(output) < self.steer_min_pwm:
            output = self.steer_min_pwm if output > 0.0 else -self.steer_min_pwm
        return int(round(output))

    def reset_pid_controller(self):
        self.pid_integral_error = 0.0
        self.last_pid_update_time = None

    def raw_to_steer_angle(self, raw_value):
        raw_value = max(
            self.steer_raw_right,
            min(self.steer_raw_left, int(raw_value)),
        )
        if raw_value >= self.steer_raw_center:
            raw_span = self.steer_raw_left - self.steer_raw_center
            angle = -(
                (raw_value - self.steer_raw_center)
                * self.steer_max_angle_deg
                / raw_span
            )
        else:
            raw_span = self.steer_raw_center - self.steer_raw_right
            angle = (
                (self.steer_raw_center - raw_value)
                * self.steer_max_angle_deg
                / raw_span
            )
        return self.clamp_steer_angle(angle)

    def is_input_stale(self, now=None):
        if self.last_input_time is None:
            return True
        if now is None:
            now = self.get_clock().now()
        age = (now - self.last_input_time).nanoseconds / 1e9
        return age > self.input_timeout

    def is_steering_feedback_stale(self, now=None):
        if self.last_steering_feedback_time is None:
            return True
        if now is None:
            now = self.get_clock().now()
        age = (
            now - self.last_steering_feedback_time
        ).nanoseconds / 1e9
        return age > self.steering_feedback_timeout

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
