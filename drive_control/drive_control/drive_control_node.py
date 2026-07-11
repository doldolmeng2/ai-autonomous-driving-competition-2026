"""drive_control_node.

역할(설계 PDF 기준):
    /motor_control 토픽으로 들어온 (steer, speed) 값을 받아서
    -> 조향 모터 제어값 / 구동 모터 제어값을 계산하고
    -> 아두이노로 시리얼 전송한다.

입력 토픽:
    /motor_control  (std_msgs/msg/Int16MultiArray)
        data = [steer, speed]
        - steer : 목표 조향각(deg). 노드에서 -steer_max_angle_deg~
                  +steer_max_angle_deg 범위로 제한한다.
        - speed : 구동 모터 PWM (부호 = 전/후진, 크기 = 세기).
                  노드에서 -max_drive_pwm~+max_drive_pwm 범위로 제한한다.

    조이스틱(수동)과 자율주행 알고리즘(lane_main / parking 등)이
    "동일한" /motor_control 토픽으로 발행하면, 이 노드는 발행 주체와 무관하게
    똑같이 동작한다.

출력(아두이노 시리얼):
    "steer speed\n"  (공백 구분, 부호 있는 정수 PWM)
    예) "-30 30\n"

조향 처리:
    조향 모터는 위치 센서가 없어서, 조향 모터를 돌린 시간을 누적해 현재 위치를
    각도 단위로 추정한다. steer 값을 목표 각도로 사용하고, 현재 추정
    각도와 목표각의 차이로 이동 시간 n초를 계산한다. n초 동안만 고정
    PWM(steer_pwm)으로 조향한 뒤에는 반드시 0을 전송한다.
"""

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray

from drive_control.serial_port import open_serial


MOTOR_CONTROL_TOPIC = '/motor_control'
SERIAL_PORT = 'auto'
BAUDRATE = 115200
COMMAND_RATE = 20.0
COMMAND_RESEND_INTERVAL = 0.1
INPUT_TIMEOUT = 0.5
ARDUINO_BOOT_DELAY = 4.0
ENABLE_ARDUINO_DEBUG_LOG = True
ENABLE_TX_DEBUG_LOG = False

MAX_DRIVE_PWM = 130
STEER_PWM = 150
STEER_MAX_ANGLE_DEG = 45.0
STEER_CENTER_TIME = 0.45
STEER_ANGLE_TOLERANCE_DEG = 1.0


class DriveControlNode(Node):
    """/motor_control (steer, speed)을 받아 아두이노로 안전하게 전달하는 노드."""

    def __init__(self):
        super().__init__('drive_control_node')

        self.motor_control_topic = MOTOR_CONTROL_TOPIC
        self.serial_port_param = SERIAL_PORT
        self.baudrate = BAUDRATE
        self.command_resend_interval = COMMAND_RESEND_INTERVAL
        self.input_timeout = INPUT_TIMEOUT
        self.arduino_boot_delay = ARDUINO_BOOT_DELAY
        self.enable_arduino_debug_log = ENABLE_ARDUINO_DEBUG_LOG
        self.enable_tx_debug_log = ENABLE_TX_DEBUG_LOG

        self.declare_parameter('max_drive_pwm', MAX_DRIVE_PWM)
        self.declare_parameter('steer_pwm', STEER_PWM)
        self.declare_parameter('steer_max_angle_deg', STEER_MAX_ANGLE_DEG)
        self.declare_parameter('steer_center_time', STEER_CENTER_TIME)
        self.declare_parameter(
            'steer_angle_tolerance_deg',
            STEER_ANGLE_TOLERANCE_DEG,
        )
        self.max_drive_pwm = max(
            0, min(255, int(self.get_parameter('max_drive_pwm').value))
        )
        self.steer_pwm = abs(
            self.limit_steer_pwm(self.get_parameter('steer_pwm').value)
        )
        self.steer_max_angle_deg = max(
            1.0, abs(float(self.get_parameter('steer_max_angle_deg').value))
        )
        self.steer_center_time = max(
            0.01, float(self.get_parameter('steer_center_time').value)
        )
        self.steer_angle_tolerance_deg = max(
            0.0, float(self.get_parameter('steer_angle_tolerance_deg').value)
        )
        self.steer_speed_deg_per_sec = self.steer_max_angle_deg / self.steer_center_time

        # ---- 내부 상태 -----------------------------------------------------
        # 시리얼 연결 상태
        self.serial = None
        self.active_serial_port = ''
        self.last_serial_attempt_time = 0.0
        self.last_error_log_time = 0.0
        self.last_command = None            # 마지막으로 아두이노에 보낸 (steer, speed)
        self.last_command_sent_time = 0.0

        # 최근 입력 상태
        self.last_input_time = None         # 마지막 /motor_control 수신 시각
        self.drive_pwm = 0                  # 목표 구동 PWM (speed에서 계산)

        # 조향 각도 추정 상태.
        # +steer_max_angle_deg=오른쪽 한계, -steer_max_angle_deg=왼쪽 한계, 0=중앙이다.
        self.target_steer_angle_deg = 0.0
        self.steer_angle_deg = 0.0
        self.last_steer_update_time = None
        self.steer_motion_direction = 0
        self.steer_motion_end_time = None
        self.steer_motion_target_angle_deg = 0.0
        self.steer_plan_dirty = False

        # ---- 통신 설정 -----------------------------------------------------
        # 조이스틱/알고리즘이 공통으로 쓰는 /motor_control 구독
        self.create_subscription(
            Int16MultiArray, self.motor_control_topic, self.motor_control_callback, 10
        )
        # 일정 주기로 아두이노에 명령을 밀어 넣는 타이머
        period = 1.0 / COMMAND_RATE if COMMAND_RATE > 0.0 else 1.0 / 20.0
        self.timer = self.create_timer(period, self.timer_callback)

        self.get_logger().info(
            f'Subscribing {self.motor_control_topic} (Int16MultiArray [steer, speed]), '
            f'sending Arduino commands at {self.baudrate} baud, '
            f'drive <= {self.max_drive_pwm}, steer {self.steer_pwm}, '
            f'steer angle +/-{self.steer_max_angle_deg:.1f}deg, '
            f'center_time {self.steer_center_time:.2f}s'
        )

    # ======================================================================
    # 입력 처리
    # ======================================================================
    def motor_control_callback(self, msg):
        """/motor_control 수신: data=[steer, speed]에서 목표 조향/구동 값을 추출."""
        self.last_input_time = self.get_clock().now()

        # data가 짧게 들어올 경우를 대비해 안전하게 인덱싱
        steer = int(msg.data[0]) if len(msg.data) > 0 else 0
        speed = int(msg.data[1]) if len(msg.data) > 1 else 0

        # 속도는 크기만 제한하고, 출력 타이머의 고정 주기는 유지한다.
        self.drive_pwm = self.limit_drive_pwm(speed)

        # 조향은 목표 각도만 갱신한다. 실제 n초 이동 계획은 타이머에서 세운다.
        target = self.clamp_steer_angle(steer)
        if target != self.target_steer_angle_deg:
            self.target_steer_angle_deg = target
            self.steer_plan_dirty = True

    # ======================================================================
    # 주기 처리 (아두이노로 실제 전송)
    # ======================================================================
    def timer_callback(self):
        # 시리얼이 아직 안 열렸으면 열기를 시도하고, 실패하면 이번 주기는 넘어감
        if self.serial is None and not self.open_serial():
            return

        now = self.get_clock().now()
        self.update_steer_position(now)

        # 입력이 끊긴 지 오래면(통신 두절 등) 안전을 위해 정지
        if self.is_input_stale():
            self.stop_steer_motion()
            steer_pwm, drive_pwm = 0, 0
        else:
            if self.steer_plan_dirty:
                self.start_steer_motion(now)
            # 조향 PWM은 이동 시간 동안 고정이고, 종료 시 반드시 0이다.
            steer_pwm = self.steer_motion_direction * self.steer_pwm
            drive_pwm = self.drive_pwm

        self.write_command(steer_pwm, drive_pwm)
        self.read_arduino_debug()

    def update_steer_position(self, now):
        """고정 PWM을 실제로 보낸 시간만큼 현재 조향각을 추정한다."""
        if self.last_steer_update_time is None:
            self.last_steer_update_time = now
            return

        if self.steer_motion_direction == 0 or self.steer_motion_end_time is None:
            self.last_steer_update_time = now
            return

        # 종료 시각 이후의 시간은 조향이 멈춰 있으므로 위치 추정에 포함하지 않는다.
        move_until = min(now, self.steer_motion_end_time)
        dt = (move_until - self.last_steer_update_time).nanoseconds / 1e9
        self.last_steer_update_time = now
        if dt > 0.0:
            self.steer_angle_deg = self.clamp_steer_angle(
                self.steer_angle_deg
                + self.steer_motion_direction * self.steer_speed_deg_per_sec * dt
            )

        if now >= self.steer_motion_end_time:
            # 계산한 n초가 지나면 목표 위치에 도달했다고 보고 PWM을 0으로 만든다.
            self.steer_angle_deg = self.steer_motion_target_angle_deg
            self.steer_motion_direction = 0
            self.steer_motion_end_time = None

    def start_steer_motion(self, now):
        """목표 위치까지 고정 조향 PWM을 보낼 시간 n초를 계산한다."""
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
        """조향 모터를 즉시 멈추고 현재 추정 위치를 새 목표로 둔다."""
        self.steer_motion_direction = 0
        self.steer_motion_end_time = None
        self.target_steer_angle_deg = self.steer_angle_deg
        self.steer_motion_target_angle_deg = self.steer_angle_deg
        self.steer_plan_dirty = False

    def is_input_stale(self):
        """마지막 /motor_control 수신 이후 input_timeout 이상 지났는지."""
        if self.last_input_time is None:
            return True
        age = (self.get_clock().now() - self.last_input_time).nanoseconds / 1e9
        return age > self.input_timeout

    # ======================================================================
    # 시리얼 (아두이노) 통신
    # ======================================================================
    def open_serial(self):
        """아두이노 시리얼 포트를 연다. 성공하면 True."""
        import time

        # 실패 시 매 주기마다 재시도하지 않도록 1초 간격으로 제한
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
                    port, self.baudrate, timeout=0.02, write_timeout=0.2
                )
            except Exception as exc:
                last_error = exc
                continue

            # 연결 성공: 상태 초기화 후 아두이노 부팅을 기다리고 정지 명령을 한번 보냄
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
        """연결을 시도할 시리얼 포트 후보 목록."""
        import glob

        if self.serial_port_param != 'auto':
            return [self.serial_port_param]
        # 아두이노는 보통 /dev/ttyACM* 또는 /dev/ttyUSB* 로 잡힌다
        return sorted(glob.glob('/dev/ttyACM*')) + sorted(glob.glob('/dev/ttyUSB*'))

    def write_command(self, steer_pwm, drive_pwm):
        """아두이노로 'steer speed\\n' 전송. 같은 명령의 과도한 반복 전송은 억제."""
        import time

        steer_pwm = self.limit_steer_pwm(steer_pwm)
        drive_pwm = self.limit_drive_pwm(drive_pwm)
        command = (steer_pwm, drive_pwm)

        # 명령이 그대로이고 재전송 간격이 안 지났으면 생략(시리얼 트래픽 절약)
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
        """아두이노가 보내는 디버그 문자열을 읽어 로그로 출력(옵션)."""
        if not self.enable_arduino_debug_log or self.serial is None:
            return

        try:
            waiting = getattr(self.serial, 'in_waiting', None)
            if waiting is None:
                # in_waiting을 지원하지 않는 구현: 한 줄만 시도
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
        """시리얼 연결 직후 아두이노가 리셋/부팅되는 시간을 기다린다."""
        import time

        if self.arduino_boot_delay > 0.0:
            time.sleep(self.arduino_boot_delay)

        try:
            if hasattr(self.serial, 'reset_input_buffer'):
                self.serial.reset_input_buffer()
        except Exception as exc:
            self.log_error_throttled(f'Arduino serial buffer reset failed: {exc}')

    def close_serial(self):
        """시리얼을 닫고 관련 상태를 초기화."""
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
        # 종료 시 모터를 반드시 정지시킨다
        if self.serial is not None:
            self.write_command(0, 0)
        self.close_serial()
        super().destroy_node()

    # ======================================================================
    # 유틸
    # ======================================================================
    def limit_steer_pwm(self, value):
        """조향 PWM을 아두이노가 허용하는 -255~255 범위로 제한한다."""
        return max(-255, min(255, int(value)))

    def limit_drive_pwm(self, value):
        """구동 PWM을 -max_drive_pwm~+max_drive_pwm 안전 범위로 제한한다."""
        return max(-self.max_drive_pwm, min(self.max_drive_pwm, int(value)))

    def clamp_steer_angle(self, value):
        """추정 조향각을 왼쪽/오른쪽 한계 안으로 제한한다."""
        return max(
            -self.steer_max_angle_deg,
            min(self.steer_max_angle_deg, float(value)),
        )

    def log_error_throttled(self, message):
        """같은 에러 로그를 5초에 한 번만 출력(로그 폭주 방지)."""
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
