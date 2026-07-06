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
    각도가 목표에 도달할 때까지 고정 PWM(steer_pwm)으로 조향한다.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray

from drive_control.serial_port import open_serial


class DriveControlNode(Node):
    """/motor_control (steer, speed)을 받아 아두이노로 안전하게 전달하는 노드."""

    def __init__(self):
        super().__init__('drive_control_node')

        # ---- 파라미터 선언 -------------------------------------------------
        # 구독 토픽 / 시리얼 연결 관련
        self.declare_parameter('motor_control_topic', '/motor_control')
        self.declare_parameter('serial_port', 'auto')       # 'auto' 또는 '/dev/ttyACM0' 등
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('command_rate', 20.0)        # 아두이노로 명령 보내는 주기(Hz)
        self.declare_parameter('command_resend_interval', 0.1)  # 같은 명령 재전송 간격(s)
        self.declare_parameter('input_timeout', 0.5)        # 이 시간 동안 입력 없으면 정지(s)
        self.declare_parameter('arduino_boot_delay', 4.0)   # 시리얼 연결 후 아두이노 부팅 대기(s)
        self.declare_parameter('enable_arduino_debug_log', True)  # 아두이노 -> PC 로그 출력
        self.declare_parameter('enable_tx_debug_log', False)      # PC -> 아두이노 송신 로그 출력

        # 구동 / 조향 제어값 관련
        self.declare_parameter('max_drive_pwm', 130)        # 구동 모터 안전 상한 PWM
        self.declare_parameter('steer_pwm', 150)             # 조향할 때 넣는 PWM 크기
        self.declare_parameter('steer_max_angle_deg', 45.0)  # 최대 좌/우 조향각(deg)
        self.declare_parameter('steer_center_time', 0.45)    # 최대 꺾임에서 중앙까지 걸리는 시간(s)
        self.declare_parameter('steer_angle_tolerance_deg', 1.0)  # 목표 도달 판정 여유(deg)

        # ---- 파라미터 읽기 -------------------------------------------------
        self.motor_control_topic = self.get_parameter('motor_control_topic').value
        self.serial_port_param = self.get_parameter('serial_port').value
        self.baudrate = int(self.get_parameter('baudrate').value)
        command_rate = float(self.get_parameter('command_rate').value)
        self.command_resend_interval = float(
            self.get_parameter('command_resend_interval').value
        )
        self.input_timeout = float(self.get_parameter('input_timeout').value)
        self.arduino_boot_delay = float(self.get_parameter('arduino_boot_delay').value)
        self.enable_arduino_debug_log = bool(
            self.get_parameter('enable_arduino_debug_log').value
        )
        self.enable_tx_debug_log = bool(self.get_parameter('enable_tx_debug_log').value)
        self.max_drive_pwm = self.clamp_pwm(int(self.get_parameter('max_drive_pwm').value))
        self.steer_pwm = self.clamp_pwm(int(self.get_parameter('steer_pwm').value))
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
        self.active_steer_direction = 0     # 직전 주기에 실제로 출력한 방향(+1/-1/0)
        self.steer_angle_deg = 0.0
        self.last_steer_update_time = None

        # ---- 통신 설정 -----------------------------------------------------
        # 조이스틱/알고리즘이 공통으로 쓰는 /motor_control 구독
        self.create_subscription(
            Int16MultiArray, self.motor_control_topic, self.motor_control_callback, 10
        )
        # 일정 주기로 아두이노에 명령을 밀어 넣는 타이머
        period = 1.0 / command_rate if command_rate > 0.0 else 1.0 / 20.0
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

        # 구동: speed를 안전 상한으로 제한한다.
        self.drive_pwm = self.clamp_drive_pwm(speed)

        # 조향: steer를 목표 각도(deg)로 사용한다. 안전을 위해 +/- 최대각으로 제한한다.
        self.target_steer_angle_deg = self.steer_to_target_angle(steer)

    def steer_to_target_angle(self, steer):
        """steer 입력값(deg)을 안전 범위 안의 목표 조향각(deg)으로 제한한다."""
        return self.clamp_steer_angle(steer)

    # ======================================================================
    # 주기 처리 (아두이노로 실제 전송)
    # ======================================================================
    def timer_callback(self):
        # 시리얼이 아직 안 열렸으면 열기를 시도하고, 실패하면 이번 주기는 넘어감
        if self.serial is None and not self.open_serial():
            return

        now = self.get_clock().now()
        self.update_steer_position(now)
        steer_pwm = self.current_steer_pwm()
        drive_pwm = self.drive_pwm

        # 입력이 끊긴 지 오래면(통신 두절 등) 안전을 위해 정지
        if self.is_input_stale():
            steer_pwm = 0
            drive_pwm = 0
            self.target_steer_angle_deg = self.steer_angle_deg
            self.active_steer_direction = 0

        self.write_command(steer_pwm, drive_pwm)
        self.read_arduino_debug()

    def update_steer_position(self, now):
        """직전 출력 방향과 경과 시간으로 현재 조향각을 추정한다."""
        if self.last_steer_update_time is None:
            self.last_steer_update_time = now
            return

        dt = (now - self.last_steer_update_time).nanoseconds / 1e9
        self.last_steer_update_time = now
        if dt <= 0.0 or self.active_steer_direction == 0:
            return

        self.steer_angle_deg = self.clamp_steer_angle(
            self.steer_angle_deg
            + self.active_steer_direction * self.steer_speed_deg_per_sec * dt
        )

    def current_steer_pwm(self):
        """요청 방향과 추정 위치로 이번 주기에 보낼 조향 PWM을 계산한다."""
        direction = self.steer_output_direction()
        self.active_steer_direction = direction
        return direction * self.steer_pwm

    def steer_output_direction(self):
        """현재 추정 각도가 목표 각도에 가까워질 때까지 조향한다."""
        tolerance = self.steer_angle_tolerance_deg
        error = self.target_steer_angle_deg - self.steer_angle_deg

        if abs(error) <= tolerance:
            self.steer_angle_deg = self.target_steer_angle_deg
            return 0
        return 1 if error > 0.0 else -1

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

        steer_pwm = self.clamp_signed_pwm(steer_pwm)
        drive_pwm = self.clamp_drive_pwm(drive_pwm)
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
    def clamp_pwm(self, value):
        """0~255 범위로 제한(양수 상한값)."""
        return max(0, min(255, int(value)))

    def clamp_signed_pwm(self, value):
        """-255~255 범위로 제한(부호 있는 PWM)."""
        return max(-255, min(255, int(value)))

    def clamp_drive_pwm(self, value):
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
