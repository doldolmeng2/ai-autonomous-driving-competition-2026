"""drive_control_node.

역할(설계 PDF 기준):
    /motor_control 토픽으로 들어온 (steer, speed) 값을 받아서
    -> 조향 모터 제어값 / 구동 모터 제어값을 계산하고
    -> 아두이노로 시리얼 전송한다.

입력 토픽:
    /motor_control  (std_msgs/msg/Int16MultiArray)
        data = [steer, speed]
        - steer : 조향 명령. 부호(+/-)로 방향을 결정한다.
                  steer > 0 : 오른쪽,  steer < 0 : 왼쪽,  steer == 0(데드밴드 이내) : 조향 없음
        - speed : 구동 모터 PWM (부호 = 전/후진, 크기 = 세기)

    조이스틱(수동)과 자율주행 알고리즘(lane_main / parking 등)이
    "동일한" /motor_control 토픽으로 발행하면, 이 노드는 발행 주체와 무관하게
    똑같이 동작한다.

출력(아두이노 시리얼):
    "steer speed\n"  (공백 구분, 부호 있는 정수 PWM)
    예) "-150 30\n"

조향 처리(기존에 검증된 방식 그대로):
    조향 모터는 위치 센서가 없어서, "방향이 새로 바뀔 때마다 일정 PWM으로
    일정 시간(steer_pulse_duration) 펄스를 준다"는 방식으로 조향한다.
    같은 방향을 계속 유지해도 펄스는 한 번만 나가고(=재펄스 없음),
    방향이 0(중앙 명령)이 되면 즉시 조향 PWM을 멈춘다.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray

from hardware.serial_port import open_serial


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
        self.declare_parameter('arduino_boot_delay', 2.0)   # 시리얼 연결 후 아두이노 부팅 대기(s)
        self.declare_parameter('enable_arduino_debug_log', True)  # 아두이노 -> PC 로그 출력
        self.declare_parameter('enable_tx_debug_log', False)      # PC -> 아두이노 송신 로그 출력

        # 구동 / 조향 제어값 관련
        self.declare_parameter('max_drive_pwm', 30)         # 구동 모터 안전 상한 PWM
        self.declare_parameter('steer_pwm', 40)             # 조향할 때 넣는 PWM 크기
        self.declare_parameter('steer_deadband', 0)         # steer 값이 이 이하면 조향 없음
        self.declare_parameter('steer_pulse_duration', 1.0)  # 방향 전환 시 조향 펄스 시간(s)

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
        self.steer_deadband = abs(int(self.get_parameter('steer_deadband').value))
        self.steer_pulse_duration = float(
            self.get_parameter('steer_pulse_duration').value
        )

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

        # 조향 펄스 상태(방향이 새로 바뀔 때만 일정 시간 펄스)
        self.requested_steer_direction = 0  # 마지막으로 요청된 방향(+1/-1/0)
        self.active_steer_direction = 0     # 현재 펄스 중인 방향
        self.steer_pulse_until = None       # 펄스 종료 시각(ns)

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
            f'drive <= {self.max_drive_pwm}, steer pulse {self.steer_pwm} '
            f'for {self.steer_pulse_duration:.1f}s'
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

        # 구동: speed를 안전 상한으로 clamp 한 PWM
        self.drive_pwm = max(-self.max_drive_pwm, min(self.max_drive_pwm, speed))

        # 조향: 부호로 방향 결정 후, 방향이 바뀌었을 때만 펄스 시작
        self.update_steer_request(self.steer_direction(steer))

    def steer_direction(self, steer):
        """steer 값 -> 방향(+1 오른쪽 / -1 왼쪽 / 0 없음). 데드밴드 이내면 0."""
        if abs(steer) <= self.steer_deadband:
            return 0
        return 1 if steer > 0 else -1

    def update_steer_request(self, direction):
        """방향이 새로 바뀔 때만 steer_pulse_duration 동안의 조향 펄스를 시작."""
        now = self.get_clock().now()
        if direction == 0:
            # 중앙(조향 없음) 명령: 진행 중이던 펄스도 즉시 종료
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

    # ======================================================================
    # 주기 처리 (아두이노로 실제 전송)
    # ======================================================================
    def timer_callback(self):
        # 시리얼이 아직 안 열렸으면 열기를 시도하고, 실패하면 이번 주기는 넘어감
        if self.serial is None and not self.open_serial():
            return

        steer_pwm = self.current_steer_pwm()
        drive_pwm = self.drive_pwm

        # 입력이 끊긴 지 오래면(통신 두절 등) 안전을 위해 정지
        if self.is_input_stale():
            steer_pwm = 0
            drive_pwm = 0
            self.requested_steer_direction = 0
            self.active_steer_direction = 0
            self.steer_pulse_until = None

        self.write_command(steer_pwm, drive_pwm)
        self.read_arduino_debug()

    def current_steer_pwm(self):
        """진행 중인 조향 펄스가 있으면 그 PWM을, 펄스가 끝났으면 0을 반환."""
        if self.active_steer_direction == 0 or self.steer_pulse_until is None:
            return 0

        now_ns = self.get_clock().now().nanoseconds
        if now_ns >= self.steer_pulse_until:
            # 펄스 시간이 끝났으므로 조향 정지
            self.active_steer_direction = 0
            self.steer_pulse_until = None
            return 0

        return self.active_steer_direction * self.steer_pwm

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
        drive_pwm = self.clamp_signed_pwm(drive_pwm)
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
