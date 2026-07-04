"""joy_to_motor_node.

역할:
    조이스틱(sensor_msgs/Joy)을 자율주행 알고리즘과 "동일한" 제어 토픽
    /motor_control (std_msgs/msg/Int16MultiArray, data=[steer, speed]) 로 변환해
    발행한다.

설계 배경(PDF 기준):
    - /motor_control 은 조이스틱과 자율주행 알고리즘이 공통으로 사용하는
      제어 토픽이다. drive_control 노드는 이 토픽만 구독한다.
    - 조이스틱/컨트롤러 관련 토픽 발행은 hardware 패키지의 역할이므로,
      Joy -> /motor_control 변환도 hardware 패키지에 둔다.
    - manual_controller_node 의 원본 Joy 발행은 시각화(controller_viewer_node)
      용으로 그대로 유지하고, 이 노드가 그 위에서 변환만 담당한다.

동작:
    manual_controller_node 는 일정 주기(publish_rate)로 Joy 를 계속 발행하므로,
    이 노드는 Joy 콜백마다 /motor_control 을 발행한다. 따라서 스틱을 가만히
    쥐고 있어도 명령이 꾸준히 나가고, drive_control 의 input_timeout 안전정지와
    충돌하지 않는다.

출력 값의 의미(drive_control 과 맞춤):
    - speed : 구동 모터 PWM (부호=전/후진). drive_control 이 max_drive_pwm 으로
              최종 안전 clamp 한다.
    - steer : 목표 조향각(deg). drive_control 이 +/-45도 안전 범위로 최종 clamp 한다.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Int16MultiArray


class JoyToMotorNode(Node):
    """Joy 입력을 /motor_control (steer, speed)로 변환해 발행하는 노드."""

    def __init__(self):
        super().__init__('joy_to_motor_node')

        # ---- 파라미터 -----------------------------------------------------
        self.declare_parameter('joy_topic', '/manual_controller/joy')
        self.declare_parameter('motor_control_topic', '/motor_control')
        # 어떤 조이스틱 축을 조향/구동으로 쓸지 (manual_controller 축 순서 기준)
        self.declare_parameter('steer_axis', 3)
        self.declare_parameter('drive_axis', 1)
        # 축 방향이 반대로 느껴지면 True 로 뒤집는다
        self.declare_parameter('invert_steer_axis', False)
        self.declare_parameter('invert_drive_axis', True)
        # 미세한 스틱 흔들림 무시(중립 근처 데드존)
        self.declare_parameter('deadzone', 0.2)
        # 스틱 최대치일 때 내보낼 speed PWM / steer 목표각(deg)
        self.declare_parameter('max_speed', 255)
        self.declare_parameter('max_steer', 45)

        self.steer_axis = int(self.get_parameter('steer_axis').value)
        self.drive_axis = int(self.get_parameter('drive_axis').value)
        self.invert_steer_axis = bool(self.get_parameter('invert_steer_axis').value)
        self.invert_drive_axis = bool(self.get_parameter('invert_drive_axis').value)
        self.deadzone = float(self.get_parameter('deadzone').value)
        self.max_speed = int(self.get_parameter('max_speed').value)
        self.max_steer = int(self.get_parameter('max_steer').value)

        motor_control_topic = self.get_parameter('motor_control_topic').value
        joy_topic = self.get_parameter('joy_topic').value

        # ---- 통신 ---------------------------------------------------------
        self.publisher = self.create_publisher(Int16MultiArray, motor_control_topic, 10)
        self.create_subscription(Joy, joy_topic, self.joy_callback, 10)

        self.get_logger().info(
            f'Converting {joy_topic} -> {motor_control_topic} '
            f'(steer_axis={self.steer_axis}, drive_axis={self.drive_axis}, '
            f'max_speed={self.max_speed}, max_steer={self.max_steer})'
        )

    def joy_callback(self, msg):
        """Joy 수신 -> 축 값을 steer/speed 로 변환 -> /motor_control 발행."""
        steer_value = self.read_axis(msg, self.steer_axis)
        drive_value = self.read_axis(msg, self.drive_axis)
        if self.invert_steer_axis:
            steer_value = -steer_value
        if self.invert_drive_axis:
            drive_value = -drive_value

        steer = self.axis_to_int(steer_value, self.max_steer)
        speed = self.axis_to_int(drive_value, self.max_speed)

        out = Int16MultiArray()
        out.data = [steer, speed]   # 형식: [steer, speed]
        self.publisher.publish(out)

    def read_axis(self, msg, index):
        """해당 축 값을 -1.0~1.0 범위로 안전하게 읽는다(없는 축이면 0)."""
        if index < 0 or len(msg.axes) <= index:
            return 0.0
        return max(-1.0, min(1.0, float(msg.axes[index])))

    def axis_to_int(self, value, max_value):
        """축 값(-1.0~1.0)을 데드존 적용 후 정수 명령(-max~max)으로 변환.

        예전 drive_control 의 axis_to_pwm 과 동일한 계산식(데드존/반올림/
        최소 1 보장)을 그대로 사용해, 예전과 같은 값이 나오도록 맞췄다.
        """
        if abs(value) < self.deadzone:
            return 0
        pwm = int(round(abs(value) * max_value))
        pwm = max(1, min(max_value, pwm))
        return pwm if value > 0.0 else -pwm


def main(args=None):
    rclpy.init(args=args)
    node = JoyToMotorNode()
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
