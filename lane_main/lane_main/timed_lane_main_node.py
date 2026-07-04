"""timed_lane_main_node.

역할:
    시간주행(2차로 고정 주행) 미션에서 /lane_offset(오른쪽 차선 기준 offset, px)을
    받아 steer/speed 를 계산하고 /motor_control 로 발행한다.

제어 방식:
    offset > 0 : 차가 왼쪽으로 치우쳐서(오른쪽 차선이 기준보다 더 오른쪽에 보임)
                 오른쪽으로 조향해야 함 -> steer > 0 (drive_control 기준과 동일)
    offset < 0 : 차가 오른쪽 차선에 너무 붙어서 왼쪽으로 조향해야 함 -> steer < 0
    steer = clamp(steer_kp * offset, -max_steer, max_steer) 의 단순 비례(P) 제어.

    steer_kp / base_speed 는 실차 트랙에서 반드시 재조정이 필요한 값이다.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16, Int16MultiArray

# ============================================================================
# 파라미터 기본값 - 튜닝은 대부분 여기서만 하면 된다.
# (전부 ROS 파라미터로도 선언되므로 --ros-args -p 로 실행 중 덮어쓰기도 가능)
# ============================================================================

# 구독/발행 토픽
LANE_OFFSET_TOPIC = '/lane_offset'
MOTOR_CONTROL_TOPIC = '/motor_control'

# P게인: offset(px) -> steer 변환 비율. 실차 트랙에서 반드시 재조정 필요.
STEER_KP = 0.6
# 순항 속도(PWM). 실차 트랙에서 반드시 재조정 필요.
BASE_SPEED = 30
# steer 값 clamp 범위 (-MAX_STEER ~ +MAX_STEER)
MAX_STEER = 200


class TimedLaneMainNode(Node):
    """/lane_offset -> steer/speed 계산 후 /motor_control 발행."""

    def __init__(self):
        super().__init__('timed_lane_main_node')

        self.declare_parameter('lane_offset_topic', LANE_OFFSET_TOPIC)
        self.declare_parameter('motor_control_topic', MOTOR_CONTROL_TOPIC)
        self.declare_parameter('steer_kp', STEER_KP)
        self.declare_parameter('base_speed', BASE_SPEED)
        self.declare_parameter('max_steer', MAX_STEER)

        self.lane_offset_topic = self.get_parameter('lane_offset_topic').value
        self.motor_control_topic = self.get_parameter('motor_control_topic').value
        self.steer_kp = float(self.get_parameter('steer_kp').value)
        self.base_speed = int(self.get_parameter('base_speed').value)
        self.max_steer = int(self.get_parameter('max_steer').value)

        self.motor_pub = self.create_publisher(
            Int16MultiArray, self.motor_control_topic, 10
        )
        self.create_subscription(
            Int16, self.lane_offset_topic, self.lane_offset_callback, 10
        )

        self.get_logger().info(
            f'Subscribing {self.lane_offset_topic}, publishing {self.motor_control_topic}, '
            f'steer_kp={self.steer_kp}, base_speed={self.base_speed}'
        )

    def lane_offset_callback(self, msg):
        offset = msg.data
        steer = int(round(self.steer_kp * offset))
        steer = max(-self.max_steer, min(self.max_steer, steer))

        command = Int16MultiArray()
        command.data = [steer, self.base_speed]
        self.motor_pub.publish(command)


def main(args=None):
    rclpy.init(args=args)
    node = TimedLaneMainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
