"""timed_lane_main_node.

역할:
    시간주행(2차로 고정 주행) 미션에서 /lane_offset(오른쪽 차선 기준 offset, px)을
    받아 steer/speed 를 계산하고 /motor_control 로 발행한다.

제어 방식:
    offset > 0 : 차가 왼쪽으로 치우쳐서(오른쪽 차선이 기준보다 더 오른쪽에 보임)
                 오른쪽으로 조향해야 함 -> steer > 0 (drive_control 기준과 동일)
    offset < 0 : 차가 오른쪽 차선에 너무 붙어서 왼쪽으로 조향해야 함 -> steer < 0
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16, Int16MultiArray

# 구독/발행 토픽
LANE_OFFSET_TOPIC = '/lane_offset'
MOTOR_CONTROL_TOPIC = '/motor_control'

# 순항 속도(PWM). 실차 트랙에서 반드시 재조정 필요.
BASE_SPEED = 30
# steer 값 clamp 범위 (-MAX_STEER ~ +MAX_STEER)
MAX_STEER = 45


class TimedLaneMainNode(Node):
    """/lane_offset -> steer/speed 계산 후 /motor_control 발행."""

    def __init__(self):
        super().__init__('timed_lane_main_node')

        self.declare_parameter('base_speed', BASE_SPEED)
        self.declare_parameter('max_steer', MAX_STEER)

        self.lane_offset_topic = LANE_OFFSET_TOPIC
        self.motor_control_topic = MOTOR_CONTROL_TOPIC
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
        )

    def lane_offset_callback(self, msg):
        offset = msg.data
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
