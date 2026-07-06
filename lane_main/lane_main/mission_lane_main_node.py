import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16, Int16MultiArray


LANE_INFO_TOPIC = '/lane_info'
LANE_OFFSET_TOPIC = '/lane_offset'
MOTOR_CONTROL_TOPIC = '/motor_control'
BASE_SPEED = 30
STEER_KP = 0.6
MAX_STEER = 45


class MissionLaneMainNode(Node):
    """Mission lane driving node.

    PDF flow:
        /lane_info, /lane_offset -> mission_lane_main_node -> /motor_control
    """

    def __init__(self):
        super().__init__('mission_lane_main_node')

        self.declare_parameter('base_speed', BASE_SPEED)
        self.declare_parameter('steer_kp', STEER_KP)
        self.declare_parameter('max_steer', MAX_STEER)

        self.base_speed = int(self.get_parameter('base_speed').value)
        self.steer_kp = float(self.get_parameter('steer_kp').value)
        self.max_steer = int(self.get_parameter('max_steer').value)
        self.lane_number = 2

        self.motor_pub = self.create_publisher(
            Int16MultiArray, MOTOR_CONTROL_TOPIC, 10
        )
        self.create_subscription(Int16, LANE_INFO_TOPIC, self.lane_info_callback, 10)
        self.create_subscription(
            Int16, LANE_OFFSET_TOPIC, self.lane_offset_callback, 10
        )

        self.get_logger().info(
            f'Subscribing {LANE_INFO_TOPIC}, {LANE_OFFSET_TOPIC}; '
            f'publishing {MOTOR_CONTROL_TOPIC}'
        )

    def lane_info_callback(self, msg):
        self.lane_number = int(msg.data)

    def lane_offset_callback(self, msg):
        offset = int(msg.data)
        steer = int(round(self.steer_kp * offset))
        steer = max(-self.max_steer, min(self.max_steer, steer))

        # TODO: Use lane_number for mission-specific lane changes.
        command = Int16MultiArray()
        command.data = [steer, self.base_speed]
        self.motor_pub.publish(command)


def main(args=None):
    rclpy.init(args=args)
    node = MissionLaneMainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
