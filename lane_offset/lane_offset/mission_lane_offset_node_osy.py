import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16


HIGH_IMAGE_TOPIC = '/camera/high/image_raw'
LANE_INFO_TOPIC = '/lane_info'
LANE_OFFSET_TOPIC = '/lane_offset'
DEFAULT_OFFSET = 0


class MissionLaneOffsetNode(Node):
    """Mission lane offset node.

    PDF flow:
        /camera/high/image_raw
        /lane_info
            -> mission_lane_offset_node
            -> /lane_offset
    """

    def __init__(self):
        super().__init__('mission_lane_offset_node_osy')

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.offset_pub = self.create_publisher(Int16, LANE_OFFSET_TOPIC, 10)
        self.create_subscription(
            Image,
            HIGH_IMAGE_TOPIC,
            self.high_image_callback,
            qos,
        )
        self.create_subscription(Int16, LANE_INFO_TOPIC, self.lane_info_callback, 10)

        self.high_image = None
        self.lane_number = 2
        self.timer = self.create_timer(0.1, self.publish_outputs)

        self.get_logger().info(
            f'Subscribing {HIGH_IMAGE_TOPIC}, {LANE_INFO_TOPIC}; '
            f'publishing {LANE_OFFSET_TOPIC}'
        )

    def high_image_callback(self, msg):
        self.high_image = msg

    def lane_info_callback(self, msg):
        self.lane_number = int(msg.data)

    def publish_outputs(self):
        # TODO: Use high_image and lane_number to calculate mission lane offset.
        offset = Int16()
        offset.data = DEFAULT_OFFSET
        self.offset_pub.publish(offset)


def main(args=None):
    rclpy.init(args=args)
    node = MissionLaneOffsetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
