import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, Range
from std_msgs.msg import Int16


HIGH_IMAGE_TOPIC = '/camera/high/image_raw'
LOW_IMAGE_TOPIC = '/camera/low/image_raw'
ULTRASONIC_TOPICS = [f'/ultrasonic/range_{index}' for index in range(1, 7)]
LANE_INFO_TOPIC = '/lane_info'
LANE_OFFSET_TOPIC = '/lane_offset'
DEFAULT_LANE_NUMBER = 2
DEFAULT_OFFSET = 0


class MissionLaneOffsetNode(Node):
    """Mission lane offset node.

    PDF flow:
        /camera/high/image_raw
        /camera/low/image_raw
        /ultrasonic/range_1 ... /ultrasonic/range_6
            -> mission_lane_offset_node
            -> /lane_info, /lane_offset
    """

    def __init__(self):
        super().__init__('mission_lane_offset_node')

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.lane_info_pub = self.create_publisher(Int16, LANE_INFO_TOPIC, 10)
        self.offset_pub = self.create_publisher(Int16, LANE_OFFSET_TOPIC, 10)
        self.create_subscription(
            Image,
            HIGH_IMAGE_TOPIC,
            self.high_image_callback,
            qos,
        )
        self.create_subscription(
            Image,
            LOW_IMAGE_TOPIC,
            self.low_image_callback,
            qos,
        )
        for topic in ULTRASONIC_TOPICS:
            self.create_subscription(Range, topic, self.ultrasonic_callback, 10)

        self.high_image = None
        self.low_image = None
        self.ultrasonic_ranges = {}
        self.timer = self.create_timer(0.1, self.publish_outputs)

        self.get_logger().info(
            f'Subscribing {HIGH_IMAGE_TOPIC}, {LOW_IMAGE_TOPIC}, '
            f'{ULTRASONIC_TOPICS}; publishing {LANE_INFO_TOPIC}, '
            f'{LANE_OFFSET_TOPIC}'
        )

    def high_image_callback(self, msg):
        self.high_image = msg

    def low_image_callback(self, msg):
        self.low_image = msg

    def ultrasonic_callback(self, msg):
        self.ultrasonic_ranges[msg.header.frame_id] = msg.range

    def publish_outputs(self):
        # TODO: Replace these defaults with mission lane recognition and
        # obstacle-aware lane selection.
        lane_info = Int16()
        lane_info.data = DEFAULT_LANE_NUMBER
        self.lane_info_pub.publish(lane_info)

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
