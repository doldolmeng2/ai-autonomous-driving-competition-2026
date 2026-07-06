import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Float32MultiArray, Int16, String


class MissionLaneOffsetNode(Node):
    """Mission lane-offset topic contract node.

    Subscribes to the mission sensors and publishes /lane_offset for lane_main.
    The image/lidar/ultrasonic callbacks are intentionally lightweight so the
    real lane-change and obstacle-aware offset algorithm can be dropped in here.
    """

    def __init__(self):
        super().__init__('mission_lane_offset_node')

        self.declare_parameter('high_image_topic', '/camera/high/image_raw')
        self.declare_parameter('low_image_topic', '/camera/low/image_raw')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('ultrasonic_topic', '/ultrasonic/ranges')
        self.declare_parameter('lane_offset_topic', '/lane_offset')
        self.declare_parameter('status_topic', '/mission/lane_offset/status')
        self.declare_parameter('default_offset', 0)

        self.lane_offset_topic = self.get_parameter('lane_offset_topic').value
        self.status_topic = self.get_parameter('status_topic').value
        self.default_offset = int(self.get_parameter('default_offset').value)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.offset_pub = self.create_publisher(Int16, self.lane_offset_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.create_subscription(
            Image, self.get_parameter('high_image_topic').value,
            self.high_image_callback, qos
        )
        self.create_subscription(
            Image, self.get_parameter('low_image_topic').value,
            self.low_image_callback, qos
        )
        self.create_subscription(
            LaserScan, self.get_parameter('scan_topic').value,
            self.scan_callback, qos
        )
        self.create_subscription(
            Float32MultiArray, self.get_parameter('ultrasonic_topic').value,
            self.ultrasonic_callback, 10
        )

        self.last_high_stamp = None
        self.last_low_stamp = None
        self.last_scan_count = 0
        self.last_ultrasonic = []
        self.timer = self.create_timer(0.1, self.publish_contract_output)

        self.get_logger().info(
            'Subscribing camera high/low, /scan, /ultrasonic/ranges; '
            f'publishing {self.lane_offset_topic} and {self.status_topic}'
        )

    def high_image_callback(self, msg):
        self.last_high_stamp = msg.header.stamp

    def low_image_callback(self, msg):
        self.last_low_stamp = msg.header.stamp

    def scan_callback(self, msg):
        self.last_scan_count = len(msg.ranges)

    def ultrasonic_callback(self, msg):
        self.last_ultrasonic = list(msg.data)

    def publish_contract_output(self):
        offset = Int16()
        offset.data = self.default_offset
        self.offset_pub.publish(offset)

        status = String()
        status.data = (
            f'high={self.last_high_stamp is not None}, '
            f'low={self.last_low_stamp is not None}, '
            f'scan_ranges={self.last_scan_count}, '
            f'ultrasonic_count={len(self.last_ultrasonic)}'
        )
        self.status_pub.publish(status)


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
