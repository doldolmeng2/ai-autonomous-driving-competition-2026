import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan, Range
from std_msgs.msg import Int16MultiArray


IMAGE_TOPIC = '/camera/high/image_raw'
SCAN_TOPIC = '/scan'
ULTRASONIC_TOPICS = [f'/ultrasonic/range_{index}' for index in range(1, 7)]
MOTOR_CONTROL_TOPIC = '/motor_control'
PUBLISH_STOP_COMMAND = True


class ParkingNode(Node):
    """Parking node.

    PDF flow:
        /camera/high/image_raw
        /ultrasonic/range_1 ... /ultrasonic/range_6
        /scan
            -> parking_node
            -> /motor_control
    """

    def __init__(self):
        super().__init__('parking_node')

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.motor_pub = self.create_publisher(
            Int16MultiArray, MOTOR_CONTROL_TOPIC, 10
        )
        self.create_subscription(Image, IMAGE_TOPIC, self.image_callback, qos)
        self.create_subscription(LaserScan, SCAN_TOPIC, self.scan_callback, qos)
        for topic in ULTRASONIC_TOPICS:
            self.create_subscription(Range, topic, self.ultrasonic_callback, 10)

        self.image = None
        self.scan = None
        self.ultrasonic_ranges = {}
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info(
            f'Subscribing {IMAGE_TOPIC}, {SCAN_TOPIC}, {ULTRASONIC_TOPICS}; '
            f'publishing {MOTOR_CONTROL_TOPIC}'
        )

    def image_callback(self, msg):
        self.image = msg

    def scan_callback(self, msg):
        self.scan = msg

    def ultrasonic_callback(self, msg):
        self.ultrasonic_ranges[msg.header.frame_id] = msg.range

    def timer_callback(self):
        if not PUBLISH_STOP_COMMAND:
            return

        # TODO: Replace safe stop with parking-space detection and parking path
        # control.
        command = Int16MultiArray()
        command.data = [0, 0]
        self.motor_pub.publish(command)


def main(args=None):
    rclpy.init(args=args)
    node = ParkingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
