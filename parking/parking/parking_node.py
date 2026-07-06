import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Float32MultiArray, Int16MultiArray, String


class ParkingNode(Node):
    """Parking mission topic contract node.

    Subscribes to camera/lidar/ultrasonic topics and publishes /motor_control.
    The default command is a safe stop until parking detection/path logic is
    implemented.
    """

    def __init__(self):
        super().__init__('parking_node')

        self.declare_parameter('high_image_topic', '/camera/high/image_raw')
        self.declare_parameter('low_image_topic', '/camera/low/image_raw')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('ultrasonic_topic', '/ultrasonic/ranges')
        self.declare_parameter('motor_control_topic', '/motor_control')
        self.declare_parameter('status_topic', '/mission/parking/status')
        self.declare_parameter('publish_stop_command', True)

        self.motor_control_topic = self.get_parameter('motor_control_topic').value
        self.status_topic = self.get_parameter('status_topic').value
        self.publish_stop_command = bool(
            self.get_parameter('publish_stop_command').value
        )

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.motor_pub = self.create_publisher(
            Int16MultiArray, self.motor_control_topic, 10
        )
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

        self.high_seen = False
        self.low_seen = False
        self.last_scan_count = 0
        self.last_ultrasonic = []
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info(
            'Subscribing camera high/low, /scan, /ultrasonic/ranges; '
            f'publishing {self.motor_control_topic} and {self.status_topic}'
        )

    def high_image_callback(self, _msg):
        self.high_seen = True

    def low_image_callback(self, _msg):
        self.low_seen = True

    def scan_callback(self, msg):
        self.last_scan_count = len(msg.ranges)

    def ultrasonic_callback(self, msg):
        self.last_ultrasonic = list(msg.data)

    def timer_callback(self):
        if self.publish_stop_command:
            command = Int16MultiArray()
            command.data = [0, 0]
            self.motor_pub.publish(command)

        status = String()
        status.data = (
            f'high={self.high_seen}, low={self.low_seen}, '
            f'scan_ranges={self.last_scan_count}, '
            f'ultrasonic_count={len(self.last_ultrasonic)}'
        )
        self.status_pub.publish(status)


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
