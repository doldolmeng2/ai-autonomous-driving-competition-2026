import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, Int16, Int16MultiArray, String


class MissionLaneMainNode(Node):
    """Mission lane-driving topic contract node.

    Subscribes to lane offset and obstacle sensors, then publishes the common
    /motor_control command consumed by drive_control_node.
    """

    def __init__(self):
        super().__init__('mission_lane_main_node')

        self.declare_parameter('lane_offset_topic', '/lane_offset')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('ultrasonic_topic', '/ultrasonic/ranges')
        self.declare_parameter('motor_control_topic', '/motor_control')
        self.declare_parameter('status_topic', '/mission/lane_main/status')
        self.declare_parameter('base_speed', 30)
        self.declare_parameter('steer_kp', 0.6)
        self.declare_parameter('max_steer', 45)

        self.base_speed = int(self.get_parameter('base_speed').value)
        self.steer_kp = float(self.get_parameter('steer_kp').value)
        self.max_steer = int(self.get_parameter('max_steer').value)
        self.motor_control_topic = self.get_parameter('motor_control_topic').value
        self.status_topic = self.get_parameter('status_topic').value

        self.motor_pub = self.create_publisher(
            Int16MultiArray, self.motor_control_topic, 10
        )
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.create_subscription(
            Int16, self.get_parameter('lane_offset_topic').value,
            self.lane_offset_callback, 10
        )
        self.create_subscription(
            LaserScan, self.get_parameter('scan_topic').value,
            self.scan_callback, 10
        )
        self.create_subscription(
            Float32MultiArray, self.get_parameter('ultrasonic_topic').value,
            self.ultrasonic_callback, 10
        )

        self.last_offset = 0
        self.last_scan_count = 0
        self.last_ultrasonic = []
        self.timer = self.create_timer(0.5, self.publish_status)

        self.get_logger().info(
            'Subscribing /lane_offset, /scan, /ultrasonic/ranges; '
            f'publishing {self.motor_control_topic} and {self.status_topic}'
        )

    def lane_offset_callback(self, msg):
        self.last_offset = int(msg.data)
        steer = int(round(self.steer_kp * self.last_offset))
        steer = max(-self.max_steer, min(self.max_steer, steer))

        command = Int16MultiArray()
        command.data = [steer, self.base_speed]
        self.motor_pub.publish(command)

    def scan_callback(self, msg):
        self.last_scan_count = len(msg.ranges)

    def ultrasonic_callback(self, msg):
        self.last_ultrasonic = list(msg.data)

    def publish_status(self):
        status = String()
        status.data = (
            f'offset={self.last_offset}, scan_ranges={self.last_scan_count}, '
            f'ultrasonic_count={len(self.last_ultrasonic)}'
        )
        self.status_pub.publish(status)


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
