import rclpy
from rclpy.node import Node


class UltrasonicNode(Node):
    """Ultrasonic sensor topic publisher placeholder."""

    def __init__(self):
        super().__init__('ultrasonic_node')
        self.get_logger().info('ultrasonic_node placeholder is ready.')


def main(args=None):
    rclpy.init(args=args)
    node = UltrasonicNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
