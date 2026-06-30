import rclpy
from rclpy.node import Node


class UltrasonicViewerNode(Node):
    """Ultrasonic visualization placeholder."""

    def __init__(self):
        super().__init__('ultrasonic_viewer_node')
        self.get_logger().info('ultrasonic_viewer_node placeholder is ready.')


def main(args=None):
    rclpy.init(args=args)
    node = UltrasonicViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
