import rclpy
from rclpy.node import Node


class CameraNode(Node):
    """Camera topic publisher placeholder."""

    def __init__(self):
        super().__init__('camera_node')
        self.get_logger().info('camera_node placeholder is ready.')


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
