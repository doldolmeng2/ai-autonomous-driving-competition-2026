import rclpy
from rclpy.node import Node


class CameraViewerNode(Node):
    """Camera visualization placeholder."""

    def __init__(self):
        super().__init__('camera_viewer_node')
        self.get_logger().info('camera_viewer_node placeholder is ready.')


def main(args=None):
    rclpy.init(args=args)
    node = CameraViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
