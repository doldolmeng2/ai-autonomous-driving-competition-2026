import rclpy
from rclpy.node import Node


class LidarViewerNode(Node):
    """LiDAR visualization placeholder."""

    def __init__(self):
        super().__init__('lidar_viewer_node')
        self.get_logger().info('lidar_viewer_node placeholder is ready.')


def main(args=None):
    rclpy.init(args=args)
    node = LidarViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
