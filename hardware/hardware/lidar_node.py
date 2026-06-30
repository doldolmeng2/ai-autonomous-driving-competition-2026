import rclpy
from rclpy.node import Node


class LidarNode(Node):
    """LiDAR topic publisher placeholder."""

    def __init__(self):
        super().__init__('lidar_node')
        self.get_logger().info('lidar_node placeholder is ready.')


def main(args=None):
    rclpy.init(args=args)
    node = LidarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
