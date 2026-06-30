import rclpy
from rclpy.node import Node


class CameraCalibrationNode(Node):
    """Camera calibration placeholder."""

    def __init__(self):
        super().__init__('camera_calibration_node')
        self.get_logger().info('camera_calibration_node placeholder is ready.')


def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
