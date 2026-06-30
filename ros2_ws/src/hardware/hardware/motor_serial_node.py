import rclpy
from rclpy.node import Node


class MotorSerialNode(Node):
    """Motor serial output placeholder for forwarding drive commands to Arduino."""

    def __init__(self):
        super().__init__('motor_serial_node')
        self.get_logger().info('motor_serial_node placeholder is ready.')


def main(args=None):
    rclpy.init(args=args)
    node = MotorSerialNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
