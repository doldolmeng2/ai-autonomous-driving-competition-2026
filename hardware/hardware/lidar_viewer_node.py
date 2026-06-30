import math

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class LidarViewerNode(Node):
    """Render LaserScan as a radar-style top-down display."""

    def __init__(self):
        super().__init__('lidar_viewer_node')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('window_name', 'lidar_radar')
        self.declare_parameter('max_range_m', 8.0)
        self.declare_parameter('image_size', 800)
        self.declare_parameter('range_ring_step_m', 1.0)

        self.window_name = self.get_parameter('window_name').value
        self.max_range = float(self.get_parameter('max_range_m').value)
        self.image_size = int(self.get_parameter('image_size').value)
        self.ring_step = float(self.get_parameter('range_ring_step_m').value)
        self.latest_scan = None

        self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').value,
            self.scan_callback,
            10,
        )
        self.timer = self.create_timer(0.033, self.draw)

    def scan_callback(self, msg):
        self.latest_scan = msg

    def draw(self):
        size = self.image_size
        center = size // 2
        scale = (size * 0.45) / self.max_range
        image = np.zeros((size, size, 3), dtype=np.uint8)

        for radius_m in np.arange(self.ring_step, self.max_range + 0.01, self.ring_step):
            radius_px = int(radius_m * scale)
            cv2.circle(image, (center, center), radius_px, (0, 90, 0), 1)
            cv2.putText(
                image,
                f'{radius_m:.0f}m',
                (center + 5, center - radius_px - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 180, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.line(image, (center, 20), (center, size - 20), (0, 70, 0), 1)
        cv2.line(image, (20, center), (size - 20, center), (0, 70, 0), 1)
        cv2.circle(image, (center, center), 7, (0, 255, 0), -1)
        cv2.putText(
            image,
            'CAR',
            (center + 12, center + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        if self.latest_scan is not None:
            self.draw_scan(image, center, scale)
        else:
            cv2.putText(
                image,
                'Waiting for /scan',
                (25, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 180, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(self.window_name, image)
        cv2.waitKey(1)

    def draw_scan(self, image, center, scale):
        msg = self.latest_scan
        for index, distance in enumerate(msg.ranges):
            if not math.isfinite(distance):
                continue
            if distance < msg.range_min or distance > min(msg.range_max, self.max_range):
                continue

            angle = msg.angle_min + index * msg.angle_increment
            x = int(center + math.sin(angle) * distance * scale)
            y = int(center - math.cos(angle) * distance * scale)
            intensity = 120
            if index < len(msg.intensities):
                intensity = min(255, 80 + int(msg.intensities[index] * 4))
            cv2.circle(image, (x, y), 2, (0, intensity, 30), -1)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LidarViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
