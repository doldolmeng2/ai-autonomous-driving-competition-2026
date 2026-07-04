"""lane_offset_debug_viewer_node.

timed_lane_offset_node 가 발행하는 /lane_offset/debug_image (ROI/차선/슬라이딩
윈도우가 그려진 이미지)를 OpenCV 창으로 띄워서 실시간으로 확인할 수 있게 한다.
rosbag 재생 중에도, 실제 카메라 주행 중에도 동일하게 사용 가능하다.
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class LaneOffsetDebugViewerNode(Node):
    """/lane_offset/debug_image 를 OpenCV 창으로 표시."""

    def __init__(self):
        super().__init__('lane_offset_debug_viewer_node')
        self.declare_parameter('debug_image_topic', '/lane_offset/debug_image')
        self.declare_parameter('window_name', 'lane_offset_debug')

        self.window_name = self.get_parameter('window_name').value

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(
            Image, self.get_parameter('debug_image_topic').value, self.image_callback, qos
        )
        self.timer = self.create_timer(0.03, lambda: cv2.waitKey(1))

    def image_callback(self, msg):
        if msg.encoding != 'bgr8':
            self.get_logger().warn(
                f'Unsupported encoding: {msg.encoding}', throttle_duration_sec=5.0
            )
            return
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        cv2.imshow(self.window_name, frame)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LaneOffsetDebugViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
