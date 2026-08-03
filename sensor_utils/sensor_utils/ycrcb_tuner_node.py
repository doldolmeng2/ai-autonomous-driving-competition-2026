"""ycrcb_tuner_node.

카메라 토픽을 구독해 트랙바로 Y/Cr/Cb 최소·최대값을 조절하면서 원본,
마스크, 마스킹 결과를 실시간으로 보여준다.

사용 예:
    ros2 run sensor_utils ycrcb_tuner_node --ros-args \
        -p image_topic:=/camera/high/image_raw
    ros2 run sensor_utils ycrcb_tuner_node --ros-args -p preset:=white
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


IMAGE_TOPIC = '/camera/high/image_raw'
PRESET = 'white'

WIN_CONTROLS = 'ycrcb_tuner_controls'
WIN_ORIGINAL = 'ycrcb_tuner_original'
WIN_MASK = 'ycrcb_tuner_mask'
WIN_RESULT = 'ycrcb_tuner_result'

# lane_offset의 색상 분류값을 바로 확인할 수 있도록 같은 범위를 제공한다.
# OpenCV YCrCb 채널 순서는 Y, Cr, Cb이다.
PRESETS = {
    'white': {'y': (135, 255), 'cr': (81, 165), 'cb': (122, 170)},
    'light_gray': {'y': (0, 163), 'cr': (0, 255), 'cb': (0, 255)},
    'full': {'y': (0, 255), 'cr': (0, 255), 'cb': (0, 255)},
}


class YCrCbTunerNode(Node):
    """트랙바로 YCrCb 범위를 조절하며 원본/마스크/결과를 보여준다."""

    def __init__(self):
        super().__init__('ycrcb_tuner_node')

        self.declare_parameter('image_topic', IMAGE_TOPIC)
        self.declare_parameter('preset', PRESET)

        self.image_topic = str(self.get_parameter('image_topic').value)
        preset_name = str(self.get_parameter('preset').value).lower()
        if preset_name not in PRESETS:
            self.get_logger().warn(
                f"Unknown preset='{preset_name}'; using '{PRESET}'"
            )
            preset_name = PRESET

        self.frame = None
        self.setup_windows(PRESETS[preset_name])

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(Image, self.image_topic, self.image_callback, qos)
        self.create_timer(1.0 / 30.0, self.tick)

        self.get_logger().info(
            f'Subscribing {self.image_topic}. Adjust Y/Cr/Cb trackbars in '
            f'"{WIN_CONTROLS}" and watch the mask window.'
        )

    def setup_windows(self, preset):
        cv2.namedWindow(WIN_CONTROLS, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_CONTROLS, 420, 260)
        cv2.namedWindow(WIN_ORIGINAL, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_MASK, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_RESULT, cv2.WINDOW_NORMAL)

        def nothing(_):
            pass

        y_lo, y_hi = preset['y']
        cr_lo, cr_hi = preset['cr']
        cb_lo, cb_hi = preset['cb']
        cv2.createTrackbar('Y min', WIN_CONTROLS, y_lo, 255, nothing)
        cv2.createTrackbar('Y max', WIN_CONTROLS, y_hi, 255, nothing)
        cv2.createTrackbar('Cr min', WIN_CONTROLS, cr_lo, 255, nothing)
        cv2.createTrackbar('Cr max', WIN_CONTROLS, cr_hi, 255, nothing)
        cv2.createTrackbar('Cb min', WIN_CONTROLS, cb_lo, 255, nothing)
        cv2.createTrackbar('Cb max', WIN_CONTROLS, cb_hi, 255, nothing)

    def image_callback(self, msg):
        self.frame = self.to_bgr(msg)

    def tick(self):
        if self.frame is None:
            cv2.waitKey(1)
            return

        y_min = cv2.getTrackbarPos('Y min', WIN_CONTROLS)
        y_max = cv2.getTrackbarPos('Y max', WIN_CONTROLS)
        cr_min = cv2.getTrackbarPos('Cr min', WIN_CONTROLS)
        cr_max = cv2.getTrackbarPos('Cr max', WIN_CONTROLS)
        cb_min = cv2.getTrackbarPos('Cb min', WIN_CONTROLS)
        cb_max = cv2.getTrackbarPos('Cb max', WIN_CONTROLS)

        ycrcb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(
            ycrcb,
            (y_min, cr_min, cb_min),
            (y_max, cr_max, cb_max),
        )
        result = cv2.bitwise_and(self.frame, self.frame, mask=mask)
        pixel_count = int(cv2.countNonZero(mask))
        ratio = pixel_count / mask.size

        overlay = self.frame.copy()
        cv2.putText(
            overlay,
            f'Y[{y_min},{y_max}] Cr[{cr_min},{cr_max}] '
            f'Cb[{cb_min},{cb_max}] px={pixel_count} ratio={ratio:.3f}',
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WIN_ORIGINAL, overlay)
        cv2.imshow(WIN_MASK, mask)
        cv2.imshow(WIN_RESULT, result)
        cv2.waitKey(1)

    def to_bgr(self, msg):
        """sensor_utils 카메라 노드들이 사용하는 인코딩을 BGR로 변환한다."""
        data = np.frombuffer(msg.data, dtype=np.uint8)
        try:
            if msg.encoding in ('yuv422_yuy2', 'yuyv', 'yuyv422'):
                yuyv = data.reshape((msg.height, msg.width, 2))
                return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
            if msg.encoding in ('bgr8', '8UC3'):
                return data.reshape((msg.height, msg.width, 3))
            if msg.encoding == 'rgb8':
                rgb = data.reshape((msg.height, msg.width, 3))
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if msg.encoding in ('mono8', '8UC1'):
                mono = data.reshape((msg.height, msg.width))
                return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
        except ValueError as error:
            self.get_logger().warn(
                f'Invalid camera image buffer: {error}',
                throttle_duration_sec=5.0,
            )
            return None

        self.get_logger().warn(
            f'Unsupported camera encoding: {msg.encoding}',
            throttle_duration_sec=5.0,
        )
        return None

    def destroy_node(self):
        cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YCrCbTunerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
