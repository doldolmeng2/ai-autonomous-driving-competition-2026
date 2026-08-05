"""HSV/YCrCb AND-mask tuner.

카메라 토픽을 구독해 HSV와 YCrCb 범위를 동시에 조절하며 각각의
마스크와 두 마스크의 교집(AND), 교집을 적용한 영상을 보여준다.
lane_offset에서 YCrCb가 설정된 색을 분류하는 방식과 같다.

사용 예:
    ros2 run sensor_utils hsv_ycrcb_tuner_node --ros-args \\
        -p image_topic:=/camera/high/image_raw
    ros2 run sensor_utils hsv_ycrcb_tuner_node --ros-args -p preset:=white
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


IMAGE_TOPIC = '/camera/high/image_raw'
PRESET = 'white'

WIN_CONTROLS = 'hsv_ycrcb_tuner_controls'
WIN_ORIGINAL = 'hsv_ycrcb_tuner_original'
WIN_HSV_MASK = 'hsv_ycrcb_tuner_hsv_mask'
WIN_YCRCB_MASK = 'hsv_ycrcb_tuner_ycrcb_mask'
WIN_AND_MASK = 'hsv_ycrcb_tuner_and_mask'
WIN_RESULT = 'hsv_ycrcb_tuner_result'

# lane_offset/config/color_profiles.yaml의 기본 범위를 튜닝 시작점으로 사용한다.
# OpenCV 채널 순서는 HSV=(H, S, V), YCrCb=(Y, Cr, Cb)이다.
PRESETS = {
    'white': {
        'hsv': {'h': (0, 179), 's': (0, 44), 'v': (161, 255)},
        'ycrcb': {'y': (135, 255), 'cr': (81, 165), 'cb': (122, 170)},
    },
    'green': {
        'hsv': {'h': (31, 60), 's': (110, 200), 'v': (0, 255)},
        'ycrcb': {'y': (0, 255), 'cr': (0, 255), 'cb': (0, 255)},
    },
    'light_gray': {
        'hsv': {'h': (56, 179), 's': (0, 64), 'v': (119, 165)},
        'ycrcb': {'y': (0, 163), 'cr': (0, 255), 'cb': (0, 255)},
    },
    'dark_gray': {
        'hsv': {'h': (0, 179), 's': (0, 102), 'v': (0, 144)},
        'ycrcb': {'y': (0, 144), 'cr': (0, 255), 'cb': (0, 142)},
    },
    'full': {
        'hsv': {'h': (0, 179), 's': (0, 255), 'v': (0, 255)},
        'ycrcb': {'y': (0, 255), 'cr': (0, 255), 'cb': (0, 255)},
    },
}


def make_masks(frame, hsv_bounds, ycrcb_bounds):
    """HSV, YCrCb, AND 마스크를 lane_offset과 같은 순서로 만든다."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    hsv_mask = cv2.inRange(hsv, hsv_bounds[0], hsv_bounds[1])
    ycrcb_mask = cv2.inRange(ycrcb, ycrcb_bounds[0], ycrcb_bounds[1])
    and_mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)
    return hsv_mask, ycrcb_mask, and_mask


class HsvYCrCbTunerNode(Node):
    """HSV/YCrCb 임계값과 교집 마스크를 실시간으로 보여준다."""

    def __init__(self):
        super().__init__('hsv_ycrcb_tuner_node')

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
            f'Subscribing {self.image_topic}. Adjust HSV and Y/Cr/Cb trackbars '
            f'in "{WIN_CONTROLS}" and watch "{WIN_AND_MASK}".'
        )

    def setup_windows(self, preset):
        cv2.namedWindow(WIN_CONTROLS, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_CONTROLS, 480, 520)
        for window in (
            WIN_ORIGINAL,
            WIN_HSV_MASK,
            WIN_YCRCB_MASK,
            WIN_AND_MASK,
            WIN_RESULT,
        ):
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        def nothing(_):
            pass

        hsv = preset['hsv']
        ycrcb = preset['ycrcb']
        trackbars = (
            ('H min', hsv['h'][0], 179),
            ('H max', hsv['h'][1], 179),
            ('S min', hsv['s'][0], 255),
            ('S max', hsv['s'][1], 255),
            ('V min', hsv['v'][0], 255),
            ('V max', hsv['v'][1], 255),
            ('Y min', ycrcb['y'][0], 255),
            ('Y max', ycrcb['y'][1], 255),
            ('Cr min', ycrcb['cr'][0], 255),
            ('Cr max', ycrcb['cr'][1], 255),
            ('Cb min', ycrcb['cb'][0], 255),
            ('Cb max', ycrcb['cb'][1], 255),
        )
        for name, initial, maximum in trackbars:
            cv2.createTrackbar(name, WIN_CONTROLS, initial, maximum, nothing)

    def image_callback(self, msg):
        self.frame = self.to_bgr(msg)

    @staticmethod
    def trackbar_bounds():
        h_min = cv2.getTrackbarPos('H min', WIN_CONTROLS)
        h_max = cv2.getTrackbarPos('H max', WIN_CONTROLS)
        s_min = cv2.getTrackbarPos('S min', WIN_CONTROLS)
        s_max = cv2.getTrackbarPos('S max', WIN_CONTROLS)
        v_min = cv2.getTrackbarPos('V min', WIN_CONTROLS)
        v_max = cv2.getTrackbarPos('V max', WIN_CONTROLS)
        y_min = cv2.getTrackbarPos('Y min', WIN_CONTROLS)
        y_max = cv2.getTrackbarPos('Y max', WIN_CONTROLS)
        cr_min = cv2.getTrackbarPos('Cr min', WIN_CONTROLS)
        cr_max = cv2.getTrackbarPos('Cr max', WIN_CONTROLS)
        cb_min = cv2.getTrackbarPos('Cb min', WIN_CONTROLS)
        cb_max = cv2.getTrackbarPos('Cb max', WIN_CONTROLS)
        return (
            ((h_min, s_min, v_min), (h_max, s_max, v_max)),
            ((y_min, cr_min, cb_min), (y_max, cr_max, cb_max)),
        )

    def tick(self):
        if self.frame is None:
            cv2.waitKey(1)
            return

        hsv_bounds, ycrcb_bounds = self.trackbar_bounds()
        hsv_mask, ycrcb_mask, and_mask = make_masks(
            self.frame, hsv_bounds, ycrcb_bounds
        )
        result = cv2.bitwise_and(self.frame, self.frame, mask=and_mask)

        hsv_pixels = int(cv2.countNonZero(hsv_mask))
        ycrcb_pixels = int(cv2.countNonZero(ycrcb_mask))
        and_pixels = int(cv2.countNonZero(and_mask))
        ratio = and_pixels / and_mask.size

        (h_min, s_min, v_min), (h_max, s_max, v_max) = hsv_bounds
        (y_min, cr_min, cb_min), (y_max, cr_max, cb_max) = ycrcb_bounds
        overlay = self.frame.copy()
        cv2.putText(
            overlay,
            f'H[{h_min},{h_max}] S[{s_min},{s_max}] V[{v_min},{v_max}] '
            f'px={hsv_pixels}',
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f'Y[{y_min},{y_max}] Cr[{cr_min},{cr_max}] '
            f'Cb[{cb_min},{cb_max}] px={ycrcb_pixels}',
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f'AND px={and_pixels} ratio={ratio:.3f}',
            (10, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WIN_ORIGINAL, overlay)
        cv2.imshow(WIN_HSV_MASK, hsv_mask)
        cv2.imshow(WIN_YCRCB_MASK, ycrcb_mask)
        cv2.imshow(WIN_AND_MASK, and_mask)
        cv2.imshow(WIN_RESULT, result)
        cv2.waitKey(1)

    def to_bgr(self, msg):
        """sensor_utils 카메라 노드가 사용하는 인코딩을 BGR로 변환한다."""
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
    node = HsvYCrCbTunerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
