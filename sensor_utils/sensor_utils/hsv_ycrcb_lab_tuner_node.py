"""HSV/YCrCb/LAB AND-mask tuner.

카메라 토픽을 구독해 HSV, YCrCb, LAB 범위를 실시간으로 조절하고,
각 색 공간의 마스크와 세 마스크의 교집(AND) 결과를 보여준다.

사용 예:
    ros2 run sensor_utils hsv_ycrcb_lab_tuner_node --ros-args \\
        -p image_topic:=/camera/high/image_raw
    ros2 run sensor_utils hsv_ycrcb_lab_tuner_node --ros-args -p preset:=white
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


IMAGE_TOPIC = '/camera/high/image_raw'
PRESET = 'white_auto_exposure'

WIN_CONTROLS = 'hsv_ycrcb_lab_tuner_controls'
WIN_ORIGINAL = 'hsv_ycrcb_lab_tuner_original'
WIN_HSV_MASK = 'hsv_ycrcb_lab_tuner_hsv_mask'
WIN_YCRCB_MASK = 'hsv_ycrcb_lab_tuner_ycrcb_mask'
WIN_LAB_MASK = 'hsv_ycrcb_lab_tuner_lab_mask'
WIN_AND_MASK = 'hsv_ycrcb_lab_tuner_and_mask'
WIN_RESULT = 'hsv_ycrcb_lab_tuner_result'

# HSV/YCrCb는 lane_offset/config/color_profiles.yaml의 기본 범위를
# 튜닝 시작점으로 사용한다. LAB 기준값은 아직 없으므로 전체 범위로 시작한다.
# OpenCV 채널 순서는 HSV=(H, S, V), YCrCb=(Y, Cr, Cb), LAB=(L, A, B)이다.
PRESETS = {
    # rosbag2_2026_08_05-11_29_45의 /camera/high/image_raw 2,075프레임으로
    # 검증한 실내 자동 노출/화이트밸런스 변화 대응 흰 차선 범위다.
    'white_auto_exposure': {
        'hsv': {'h': (0, 179), 's': (0, 140), 'v': (155, 255)},
        'ycrcb': {'y': (150, 255), 'cr': (80, 134), 'cb': (110, 158)},
        'lab': {'l': (160, 255), 'a': (106, 134), 'b': (102, 134)},
    },
    'white': {
        'hsv': {'h': (0, 179), 's': (0, 44), 'v': (161, 255)},
        'ycrcb': {'y': (135, 255), 'cr': (81, 165), 'cb': (122, 170)},
        'lab': {'l': (0, 255), 'a': (0, 255), 'b': (0, 255)},
    },
    'green': {
        'hsv': {'h': (31, 60), 's': (110, 200), 'v': (0, 255)},
        'ycrcb': {'y': (0, 255), 'cr': (0, 255), 'cb': (0, 255)},
        'lab': {'l': (0, 255), 'a': (0, 255), 'b': (0, 255)},
    },
    'light_gray': {
        'hsv': {'h': (56, 179), 's': (0, 64), 'v': (119, 165)},
        'ycrcb': {'y': (0, 163), 'cr': (0, 255), 'cb': (0, 255)},
        'lab': {'l': (0, 255), 'a': (0, 255), 'b': (0, 255)},
    },
    'dark_gray': {
        'hsv': {'h': (0, 179), 's': (0, 102), 'v': (0, 144)},
        'ycrcb': {'y': (0, 144), 'cr': (0, 255), 'cb': (0, 142)},
        'lab': {'l': (0, 255), 'a': (0, 255), 'b': (0, 255)},
    },
    'full': {
        'hsv': {'h': (0, 179), 's': (0, 255), 'v': (0, 255)},
        'ycrcb': {'y': (0, 255), 'cr': (0, 255), 'cb': (0, 255)},
        'lab': {'l': (0, 255), 'a': (0, 255), 'b': (0, 255)},
    },
}


def make_masks(frame, hsv_bounds, ycrcb_bounds, lab_bounds):
    """HSV, YCrCb, LAB, 세 마스크의 AND 마스크를 만든다."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    hsv_mask = cv2.inRange(hsv, hsv_bounds[0], hsv_bounds[1])
    ycrcb_mask = cv2.inRange(ycrcb, ycrcb_bounds[0], ycrcb_bounds[1])
    lab_mask = cv2.inRange(lab, lab_bounds[0], lab_bounds[1])
    and_mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)
    and_mask = cv2.bitwise_and(and_mask, lab_mask)
    return hsv_mask, ycrcb_mask, lab_mask, and_mask


class HsvYCrCbLabTunerNode(Node):
    """HSV/YCrCb/LAB 임계값과 교집 마스크를 실시간으로 보여준다."""

    def __init__(self):
        super().__init__('hsv_ycrcb_lab_tuner_node')

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
        self.active_trackbar = None
        self.input_buffer = ''
        self.last_input_status = ''
        self.trackbar_maximum = {}
        self.setup_windows(PRESETS[preset_name])

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(Image, self.image_topic, self.image_callback, qos)
        self.create_timer(1.0 / 30.0, self.tick)

        self.get_logger().info(
            f'Subscribing {self.image_topic}. Adjust HSV, Y/Cr/Cb and L/A/B '
            f'trackbars in "{WIN_CONTROLS}" and watch "{WIN_AND_MASK}".'
        )

    def setup_windows(self, preset):
        cv2.namedWindow(WIN_CONTROLS, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_CONTROLS, 520, 760)
        for window in (
            WIN_ORIGINAL,
            WIN_HSV_MASK,
            WIN_YCRCB_MASK,
            WIN_LAB_MASK,
            WIN_AND_MASK,
            WIN_RESULT,
        ):
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        def select_trackbar(name):
            def selected(_):
                if self.active_trackbar != name:
                    self.input_buffer = ''
                self.active_trackbar = name
                self.last_input_status = ''

            return selected

        hsv = preset['hsv']
        ycrcb = preset['ycrcb']
        lab = preset['lab']
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
            ('L min', lab['l'][0], 255),
            ('L max', lab['l'][1], 255),
            ('A min', lab['a'][0], 255),
            ('A max', lab['a'][1], 255),
            ('B min', lab['b'][0], 255),
            ('B max', lab['b'][1], 255),
        )
        for name, initial, maximum in trackbars:
            self.trackbar_maximum[name] = maximum
            cv2.createTrackbar(
                name,
                WIN_CONTROLS,
                initial,
                maximum,
                select_trackbar(name),
            )

        # createTrackbar() may invoke callbacks while creating the controls.
        self.active_trackbar = 'H min'
        self.input_buffer = ''
        self.last_input_status = ''

    def draw_control_status(self):
        """Show numeric-entry help and the value currently being typed."""
        panel = np.full((105, 520, 3), 35, dtype=np.uint8)
        value = self.input_buffer if self.input_buffer else '-'
        lines = (
            'Move a slider to select it, then type a number.',
            f'Selected: {self.active_trackbar}    Input: {value}',
            self.last_input_status or 'Enter: apply    Backspace: erase    Esc: clear',
        )
        for index, text in enumerate(lines, start=1):
            cv2.putText(
                panel,
                text,
                (10, index * 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
        cv2.imshow(WIN_CONTROLS, panel)

    def handle_key(self, key):
        """Apply keyboard numeric input to the most recently selected slider."""
        if ord('0') <= key <= ord('9'):
            self.input_buffer += chr(key)
            self.last_input_status = ''
            return

        if key in (8, 127):  # Backspace/Delete on common OpenCV backends.
            self.input_buffer = self.input_buffer[:-1]
            self.last_input_status = ''
            return

        if key == 27:  # Escape
            self.input_buffer = ''
            self.last_input_status = 'Input cleared'
            return

        if key not in (10, 13) or not self.input_buffer:
            return

        maximum = self.trackbar_maximum[self.active_trackbar]
        requested = int(self.input_buffer)
        applied = min(requested, maximum)
        cv2.setTrackbarPos(self.active_trackbar, WIN_CONTROLS, applied)
        if applied == requested:
            self.last_input_status = f'Applied {self.active_trackbar} = {applied}'
        else:
            self.last_input_status = (
                f'Applied {self.active_trackbar} = {applied} (max {maximum})'
            )
        self.input_buffer = ''

    def image_callback(self, msg):
        self.frame = self.to_bgr(msg)

    @staticmethod
    def trackbar_bounds():
        def bounds(channels):
            minimum = tuple(
                cv2.getTrackbarPos(f'{channel} min', WIN_CONTROLS)
                for channel in channels
            )
            maximum = tuple(
                cv2.getTrackbarPos(f'{channel} max', WIN_CONTROLS)
                for channel in channels
            )
            return minimum, maximum

        return bounds(('H', 'S', 'V')), bounds(('Y', 'Cr', 'Cb')), bounds(('L', 'A', 'B'))

    def tick(self):
        self.draw_control_status()
        if self.frame is None:
            self.handle_key(cv2.waitKey(1) & 0xFF)
            return

        hsv_bounds, ycrcb_bounds, lab_bounds = self.trackbar_bounds()
        hsv_mask, ycrcb_mask, lab_mask, and_mask = make_masks(
            self.frame, hsv_bounds, ycrcb_bounds, lab_bounds
        )
        result = cv2.bitwise_and(self.frame, self.frame, mask=and_mask)

        hsv_pixels = int(cv2.countNonZero(hsv_mask))
        ycrcb_pixels = int(cv2.countNonZero(ycrcb_mask))
        lab_pixels = int(cv2.countNonZero(lab_mask))
        and_pixels = int(cv2.countNonZero(and_mask))
        ratio = and_pixels / and_mask.size

        (h_min, s_min, v_min), (h_max, s_max, v_max) = hsv_bounds
        (y_min, cr_min, cb_min), (y_max, cr_max, cb_max) = ycrcb_bounds
        (l_min, a_min, b_min), (l_max, a_max, b_max) = lab_bounds
        overlay = self.frame.copy()
        lines = (
            (
                f'H[{h_min},{h_max}] S[{s_min},{s_max}] V[{v_min},{v_max}] '
                f'px={hsv_pixels}',
                (0, 255, 0),
            ),
            (
                f'Y[{y_min},{y_max}] Cr[{cr_min},{cr_max}] '
                f'Cb[{cb_min},{cb_max}] px={ycrcb_pixels}',
                (0, 255, 255),
            ),
            (
                f'L[{l_min},{l_max}] A[{a_min},{a_max}] '
                f'B[{b_min},{b_max}] px={lab_pixels}',
                (255, 0, 255),
            ),
            (f'AND px={and_pixels} ratio={ratio:.3f}', (255, 255, 0)),
        )
        for index, (text, color) in enumerate(lines, start=1):
            cv2.putText(
                overlay,
                text,
                (10, index * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(WIN_ORIGINAL, overlay)
        cv2.imshow(WIN_HSV_MASK, hsv_mask)
        cv2.imshow(WIN_YCRCB_MASK, ycrcb_mask)
        cv2.imshow(WIN_LAB_MASK, lab_mask)
        cv2.imshow(WIN_AND_MASK, and_mask)
        cv2.imshow(WIN_RESULT, result)
        self.handle_key(cv2.waitKey(1) & 0xFF)

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
    node = HsvYCrCbLabTunerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
