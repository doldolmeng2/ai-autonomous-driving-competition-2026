"""ycrcb_hsv_tuner_node.

카메라 토픽(bag 재생이든 실제 카메라든 상관없음)을 구독해서 트랙바로
HSV(H/S/V)와 YCrCb(Y/Cr/Cb) 6개씩, 총 12개 값을 모두 조절하며
두 마스크를 AND로 합친 결과를 실시간으로 보여준다.
HSV가 잘 잡는 부분과 YCrCb가 잘 잡는 부분을 각각 튜닝한 뒤 교집합으로
합쳐서, 어느 한쪽만으로는 안 걸러지는 노이즈를 서로 보완할 때 사용.

사용 예:
    ros2 run sensor_utils ycrcb_hsv_tuner_node --ros-args -p image_topic:=/camera/high/image_raw
    ros2 run sensor_utils ycrcb_hsv_tuner_node --ros-args -p preset:=green
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

# ============================================================================
# 파라미터 기본값 - 튜닝은 대부분 여기서만 하면 된다.
# (전부 ROS 파라미터로도 선언되므로 --ros-args -p 로 실행 중 덮어쓰기도 가능)
# ============================================================================
IMAGE_TOPIC = '/camera/high/image_raw'
PRESET = 'white'

WIN_CONTROLS = 'ycrcb_hsv_tuner_controls'
WIN_VALUES = 'ycrcb_hsv_tuner_values'
WIN_ORIGINAL = 'ycrcb_hsv_tuner_original'
WIN_HSV_MASK = 'ycrcb_hsv_tuner_hsv_mask'
WIN_YCRCB_MASK = 'ycrcb_hsv_tuner_ycrcb_mask'
WIN_COMBINED_MASK = 'ycrcb_hsv_tuner_combined_mask'
WIN_RESULT = 'ycrcb_hsv_tuner_result'

# lane_offset/sensor_utils 의 color_segment(_bev)_node.py CLASSES 와 동일한 값
HSV_PRESETS = {
    'white': {'h': (0, 179), 's': (0, 44), 'v': (178, 255)},
    'light_gray': {'h': (56, 179), 's': (0, 64), 'v': (119, 165)},
    'dark_gray': {'h': (0, 179), 's': (0, 102), 'v': (0, 144)},
    'green': {'h': (31, 60), 's': (48, 200), 'v': (0, 255)},
    'full': {'h': (0, 179), 's': (0, 255), 'v': (0, 255)},
}

# YCrCb는 아직 lane_offset에서 쓰지 않으므로 대략적인 시작값만 제공.
# Y=밝기, Cr/Cb=색차이며 무채색(흰색)일수록 Cr/Cb가 128 근처에 몰린다.
YCRCB_PRESETS = {
    'white': {'y': (135, 255), 'cr': (81, 165), 'cb': (122, 170)},
    'light_gray': {'y': (0, 163), 'cr': (0, 255), 'cb': (0, 255)},
    'dark_gray': {'y': (0, 144), 'cr': (0, 255), 'cb': (0, 142)},
    'green': {'y': (0, 255), 'cr': (0, 140), 'cb': (0, 140)},
    'full': {'y': (0, 255), 'cr': (0, 255), 'cb': (0, 255)},
}


class YCrCbHsvTunerNode(Node):
    """트랙바로 HSV 6개 + YCrCb 6개 값을 모두 조절하며 AND로 합쳐진 마스크/결과를 보여준다."""

    def __init__(self):
        super().__init__('ycrcb_hsv_tuner_node')

        self.declare_parameter('image_topic', IMAGE_TOPIC)
        self.declare_parameter('preset', PRESET)

        self.image_topic = self.get_parameter('image_topic').value
        preset_name = self.get_parameter('preset').value
        hsv_preset = HSV_PRESETS.get(preset_name, HSV_PRESETS['green'])
        ycrcb_preset = YCRCB_PRESETS.get(preset_name, YCRCB_PRESETS['green'])

        self.frame = None
        self.setup_windows(hsv_preset, ycrcb_preset)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(Image, self.image_topic, self.image_callback, qos)
        self.timer = self.create_timer(1.0 / 30.0, self.tick)

        self.get_logger().info(
            f'Subscribing {self.image_topic}. Adjust trackbars in "{WIN_CONTROLS}" '
            'and watch the combined (HSV AND YCrCb) mask window.'
        )

    def setup_windows(self, hsv_preset, ycrcb_preset):
        cv2.namedWindow(WIN_CONTROLS, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_CONTROLS, 420, 480)
        # Qt 트랙바 라벨 텍스트가 시스템 테마에 따라 안 보이는 경우가 있어서,
        # 값을 직접 putText로 그려서 보여주는 창을 별도로 둔다.
        cv2.namedWindow(WIN_VALUES, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_VALUES, 260, 340)
        cv2.namedWindow(WIN_ORIGINAL, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_HSV_MASK, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_YCRCB_MASK, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_COMBINED_MASK, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_RESULT, cv2.WINDOW_NORMAL)

        def nothing(_):
            pass

        h_lo, h_hi = hsv_preset['h']
        s_lo, s_hi = hsv_preset['s']
        v_lo, v_hi = hsv_preset['v']
        cv2.createTrackbar('H min', WIN_CONTROLS, h_lo, 179, nothing)
        cv2.createTrackbar('H max', WIN_CONTROLS, h_hi, 179, nothing)
        cv2.createTrackbar('S min', WIN_CONTROLS, s_lo, 255, nothing)
        cv2.createTrackbar('S max', WIN_CONTROLS, s_hi, 255, nothing)
        cv2.createTrackbar('V min', WIN_CONTROLS, v_lo, 255, nothing)
        cv2.createTrackbar('V max', WIN_CONTROLS, v_hi, 255, nothing)

        y_lo, y_hi = ycrcb_preset['y']
        cr_lo, cr_hi = ycrcb_preset['cr']
        cb_lo, cb_hi = ycrcb_preset['cb']
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

        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))

        ycrcb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2YCrCb)
        ycrcb_mask = cv2.inRange(
            ycrcb, (y_min, cr_min, cb_min), (y_max, cr_max, cb_max)
        )

        values_view = self.render_values(
            h_min, h_max, s_min, s_max, v_min, v_max,
            y_min, y_max, cr_min, cr_max, cb_min, cb_max,
        )
        cv2.imshow(WIN_VALUES, values_view)

        combined_mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)
        result = cv2.bitwise_and(self.frame, self.frame, mask=combined_mask)

        pixel_count = int(np.count_nonzero(combined_mask))
        ratio = pixel_count / combined_mask.size

        overlay = self.frame.copy()
        cv2.putText(
            overlay,
            f'HSV H[{h_min},{h_max}] S[{s_min},{s_max}] V[{v_min},{v_max}]',
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f'YCrCb Y[{y_min},{y_max}] Cr[{cr_min},{cr_max}] Cb[{cb_min},{cb_max}]',
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f'combined px={pixel_count} ratio={ratio:.3f}',
            (10, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 128, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WIN_ORIGINAL, overlay)
        cv2.imshow(WIN_HSV_MASK, hsv_mask)
        cv2.imshow(WIN_YCRCB_MASK, ycrcb_mask)
        cv2.imshow(WIN_COMBINED_MASK, combined_mask)
        cv2.imshow(WIN_RESULT, result)
        cv2.waitKey(1)

    # ======================================================================
    # 트랙바 값을 putText로 그린 이미지를 만든다. Qt 트랙바 라벨이 시스템
    # 테마 때문에 안 보이는 환경에서도 값을 확인할 수 있게 하기 위함.
    # ======================================================================
    def render_values(self, h_min, h_max, s_min, s_max, v_min, v_max,
                       y_min, y_max, cr_min, cr_max, cb_min, cb_max):
        rows = [
            ('H', h_min, h_max), ('S', s_min, s_max), ('V', v_min, v_max),
            ('Y', y_min, y_max), ('Cr', cr_min, cr_max), ('Cb', cb_min, cb_max),
        ]
        view = np.zeros((260, 260, 3), dtype=np.uint8)
        for i, (name, lo, hi) in enumerate(rows):
            cv2.putText(
                view, f'{name}: [{lo}, {hi}]', (10, 30 + i * 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )
        return view

    # ======================================================================
    # YUYV -> BGR 변환 (sensor_utils/camera_viewer_node.py 와 동일한 방식)
    # ======================================================================
    def to_bgr(self, msg):
        data = np.frombuffer(msg.data, dtype=np.uint8)
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

        self.get_logger().warn(
            f'Unsupported camera encoding: {msg.encoding}', throttle_duration_sec=5.0
        )
        return None

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YCrCbHsvTunerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
