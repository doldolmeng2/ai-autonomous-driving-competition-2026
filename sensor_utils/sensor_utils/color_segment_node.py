"""color_segment_node.

카메라 토픽을 구독해서 원본 이미지에 4가지 색상 클래스(흰색/밝은 회색/
짙은 회색/초록색) 마스크를 각각 원본 이미지 기준으로 병렬 계산한 뒤,
1->2->3->4 순서로 겹쳐 그려서 5색(4개 클래스 + 검정 배경)짜리 분류
이미지 한 장을 만들어 보여준다.

각 클래스는 hsv_tuner_node / ycrcb_hsv_tuner_node로 먼저 값을 찾은 뒤
CLASSES 에 고정값으로 박아넣는 방식이라 트랙바는 없다. 값이 바뀌면
아래 CLASSES 딕셔너리만 수정하면 된다.

칠하는 순서(1->2->3->4)가 그대로 우선순위다: 뒤 클래스가 이미 칠해진
픽셀이라도 덮어쓴다. 어느 클래스에도 안 걸린 픽셀은 검정으로 남는다.

사용 예:
    ros2 run sensor_utils color_segment_node --ros-args -p image_topic:=/camera/left/image_raw
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

# ============================================================================
# 파라미터 기본값 - 튜닝은 대부분 여기서만 하면 된다.
# hsv_tuner_node / ycrcb_hsv_tuner_node 로 찾은 값을 그대로 옮겨 적는다.
# 'ycrcb' 가 None 인 클래스는 HSV 마스크만 사용한다.
# 여러 클래스에 걸리는 픽셀은 CLASSES 뒤쪽 항목이 앞쪽 항목을 덮어쓴다.
# ============================================================================
IMAGE_TOPIC = '/camera/high/image_raw'

BACKGROUND_COLOR_BGR = (0, 0, 0)

CLASSES = [
    {
        'name': 'white',
        'color_bgr': (255, 255, 255),
        'hsv': {'h': (0, 179), 's': (0, 44), 'v': (178, 255)},
        'ycrcb': {'y': (135, 255), 'cr': (81, 165), 'cb': (122, 170)},
    },
    {
        'name': 'light_gray',
        'color_bgr': (211, 211, 211),
        'hsv': {'h': (56, 179), 's': (0, 64), 'v': (119, 165)},
        'ycrcb': {'y': (0, 163), 'cr': (0, 255), 'cb': (0, 255)},
    },
    {
        'name': 'dark_gray',
        'color_bgr': (105, 105, 105),
        'hsv': {'h': (0, 179), 's': (0, 102), 'v': (0, 144)},
        'ycrcb': {'y': (0, 144), 'cr': (0, 255), 'cb': (0, 142)},
    },
    {
        'name': 'green',
        'color_bgr': (0, 255, 0),
        'hsv': {'h': (31, 60), 's': (48, 200), 'v': (0, 255)},
        'ycrcb': None,
    },
]

WIN_ORIGINAL = 'color_segment_original'
WIN_RESULT = 'color_segment_result'


def class_mask(hsv, ycrcb, cls):
    h_lo, h_hi = cls['hsv']['h']
    s_lo, s_hi = cls['hsv']['s']
    v_lo, v_hi = cls['hsv']['v']
    mask = cv2.inRange(hsv, (h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi))

    if cls['ycrcb'] is not None:
        y_lo, y_hi = cls['ycrcb']['y']
        cr_lo, cr_hi = cls['ycrcb']['cr']
        cb_lo, cb_hi = cls['ycrcb']['cb']
        ycrcb_mask = cv2.inRange(
            ycrcb, (y_lo, cr_lo, cb_lo), (y_hi, cr_hi, cb_hi)
        )
        mask = cv2.bitwise_and(mask, ycrcb_mask)

    return mask


class ColorSegmentNode(Node):
    """4개 색상 클래스 마스크를 순서대로 덮어 그려 5색 분류 이미지를 보여준다."""

    def __init__(self):
        super().__init__('color_segment_node')

        self.declare_parameter('image_topic', IMAGE_TOPIC)
        self.image_topic = self.get_parameter('image_topic').value

        self.frame = None

        cv2.namedWindow(WIN_ORIGINAL, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_RESULT, cv2.WINDOW_NORMAL)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(Image, self.image_topic, self.image_callback, qos)
        self.timer = self.create_timer(1.0 / 30.0, self.tick)

        legend = ', '.join(f"{i + 1}:{c['name']}" for i, c in enumerate(CLASSES))
        self.get_logger().info(
            f'Subscribing {self.image_topic}. Paint order (later overwrites earlier): '
            f'{legend}, background:black'
        )

    def image_callback(self, msg):
        self.frame = self.to_bgr(msg)

    def tick(self):
        if self.frame is None:
            cv2.waitKey(1)
            return

        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2YCrCb)

        result = np.full_like(self.frame, BACKGROUND_COLOR_BGR, dtype=np.uint8)
        for cls in CLASSES:
            mask = class_mask(hsv, ycrcb, cls)
            result[mask > 0] = cls['color_bgr']

        cv2.imshow(WIN_ORIGINAL, self.frame)
        cv2.imshow(WIN_RESULT, result)
        cv2.waitKey(1)

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
    node = ColorSegmentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
