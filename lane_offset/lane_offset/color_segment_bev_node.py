"""color_segment_bev_node.

sensor_utils/color_segment_node.py 의 색상 분류(흰색/밝은 회색/짙은 회색/
초록색 -> 5색 이미지) 알고리즘을 그대로 가져와서 쓰고, 그 결과를 BEV
(Bird's Eye View)로 펼쳐서 보여준다.

BEV 변환 방식(IPM, Inverse Perspective Mapping):
    원본 이미지에서 y가 BEV_Y_TOP_RATIO~BEV_Y_BOTTOM_RATIO 구간(위쪽=먼 곳,
    아래쪽=가까운 곳)에 해당하는 사다리꼴(윗변이 좁고 아랫변이 넓음)을
    잘라서 직사각형으로 펴는 방식. 사다리꼴 네 꼭짓점(src)을 결과
    사각형 네 꼭짓점(dst)에 대응시켜 cv2.getPerspectiveTransform +
    warpPerspective로 계산한다.

    y 범위(50%~80%)는 사용자가 지정한 값이고, 윗변/아랫변 폭 비율은
    아직 임의값이라 실제 화면(원본 창에 그려지는 사다리꼴)을 보면서
    BEV_TOP_WIDTH_RATIO / BEV_BOTTOM_WIDTH_RATIO 를 맞춰야 한다.

    분류(5색) 결과를 얻은 뒤에 그 이미지를 BEV로 워프한다(원본을 먼저
    워프하는 게 아님). 색이 섞이면 안 되므로 워프 보간은 INTER_NEAREST.

차선 추적 등 뒤 알고리즘은 아직 없음 - 이후 여기에 추가 예정.

사용 예:
    ros2 run lane_offset color_segment_bev_node --ros-args -p image_topic:=/camera/left/image_raw
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
IMAGE_TOPIC = '/camera/left/image_raw'

# ---- 색상 분류: sensor_utils/color_segment_node.py 와 동일 (그대로 가져옴) ----
# hsv_tuner_node / ycrcb_hsv_tuner_node 로 찾은 값. 'ycrcb'가 None이면 HSV만 사용.
# 여러 클래스에 걸리면 리스트 뒤쪽 항목이 앞쪽 항목을 덮어쓴다.
BACKGROUND_COLOR_BGR = (0, 0, 0)
CLASSES = [
    {
        'name': 'white',
        'color_bgr': (255, 255, 255),
        'hsv': {'h': (0, 179), 's': (0, 44), 'v': (178, 255)},
        'ycrcb': {'y': (175, 255), 'cr': (81, 165), 'cb': (122, 170)},
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
        'hsv': {'h': (0, 179), 's': (0, 102), 'v': (0, 121)},
        'ycrcb': {'y': (0, 110), 'cr': (0, 255), 'cb': (0, 142)},
    },
    {
        'name': 'green',
        'color_bgr': (0, 255, 0),
        'hsv': {'h': (40, 56), 's': (48, 200), 'v': (0, 255)},
        'ycrcb': None,
    },
]

# ---- BEV(Bird's Eye View) 변환 ----
# 사다리꼴 윗변/아랫변 y (이미지 높이 비율). 사용자가 지정한 50%~80% 구간.
BEV_Y_TOP_RATIO = 0.5
BEV_Y_BOTTOM_RATIO = 0.8
# 사다리꼴 윗변/아랫변 폭 (이미지 전체 폭 대비, 중앙 정렬). 아직 임의값 -
# 원본 창에 그려지는 사다리꼴을 보면서 맞출 것. 윗변(먼 곳)을 아랫변보다
# 좁게(안쪽으로) 잡아야 원근이 사라진 BEV가 나온다.
BEV_TOP_WIDTH_RATIO = 0.4
BEV_BOTTOM_WIDTH_RATIO = 1.0
# BEV 출력 이미지 크기 (임의값)
BEV_OUTPUT_WIDTH = 400
BEV_OUTPUT_HEIGHT = 400

WIN_ORIGINAL = 'color_segment_bev_original'
WIN_SEGMENTED = 'color_segment_bev_segmented'
WIN_BEV = 'color_segment_bev_result'


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


class ColorSegmentBevNode(Node):
    """color_segment 5색 분류 결과를 BEV로 펼쳐서 보여준다."""

    def __init__(self):
        super().__init__('color_segment_bev_node')

        self.declare_parameter('image_topic', IMAGE_TOPIC)
        self.declare_parameter('bev_y_top_ratio', BEV_Y_TOP_RATIO)
        self.declare_parameter('bev_y_bottom_ratio', BEV_Y_BOTTOM_RATIO)
        self.declare_parameter('bev_top_width_ratio', BEV_TOP_WIDTH_RATIO)
        self.declare_parameter('bev_bottom_width_ratio', BEV_BOTTOM_WIDTH_RATIO)
        self.declare_parameter('bev_output_width', BEV_OUTPUT_WIDTH)
        self.declare_parameter('bev_output_height', BEV_OUTPUT_HEIGHT)

        self.image_topic = self.get_parameter('image_topic').value
        self.bev_y_top_ratio = float(self.get_parameter('bev_y_top_ratio').value)
        self.bev_y_bottom_ratio = float(
            self.get_parameter('bev_y_bottom_ratio').value
        )
        self.bev_top_width_ratio = float(
            self.get_parameter('bev_top_width_ratio').value
        )
        self.bev_bottom_width_ratio = float(
            self.get_parameter('bev_bottom_width_ratio').value
        )
        self.bev_output_width = int(self.get_parameter('bev_output_width').value)
        self.bev_output_height = int(self.get_parameter('bev_output_height').value)

        self.frame = None

        cv2.namedWindow(WIN_ORIGINAL, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_SEGMENTED, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_BEV, cv2.WINDOW_NORMAL)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(Image, self.image_topic, self.image_callback, qos)
        self.timer = self.create_timer(1.0 / 30.0, self.tick)

        self.get_logger().info(
            f'Subscribing {self.image_topic}. '
            f'BEV y=[{self.bev_y_top_ratio},{self.bev_y_bottom_ratio}] '
            f'top_width={self.bev_top_width_ratio} bottom_width={self.bev_bottom_width_ratio}'
        )

    def image_callback(self, msg):
        self.frame = self.to_bgr(msg)

    def tick(self):
        if self.frame is None:
            cv2.waitKey(1)
            return

        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2YCrCb)

        segmented = np.full_like(self.frame, BACKGROUND_COLOR_BGR, dtype=np.uint8)
        for cls in CLASSES:
            mask = class_mask(hsv, ycrcb, cls)
            segmented[mask > 0] = cls['color_bgr']

        src_points, dst_points = self.compute_bev_points(self.frame.shape)
        transform = cv2.getPerspectiveTransform(src_points, dst_points)
        bev = cv2.warpPerspective(
            segmented,
            transform,
            (self.bev_output_width, self.bev_output_height),
            flags=cv2.INTER_NEAREST,
        )

        overlay = self.frame.copy()
        cv2.polylines(
            overlay, [src_points.astype(np.int32)], True, (0, 165, 255), 2
        )

        cv2.imshow(WIN_ORIGINAL, overlay)
        cv2.imshow(WIN_SEGMENTED, segmented)
        cv2.imshow(WIN_BEV, bev)
        cv2.waitKey(1)

    # ======================================================================
    # BEV 사다리꼴(src) <-> 결과 사각형(dst) 꼭짓점. 순서: TL, TR, BR, BL.
    # ======================================================================
    def compute_bev_points(self, frame_shape):
        height, width = frame_shape[:2]
        y_top = height * self.bev_y_top_ratio
        y_bottom = height * self.bev_y_bottom_ratio
        cx = width / 2.0
        top_half_width = width * self.bev_top_width_ratio / 2.0
        bottom_half_width = width * self.bev_bottom_width_ratio / 2.0

        src_points = np.float32([
            [cx - top_half_width, y_top],
            [cx + top_half_width, y_top],
            [cx + bottom_half_width, y_bottom],
            [cx - bottom_half_width, y_bottom],
        ])
        dst_points = np.float32([
            [0, 0],
            [self.bev_output_width, 0],
            [self.bev_output_width, self.bev_output_height],
            [0, self.bev_output_height],
        ])
        return src_points, dst_points

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
    node = ColorSegmentBevNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
