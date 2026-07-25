"""color_segment_bev_node.

sensor_utils/color_segment_node.py 의 색상 분류(흰색/밝은 회색/짙은 회색/
초록색 -> 5색 이미지) 알고리즘을 그대로 가져와서 쓰고, 그 결과를 BEV
(Bird's Eye View)로 펼쳐서 보여준다.

BEV 변환 방식(IPM, Inverse Perspective Mapping):
    원본 이미지에서 y가 BEV_Y_TOP_RATIO~BEV_Y_BOTTOM_RATIO 구간(위쪽=먼 곳,
    아래쪽=가까운 곳)에 해당하는 순수 직사각형(좌우 폭 변화 없음)을 그대로
    잘라서(src), BEV 출력 캔버스(dst)에는 사다리꼴로 눌러 넣는다. dst
    윗변(y=0)은 항상 출력 폭 그대로(원본과 동일한 폭)이고, 아랫변으로
    갈수록 하나의 변환으로 연속적으로 좁아진다(구간을 나누지 않음).
    src 네 꼭짓점을 dst 네 꼭짓점에 대응시켜 cv2.getPerspectiveTransform +
    warpPerspective로 계산한다.

    y 범위(50%~80%)는 사용자가 지정한 값이고, dst 아랫변 폭 비율은
    아직 임의값이라 실제 BEV 결과 창을 보면서
    BEV_BOTTOM_WIDTH_RATIO 를 맞춰야 한다.

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
IMAGE_TOPIC = '/camera/high/image_raw'

# ---- 색상 분류: sensor_utils/color_segment_node.py 와 동일 (그대로 가져옴) ----
# hsv_tuner_node / ycrcb_hsv_tuner_node 로 찾은 값. 'ycrcb'가 None이면 HSV만 사용.
# 여러 클래스에 걸리면 리스트 뒤쪽 항목이 앞쪽 항목을 덮어쓴다.
BACKGROUND_COLOR_BGR = (0, 0, 0)
CLASSES = [
    {
        'name': 'white',
        'color_bgr': (255, 255, 255),
        'hsv': {'h': (0, 179), 's': (0, 44), 'v': (161, 255)},
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

# ---- BEV(Bird's Eye View) 변환 ----
# 원본에서 그대로 잘라올 직사각형 영역의 y (이미지 높이 비율). 사용자가 지정한 구간.
BEV_Y_TOP_RATIO = 0.4
BEV_Y_BOTTOM_RATIO = 1.0
# BEV 출력 캔버스의 윗변/아랫변 폭 (출력 폭 대비, 중앙 정렬). 윗변은 항상
# 출력 폭 그대로(1.0)로 두고, 아랫변만 안쪽으로 눌러 좁힌다. 아직 임의값 -
# BEV 결과 창을 보면서 BEV_BOTTOM_WIDTH_RATIO를 맞출 것.
BEV_TOP_WIDTH_RATIO = 1.0
BEV_BOTTOM_WIDTH_RATIO = 0.7
# BEV 출력 이미지 크기. 높이가 0이면 실제 변환할 y 구간의 종횡비를
# 유지하도록 프레임마다 자동 계산한다.
# 예: 입력 640x480, y=0.4~1.0 -> ROI 640x288 -> 출력 400x180.
BEV_OUTPUT_WIDTH = 400
BEV_OUTPUT_HEIGHT = 0

# ---- Sliding window 시작점 검출 ----
# BEV를 아래에서 위로 훑으므로 점선이 최하단에 없어도 첫 유효 구간을 찾는다.
# 한 번에 검사하는 시작 윈도우의 세로 높이(px).
START_WINDOW_HEIGHT = 24

# 아래에서 위로 다음 후보 영역을 검사할 때 이동하는 세로 간격(px).
# START_WINDOW_HEIGHT보다 작으면 검사 영역이 서로 겹친다.
START_WINDOW_STEP = 12

# 흰색 후보 바로 좌우에서 배경색을 검사하는 각 영역의 가로 폭(px).
SIDE_WINDOW_WIDTH = 15

# 차선 후보로 인정할 흰색 연속 구간의 최소 가로 두께(px).
WHITE_WIDTH_MIN = 10

# 차선 후보로 인정할 흰색 연속 구간의 최대 가로 두께(px).
# 너무 두껍거나 좌우로 긴 흰색 영역을 제외하는 1차 조건이다.
WHITE_WIDTH_MAX = 19

# 한 x열을 흰색으로 판단하기 위한 최소 흰색 세로 점유율.
# 예: 높이 24, 비율 0.5이면 해당 열에 흰색이 12px 이상 있어야 한다.
WHITE_COLUMN_MIN_RATIO = 0.5

# 좌우 검사 영역을 요구 색상(dark/light gray 또는 green)으로 인정할
# 최소 픽셀 점유율.
SIDE_COLOR_MIN_RATIO = 0.25

# 좌우 검사 영역에서 허용할 흰색 픽셀의 최대 점유율.
# 가로로 긴 흰 선이 좌우 검사 영역까지 이어지는 경우를 제외한다.
SIDE_WHITE_MAX_RATIO = 0.1

# 위쪽 다음 윈도우가 이전 검출 중심에서 이동할 수 있는 최대 x거리(px).
SLIDING_MAX_X_DELTA = 30

# 중앙 점선의 빈 구간처럼 후보가 없는 윈도우를 연속으로 허용할 횟수.
SLIDING_MAX_MISSED = 4

# 좌·우 실선 종류를 나누는 BEV x좌표(px).
# 이 값보다 작으면 왼쪽 실선, 크면 오른쪽 실선 후보가 될 수 있다.
LANE_SPLIT_X = 240

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
        self.declare_parameter('start_window_height', START_WINDOW_HEIGHT)
        self.declare_parameter('start_window_step', START_WINDOW_STEP)
        self.declare_parameter('side_window_width', SIDE_WINDOW_WIDTH)
        self.declare_parameter('white_width_min', WHITE_WIDTH_MIN)
        self.declare_parameter('white_width_max', WHITE_WIDTH_MAX)
        self.declare_parameter(
            'white_column_min_ratio', WHITE_COLUMN_MIN_RATIO
        )
        self.declare_parameter('side_color_min_ratio', SIDE_COLOR_MIN_RATIO)
        self.declare_parameter('side_white_max_ratio', SIDE_WHITE_MAX_RATIO)
        self.declare_parameter('sliding_max_x_delta', SLIDING_MAX_X_DELTA)
        self.declare_parameter('sliding_max_missed', SLIDING_MAX_MISSED)
        self.declare_parameter('lane_split_x', LANE_SPLIT_X)

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
        self.start_window_height = int(
            self.get_parameter('start_window_height').value
        )
        self.start_window_step = int(
            self.get_parameter('start_window_step').value
        )
        self.side_window_width = int(
            self.get_parameter('side_window_width').value
        )
        self.white_width_min = int(
            self.get_parameter('white_width_min').value
        )
        self.white_width_max = int(
            self.get_parameter('white_width_max').value
        )
        self.white_column_min_ratio = float(
            self.get_parameter('white_column_min_ratio').value
        )
        self.side_color_min_ratio = float(
            self.get_parameter('side_color_min_ratio').value
        )
        self.side_white_max_ratio = float(
            self.get_parameter('side_white_max_ratio').value
        )
        self.sliding_max_x_delta = int(
            self.get_parameter('sliding_max_x_delta').value
        )
        self.sliding_max_missed = int(
            self.get_parameter('sliding_max_missed').value
        )
        self.lane_split_x = int(self.get_parameter('lane_split_x').value)

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

        bev_output_height = self.compute_bev_output_height(self.frame.shape)
        src_points, dst_points = self.compute_bev_points(
            self.frame.shape, bev_output_height
        )
        transform = cv2.getPerspectiveTransform(src_points, dst_points)
        bev = cv2.warpPerspective(
            segmented,
            transform,
            (self.bev_output_width, bev_output_height),
            flags=cv2.INTER_NEAREST,
        )
        bev_visualized = bev.copy()
        start_windows = self.find_start_windows(bev)
        sliding_windows = self.build_sliding_windows(bev, start_windows)
        self.draw_sliding_windows(bev_visualized, sliding_windows)

        overlay = self.frame.copy()
        cv2.polylines(
            overlay, [src_points.astype(np.int32)], True, (0, 165, 255), 2
        )

        cv2.imshow(WIN_ORIGINAL, overlay)
        cv2.imshow(WIN_SEGMENTED, segmented)
        cv2.imshow(WIN_BEV, bev_visualized)
        cv2.waitKey(1)

    # ======================================================================
    # Sliding window 시작점: 아래에서 위로 검색하고 차선 종류별 첫 후보만 사용.
    # ======================================================================
    def find_start_windows(self, bev):
        masks = self.make_bev_masks(bev)
        height, width = masks['white'].shape
        window_h = max(1, self.start_window_height)
        step = max(1, self.start_window_step)
        found = {}

        for y_bottom in range(height, 0, -step):
            y_top = max(0, y_bottom - window_h)
            candidates = self.find_window_candidates(
                masks, y_top, y_bottom, width
            )
            for candidate in candidates:
                lane_type = candidate['type']
                if lane_type not in found:
                    found[lane_type] = candidate

            if len(found) == 3:
                break

        return found

    def build_sliding_windows(self, bev, start_windows):
        """각 시작점에서 위로 올라가며 가장 가까운 동일 차선 후보를 잇는다."""
        masks = self.make_bev_masks(bev)
        _, width = masks['white'].shape
        window_h = max(1, self.start_window_height)
        step = max(1, self.start_window_step)
        max_delta = max(0, self.sliding_max_x_delta)
        max_missed = max(0, self.sliding_max_missed)
        tracks = {}

        for lane_type, start in start_windows.items():
            track = [start]
            previous_x = self.window_center_x(start)
            missed = 0
            y_bottom = start['white'][3] - step

            while y_bottom > 0:
                y_top = max(0, y_bottom - window_h)
                candidates = [
                    candidate
                    for candidate in self.find_window_candidates(
                        masks, y_top, y_bottom, width
                    )
                    if candidate['type'] == lane_type
                    and abs(self.window_center_x(candidate) - previous_x)
                    <= max_delta
                ]

                if candidates:
                    selected = min(
                        candidates,
                        key=lambda candidate: abs(
                            self.window_center_x(candidate) - previous_x
                        ),
                    )
                    track.append(selected)
                    previous_x = self.window_center_x(selected)
                    missed = 0
                else:
                    missed += 1
                    if missed > max_missed:
                        break

                y_bottom -= step

            tracks[lane_type] = track

        return tracks

    @staticmethod
    def make_bev_masks(bev):
        return {
            'white': np.all(bev == (255, 255, 255), axis=2),
            'dark_gray': np.all(bev == (105, 105, 105), axis=2),
            'light_gray': np.all(bev == (211, 211, 211), axis=2),
            'green': np.all(bev == (0, 255, 0), axis=2),
        }

    def find_window_candidates(self, masks, y_top, y_bottom, image_width):
        """한 y 구간에서 색상 배치 조건을 만족하는 모든 차선 후보를 찾는다."""
        window_height = y_bottom - y_top
        min_column_pixels = max(
            1, round(window_height * self.white_column_min_ratio)
        )
        active_columns = (
            np.count_nonzero(masks['white'][y_top:y_bottom], axis=0)
            >= min_column_pixels
        )
        candidates = []

        for x_left, x_right in self.contiguous_runs(active_columns):
            white_width = int(x_right - x_left)
            if not self.white_width_min <= white_width <= self.white_width_max:
                continue

            x_left = int(x_left)
            x_right = int(x_right)
            center_x = (x_left + x_right) // 2
            side_w = max(1, self.side_window_width)
            left_bounds = (max(0, x_left - side_w), x_left)
            right_bounds = (x_right, min(image_width, x_right + side_w))

            ratios = {
                f'{side}_{color}': self.region_ratio(
                    masks[color], y_top, y_bottom, *bounds
                )
                for side, bounds in (
                    ('left', left_bounds),
                    ('right', right_bounds),
                )
                for color in ('white', 'dark_gray', 'light_gray', 'green')
            }

            # 정상 차선의 좌우에는 배경색이 있어야 한다. 흰색이 옆
            # 영역까지 이어지면 가로선일 가능성이 높으므로 제외한다.
            if (
                ratios['left_white'] >= self.side_white_max_ratio
                or ratios['right_white'] >= self.side_white_max_ratio
            ):
                continue

            threshold = self.side_color_min_ratio
            lane_type = None
            if (
                ratios['left_dark_gray'] >= threshold
                and ratios['right_dark_gray'] >= threshold
            ):
                lane_type = 'center'
            elif (
                center_x < self.lane_split_x
                and ratios['left_light_gray'] >= threshold
                and ratios['right_dark_gray'] >= threshold
            ):
                lane_type = 'left'
            elif (
                center_x > self.lane_split_x
                and ratios['left_dark_gray'] >= threshold
                and ratios['right_green'] >= threshold
            ):
                lane_type = 'right'

            if lane_type is not None:
                candidates.append({
                    'type': lane_type,
                    'white': (x_left, y_top, x_right, y_bottom),
                    'left': (left_bounds[0], y_top, left_bounds[1], y_bottom),
                    'right': (
                        right_bounds[0], y_top, right_bounds[1], y_bottom
                    ),
                    'white_width': white_width,
                })

        return candidates

    @staticmethod
    def window_center_x(window):
        x_left, _, x_right, _ = window['white']
        return (x_left + x_right) // 2

    @staticmethod
    def contiguous_runs(active_columns):
        padded = np.pad(active_columns.astype(np.int8), (1, 1))
        edges = np.diff(padded)
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        return zip(starts, ends)

    @staticmethod
    def region_ratio(mask, y_top, y_bottom, x_left, x_right):
        if y_bottom <= y_top or x_right <= x_left:
            return 0.0
        return float(np.mean(mask[y_top:y_bottom, x_left:x_right]))

    @staticmethod
    def draw_sliding_windows(image, tracks):
        label_colors = {
            'left': (255, 128, 0),
            'center': (255, 128, 255),
            'right': (255, 0, 255),
        }
        for lane_type, windows in tracks.items():
            for window in windows:
                # 파랑=왼쪽 색상 검사, 노랑=흰색, 빨강=오른쪽 색상 검사
                ColorSegmentBevNode.draw_box(
                    image, window['left'], (255, 0, 0)
                )
                ColorSegmentBevNode.draw_box(
                    image, window['white'], (0, 255, 255)
                )
                ColorSegmentBevNode.draw_box(
                    image, window['right'], (0, 0, 255)
                )

            start = windows[0]
            x_left, y_top, _, _ = start['white']
            label_y = max(12, y_top - 4)
            label = f"{lane_type} n={len(windows)} w={start['white_width']}"
            cv2.putText(
                image,
                label,
                (x_left, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                label_colors[lane_type],
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def draw_box(image, bounds, color):
        x_left, y_top, x_right, y_bottom = bounds
        if x_right > x_left and y_bottom > y_top:
            cv2.rectangle(
                image,
                (x_left, y_top),
                (x_right - 1, y_bottom - 1),
                color,
                1,
            )

    # ======================================================================
    # 원본 직사각형(src) <-> BEV 사다리꼴(dst) 꼭짓점. 순서: TL, TR, BR, BL.
    # ======================================================================
    def compute_bev_output_height(self, frame_shape):
        """출력 높이 0이면 실제 BEV 입력 ROI의 종횡비에 맞춘다."""
        if self.bev_output_height > 0:
            return self.bev_output_height

        height, width = frame_shape[:2]
        roi_height = height * (
            self.bev_y_bottom_ratio - self.bev_y_top_ratio
        )
        return max(1, round(self.bev_output_width * roi_height / width))

    def compute_bev_points(self, frame_shape, bev_output_height):
        height, width = frame_shape[:2]
        y_top = height * self.bev_y_top_ratio
        y_bottom = height * self.bev_y_bottom_ratio

        # src: 폭 변화 없는 순수 직사각형 (y 구간, 좌우 전체 폭)
        src_points = np.float32([
            [0, y_top],
            [width, y_top],
            [width, y_bottom],
            [0, y_bottom],
        ])

        # dst: 윗변(y=0)은 출력 폭 그대로, 아랫변으로 갈수록 안쪽으로 눌림
        out_w = self.bev_output_width
        out_h = bev_output_height
        cx = out_w / 2.0
        top_half_width = out_w * self.bev_top_width_ratio / 2.0
        bottom_half_width = out_w * self.bev_bottom_width_ratio / 2.0

        dst_points = np.float32([
            [cx - top_half_width, 0],
            [cx + top_half_width, 0],
            [cx + bottom_half_width, out_h],
            [cx - bottom_half_width, out_h],
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
