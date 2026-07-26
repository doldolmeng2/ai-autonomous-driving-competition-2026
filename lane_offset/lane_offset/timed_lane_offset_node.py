"""timed_lane_offset_node_ngg.

역할:
    /camera/high/image_raw 를 받아 "오른쪽 실선"을 찾고, 그 실선이 화면의 특정
    x좌표(target_right_x)에 오도록 /lane_offset 을 발행한다.

설계 요점(기존 노드와 다른 점):
    1) 기준은 중앙 점선이 아니라 **오른쪽 실선**이다.
    2) 조향에 쓰는 x는 실선 전체가 아니라 **차량과 y축으로 가장 가까운 부분**만
       사용한다. 실선은 커브에서 휘기 때문에, 전체 평균을 기준 x에 맞추려 하면
       먼 쪽 곡률에 끌려가 오히려 차선을 이탈할 수 있다. 그래서 화면 아래쪽
       근접 밴드(near band)의 픽셀만으로 x를 잰다.
    3) 중앙 점선을 오른쪽 실선으로 오인하면 안 된다. 트랙에서 오른쪽 실선
       **바깥(오른쪽)에는 초록 매트**가 있으므로, 흰색 덩어리 주변에 초록색이
       충분히 있고 그 초록이 덩어리보다 **오른쪽**에 있을 때만 오른쪽 실선으로
       인정한다. (중앙 점선은 양옆이 회색 아스팔트라 걸러진다.)

좌표/부호 규약:


    화면 x는 차가 오른쪽으로 치우칠수록 작아진다(오른쪽 실선이 화면 안쪽으로
    들어옴). 따라서
        error = measured_x - target_right_x
        error < 0  -> 차가 오른쪽으로 치우침 -> offset < 0 -> 좌조향
    으로, 기존 노드들과 동일한 부호 규약을 따른다.

디버그 시각화(debug_view:=True):
    - 기준 x 세로선(주황)과 라벨
    - 조향 x를 재는 근접 밴드(초록 반투명 띠)
    - 오른쪽 실선으로 인정된 덩어리 **전체**를 빨강으로 칠함(마스킹 확인용)
    - 초록 매트 근거 영역(파랑 점)과 측정 x(노란 원), 상태 텍스트
"""

import numpy as np
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16

import cv2

# ============================================================================
# 파라미터 기본값 - 튜닝은 대부분 여기서만 하면 된다.
# (전부 ROS 파라미터로도 선언되므로 --ros-args -p 로 실행 중 덮어쓰기도 가능)
# ============================================================================

IMAGE_TOPIC = '/camera/high/image_raw'
LANE_OFFSET_TOPIC = '/lane_offset'
DEBUG_IMAGE_TOPIC = '/lane_offset/debug_image_ngg'

# 색상 분류. color_segment_bev_node와 동일하게 HSV와 YCrCb를 함께 사용한다.
# 뒤쪽 클래스가 먼저 적용된 클래스를 덮어쓴다.
BACKGROUND_COLOR_BGR = (0, 0, 0)
COLOR_CLASSES = [
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

# BEV(Bird's Eye View): 원본의 아래쪽 직사각형을 출력 사다리꼴로 워프한다.
BEV_Y_TOP_RATIO = 0.4
BEV_Y_BOTTOM_RATIO = 1.0
BEV_TOP_WIDTH_RATIO = 1.0
BEV_BOTTOM_WIDTH_RATIO = 0.7
# 출력 폭은 기존 640px 좌표계와 맞추고, 높이 0이면 BEV 입력 ROI 비율로 계산한다.
BEV_OUTPUT_WIDTH = 640
BEV_OUTPUT_HEIGHT = 0

# 횡단보도/정지선처럼 가로로 긴 흰색 영역 제거
# 한 행에서 이 길이 이상 연속된 흰색 픽셀이 있으면 가로선으로 판단한다.
HORIZONTAL_RUN_MIN_PX = 50
# 검출된 가로선 행을 기준으로 위/아래 이 픽셀 수만큼 흰색 마스크를 지운다.
HORIZONTAL_ERASE_HALF_BAND_PX = 5

# ---------------------------------------------------------------------------
# 오른쪽 실선 판정: 흰색 덩어리 주변 초록 매트 검증
# ---------------------------------------------------------------------------
# 덩어리를 이 픽셀 수만큼 부풀린 이웃 영역에서 초록을 찾는다. 커브에서 실선과
# 매트가 같은 x strip에 안 겹쳐도 잡히도록 고정 strip 대신 팽창을 쓴다.
GREEN_NEAR_DISTANCE_PX = 20
# 이웃 영역 안 초록 픽셀이 이 수 이상이어야 "매트 옆 실선"으로 인정한다.
GREEN_MIN_PIXELS = 10
# 초록이 덩어리보다 오른쪽에 있어야 한다(중앙 점선/왼쪽 실선 배제).
# 이웃 초록의 평균 x가 덩어리 평균 x보다 이 값 이상 커야 한다.
GREEN_RIGHT_MARGIN_PX = -1000

# ---------------------------------------------------------------------------
# 흰색 덩어리 모양 필터(초록 매트 위 흰 꽃 그림 등 제외)
# ---------------------------------------------------------------------------
MIN_COMPONENT_AREA = 50      # 너무 작은 덩어리 제외
MIN_LINE_HEIGHT_PX = 25       # 세로로 어느 정도 길어야 실선
MIN_LINE_ASPECT_RATIO = 0.0   # 세로/가로 비. 동글동글한 꽃 그림 배제

# ---------------------------------------------------------------------------
# 조향 기준
# ---------------------------------------------------------------------------
# 차가 정상 위치일 때 오른쪽 실선이 있어야 할 BEV 화면 x.
# 기존 카메라 좌표 보정값을 유지한 값이므로, BEV 적용 후에는 디버그 화면의
# right_x를 보고 다시 조정해야 한다.
TARGET_RIGHT_X = 510
# 조향에 쓰는 x는 BEV 바닥에서 이 픽셀 수 이내(차량과 가장 가까운 구간)의
# 실선 픽셀만 사용한다. 커브에서 먼 쪽 곡률에 끌려가지 않게 하는 핵심 값.
NEAR_ROWS = 80
# 근접 밴드에서 이 수 이상 픽셀이 있어야 측정을 신뢰한다.
NEAR_MIN_PIXELS = 20
# 근접 밴드가 비면(실선이 화면 위쪽에서만 보임) 덩어리 자체의 아래쪽
# NEAR_ROWS 행을 대신 쓴다. True면 폴백 허용.
ALLOW_LINE_BOTTOM_FALLBACK = True

# 기준선과 이만큼 차이 나면 lane_offset 최대/최소(+/-45)에 도달한다.
OFFSET_ERROR_LIMIT_PX = 130
LANE_OFFSET_LIMIT = 45
OFFSET_KP = 1.0
# 한 프레임 사이 offset이 이보다 크게 튀면 오검출로 보고 직전 값을 유지한다.
MAX_OFFSET_JUMP = 2000
# 발행 offset 저역통과(EMA) 계수. 1.0이면 필터 없음.
OFFSET_SMOOTHING_ALPHA = 1.0

# 디버그 시각화
DEBUG_VIEW = False
WINDOW_NAME = 'timed_lane_offset_ngg'
WHITE_MASK_WINDOW_NAME = 'ngg_white_mask'
GREEN_MASK_WINDOW_NAME = 'ngg_green_mask'


def class_mask(hsv, ycrcb, color_class):
    """color_segment_bev_node와 같은 HSV/YCrCb 교집합 마스크."""
    h_lo, h_hi = color_class['hsv']['h']
    s_lo, s_hi = color_class['hsv']['s']
    v_lo, v_hi = color_class['hsv']['v']
    mask = cv2.inRange(hsv, (h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi))

    ycrcb_range = color_class['ycrcb']
    if ycrcb_range is not None:
        y_lo, y_hi = ycrcb_range['y']
        cr_lo, cr_hi = ycrcb_range['cr']
        cb_lo, cb_hi = ycrcb_range['cb']
        ycrcb_mask = cv2.inRange(
            ycrcb, (y_lo, cr_lo, cb_lo), (y_hi, cr_hi, cb_hi)
        )
        mask = cv2.bitwise_and(mask, ycrcb_mask)
    return mask


class TimedLaneOffsetNggNode(Node):
    """오른쪽 실선을 기준 x에 맞추는 조향 offset 발행 노드."""

    def __init__(self):
        super().__init__('timed_lane_offset_node_ngg')

        self.declare_parameter('bev_y_top_ratio', BEV_Y_TOP_RATIO)
        self.declare_parameter('bev_y_bottom_ratio', BEV_Y_BOTTOM_RATIO)
        self.declare_parameter('bev_top_width_ratio', BEV_TOP_WIDTH_RATIO)
        self.declare_parameter('bev_bottom_width_ratio', BEV_BOTTOM_WIDTH_RATIO)
        self.declare_parameter('bev_output_width', BEV_OUTPUT_WIDTH)
        self.declare_parameter('bev_output_height', BEV_OUTPUT_HEIGHT)
        self.declare_parameter('horizontal_run_min_px', HORIZONTAL_RUN_MIN_PX)
        self.declare_parameter(
            'horizontal_erase_half_band_px', HORIZONTAL_ERASE_HALF_BAND_PX
        )
        self.declare_parameter('green_near_distance_px', GREEN_NEAR_DISTANCE_PX)
        self.declare_parameter('green_min_pixels', GREEN_MIN_PIXELS)
        self.declare_parameter('green_right_margin_px', GREEN_RIGHT_MARGIN_PX)
        self.declare_parameter('min_component_area', MIN_COMPONENT_AREA)
        self.declare_parameter('min_line_height_px', MIN_LINE_HEIGHT_PX)
        self.declare_parameter('min_line_aspect_ratio', MIN_LINE_ASPECT_RATIO)
        self.declare_parameter('target_right_x', TARGET_RIGHT_X)
        self.declare_parameter('near_rows', NEAR_ROWS)
        self.declare_parameter('near_min_pixels', NEAR_MIN_PIXELS)
        self.declare_parameter(
            'allow_line_bottom_fallback', ALLOW_LINE_BOTTOM_FALLBACK
        )
        self.declare_parameter('offset_error_limit_px', OFFSET_ERROR_LIMIT_PX)
        self.declare_parameter('lane_offset_limit', LANE_OFFSET_LIMIT)
        self.declare_parameter('offset_kp', OFFSET_KP)
        self.declare_parameter('max_offset_jump', MAX_OFFSET_JUMP)
        self.declare_parameter('offset_smoothing_alpha', OFFSET_SMOOTHING_ALPHA)
        self.declare_parameter('debug_view', DEBUG_VIEW)

        self.image_topic = IMAGE_TOPIC
        self.lane_offset_topic = LANE_OFFSET_TOPIC
        self.debug_image_topic = DEBUG_IMAGE_TOPIC
        self._load_parameters(lambda name: self.get_parameter(name).value)

        # 런타임 상태
        self.last_offset = 0.0
        self.last_line_x = None          # 직전 프레임의 오른쪽 실선 측정 x
        self.publish_debug_image = True

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.offset_pub = self.create_publisher(Int16, self.lane_offset_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, qos)
        self.create_subscription(Image, self.image_topic, self.image_callback, qos)
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.get_logger().info(
            f'Subscribing {self.image_topic}, publishing {self.lane_offset_topic}, '
            f'target_right_x={self.target_right_x}, near_rows={self.near_rows}, '
            f'bev_y=({self.bev_y_top_ratio},{self.bev_y_bottom_ratio}), '
            f'bev_bottom_width={self.bev_bottom_width_ratio}, '
            f'debug_view={self.debug_view}'
        )

    # ======================================================================
    # 파라미터
    # ======================================================================
    def _load_parameters(self, get):
        self.bev_y_top_ratio = float(np.clip(get('bev_y_top_ratio'), 0.0, 1.0))
        self.bev_y_bottom_ratio = float(
            np.clip(get('bev_y_bottom_ratio'), self.bev_y_top_ratio, 1.0)
        )
        self.bev_top_width_ratio = float(
            np.clip(get('bev_top_width_ratio'), 0.0, 1.0)
        )
        self.bev_bottom_width_ratio = float(
            np.clip(get('bev_bottom_width_ratio'), 0.0, 1.0)
        )
        self.bev_output_width = max(1, int(get('bev_output_width')))
        self.bev_output_height = max(0, int(get('bev_output_height')))
        self.horizontal_run_min_px = max(1, int(get('horizontal_run_min_px')))
        self.horizontal_erase_half_band_px = max(
            0, int(get('horizontal_erase_half_band_px'))
        )
        self.green_near_distance_px = max(1, int(get('green_near_distance_px')))
        self.green_min_pixels = int(get('green_min_pixels'))
        self.green_right_margin_px = float(get('green_right_margin_px'))
        self.min_component_area = int(get('min_component_area'))
        self.min_line_height_px = int(get('min_line_height_px'))
        self.min_line_aspect_ratio = float(get('min_line_aspect_ratio'))
        self.target_right_x = int(get('target_right_x'))
        self.near_rows = max(1, int(get('near_rows')))
        self.near_min_pixels = int(get('near_min_pixels'))
        self.allow_line_bottom_fallback = bool(get('allow_line_bottom_fallback'))
        self.offset_error_limit_px = max(1, int(get('offset_error_limit_px')))
        self.lane_offset_limit = max(1, int(get('lane_offset_limit')))
        self.offset_kp = max(0.0, float(get('offset_kp')))
        self.max_offset_jump = int(get('max_offset_jump'))
        self.offset_smoothing_alpha = float(
            np.clip(get('offset_smoothing_alpha'), 0.0, 1.0)
        )
        self.debug_view = bool(get('debug_view'))

    def _on_set_parameters(self, params):
        incoming = {p.name: p.value for p in params}

        def get(name):
            return incoming[name] if name in incoming else self.get_parameter(name).value

        try:
            self._load_parameters(get)
        except Exception as exc:  # noqa: BLE001 - 잘못된 값은 거부만, 노드는 유지
            return SetParametersResult(successful=False, reason=str(exc))
        self.get_logger().info(
            'Params updated live: ' + ', '.join(f'{p.name}={p.value}' for p in params)
        )
        return SetParametersResult(successful=True)

    # ======================================================================
    # 이미지 콜백
    # ======================================================================
    def image_callback(self, msg):
        frame = self.to_bgr(msg)
        if frame is None:
            return

        segmented = self.segment_colors(frame)
        bev = self.warp_to_bev(segmented)
        white_mask = self.make_white_mask(bev)
        green_mask = self.make_green_mask(bev)
        self.show_mask_windows(white_mask, green_mask)

        line_mask, measured_x, measure_mode, reject_reason = self.find_right_solid_line(
            white_mask, green_mask
        )

        if measured_x is None:
            # 오른쪽 실선을 못 찾음 -> 직전 offset 유지(급변 방지)
            self.get_logger().warn(
                f'Right solid line not found ({reject_reason}); holding last offset',
                throttle_duration_sec=1.0,
            )
            self.publish_offset(self.last_offset)
            self.publish_debug(
                msg, bev, 'NO RIGHT LINE', line_mask, None, measure_mode,
            )
            return

        raw_offset = self.map_line_x_to_offset(measured_x)

        if abs(raw_offset - self.last_offset) > self.max_offset_jump:
            self.get_logger().warn(
                f'Offset jump too large ({self.last_offset:.0f} -> {raw_offset}), '
                'holding last offset',
                throttle_duration_sec=1.0,
            )
            self.publish_offset(self.last_offset)
            self.publish_debug(
                msg, bev, 'JUMP REJECTED', line_mask, measured_x, measure_mode,
                raw_offset,
            )
            return

        self.last_offset = (
            self.offset_smoothing_alpha * raw_offset
            + (1.0 - self.offset_smoothing_alpha) * self.last_offset
        )
        self.last_line_x = measured_x
        self.publish_offset(self.last_offset)
        self.publish_debug(
            msg, bev, 'OK', line_mask, measured_x, measure_mode, raw_offset,
        )

    # ======================================================================
    # 마스크
    # ======================================================================
    def segment_colors(self, frame):
        """입력을 5색 분류 이미지로 바꾼다. BEV는 이 결과에 적용한다."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        segmented = np.full_like(frame, BACKGROUND_COLOR_BGR, dtype=np.uint8)
        for color_class in COLOR_CLASSES:
            mask = class_mask(hsv, ycrcb, color_class)
            segmented[mask > 0] = color_class['color_bgr']
        return segmented

    def make_white_mask(self, segmented_bev):
        white_mask = (
            np.all(segmented_bev == (255, 255, 255), axis=2).astype(np.uint8) * 255
        )
        return self.remove_horizontal_white_bands(white_mask)

    def remove_horizontal_white_bands(self, white_mask):
        """50px 이상 연속된 가로 흰 선과 그 위아래 밴드를 마스크에서 지운다."""
        height, width = white_mask.shape
        min_run = self.horizontal_run_min_px
        if width < min_run or height == 0:
            return white_mask

        # 각 행의 길이 min_run짜리 구간 합을 구한다. 합이 min_run이면 그 구간은
        # 전부 흰색이므로 해당 행에 연속된 가로선이 있다고 판단할 수 있다.
        binary = (white_mask > 0).astype(np.int32)
        row_prefix = np.pad(
            np.cumsum(binary, axis=1), ((0, 0), (1, 0)), mode='constant'
        )
        window_sums = row_prefix[:, min_run:] - row_prefix[:, :-min_run]
        horizontal_rows = np.any(window_sums >= min_run, axis=1)
        if not np.any(horizontal_rows):
            return white_mask

        # 검출 행마다 위/아래 half_band를 더한 행 전체를 0으로 만든다.
        half_band = self.horizontal_erase_half_band_px
        erase_rows = cv2.dilate(
            horizontal_rows.astype(np.uint8).reshape(-1, 1),
            np.ones((half_band * 2 + 1, 1), dtype=np.uint8),
        ).reshape(-1) > 0

        filtered = white_mask.copy()
        filtered[erase_rows, :] = 0
        return filtered

    @staticmethod
    def make_green_mask(segmented_bev):
        return (
            np.all(segmented_bev == (0, 255, 0), axis=2).astype(np.uint8) * 255
        )

    # ======================================================================
    # BEV(Bird's Eye View)
    # ======================================================================
    def warp_to_bev(self, segmented):
        output_height = self.compute_bev_output_height(segmented.shape)
        src_points, dst_points = self.compute_bev_points(
            segmented.shape, output_height
        )
        transform = cv2.getPerspectiveTransform(src_points, dst_points)
        # 분류 색이 섞여 새 색으로 생기지 않도록 최근접 보간을 사용한다.
        return cv2.warpPerspective(
            segmented,
            transform,
            (self.bev_output_width, output_height),
            flags=cv2.INTER_NEAREST,
        )

    def compute_bev_output_height(self, frame_shape):
        if self.bev_output_height > 0:
            return self.bev_output_height

        height, width = frame_shape[:2]
        source_height = height * (
            self.bev_y_bottom_ratio - self.bev_y_top_ratio
        )
        return max(1, round(self.bev_output_width * source_height / width))

    def compute_bev_points(self, frame_shape, output_height):
        height, width = frame_shape[:2]
        y_top = height * self.bev_y_top_ratio
        y_bottom = height * self.bev_y_bottom_ratio
        src_points = np.float32([
            [0, y_top],
            [width, y_top],
            [width, y_bottom],
            [0, y_bottom],
        ])

        center_x = self.bev_output_width / 2.0
        top_half_width = self.bev_output_width * self.bev_top_width_ratio / 2.0
        bottom_half_width = (
            self.bev_output_width * self.bev_bottom_width_ratio / 2.0
        )
        dst_points = np.float32([
            [center_x - top_half_width, 0],
            [center_x + top_half_width, 0],
            [center_x + bottom_half_width, output_height],
            [center_x - bottom_half_width, output_height],
        ])
        return src_points, dst_points

    # ======================================================================
    # 오른쪽 실선 찾기
    # ======================================================================
    def find_right_solid_line(self, white_mask, green_mask):
        """초록 매트가 오른쪽에 붙은 흰 실선을 찾아 (마스크, 측정x, 모드, 사유) 반환.

        - 흰색 덩어리 중 크기/모양 필터를 통과한 것만 후보로 본다.
        - 후보를 green_near_distance_px 만큼 팽창시킨 이웃에서 초록 픽셀을 세고,
          그 초록의 평균 x가 덩어리 평균 x보다 오른쪽이어야 오른쪽 실선으로 인정한다.
          (중앙 점선은 양옆이 아스팔트라 여기서 걸러진다.)
        - 여러 개면 직전 측정 x에 가장 가까운 것을, 없으면 가장 큰 것을 고른다.
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            white_mask, connectivity=8
        )
        if num_labels <= 1:
            return None, None, 'none', 'no white component'

        kernel_size = self.green_near_distance_px * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        green_bool = green_mask > 0

        best = None  # (score, label)
        shape_pass = 0
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            if area < self.min_component_area:
                continue
            if h < self.min_line_height_px:
                continue
            if w > 0 and (h / float(w)) < self.min_line_aspect_ratio:
                continue
            shape_pass += 1

            comp = labels == label
            neighborhood = cv2.dilate(comp.astype(np.uint8), kernel) > 0
            green_near = green_bool & neighborhood
            green_count = int(np.count_nonzero(green_near))
            if green_count < self.green_min_pixels:
                continue
            # 초록이 덩어리보다 오른쪽에 있어야 한다.
            green_mean_x = float(np.nonzero(green_near)[1].mean())
            comp_mean_x = float(centroids[label][0])
            if green_mean_x < comp_mean_x + self.green_right_margin_px:
                continue

            if self.last_line_x is not None:
                score = -abs(comp_mean_x - self.last_line_x)  # 가까울수록 높은 점수
            else:
                score = float(area)
            if best is None or score > best[0]:
                best = (score, label)

        if best is None:
            reason = (
                'no green-backed line' if shape_pass else 'no line-shaped component'
            )
            return None, None, 'none', reason

        line_mask = (labels == best[1]).astype(np.uint8) * 255
        measured_x, mode = self.measure_near_x(line_mask)
        if measured_x is None:
            return line_mask, None, mode, 'near band empty'
        return line_mask, measured_x, mode, ''

    def measure_near_x(self, line_mask):
        """실선 중 차량과 y축으로 가장 가까운 구간의 x를 median으로 잰다.

        커브에서 실선 전체를 평균 내면 먼 쪽 곡률에 끌려 기준 x가 왜곡되므로,
        ROI 바닥에서 near_rows 이내의 픽셀만 사용한다. 그 구간이 비어 있으면
        (실선이 화면 위쪽에서만 보이는 경우) 실선 자체의 아래쪽 near_rows 행으로
        폴백한다. 어느 쪽을 썼는지 모드로 돌려 디버그에 표시한다.
        """
        height = line_mask.shape[0]
        ys, xs = np.nonzero(line_mask)
        if ys.size == 0:
            return None, 'none'

        near = ys >= (height - self.near_rows)
        if int(near.sum()) >= self.near_min_pixels:
            return float(np.median(xs[near])), 'near band'

        if self.allow_line_bottom_fallback:
            bottom_cut = ys.max() - self.near_rows
            bottom = ys >= bottom_cut
            if int(bottom.sum()) >= self.near_min_pixels:
                return float(np.median(xs[bottom])), 'line bottom'
        return None, 'none'

    # ======================================================================
    # 조향 매핑 / 발행
    # ======================================================================
    def map_line_x_to_offset(self, measured_x):
        """오른쪽 실선 x를 기준 x로 되돌리는 offset(+/-45)을 만든다.

        measured_x < target(차가 오른쪽으로 치우침) -> error<0 -> offset<0 -> 좌조향.
        """
        error_px = float(measured_x) - self.target_right_x
        normalized = np.clip(error_px / self.offset_error_limit_px, -1.0, 1.0)
        scaled = normalized * self.lane_offset_limit * self.offset_kp
        return int(round(np.clip(
            scaled, -self.lane_offset_limit, self.lane_offset_limit
        )))

    def publish_offset(self, value):
        msg = Int16()
        msg.data = int(np.clip(value, -self.lane_offset_limit, self.lane_offset_limit))
        self.offset_pub.publish(msg)

    # ======================================================================
    # 디버그 시각화
    # ======================================================================
    def show_mask_windows(self, white_mask, green_mask):
        if not self.debug_view:
            return
        green_debug = np.zeros((*green_mask.shape, 3), dtype=np.uint8)
        green_debug[green_mask > 0] = (0, 255, 0)
        cv2.imshow(WHITE_MASK_WINDOW_NAME, cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR))
        cv2.imshow(GREEN_MASK_WINDOW_NAME, green_debug)

    def publish_debug(
        self, src_msg, frame, status, line_mask, measured_x, measure_mode,
        raw_offset=None,
    ):
        if not (self.debug_view or self.publish_debug_image):
            return

        debug = frame.copy()
        height, width = debug.shape[:2]
        # BEV 전체가 검출 ROI다.
        roi_top_y = 0
        roi_bottom_y = height - 1

        # ROI 경계(노랑)
        cv2.rectangle(debug, (0, roi_top_y), (width - 1, roi_bottom_y), (0, 255, 255), 1)

        # 조향 x를 재는 근접 밴드(초록 반투명) — "어느 정도 가까운 지점만 보는지"
        near_top_y = int(np.clip(height - self.near_rows, 0, height - 1))
        overlay = debug.copy()
        cv2.rectangle(
            overlay, (0, near_top_y), (width - 1, roi_bottom_y), (0, 220, 0), -1
        )
        cv2.addWeighted(overlay, 0.28, debug, 0.72, 0, debug)
        cv2.line(debug, (0, near_top_y), (width - 1, near_top_y), (0, 220, 0), 1)
        cv2.putText(
            debug, f'near_rows={self.near_rows}', (8, near_top_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA,
        )

        # 인식된 오른쪽 실선 "전체"를 빨강으로 칠한다(마스킹 정확도 확인용)
        if line_mask is not None and line_mask.any():
            ys, xs = np.nonzero(line_mask)
            debug[ys, xs] = (0, 0, 255)

        # 기준 x 세로선(주황) — "여기 오면 offset=0"
        self.draw_dashed_vline(
            debug, self.target_right_x, roi_top_y, roi_bottom_y, (0, 165, 255)
        )
        cv2.putText(
            debug, f'target_x={self.target_right_x}',
            (min(self.target_right_x + 6, width - 150), roi_top_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2, cv2.LINE_AA,
        )

        # 측정된 x(노란 원 + 세로선)와 기준선까지의 오차
        if measured_x is not None:
            mx = int(round(measured_x))
            cv2.circle(debug, (mx, roi_bottom_y - 4), 7, (0, 255, 255), -1)
            self.draw_dashed_vline(
                debug, mx, near_top_y, roi_bottom_y, (0, 255, 255), dash=5
            )
            cv2.line(
                debug, (mx, roi_bottom_y - 4),
                (int(self.target_right_x), roi_bottom_y - 4), (255, 255, 255), 1,
            )

        color = (0, 255, 0) if status == 'OK' else (0, 0, 255)
        err = '--' if measured_x is None else f'{measured_x - self.target_right_x:+.0f}'
        lines = [
            f'status: {status}   mode: {measure_mode}',
            f'right_x: {"--" if measured_x is None else f"{measured_x:.0f}"} '
            f'-> target {self.target_right_x} (err {err})',
            f'offset: {raw_offset if raw_offset is not None else "--"} '
            f'(smoothed {self.last_offset:.1f})',
            f'green: min_px={self.green_min_pixels} near={self.green_near_distance_px}',
        ]
        for i, text in enumerate(lines):
            cv2.putText(
                debug, text, (10, 20 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
            )

        if self.debug_view:
            cv2.imshow(WINDOW_NAME, debug)
            cv2.waitKey(1)
        if self.publish_debug_image:
            self.debug_pub.publish(self.to_image_msg(debug, src_msg.header.stamp))

    def draw_dashed_vline(self, img, x, y_top, y_bottom, color, dash=8):
        x = int(x)
        if x < 0 or x >= img.shape[1]:
            return
        for y in range(y_top, y_bottom, dash * 2):
            cv2.line(img, (x, y), (x, min(y + dash, y_bottom)), color, 2)

    def to_image_msg(self, bgr_img, stamp):
        msg = Image()
        msg.header.stamp = stamp
        msg.height, msg.width = bgr_img.shape[:2]
        msg.encoding = 'bgr8'
        msg.is_bigendian = False
        msg.step = msg.width * 3
        msg.data = np.ascontiguousarray(bgr_img).tobytes()
        return msg

    # ======================================================================
    # YUYV -> BGR
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
        if self.debug_view:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TimedLaneOffsetNggNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
