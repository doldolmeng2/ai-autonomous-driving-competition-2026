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

# ROI. 카메라를 차량 앞쪽으로 옮겨 거의 top-down(바로 아래를 내려다보는) 뷰가
# 되면서 화면 전체가 노면이다(지평선/배경/후드 없음). 그래서 크롭하지 않고
# 전체를 쓴다. cam_new bag(640x360) 실측 기준.
ROI_TOP = 0
ROI_BOTTOM = 360

# 흰색(차선) HSV 임계값
WHITE_S_MAX = 25
WHITE_V_MIN = 160

# 초록색(오른쪽 실선 바깥 매트) HSV 임계값
GREEN_H_MIN = 30
GREEN_H_MAX = 90
GREEN_S_MIN = 40
GREEN_V_MIN = 70

# ---------------------------------------------------------------------------
# 오른쪽 실선 판정: 흰색 덩어리 주변 초록 매트 검증
# ---------------------------------------------------------------------------
# 덩어리를 이 픽셀 수만큼 부풀린 이웃 영역에서 초록을 찾는다. 커브에서 실선과
# 매트가 같은 x strip에 안 겹쳐도 잡히도록 고정 strip 대신 팽창을 쓴다.
GREEN_NEAR_DISTANCE_PX = 40
# 이웃 영역 안 초록 픽셀이 이 수 이상이어야 "매트 옆 실선"으로 인정한다.
GREEN_MIN_PIXELS = 150
# 초록이 덩어리보다 오른쪽에 있어야 한다(중앙 점선/왼쪽 실선 배제).
# 이웃 초록의 평균 x가 덩어리 평균 x보다 이 값 이상 커야 한다.
GREEN_RIGHT_MARGIN_PX = 5

# ---------------------------------------------------------------------------
# 흰색 덩어리 모양 필터(초록 매트 위 흰 꽃 그림 등 제외)
# ---------------------------------------------------------------------------
MIN_COMPONENT_AREA = 150      # 너무 작은 덩어리 제외
MIN_LINE_HEIGHT_PX = 25       # 세로로 어느 정도 길어야 실선
MIN_LINE_ASPECT_RATIO = 1.2   # 세로/가로 비. 동글동글한 꽃 그림 배제

# ---------------------------------------------------------------------------
# 조향 기준
# ---------------------------------------------------------------------------
# 차가 정상 위치일 때 오른쪽 실선이 있어야 할 화면 x.
# cam_new bag 실측: 차량을 원하는 주행 위치에 세워둔 상태에서 122프레임 전부
# right_x = 563 (std 0.0) 으로 측정되어 이 값을 기준으로 삼는다.
TARGET_RIGHT_X = 563
# 조향에 쓰는 x는 ROI 바닥에서 이 픽셀 수 이내(차량과 가장 가까운 구간)의
# 실선 픽셀만 사용한다. 커브에서 먼 쪽 곡률에 끌려가지 않게 하는 핵심 값.
NEAR_ROWS = 45
# 근접 밴드에서 이 수 이상 픽셀이 있어야 측정을 신뢰한다.
NEAR_MIN_PIXELS = 20
# 근접 밴드가 비면(실선이 화면 위쪽에서만 보임) 덩어리 자체의 아래쪽
# NEAR_ROWS 행을 대신 쓴다. True면 폴백 허용.
ALLOW_LINE_BOTTOM_FALLBACK = True

# 기준선과 이만큼 차이 나면 lane_offset 최대/최소(+/-45)에 도달한다.
OFFSET_ERROR_LIMIT_PX = 120
LANE_OFFSET_LIMIT = 45
OFFSET_KP = 1.0
# 한 프레임 사이 offset이 이보다 크게 튀면 오검출로 보고 직전 값을 유지한다.
MAX_OFFSET_JUMP = 40
# 발행 offset 저역통과(EMA) 계수. 1.0이면 필터 없음.
OFFSET_SMOOTHING_ALPHA = 0.6

# 디버그 시각화
DEBUG_VIEW = False
WINDOW_NAME = 'timed_lane_offset_ngg'
WHITE_MASK_WINDOW_NAME = 'ngg_white_mask'
GREEN_MASK_WINDOW_NAME = 'ngg_green_mask'


class TimedLaneOffsetNggNode(Node):
    """오른쪽 실선을 기준 x에 맞추는 조향 offset 발행 노드."""

    def __init__(self):
        super().__init__('timed_lane_offset_node_ngg')

        self.declare_parameter('roi_top', ROI_TOP)
        self.declare_parameter('roi_bottom', ROI_BOTTOM)
        self.declare_parameter('white_s_max', WHITE_S_MAX)
        self.declare_parameter('white_v_min', WHITE_V_MIN)
        self.declare_parameter('green_h_min', GREEN_H_MIN)
        self.declare_parameter('green_h_max', GREEN_H_MAX)
        self.declare_parameter('green_s_min', GREEN_S_MIN)
        self.declare_parameter('green_v_min', GREEN_V_MIN)
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
            f'roi=({self.roi_top},{self.roi_bottom}), debug_view={self.debug_view}'
        )

    # ======================================================================
    # 파라미터
    # ======================================================================
    def _load_parameters(self, get):
        self.roi_top = int(get('roi_top'))
        self.roi_bottom = int(get('roi_bottom'))
        self.white_s_max = int(get('white_s_max'))
        self.white_v_min = int(get('white_v_min'))
        self.green_h_min = int(get('green_h_min'))
        self.green_h_max = int(get('green_h_max'))
        self.green_s_min = int(get('green_s_min'))
        self.green_v_min = int(get('green_v_min'))
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

        roi = frame[self.roi_top:self.roi_bottom, :]
        if roi.size == 0:
            return

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white_mask = self.make_white_mask(hsv)
        green_mask = self.make_green_mask(hsv)
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
                msg, frame, 'NO RIGHT LINE', line_mask, None, measure_mode,
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
                msg, frame, 'JUMP REJECTED', line_mask, measured_x, measure_mode,
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
            msg, frame, 'OK', line_mask, measured_x, measure_mode, raw_offset,
        )

    # ======================================================================
    # 마스크
    # ======================================================================
    def make_white_mask(self, hsv):
        _h, s, v = cv2.split(hsv)
        mask = (s < self.white_s_max) & (v > self.white_v_min)
        return (mask.astype(np.uint8)) * 255

    def make_green_mask(self, hsv):
        h, s, v = cv2.split(hsv)
        mask = (
            (h > self.green_h_min)
            & (h < self.green_h_max)
            & (s > self.green_s_min)
            & (v > self.green_v_min)
        )
        return (mask.astype(np.uint8)) * 255

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
        roi_top_y = int(np.clip(self.roi_top, 0, height - 1))
        roi_bottom_y = int(np.clip(self.roi_bottom - 1, 0, height - 1))

        # ROI 경계(노랑)
        cv2.rectangle(debug, (0, roi_top_y), (width - 1, roi_bottom_y), (0, 255, 255), 1)

        # 조향 x를 재는 근접 밴드(초록 반투명) — "어느 정도 가까운 지점만 보는지"
        near_top_y = int(np.clip(self.roi_bottom - self.near_rows, 0, height - 1))
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
            debug[ys + self.roi_top, xs] = (0, 0, 255)

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
