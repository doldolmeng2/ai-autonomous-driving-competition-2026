"""Mission lane offset node.

하나의 파일 안에서 다음을 모두 수행한다.
- HSV+YCrCb 흰색 분류와 최근접 보간 BEV 마스킹
- 횡단보도/정지선 가로 밴드 제거
- OSY 중앙 점선 슬라이딩 윈도우 추적
- /lane_info(1/2)에 따른 1lane/2lane 기준 전환
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16

import cv2

# ============================================================================
# 파라미터 기본값 - 튜닝은 대부분 여기서만 하면 된다.
# (전부 ROS 파라미터로도 선언되므로 --ros-args -p 로 실행 중 덮어쓰기도 가능)
# ============================================================================

# 구독/발행 토픽
IMAGE_TOPIC = '/camera/high/image_raw'
LANE_OFFSET_TOPIC = '/lane_offset'
LANE_INFO_TOPIC = '/lane_info'

# 중앙 흰색 점선 분류 설정.
BACKGROUND_COLOR_BGR = (0, 0, 0)
COLOR_CLASSES = [
    {
        'name': 'white',
        'color_bgr': (255, 255, 255),
        'hsv': {'h': (0, 179), 's': (0, 44), 'v': (161, 255)},
        'ycrcb': {'y': (135, 255), 'cr': (81, 165), 'cb': (122, 170)},
    },
]

# 원본 영상 높이의 20% 지점부터 최하단까지 BEV로 펼친다.
BEV_Y_TOP_RATIO = 0.2
BEV_Y_BOTTOM_RATIO = 1.0
BEV_TOP_WIDTH_RATIO = 1.0
BEV_BOTTOM_WIDTH_RATIO = 0.7
BEV_OUTPUT_WIDTH = 640
BEV_OUTPUT_HEIGHT = 0

# 차선보다 훨씬 긴 가로 성분(정지선/횡단보도)을 제거한다.
HORIZONTAL_RUN_MIN_PX = 80
HORIZONTAL_ERASE_HALF_BAND_PX = 3

# ROI: 이미지 상단(배경)과 하단(차량 후드)을 잘라낸다. (640x480 기준)
ROI_TOP = 250
ROI_BOTTOM = 480
# ROI 안에서 사용할 사다리꼴의 좌/우 inset. 아래쪽은 경계 차선을 보존하기 위해
# 거의 자르지 않고, 위쪽만 좁혀 먼 거리의 양옆 잡음을 제외한다. (640px 폭 기준)
ROI_TRAPEZOID_TOP_INSET_PX = 150
ROI_TRAPEZOID_BOTTOM_INSET_PX = 0

# 흰색 과검출 검사에 쓰는 근접(ROI 하단) 밴드 높이
NEAR_FIELD_ROWS = 100
# 근접 밴드에서 흰색 비율이 이 값을 넘으면 횡단보도 등으로 판단하고 무시
WHITE_OVERLOAD_RATIO = 0.15

# 2차선에서 시작하며 /lane_info가 바뀌면 모드별 기준값으로 전환한다.
DRIVING_MODE = '1lane'
# 640px BEV에서 차가 정상 위치일 때 보이는 중앙 점선의 x좌표.
# 1차선은 화면 오른쪽 점선, 2차선은 화면 왼쪽 점선을 각각 기준으로 삼는다.
DASHED_REFERENCE_X_PX_2LANE = 108
DASHED_REFERENCE_X_PX_1LANE = 570
# 기준선과 이만큼 차이 나면 lane_offset의 최대/최소값(+/-45)에 도달한다.
OFFSET_ERROR_LIMIT_PX = 195
LANE_OFFSET_LIMIT = 45
# 기준선 오차에 비례해 조향하도록 1.0을 사용한다. 10.0은 약 20px 오차만
# 생겨도 바로 +/-45로 포화되어 "떨어진 만큼" 조향하지 못한다.
OFFSET_KP = 2.0
# 한 프레임 사이 offset이 이 값보다 더 튀면 오검출로 보고 이전 값 유지
MAX_OFFSET_JUMP_PX = 80

# 슬라이딩 윈도우. BEV 세로 방향 박스를 약 20px로 얇게 나눈다.
NUM_WINDOWS = 15
# 점선은 빈 구간을 넘어 다음 조각을 잡아야 하므로 실선보다 넓게 탐색한다.
DASHED_WINDOW_MARGIN = 160
WINDOW_MIN_COMPONENT_PIXELS = 30
# 중앙 점선 성분 검증. 기준 x 주변에서 세로로 50px 이상 이어지는 흰 성분
# 하나를 선택한다.
DASH_MIN_VERTICAL_SPAN_PX = 50
DASH_MIN_WIDTH_PX = 10
DASH_MIN_AVERAGE_WIDTH_PX = 8.0
DASH_MAX_AVERAGE_WIDTH_PX = 40.0
DASH_MIN_COMPONENT_PIXELS = 80
# ROI 높이 대부분을 계속 잇는 성분은 점선이 아니라 실선으로 본다.
DASH_MAX_VERTICAL_SPAN_RATIO = 0.92
# 서로 다른 높이에 나타나는 위/아래 점선 조각을 한 트랙으로 묶는다.
DASH_MAX_TRACKED_PIECES = 2
DASH_MAX_VERTICAL_OVERLAP_RATIO = 0.25
# 이전 프레임의 검출 위치를 다음 프레임 윈도우 시작점에 반영하는 비율.
# 0.20이면 한 프레임에 차이의 20%만 움직여 급격한 점프를 막는다.
WINDOW_START_ADAPT_RATE = 0.7
# 새 검출 위치가 이전 박스 시작점에서 이 거리보다 크게 튀면 오검출로 보고
# 시작점을 갱신하지 않는다.
MAX_WINDOW_START_JUMP_PX = 180
# 디버그 시각화: ROI/차선/슬라이딩 윈도우를 그린 화면을 바로 OpenCV 창으로 띄운다.
# (bag/카메라 토픽만 켜져 있으면, 이 노드 실행만으로 인식 화면이 뜬다.)
# 실차 대회 주행 시에는 CPU 절약을 위해 False로 끄는 것을 권장.
DEBUG_VIEW = True
WINDOW_NAME = 'timed_lane_offset_debug'
WHITE_MASK_WINDOW_NAME = 'timed_lane_offset_white_mask_osy'
DEBUG_IMAGE_TOPIC = '/lane_offset/debug_image'


def class_mask(hsv, ycrcb, color_class):
    """한 색상 클래스의 HSV/YCrCb 교집합 마스크를 만든다."""
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


class MissionLaneOffsetNode(Node):
    """/camera/high/image_raw -> 중앙 점선 기준 /lane_offset 발행."""

    def __init__(self):
        super().__init__('mission_lane_offset_node')

        # ---- 파라미터 ------------------------------------------------------
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
        self.declare_parameter('roi_top', ROI_TOP)
        self.declare_parameter('roi_bottom', ROI_BOTTOM)
        self.declare_parameter('roi_trapezoid_top_inset_px', ROI_TRAPEZOID_TOP_INSET_PX)
        self.declare_parameter(
            'roi_trapezoid_bottom_inset_px', ROI_TRAPEZOID_BOTTOM_INSET_PX
        )
        self.declare_parameter('near_field_rows', NEAR_FIELD_ROWS)
        self.declare_parameter('white_overload_ratio', WHITE_OVERLOAD_RATIO)
        self.declare_parameter('driving_mode', DRIVING_MODE)
        self.declare_parameter('dashed_reference_x_px_2lane', DASHED_REFERENCE_X_PX_2LANE)
        self.declare_parameter('dashed_reference_x_px_1lane', DASHED_REFERENCE_X_PX_1LANE)
        self.declare_parameter('offset_error_limit_px', OFFSET_ERROR_LIMIT_PX)
        self.declare_parameter('lane_offset_limit', LANE_OFFSET_LIMIT)
        self.declare_parameter('offset_kp', OFFSET_KP)
        self.declare_parameter('max_offset_jump_px', MAX_OFFSET_JUMP_PX)
        self.declare_parameter('num_windows', NUM_WINDOWS)
        self.declare_parameter('dashed_window_margin', DASHED_WINDOW_MARGIN)
        self.declare_parameter(
            'dash_min_vertical_span_px', DASH_MIN_VERTICAL_SPAN_PX
        )
        self.declare_parameter('dash_min_width_px', DASH_MIN_WIDTH_PX)
        self.declare_parameter(
            'dash_min_average_width_px', DASH_MIN_AVERAGE_WIDTH_PX
        )
        self.declare_parameter(
            'dash_max_average_width_px', DASH_MAX_AVERAGE_WIDTH_PX
        )
        self.declare_parameter(
            'dash_min_component_pixels', DASH_MIN_COMPONENT_PIXELS
        )
        self.declare_parameter(
            'dash_max_vertical_span_ratio', DASH_MAX_VERTICAL_SPAN_RATIO
        )
        self.declare_parameter(
            'dash_max_tracked_pieces', DASH_MAX_TRACKED_PIECES
        )
        self.declare_parameter(
            'dash_max_vertical_overlap_ratio',
            DASH_MAX_VERTICAL_OVERLAP_RATIO,
        )
        self.declare_parameter('window_min_component_pixels', WINDOW_MIN_COMPONENT_PIXELS)
        self.declare_parameter('window_start_adapt_rate', WINDOW_START_ADAPT_RATE)
        self.declare_parameter('max_window_start_jump_px', MAX_WINDOW_START_JUMP_PX)
        self.declare_parameter('debug_view', DEBUG_VIEW)

        self.bev_y_top_ratio = float(np.clip(
            self.get_parameter('bev_y_top_ratio').value, 0.0, 1.0
        ))
        self.bev_y_bottom_ratio = float(np.clip(
            self.get_parameter('bev_y_bottom_ratio').value,
            self.bev_y_top_ratio,
            1.0,
        ))
        self.bev_top_width_ratio = float(np.clip(
            self.get_parameter('bev_top_width_ratio').value, 0.0, 1.0
        ))
        self.bev_bottom_width_ratio = float(np.clip(
            self.get_parameter('bev_bottom_width_ratio').value, 0.0, 1.0
        ))
        self.bev_output_width = max(
            1, int(self.get_parameter('bev_output_width').value)
        )
        self.bev_output_height = max(
            0, int(self.get_parameter('bev_output_height').value)
        )
        self.horizontal_run_min_px = max(
            1, int(self.get_parameter('horizontal_run_min_px').value)
        )
        self.horizontal_erase_half_band_px = max(
            0, int(self.get_parameter('horizontal_erase_half_band_px').value)
        )

        self.image_topic = IMAGE_TOPIC
        self.lane_offset_topic = LANE_OFFSET_TOPIC
        self.roi_top = int(self.get_parameter('roi_top').value)
        self.roi_bottom = int(self.get_parameter('roi_bottom').value)
        self.roi_trapezoid_top_inset_px = int(
            self.get_parameter('roi_trapezoid_top_inset_px').value
        )
        self.roi_trapezoid_bottom_inset_px = int(
            self.get_parameter('roi_trapezoid_bottom_inset_px').value
        )
        self.near_field_rows = int(self.get_parameter('near_field_rows').value)
        self.white_overload_ratio = float(
            self.get_parameter('white_overload_ratio').value
        )
        self.driving_mode = str(self.get_parameter('driving_mode').value).lower()
        if self.driving_mode not in ('1lane', '2lane'):
            self.get_logger().warn(
                f"Unknown driving_mode='{self.driving_mode}'; using '1lane'"
            )
            self.driving_mode = '1lane'
        self.dashed_reference_x_px_2lane = int(
            self.get_parameter('dashed_reference_x_px_2lane').value
        )
        self.dashed_reference_x_px_1lane = int(
            self.get_parameter('dashed_reference_x_px_1lane').value
        )
        self.dashed_reference_x_px = (
            self.dashed_reference_x_px_2lane
            if self.driving_mode == '2lane'
            else self.dashed_reference_x_px_1lane
        )
        self.offset_error_limit_px = max(
            1, int(self.get_parameter('offset_error_limit_px').value)
        )
        self.lane_offset_limit = max(
            1, int(self.get_parameter('lane_offset_limit').value)
        )
        self.offset_kp = max(0.0, float(self.get_parameter('offset_kp').value))
        self.max_offset_jump_px = int(
            self.get_parameter('max_offset_jump_px').value
        )
        self.num_windows = int(self.get_parameter('num_windows').value)
        self.dashed_window_margin = int(
            self.get_parameter('dashed_window_margin').value
        )
        self.dash_min_vertical_span_px = max(
            1, int(self.get_parameter('dash_min_vertical_span_px').value)
        )
        self.dash_min_width_px = max(
            1, int(self.get_parameter('dash_min_width_px').value)
        )
        self.dash_min_average_width_px = max(
            0.0, float(self.get_parameter('dash_min_average_width_px').value)
        )
        self.dash_max_average_width_px = max(
            self.dash_min_average_width_px,
            float(self.get_parameter('dash_max_average_width_px').value),
        )
        self.dash_min_component_pixels = max(
            1, int(self.get_parameter('dash_min_component_pixels').value)
        )
        self.dash_max_vertical_span_ratio = float(np.clip(
            self.get_parameter('dash_max_vertical_span_ratio').value,
            0.0,
            1.0,
        ))
        self.dash_max_tracked_pieces = max(
            1, int(self.get_parameter('dash_max_tracked_pieces').value)
        )
        self.dash_max_vertical_overlap_ratio = float(np.clip(
            self.get_parameter('dash_max_vertical_overlap_ratio').value,
            0.0,
            1.0,
        ))
        self.window_min_component_pixels = int(
            self.get_parameter('window_min_component_pixels').value
        )
        self.window_start_adapt_rate = float(np.clip(
            self.get_parameter('window_start_adapt_rate').value, 0.0, 1.0
        ))
        self.max_window_start_jump_px = max(0, int(
            self.get_parameter('max_window_start_jump_px').value
        ))
        self.debug_view = bool(self.get_parameter('debug_view').value)
        self.window_name = WINDOW_NAME
        self.white_mask_window_name = WHITE_MASK_WINDOW_NAME
        self.publish_debug_image = False
        self.debug_image_topic = DEBUG_IMAGE_TOPIC

        # 마지막으로 발행한(유효했던) offset. 오검출 프레임에서는 이 값을 그대로 재사용.
        self.last_offset = 0
        # 다음 프레임의 중앙 점선 슬라이딩 윈도우 시작점.
        self.window_start_x = {'dashed': float(self.dashed_reference_x_px)}
        # 마지막으로 유효했던 중앙 점선 슬라이딩 윈도우. 인식 공백에서도 디버그
        # 박스가 사라지지 않도록 유지한다.
        self.last_lane_tracks = None

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.offset_pub = self.create_publisher(Int16, self.lane_offset_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, qos)
        self.create_subscription(Image, self.image_topic, self.image_callback, qos)
        self.create_subscription(Int16, LANE_INFO_TOPIC, self.lane_info_callback, 10)

        self.get_logger().info(
            f'Subscribing {self.image_topic}, publishing {self.lane_offset_topic}, '
            f'driving_mode={self.driving_mode}, '
            f'dashed_reference_x_px={self.dashed_reference_x_px}, '
            f'offset_kp={self.offset_kp:.2f}, '
            f'debug_view={self.debug_view}, '
            f'offset range=+/-{self.lane_offset_limit}'
        )

    # ======================================================================
    # 이미지 콜백
    # ======================================================================
    def lane_info_callback(self, msg):
        lane_number = int(msg.data)
        if lane_number not in (1, 2):
            self.get_logger().warn(
                f'Ignoring invalid lane_info={lane_number}',
                throttle_duration_sec=1.0,
            )
            return
        self.set_driving_mode(f'{lane_number}lane')

    def set_driving_mode(self, mode, force=False):
        """차로 기준만 바꾸고 현재 추적 중인 물리적 중앙선은 이어간다."""
        mode = str(mode).lower()
        if mode not in ('1lane', '2lane'):
            return
        if not force and mode == self.driving_mode:
            return

        previous_track_x = self.window_start_x.get('dashed')
        self.driving_mode = mode
        self.dashed_reference_x_px = (
            self.dashed_reference_x_px_1lane
            if mode == '1lane'
            else self.dashed_reference_x_px_2lane
        )
        # 기준 x는 조향 목표일 뿐 검출 시작점이 아니다. 차선 전환 중에도 직전에
        # 보던 동일한 점선 주변에서 다음 프레임 탐색을 계속한다.
        if force or previous_track_x is None:
            previous_track_x = float(self.dashed_reference_x_px)
        self.window_start_x = {'dashed': float(previous_track_x)}
        self.get_logger().info(
            f'Driving mode changed to {mode}; '
            f'dashed_reference_x={self.dashed_reference_x_px}, '
            f'continuing_track_x={previous_track_x:.1f}'
        )

    def image_callback(self, msg):
        frame = self.to_bgr(msg)
        if frame is None:
            return

        segmented = self.segment_white(frame)
        frame = self.warp_to_bev(segmented)
        if frame.size == 0:
            return

        # BEV 전체를 중앙선 추적 ROI로 사용한다.
        self.roi_top = 0
        self.roi_bottom = frame.shape[0]
        self.roi_trapezoid_top_inset_px = 0
        self.roi_trapezoid_bottom_inset_px = 0
        roi = frame

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        trapezoid_mask = self.make_roi_trapezoid_mask(roi.shape[:2])
        white_mask = cv2.bitwise_and(self.make_white_mask(hsv), trapezoid_mask)
        self.show_debug_mask(white_mask)

        near = white_mask[-self.near_field_rows:, :]
        near_white_ratio = float((near > 0).mean()) if near.size else 0.0

        if near_white_ratio > self.white_overload_ratio:
            # 횡단보도 등으로 흰색이 과도하게 잡힘: 이번 측정은 버리고 직전 값 유지
            self.get_logger().info(
                f'White overload (ratio={near_white_ratio:.2f}), '
                f'holding last offset={self.last_offset}',
                throttle_duration_sec=1.0,
            )
            self.publish_offset(self.last_offset)
            self.publish_debug(msg, frame, 'WHITE OVERLOAD', near_white_ratio)
            return

        # 기준 x 주변에서 세로로 충분히 긴 흰색 성분 하나를 중앙 점선으로 쓴다.
        dashed_start_x = self.get_window_start_x(
            'dashed', self.dashed_reference_x_px
        )
        dashed_mask, dashed_components = self.find_center_dashed_component(
            white_mask, dashed_start_x
        )
        if dashed_mask is None:
            self.get_logger().warn(
                'No center-dash component; holding last offset',
                throttle_duration_sec=1.0,
            )
            self.publish_offset(self.last_offset)
            self.publish_debug(
                msg, frame, 'NO DASH', near_white_ratio,
                base_x=dashed_start_x,
            )
            return

        dashed_track = self.track_lane_with_sliding_window(
            dashed_mask, dashed_start_x, self.dashed_window_margin,
            allow_gaps=True,
        )
        dashed_x, windows, points_x, points_y = dashed_track
        lane_tracks = {'dashed': dashed_track}
        if dashed_x is None:
            self.get_logger().warn(
                'Center dash sliding-window detection failed; holding last offset',
                throttle_duration_sec=1.0,
            )
            self.publish_offset(self.last_offset)
            self.publish_debug(
                msg, frame, 'DASH TRACK FAILED', near_white_ratio,
                base_x=dashed_start_x, windows=windows,
                lane_tracks=lane_tracks,
            )
            return

        line_x = dashed_x
        lane_offset = self.map_lane_x_to_offset(line_x)

        if abs(lane_offset - self.last_offset) > self.max_offset_jump_px:
            # 한 프레임 만에 비정상적으로 튀면 오검출로 보고 이전 값 유지
            self.get_logger().warn(
                f'Offset jump too large ({self.last_offset} -> {lane_offset}), '
                'holding last offset',
                throttle_duration_sec=1.0,
            )
            self.publish_offset(self.last_offset)
            self.publish_debug(
                msg, frame, 'JUMP REJECTED', near_white_ratio,
                base_x=dashed_start_x, line_x=line_x, windows=windows,
                points_x=points_x, points_y=points_y, lane_offset=lane_offset,
                lane_tracks=lane_tracks,
            )
            return

        self.last_offset = lane_offset
        self.update_window_start_x('dashed', dashed_x, white_mask.shape[1])
        self.last_lane_tracks = lane_tracks
        self.publish_offset(lane_offset)
        self.publish_debug(
            msg, frame, 'OK', near_white_ratio,
            base_x=dashed_start_x, line_x=line_x, windows=windows,
            points_x=points_x, points_y=points_y, lane_offset=lane_offset,
            lane_tracks=lane_tracks, dashed_components=dashed_components,
        )

    def publish_offset(self, value):
        msg = Int16()
        msg.data = int(np.clip(value, -self.lane_offset_limit, self.lane_offset_limit))
        self.offset_pub.publish(msg)

    def map_lane_x_to_offset(self, detected_lane_x):
        """점선 오차에 Kp를 적용해 -45~45 offset으로 매핑한다."""
        error_px = float(detected_lane_x) - self.dashed_reference_x_px
        normalized = np.clip(
            error_px / self.offset_error_limit_px,
            -1.0,
            1.0,
        )
        scaled_offset = normalized * self.lane_offset_limit * self.offset_kp
        return int(round(np.clip(
            scaled_offset, -self.lane_offset_limit, self.lane_offset_limit
        )))

    # ======================================================================
    # 색상 마스크
    # ======================================================================
    def segment_white(self, frame):
        """원본에서 흰색 후보만 분류하고 나머지는 검정으로 만든다."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        segmented = np.full_like(
            frame, BACKGROUND_COLOR_BGR, dtype=np.uint8
        )
        for color_class in COLOR_CLASSES:
            mask = class_mask(hsv, ycrcb, color_class)
            segmented[mask > 0] = color_class['color_bgr']
        return segmented

    def warp_to_bev(self, segmented):
        output_height = self.compute_bev_output_height(segmented.shape)
        src_points, dst_points = self.compute_bev_points(
            segmented.shape, output_height
        )
        transform = cv2.getPerspectiveTransform(src_points, dst_points)
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
        return max(
            1, round(self.bev_output_width * source_height / width)
        )

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
        top_half_width = (
            self.bev_output_width * self.bev_top_width_ratio / 2.0
        )
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

    # BEV 입력은 흰색/검정 이진 색상이므로 흰색만 정확히 꺼낸다.
    def make_white_mask(self, hsv):
        _h, s, v = cv2.split(hsv)
        white_mask = ((s == 0) & (v == 255)).astype(np.uint8) * 255
        return self.remove_horizontal_white_bands(white_mask)

    def remove_horizontal_white_bands(self, white_mask):
        height, width = white_mask.shape
        min_run = self.horizontal_run_min_px
        if height == 0 or width < min_run:
            return white_mask

        # 긴 가로선이 있는 "행 전체"를 지우면 같은 높이에 있는 중앙 점선까지
        # 잘린다. 가로 opening으로 실제 정지선 픽셀의 x구간만 추출해 제거한다.
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (min_run, 1)
        )
        horizontal = cv2.morphologyEx(
            white_mask, cv2.MORPH_OPEN, horizontal_kernel
        )
        if not np.any(horizontal):
            return white_mask

        half_band = self.horizontal_erase_half_band_px
        if half_band > 0:
            horizontal = cv2.dilate(
                horizontal,
                np.ones((half_band * 2 + 1, 1), dtype=np.uint8),
            )
        filtered = white_mask.copy()
        filtered[horizontal > 0] = 0
        return filtered

    def make_roi_trapezoid_mask(self, shape):
        """위쪽만 좁히고 하단변은 영상 전체 폭인 사다리꼴 마스크를 만든다."""
        height, width = shape
        top_inset = int(np.clip(self.roi_trapezoid_top_inset_px, 0, width // 2))
        polygon = np.array([
            (top_inset, 0),
            (width - 1 - top_inset, 0),
            # 하단변은 반드시 카메라 화면의 좌/우 끝까지 사용한다.
            (width - 1, height - 1),
            (0, height - 1),
        ], dtype=np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        return mask

    def show_debug_mask(self, white_mask):
        """ROI에서 만든 흰색 마스크를 별도 디버그 창으로 표시한다."""
        if not self.debug_view:
            return

        white_debug = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        cv2.imshow(self.white_mask_window_name, white_debug)

    def find_center_dashed_component(self, white_mask, expected_x):
        """기준 x 주변에서 점선 모양에 가장 가까운 흰 성분 하나를 반환한다."""
        height, _width = white_mask.shape
        # 가로선 제거가 점선과 교차한 지점에 만든 작은 틈만 세로로 복구한다.
        reconnect_height = max(
            15, self.horizontal_erase_half_band_px * 2 + 9
        )
        candidate_mask = cv2.morphologyEx(
            white_mask,
            cv2.MORPH_CLOSE,
            np.ones((reconnect_height, 1), dtype=np.uint8),
        )
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            candidate_mask, connectivity=8
        )
        candidates = []
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            spans_full_height = y <= 2 and y + h >= height - 2
            vertical_span_ratio = h / float(height)
            average_width = area / float(h)
            if (
                h < self.dash_min_vertical_span_px
                or w < self.dash_min_width_px
                or average_width < self.dash_min_average_width_px
                or average_width > self.dash_max_average_width_px
                or area < self.dash_min_component_pixels
                or h <= w
                or spans_full_height
                or vertical_span_ratio > self.dash_max_vertical_span_ratio
            ):
                continue

            # 기울어진 점선은 bbox 중심보다 ROI 하단 연장점이 차량 위치 오차를
            # 더 잘 나타낸다. 성분 픽셀로 x=f(y)를 피팅해 하단 x를 예측한다.
            local_y, local_x = np.where(labels == label)
            if len(np.unique(local_y)) >= 3:
                slope, intercept = np.polyfit(local_y, local_x, 1)
                bottom_x = float(slope * (height - 1) + intercept)
            else:
                bottom_x = float(centroids[label][0])
            distance = abs(bottom_x - float(expected_x))
            bbox_distance = max(
                float(x) - float(expected_x),
                float(expected_x) - float(x + w - 1),
                0.0,
            )
            if min(distance, bbox_distance) <= self.dashed_window_margin:
                candidates.append((distance, bottom_x, label, x, y, w, h))

        if not candidates:
            return None, []

        # 하단 점선 조각을 기준으로 잡고, y구간이 겹치지 않는 위쪽/아래쪽
        # 조각도 같은 트랙에 추가한다. 같은 높이의 평행 실선은 제외된다.
        primary = min(candidates, key=lambda candidate: candidate[0])
        selected = [primary]
        for candidate in sorted(candidates, key=lambda item: item[0]):
            if candidate is primary:
                continue
            _distance, _bottom_x, _label, _x, y, _w, h = candidate
            vertically_separate = True
            for chosen in selected:
                chosen_y, chosen_h = chosen[4], chosen[6]
                overlap = max(
                    0,
                    min(y + h, chosen_y + chosen_h) - max(y, chosen_y),
                )
                overlap_ratio = overlap / float(min(h, chosen_h))
                if overlap_ratio > self.dash_max_vertical_overlap_ratio:
                    vertically_separate = False
                    break
            if vertically_separate:
                selected.append(candidate)
            if len(selected) >= self.dash_max_tracked_pieces:
                break

        filtered = np.zeros_like(white_mask)
        boxes = []
        for _distance, _bottom_x, label, x, y, w, h in selected:
            filtered[labels == label] = 255
            boxes.append((int(x), int(y), int(w), int(h)))
        return filtered, boxes

    def get_window_start_x(self, lane_name, fallback_x):
        """저장된 시작점이 있으면 사용하고, 첫 프레임만 색/모드 기준값을 쓴다."""
        start_x = self.window_start_x[lane_name]
        return float(fallback_x) if start_x is None and fallback_x is not None else start_x

    def update_window_start_x(self, lane_name, detected_x, image_width):
        """유효 검출값 쪽으로 다음 프레임 시작점을 조금씩 이동한다."""
        if detected_x is None:
            return
        previous_x = self.window_start_x[lane_name]
        if previous_x is None:
            next_x = float(detected_x)
        elif abs(float(detected_x) - previous_x) > self.max_window_start_jump_px:
            self.get_logger().warn(
                f'{lane_name} window start jump rejected: '
                f'{previous_x:.1f} -> {float(detected_x):.1f}',
                throttle_duration_sec=1.0,
            )
            return
        else:
            next_x = previous_x + self.window_start_adapt_rate * (
                float(detected_x) - previous_x
            )
        self.window_start_x[lane_name] = float(
            np.clip(next_x, 0, max(0, image_width - 1))
        )

    def track_lane_with_sliding_window(self, white_mask, base_x, margin, allow_gaps):
        """한 차선을 슬라이딩 윈도우로 추적한다.

        점선(`allow_gaps=True`)은 빈 창에서 박스 중심을 그대로 유지한다. 이후
        같은 위치 근처에서 흰 성분이 다시 잡힐 때만 박스 중심을 갱신한다.
        관측된 점선 조각은 직선 피팅으로 연결하며, 반환 x는 ROI 하단 위치다.
        """
        height, width = white_mask.shape
        window_height = max(1, height // self.num_windows)
        x_current = float(base_x)
        windows, collected_x, collected_y = [], [], []

        for i in range(self.num_windows):
            y_high = height - i * window_height
            y_low = max(0, height - (i + 1) * window_height)
            x_low = max(0, int(round(x_current - margin)))
            x_high = min(width, int(round(x_current + margin + 1)))
            windows.append((x_low, max(x_low, x_high - 1), y_low, y_high))
            if x_high <= x_low or y_high <= y_low:
                continue

            sub_mask = white_mask[y_low:y_high, x_low:x_high]
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                sub_mask, connectivity=8
            )
            candidates = [
                label for label in range(1, num_labels)
                if stats[label, cv2.CC_STAT_AREA] >= self.window_min_component_pixels
            ]
            if not candidates:
                # 점선 공백: 이전 슬라이딩 윈도우 박스 위치를 그대로 유지한다.
                continue

            label = min(
                candidates,
                key=lambda item: abs((centroids[item][0] + x_low) - x_current),
            )
            local_ys, local_xs = np.where(labels == label)
            xs, ys = local_xs + x_low, local_ys + y_low
            observed_x = float(xs.mean())
            x_current = observed_x
            collected_x.extend(xs.tolist())
            collected_y.extend(ys.tolist())

        if len(collected_x) < self.window_min_component_pixels:
            return None, windows, collected_x, collected_y
        unique_y = len(set(collected_y))
        if unique_y >= 3 and len(collected_x) >= 12:
            coeffs = np.polyfit(collected_y, collected_x, 1)
            line_x = float(np.polyval(coeffs, height - 1))
        else:
            line_x = float(np.mean(collected_x))
        return line_x, windows, collected_x, collected_y

    # ======================================================================
    # 디버그 시각화
    # ======================================================================
    def publish_debug(
        self, src_msg, frame, status, near_white_ratio,
        base_x=None, line_x=None, windows=None,
        points_x=None, points_y=None, lane_offset=None, lane_tracks=None,
        dashed_components=None,
    ):
        if not (self.debug_view or self.publish_debug_image):
            return

        # 이번 프레임이 미검출이면 직전에 유효했던 박스를 그대로 표시한다.
        if lane_tracks is None:
            lane_tracks = self.last_lane_tracks

        debug = frame.copy()
        height, width = debug.shape[:2]
        # ROI 영역 표시
        roi_top_y = int(np.clip(self.roi_top, 0, height - 1))
        roi_bottom_y = int(np.clip(self.roi_bottom - 1, 0, height - 1))
        cv2.rectangle(
            debug, (0, roi_top_y), (width - 1, roi_bottom_y), (0, 255, 255), 1
        )
        # 실제 차선 마스크에 적용한 사다리꼴 ROI(노랑)
        top_inset = int(np.clip(self.roi_trapezoid_top_inset_px, 0, width // 2))
        trapezoid = np.array([
            (top_inset, roi_top_y),
            (width - 1 - top_inset, roi_top_y),
            # 디버그 선도 카메라 화면의 전체 가로폭을 덮는다.
            (width - 1, roi_bottom_y),
            (0, roi_bottom_y),
        ], dtype=np.int32)
        cv2.polylines(debug, [trapezoid], True, (0, 255, 255), 2)
        # 중앙 점선으로 선택한 connected component(자홍).
        if dashed_components:
            for x, y, w, h in dashed_components:
                cv2.rectangle(
                    debug, (x, y), (x + w - 1, y + h - 1), (255, 0, 255), 2
                )
        # 현재 주행 모드의 하드코딩 중앙 점선 기준 위치(주황, "여기면 lane_offset=0")
        self.draw_dashed_vline(
            debug, self.dashed_reference_x_px,
            self.roi_top, self.roi_bottom, (0, 165, 255)
        )

        # 슬라이딩 윈도우 (파란 사각형, ROI 로컬 좌표 -> 전체 프레임 좌표로 오프셋)
        if windows:
            for x_low, x_high, y_low, y_high in windows:
                cv2.rectangle(
                    debug,
                    (x_low, y_low + self.roi_top),
                    (x_high, y_high + self.roi_top),
                    (255, 0, 0),
                    1,
                )

        # 윈도우에 잡힌 차선 픽셀 (초록 점)
        if points_x:
            for px, py in zip(points_x, points_y):
                cv2.circle(debug, (int(px), int(py) + self.roi_top), 1, (0, 255, 0), -1)

        if base_x is not None:
            cv2.circle(debug, (int(base_x), self.roi_bottom - 1), 5, (0, 0, 255), -1)
        if line_x is not None:
            cv2.circle(debug, (int(round(line_x)), self.roi_bottom - 1), 6, (0, 255, 0), 2)

        # 세 개의 독립 슬라이딩 윈도우 추적 결과: 왼 실선=파랑, 점선=노랑,
        # 오른 실선=빨강. 점선의 빈 구간도 윈도우 중심이 예측값으로 이어진다.
        if lane_tracks:
            track_colors = {
                'left': (255, 0, 0),
                'dashed': (0, 255, 255),
                'right': (0, 0, 255),
            }
            for name, (track_x, track_windows, _px, _py) in lane_tracks.items():
                color = track_colors[name]
                for x_low, x_high, y_low, y_high in track_windows:
                    cv2.rectangle(
                        debug,
                        (x_low, y_low + self.roi_top),
                        (x_high, y_high + self.roi_top),
                        color,
                        1,
                    )
                if track_x is not None:
                    cv2.circle(
                        debug, (int(round(track_x)), self.roi_bottom - 1),
                        5, color, -1,
                    )

        color = (0, 255, 0) if status == 'OK' else (0, 0, 255)
        lines = [
            f'status: {status}',
            f'lane_offset: {lane_offset if lane_offset is not None else self.last_offset}',
            f'mode: {self.driving_mode}, dashed_ref_x: {self.dashed_reference_x_px}',
            f'white_ratio: {near_white_ratio:.2f}',
        ]
        for i, text in enumerate(lines):
            cv2.putText(
                debug, text, (10, 20 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
            )

        if self.debug_view:
            cv2.imshow(self.window_name, debug)
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
        if self.debug_view:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MissionLaneOffsetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
