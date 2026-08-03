"""Mission lane offset node.

하나의 파일 안에서 다음을 모두 수행한다.
- HSV+YCrCb 흰색/초록/밝은 회색 분류와 최근접 보간 BEV 마스킹
- 횡단보도/정지선 가로 밴드 제거
- 바깥 차선(도로 밖 색이 옆에 붙은 흰 덩어리) 배제
- OSY 중앙 점선 슬라이딩 윈도우 추적
- /lane_info(1/2)에 따른 1lane/2lane 기준 전환

바깥 차선 배제(차로 변경 중 오인식 방지):
    1차로<->2차로를 옮기는 동안 중앙 점선이 화면을 가로질러 이동하는데, 이때
    1차로 왼쪽 바깥 실선이나 2차로 오른쪽 바깥 실선을 중앙 점선으로 잘못 잡는
    문제가 있었다. 트랙에서 **오른쪽 바깥 실선 밖은 초록 매트**, **왼쪽 바깥
    실선 밖은 밝은 회색 영역**이므로(중앙 점선은 양옆이 모두 아스팔트),
    흰색 덩어리를 조금 부풀린 이웃에 그 색이 임계값 이상 있고 색이 기대하는
    쪽(초록=오른쪽, 회색=왼쪽)에 있으면 바깥 실선으로 보고 후보에서 제외한다.

    timed_lane_offset_node 는 "초록이 안 보여도 직전 오른쪽 실선 근처면 실선"
    으로 유지하는데, 여기서는 그 반대 방향으로 기억을 쓴다. 즉 한 번 바깥
    실선으로 판정한 x 주변의 덩어리는 초록/회색이 화면에서 사라진 뒤에도 계속
    바깥 실선으로 취급해 중앙 점선으로 승격되지 않게 한다. 기억은 색으로 다시
    확인되지 않은 채 오래되거나(outer_memory_max_age_frames), 근처에서 아무
    덩어리도 이어지지 않으면(outer_memory_max_misses) 스스로 버려진다.
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16

import cv2

from .color_profiles import load_color_classes

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
        'ycrcb': {'y': (175, 255), 'cr': (81, 165), 'cb': (122, 170)},
    },
]

# 도로 "바깥" 색. 오른쪽 바깥 실선 밖은 초록 매트(timed_lane_offset_node와 동일
# 임계값), 왼쪽 바깥 실선 밖은 밝은 회색 영역이다. side는 흰 덩어리를 기준으로
# 그 색이 어느 쪽에 있어야 하는지를 뜻한다.
OUTER_COLOR_CLASSES = [
    {
        'name': 'green',
        'side': 'right',
        'color_bgr': (0, 255, 0),
        'hsv': {'h': (31, 60), 's': (48, 200), 'v': (0, 255)},
        'ycrcb': None,
    },
    {
        'name': 'light_gray',
        'side': 'left',
        'color_bgr': (211, 211, 211),
        'hsv': {'h': (77, 104), 's': (6, 30), 'v': (140, 180)},
        'ycrcb': None,
    },
]
# 흰 덩어리를 이 픽셀만큼 부풀린 이웃에서 바깥 색을 찾는다(BEV 좌표 기준).
OUTER_NEAR_DISTANCE_PX = 20
# 이웃 안 바깥 색 픽셀이 이 수 이상이어야 "바깥 실선"으로 인정한다.
# 진짜 초록 매트/회색 영역은 수천 픽셀이므로 넉넉히 잡아도 된다.
OUTER_MIN_PIXELS = 40
# 근거는 덩어리 bbox에서 기대하는 쪽으로 이만큼 더 바깥에서부터 센다.
OUTER_SIDE_MARGIN_PX = 2
# 밝은 흰 선 가장자리는 번짐 때문에 밝은 회색으로 분류되기 쉽다. 흰색에서 이
# 거리 안의 색은 "옆 영역" 근거로 쓰지 않는다(중앙 점선의 자기 테두리를 보고
# 바깥 실선이라고 오판하는 것을 막는 핵심 값).
OUTER_HALO_MARGIN_PX = 5
# 이보다 작은 흰 덩어리는 바깥 실선 판정 자체를 하지 않는다(노이즈).
OUTER_MIN_COMPONENT_PIXELS = 60
# 바깥 차선은 실선이므로 BEV 높이에서 이 비율 이상 이어져야 한다. 중앙 점선
# 조각 옆 아스팔트가 밝은 회색으로 잡히더라도 외곽선 기억을 만들지 않게 한다.
OUTER_MIN_VERTICAL_SPAN_RATIO = 0.60
# 기억한 바깥 실선 x에서 이 거리 안에 있는 덩어리는 색 근거가 없어도 바깥 실선.
OUTER_MEMORY_TOLERANCE_PX = 40
# 기억 위치를 이번 프레임에 이어진 덩어리 쪽으로 옮기는 비율(차선 변경 중 추적).
OUTER_MEMORY_ADAPT_RATE = 0.5
# 색 근거 없이 기억만으로 움직일 때 한 프레임 최대 이동량.
OUTER_MEMORY_MAX_DRIFT_PX = 15
# 색으로 다시 확인되지 않은 채 이 프레임 수가 지나면 기억을 버린다(자가 복구).
OUTER_MEMORY_MAX_AGE_FRAMES = 90
# 기억 근처에서 아무 덩어리도 이어지지 않은 프레임이 이만큼 쌓이면 기억을 버린다.
OUTER_MEMORY_MAX_MISSES = 10

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

# 시작 차선은 이 노드에서 정하지 않는다. mission_lane_main_node가 발행하는
# /lane_info를 단일 기준으로 사용한다.
# 640px BEV에서 차가 정상 위치일 때 보이는 중앙 점선의 x좌표.
# 1차선은 화면 오른쪽 점선, 2차선은 화면 왼쪽 점선을 각각 기준으로 삼는다.
DASHED_REFERENCE_X_PX_2LANE = 190
DASHED_REFERENCE_X_PX_1LANE = 510
# 기준선과 이만큼 차이 나면 lane_offset의 최대/최소값(+/-45)에 도달한다.
OFFSET_ERROR_LIMIT_PX = 195
LANE_OFFSET_LIMIT = 45
# 기준선 오차에 비례해 조향하도록 1.0을 사용한다. 10.0은 약 20px 오차만
# 생겨도 바로 +/-45로 포화되어 "떨어진 만큼" 조향하지 못한다.
OFFSET_KP = 2.0
# 한 프레임 사이 offset이 이 값보다 더 튀면 오검출로 보고 이전 값 유지
MAX_OFFSET_JUMP_PX = 80
# 중앙 점선 미검출 시 현재 차선 안쪽으로 꺾는 offset 누적량.
# 2차선은 좌회전(-), 1차선은 우회전(+) 방향이다.
NO_DASH_RECOVERY_STEP = 3
# 차로 전환 시 조향 기준을 한 프레임에 이동시킬 최대 픽셀 수. 검출 시작점은
# 이 값과 무관하게 기존에 추적하던 물리적 중앙선 위치를 계속 사용한다.
LANE_REFERENCE_TRANSITION_STEP_PX = 5.0

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
# 중앙선의 기준 조각은 위의 엄격한 조건으로 정하되, 그 조각과 같은 선상에 있는
# 짧은 점선 조각도 시각화/추적 연결에 사용할 수 있도록 완화한 보조 조건.
DASH_SUPPORT_MIN_VERTICAL_SPAN_PX = 15
DASH_SUPPORT_MIN_COMPONENT_PIXELS = 60
# ROI 높이 대부분을 계속 잇는 성분은 점선이 아니라 실선으로 본다.
DASH_MAX_VERTICAL_SPAN_RATIO = 0.92
# 서로 다른 높이에 나타나는 위/아래 점선 조각을 한 트랙으로 묶는다.
DASH_MAX_TRACKED_PIECES = 4
DASH_MAX_VERTICAL_OVERLAP_RATIO = 0.25
# 같은 중앙선으로 선택된 인접 점선 조각 사이를 선으로 연결하는 조건.
DASH_CONNECT_MAX_VERTICAL_GAP_PX = 200
DASH_CONNECT_MAX_HORIZONTAL_GAP_PX = 100
DASH_CONNECT_THICKNESS_PX = 8
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
WINDOW_NAME = 'mission_lane_offset_debug'
WHITE_MASK_WINDOW_NAME = 'mission_lane_offset_white_mask'
CONNECTED_DASH_WINDOW_NAME = 'mission_lane_offset_connected_dash'
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

        try:
            color_classes, self.color_profile_name, profile_path = load_color_classes(
                list(COLOR_CLASSES) + list(OUTER_COLOR_CLASSES), 'mission'
            )
            self.color_classes = [
                item for item in color_classes if item['name'] == 'white'
            ]
            self.outer_color_classes = [
                item for item in color_classes if item['name'] != 'white'
            ]
            self.get_logger().info(
                f'Color profile={self.color_profile_name} ({profile_path})'
            )
        except (OSError, ValueError) as error:
            self.color_profile_name = 'built-in fallback'
            self.color_classes = COLOR_CLASSES
            self.outer_color_classes = OUTER_COLOR_CLASSES
            self.get_logger().error(
                f'Color profile load failed; using built-in values: {error}'
            )

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
        self.declare_parameter('outer_near_distance_px', OUTER_NEAR_DISTANCE_PX)
        self.declare_parameter('outer_min_pixels', OUTER_MIN_PIXELS)
        self.declare_parameter('outer_side_margin_px', OUTER_SIDE_MARGIN_PX)
        self.declare_parameter('outer_halo_margin_px', OUTER_HALO_MARGIN_PX)
        self.declare_parameter(
            'outer_min_component_pixels', OUTER_MIN_COMPONENT_PIXELS
        )
        self.declare_parameter(
            'outer_min_vertical_span_ratio', OUTER_MIN_VERTICAL_SPAN_RATIO
        )
        self.declare_parameter(
            'outer_memory_tolerance_px', OUTER_MEMORY_TOLERANCE_PX
        )
        self.declare_parameter('outer_memory_adapt_rate', OUTER_MEMORY_ADAPT_RATE)
        self.declare_parameter(
            'outer_memory_max_drift_px', OUTER_MEMORY_MAX_DRIFT_PX
        )
        self.declare_parameter(
            'outer_memory_max_age_frames', OUTER_MEMORY_MAX_AGE_FRAMES
        )
        self.declare_parameter('outer_memory_max_misses', OUTER_MEMORY_MAX_MISSES)
        self.declare_parameter('dashed_reference_x_px_2lane', DASHED_REFERENCE_X_PX_2LANE)
        self.declare_parameter('dashed_reference_x_px_1lane', DASHED_REFERENCE_X_PX_1LANE)
        self.declare_parameter('offset_error_limit_px', OFFSET_ERROR_LIMIT_PX)
        self.declare_parameter('lane_offset_limit', LANE_OFFSET_LIMIT)
        self.declare_parameter('offset_kp', OFFSET_KP)
        self.declare_parameter('max_offset_jump_px', MAX_OFFSET_JUMP_PX)
        self.declare_parameter(
            'lane_reference_transition_step_px',
            LANE_REFERENCE_TRANSITION_STEP_PX,
        )
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
            'dash_support_min_vertical_span_px',
            DASH_SUPPORT_MIN_VERTICAL_SPAN_PX,
        )
        self.declare_parameter(
            'dash_support_min_component_pixels',
            DASH_SUPPORT_MIN_COMPONENT_PIXELS,
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
        self.declare_parameter(
            'dash_connect_max_vertical_gap_px',
            DASH_CONNECT_MAX_VERTICAL_GAP_PX,
        )
        self.declare_parameter(
            'dash_connect_max_horizontal_gap_px',
            DASH_CONNECT_MAX_HORIZONTAL_GAP_PX,
        )
        self.declare_parameter(
            'dash_connect_thickness_px', DASH_CONNECT_THICKNESS_PX
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
        self.outer_near_distance_px = max(
            1, int(self.get_parameter('outer_near_distance_px').value)
        )
        self.outer_min_pixels = max(
            1, int(self.get_parameter('outer_min_pixels').value)
        )
        self.outer_side_margin_px = max(
            0, int(self.get_parameter('outer_side_margin_px').value)
        )
        self.outer_halo_margin_px = max(
            0, int(self.get_parameter('outer_halo_margin_px').value)
        )
        self.outer_min_component_pixels = max(
            1, int(self.get_parameter('outer_min_component_pixels').value)
        )
        self.outer_min_vertical_span_ratio = float(np.clip(
            self.get_parameter('outer_min_vertical_span_ratio').value,
            0.0,
            1.0,
        ))
        self.outer_memory_tolerance_px = max(
            0.0, float(self.get_parameter('outer_memory_tolerance_px').value)
        )
        self.outer_memory_adapt_rate = float(np.clip(
            self.get_parameter('outer_memory_adapt_rate').value, 0.0, 1.0
        ))
        self.outer_memory_max_drift_px = max(
            0.0, float(self.get_parameter('outer_memory_max_drift_px').value)
        )
        self.outer_memory_max_age_frames = max(
            1, int(self.get_parameter('outer_memory_max_age_frames').value)
        )
        self.outer_memory_max_misses = max(
            1, int(self.get_parameter('outer_memory_max_misses').value)
        )
        self.dashed_reference_x_px_2lane = int(
            self.get_parameter('dashed_reference_x_px_2lane').value
        )
        self.dashed_reference_x_px_1lane = int(
            self.get_parameter('dashed_reference_x_px_1lane').value
        )
        self.driving_mode = None
        self.dashed_reference_x_px = None
        self.dashed_reference_target_x_px = None
        self.lane_reference_transition_active = False
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
        self.lane_reference_transition_step_px = max(
            0.1,
            float(
                self.get_parameter(
                    'lane_reference_transition_step_px'
                ).value
            ),
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
        self.dash_support_min_vertical_span_px = max(
            1,
            int(
                self.get_parameter(
                    'dash_support_min_vertical_span_px'
                ).value
            ),
        )
        self.dash_support_min_component_pixels = max(
            1,
            int(
                self.get_parameter(
                    'dash_support_min_component_pixels'
                ).value
            ),
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
        self.dash_connect_max_vertical_gap_px = max(
            0,
            int(
                self.get_parameter(
                    'dash_connect_max_vertical_gap_px'
                ).value
            ),
        )
        self.dash_connect_max_horizontal_gap_px = max(
            0,
            int(
                self.get_parameter(
                    'dash_connect_max_horizontal_gap_px'
                ).value
            ),
        )
        self.dash_connect_thickness_px = max(
            1, int(self.get_parameter('dash_connect_thickness_px').value)
        )
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
        self.connected_dash_window_name = CONNECTED_DASH_WINDOW_NAME
        self.publish_debug_image = False
        self.debug_image_topic = DEBUG_IMAGE_TOPIC

        # 마지막으로 발행한(유효했던) offset. 오검출 프레임에서는 이 값을 그대로 재사용.
        self.last_offset = 0
        # 다음 프레임의 중앙 점선 슬라이딩 윈도우 시작점.
        self.window_start_x = {'dashed': None}
        # 마지막으로 유효했던 중앙 점선 슬라이딩 윈도우. 인식 공백에서도 디버그
        # 박스가 사라지지 않도록 유지한다.
        self.last_lane_tracks = None
        # 바깥 실선(왼쪽/오른쪽)으로 판정한 x 기억. 초록/회색이 사라져도 그 근처
        # 덩어리를 중앙 점선으로 오인하지 않도록 유지한다.
        # {'left'|'right': {'x': float, 'age': int, 'misses': int} 또는 None}
        self.outer_memory = {
            color_class['side']: None for color_class in self.outer_color_classes
        }
        # 이번 프레임에 바깥 실선으로 제외한 덩어리 박스(디버그 표시용).
        self.last_outer_boxes = []
        # 이번 프레임에서 같은 중앙선으로 이어 붙인 점선 조각의 끝점 쌍.
        self.last_dashed_connections = []
        self.last_connected_dashed_mask = None

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.offset_pub = self.create_publisher(Int16, self.lane_offset_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, qos)
        self.create_subscription(Int16, LANE_INFO_TOPIC, self.lane_info_callback, 10)
        self.create_subscription(Image, self.image_topic, self.image_callback, qos)

        self.get_logger().info(
            f'Subscribing {self.image_topic}, publishing {self.lane_offset_topic}, '
            f'waiting for {LANE_INFO_TOPIC} to select the starting lane, '
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
        """기존 중앙선 추적을 유지한 채 새 차로 기준으로 서서히 전환한다."""
        mode = str(mode).lower()
        if mode not in ('1lane', '2lane'):
            return
        if not force and mode == self.driving_mode:
            return

        previous_track_x = self.window_start_x.get('dashed')
        self.driving_mode = mode
        new_reference_x = float(
            self.dashed_reference_x_px_1lane
            if mode == '1lane'
            else self.dashed_reference_x_px_2lane
        )

        # 최초 /lane_info에서는 해당 차로 기준으로 즉시 초기화한다. 이후 모드
        # 변경에서는 현재 기준값을 유지하고 목표값만 새 차로 기준으로 바꾼다.
        if force or self.dashed_reference_x_px is None:
            self.dashed_reference_x_px = new_reference_x
            self.lane_reference_transition_active = False
        else:
            self.lane_reference_transition_active = not np.isclose(
                self.dashed_reference_x_px, new_reference_x
            )
        self.dashed_reference_target_x_px = new_reference_x

        # 조향 기준과 검출 시작점은 독립적이다. 차로 전환 중에도 직전에 보던
        # 동일한 물리적 중앙선 주변에서 다음 프레임 탐색을 계속한다.
        if previous_track_x is None:
            previous_track_x = float(self.dashed_reference_x_px)
        self.window_start_x = {'dashed': float(previous_track_x)}
        self.get_logger().info(
            f'Driving mode changed to {mode}; '
            f'dashed_reference_x={self.dashed_reference_x_px:.1f} -> '
            f'{self.dashed_reference_target_x_px:.1f}, '
            f'continuing_track_x={previous_track_x:.1f}'
        )

    def advance_lane_reference_transition(self):
        """현재 조향 기준을 새 차로 목표 기준 쪽으로 한 단계 이동한다."""
        if not self.lane_reference_transition_active:
            return False

        delta = (
            self.dashed_reference_target_x_px
            - self.dashed_reference_x_px
        )
        step = self.lane_reference_transition_step_px
        if abs(delta) <= step:
            self.dashed_reference_x_px = self.dashed_reference_target_x_px
            self.lane_reference_transition_active = False
            self.get_logger().info(
                f'Lane reference transition complete: '
                f'{self.dashed_reference_x_px:.1f}'
            )
        else:
            self.dashed_reference_x_px += math.copysign(step, delta)
        return True

    def image_callback(self, msg):
        # 시작 차선의 단일 기준은 mission_lane_main_node의 DRIVING_MODE이다.
        # 첫 /lane_info 전에 임의 차선 기준으로 잘못된 offset을 내보내지 않는다.
        if self.driving_mode is None:
            self.get_logger().info(
                f'Waiting for {LANE_INFO_TOPIC}; skipping image',
                throttle_duration_sec=1.0,
            )
            return

        self.advance_lane_reference_transition()

        frame = self.to_bgr(msg)
        if frame is None:
            return

        segmented = self.segment_colors(frame)
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
        # 바깥 색은 사다리꼴 밖(화면 가장자리)에도 근거로 쓸 수 있어야 하므로
        # ROI 마스크를 적용하지 않는다.
        outer_masks = self.make_outer_masks(roi)
        self.show_debug_mask(white_mask, outer_masks)

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
            white_mask, dashed_start_x, outer_masks
        )
        self.show_connected_dashed_view()
        if dashed_mask is None:
            recovery_offset = self.apply_no_dash_recovery()
            self.get_logger().warn(
                f'No center-dash component; {self.driving_mode} recovery '
                f'offset={recovery_offset}',
                throttle_duration_sec=1.0,
            )
            self.publish_offset(recovery_offset)
            self.publish_debug(
                msg, frame, 'NO DASH', near_white_ratio,
                base_x=dashed_start_x, lane_offset=recovery_offset,
            )
            return

        dashed_track = self.track_lane_with_sliding_window(
            dashed_mask, dashed_start_x, self.dashed_window_margin,
            allow_gaps=True,
        )
        dashed_x, windows, points_x, points_y = dashed_track
        lane_tracks = {'dashed': dashed_track}
        if dashed_x is None:
            recovery_offset = self.apply_no_dash_recovery()
            self.get_logger().warn(
                f'Center dash sliding-window detection failed; '
                f'{self.driving_mode} recovery offset={recovery_offset}',
                throttle_duration_sec=1.0,
            )
            self.publish_offset(recovery_offset)
            self.publish_debug(
                msg, frame, 'DASH TRACK FAILED', near_white_ratio,
                base_x=dashed_start_x, windows=windows,
                lane_tracks=lane_tracks, lane_offset=recovery_offset,
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

    def apply_no_dash_recovery(self):
        """중앙선 미검출 동안 차선별 복구 방향으로 offset을 누적한다."""
        direction = -1 if self.driving_mode == '2lane' else 1
        self.last_offset = int(np.clip(
            self.last_offset + direction * NO_DASH_RECOVERY_STEP,
            -self.lane_offset_limit,
            self.lane_offset_limit,
        ))
        return self.last_offset

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
    def segment_colors(self, frame):
        """원본을 흰색 + 도로 바깥 색으로 분류하고 나머지는 검정으로 만든다.

        흰색을 마지막에 칠해 겹치는 화소는 항상 흰색이 이긴다. 바깥 색 추가가
        기존 중앙 점선 검출 결과를 바꾸지 않게 하기 위한 순서다.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        segmented = np.full_like(
            frame, BACKGROUND_COLOR_BGR, dtype=np.uint8
        )
        for color_class in self.outer_color_classes + self.color_classes:
            mask = class_mask(hsv, ycrcb, color_class)
            segmented[mask > 0] = color_class['color_bgr']
        return segmented

    def make_outer_masks(self, bev):
        """BEV에서 도로 바깥 색 마스크를 side('left'/'right')별로 만든다.

        흰 선 가장자리 번짐(halo)은 제거한다. 밝은 흰 선의 테두리는 밝은 회색
        범위에 쉽게 들어가서, 그대로 두면 중앙 점선이 자기 테두리를 근거로
        "옆에 회색이 있다 = 왼쪽 바깥 실선"으로 오판된다.
        """
        white = cv2.inRange(bev, (255, 255, 255), (255, 255, 255))
        if self.outer_halo_margin_px > 0:
            kernel_size = self.outer_halo_margin_px * 2 + 1
            halo = cv2.dilate(
                white, np.ones((kernel_size, kernel_size), dtype=np.uint8)
            )
        else:
            halo = np.zeros_like(white)

        masks = {}
        for color_class in self.outer_color_classes:
            color = color_class['color_bgr']
            mask = cv2.inRange(bev, color, color)
            masks[color_class['side']] = cv2.bitwise_and(
                mask, cv2.bitwise_not(halo)
            )
        return masks

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

    def show_debug_mask(self, white_mask, outer_masks=None):
        """흰색 마스크와 도로 바깥 색 마스크를 한 디버그 창에 겹쳐 표시한다."""
        if not self.debug_view:
            return

        white_debug = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        if outer_masks:
            for color_class in self.outer_color_classes:
                mask = outer_masks.get(color_class['side'])
                if mask is not None:
                    white_debug[mask > 0] = color_class['color_bgr']
        cv2.imshow(self.white_mask_window_name, white_debug)

    # ======================================================================
    # 바깥 차선(도로 밖 색이 옆에 붙은 흰 덩어리) 판정
    # ======================================================================
    def classify_outer_lines(self, num_labels, labels, stats, centroids, outer_masks):
        """바깥 실선인 흰 덩어리를 찾아 {label: (side, source)} 로 돌려준다.

        source='color' 는 이번 프레임에 초록/회색으로 확인한 것, 'memory' 는
        색 근거 없이 직전 기억 위치로 이어간 것이다. 차로를 옮기는 동안 바깥
        색이 화면에서 사라져도 그 덩어리가 중앙 점선 후보로 올라오지 않게 한다.
        """
        kernel_size = self.outer_near_distance_px * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        height, width = labels.shape
        big_labels = [
            label for label in range(1, num_labels)
            if (
                stats[label, cv2.CC_STAT_AREA]
                >= self.outer_min_component_pixels
                and stats[label, cv2.CC_STAT_HEIGHT] / float(height)
                >= self.outer_min_vertical_span_ratio
            )
        ]

        sides = {}       # label -> (side, source)
        confirmed = {}   # side -> x (이번 프레임 색 근거로 확인된 위치)
        matched = {}     # side -> x (색이든 기억이든 이어진 위치)

        pad = self.outer_near_distance_px
        usable = {
            color_class['side']: (
                outer_masks.get(color_class['side']) is not None
                and outer_masks[color_class['side']].any()
            )
            for color_class in self.outer_color_classes
        }

        for label in big_labels:
            comp_x = float(centroids[label][0])
            left = int(stats[label, cv2.CC_STAT_LEFT])
            top = int(stats[label, cv2.CC_STAT_TOP])
            right = left + int(stats[label, cv2.CC_STAT_WIDTH]) - 1
            bottom = top + int(stats[label, cv2.CC_STAT_HEIGHT]) - 1
            # 이웃 계산은 덩어리 bbox 주변 잘라낸 영역에서만 한다(전체 화면
            # 팽창은 프레임당 수십 ms가 든다).
            x0, x1 = max(0, left - pad), min(width, right + pad + 1)
            y0, y1 = max(0, top - pad), min(height, bottom + pad + 1)
            neighborhood = None
            for color_class in self.outer_color_classes:
                side = color_class['side']
                if not usable[side]:
                    continue
                if neighborhood is None:
                    comp = (labels[y0:y1, x0:x1] == label).astype(np.uint8)
                    neighborhood = cv2.dilate(comp, kernel)
                # 덩어리 bbox의 기대하는 쪽 "바깥"만 근거로 센다. 양쪽을 다 세고
                # 평균 x로 판정하면 대칭인 번짐에서 좌/우가 뒤집힐 수 있다.
                if side == 'left':
                    band = slice(x0, max(x0, left - self.outer_side_margin_px))
                else:
                    band = slice(min(x1, right + 1 + self.outer_side_margin_px), x1)
                if band.stop <= band.start:
                    continue
                sub_mask = outer_masks[side][y0:y1, band]
                sub_neighborhood = neighborhood[:, band.start - x0:band.stop - x0]
                near = cv2.bitwise_and(sub_mask, sub_neighborhood)
                if int(cv2.countNonZero(near)) < self.outer_min_pixels:
                    continue

                sides[label] = (side, 'color')
                # 같은 쪽에서 여러 덩어리가 확인되면 가장 바깥쪽 것이 바깥 실선이다.
                if side not in confirmed:
                    confirmed[side] = comp_x
                elif side == 'left':
                    confirmed[side] = min(confirmed[side], comp_x)
                else:
                    confirmed[side] = max(confirmed[side], comp_x)
                matched[side] = confirmed[side]
                break

        # 색 근거가 사라진 쪽은 기억한 x 주변 덩어리를 계속 바깥 실선으로 본다.
        # 여러 개가 걸리면 기억에 가장 가까운 하나만 기억 갱신에 쓴다.
        nearest = {}  # side -> (distance, x)
        for label in big_labels:
            if label in sides:
                continue
            x, _y, w, _h, _area = stats[label]
            for side, memory in self.outer_memory.items():
                if memory is None or side in confirmed:
                    continue
                distance = max(
                    float(x) - memory['x'],
                    memory['x'] - float(x + w - 1),
                    0.0,
                )
                if distance > self.outer_memory_tolerance_px:
                    continue
                sides[label] = (side, 'memory')
                comp_x = float(centroids[label][0])
                if side not in nearest or distance < nearest[side][0]:
                    nearest[side] = (distance, comp_x)
                break
        for side, (_distance, comp_x) in nearest.items():
            matched[side] = comp_x

        self.update_outer_memory(confirmed, matched)
        return sides

    def update_outer_memory(self, confirmed, matched):
        """바깥 실선 기억을 갱신하고, 오래되거나 끊긴 기억은 버린다."""
        for side in list(self.outer_memory):
            memory = self.outer_memory[side]
            if side in confirmed:
                x = confirmed[side]
                if memory is None:
                    self.outer_memory[side] = {'x': x, 'age': 0, 'misses': 0}
                else:
                    memory['x'] += self.outer_memory_adapt_rate * (x - memory['x'])
                    memory['age'] = 0
                    memory['misses'] = 0
                continue
            if memory is None:
                continue

            if side in matched:
                # 색은 못 봤지만 같은 위치의 덩어리를 이어 잡았다: 위치만 따라간다.
                # 단 도로 "바깥" 방향으로만, 그리고 한 프레임 이동량을 제한한다.
                # 색 근거 없이 안쪽으로 끌려가면 결국 중앙 점선을 삼켜버린다.
                # (바깥 실선이 실제로 안쪽으로 들어오는 상황이면 그 옆 초록/회색도
                #  화면 안에 있으므로 색으로 다시 확인된다.)
                step = float(np.clip(
                    self.outer_memory_adapt_rate * (matched[side] - memory['x']),
                    -self.outer_memory_max_drift_px,
                    self.outer_memory_max_drift_px,
                ))
                step = min(step, 0.0) if side == 'left' else max(step, 0.0)
                memory['x'] += step
                memory['misses'] = 0
            else:
                memory['misses'] += 1
            memory['age'] += 1

            if (
                memory['misses'] > self.outer_memory_max_misses
                or memory['age'] > self.outer_memory_max_age_frames
            ):
                self.get_logger().info(
                    f'Forgetting {side} outer-line memory at x={memory["x"]:.1f} '
                    f'(age={memory["age"]}, misses={memory["misses"]})'
                )
                self.outer_memory[side] = None

    def find_center_dashed_component(self, white_mask, expected_x, outer_masks=None):
        """기준 x 주변에서 점선 모양에 가장 가까운 흰 성분 하나를 반환한다."""
        self.last_dashed_connections = []
        self.last_connected_dashed_mask = np.zeros_like(white_mask)
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
        outer_sides = self.classify_outer_lines(
            num_labels, labels, stats, centroids, outer_masks or {}
        )
        self.last_outer_boxes = [
            (
                int(stats[label, cv2.CC_STAT_LEFT]),
                int(stats[label, cv2.CC_STAT_TOP]),
                int(stats[label, cv2.CC_STAT_WIDTH]),
                int(stats[label, cv2.CC_STAT_HEIGHT]),
                side,
                source,
            )
            for label, (side, source) in outer_sides.items()
        ]
        candidates = []
        support_candidates = []
        for label in range(1, num_labels):
            if label in outer_sides:
                # 바깥 실선(초록/회색이 옆에 있거나, 직전에 그렇게 판정한 위치)
                # 은 중앙 점선 후보에서 제외한다.
                continue
            x, y, w, h, area = stats[label]
            spans_full_height = y <= 2 and y + h >= height - 2
            vertical_span_ratio = h / float(height)
            average_width = area / float(h)
            common_shape_ok = (
                w >= self.dash_min_width_px
                and average_width >= self.dash_min_average_width_px
                and average_width <= self.dash_max_average_width_px
                and h > w
                and not spans_full_height
                and vertical_span_ratio <= self.dash_max_vertical_span_ratio
            )
            if not common_shape_ok:
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
            if min(distance, bbox_distance) > self.dashed_window_margin:
                continue

            candidate = (distance, bottom_x, label, x, y, w, h)
            if (
                h >= self.dash_support_min_vertical_span_px
                and area >= self.dash_support_min_component_pixels
            ):
                support_candidates.append(candidate)
            if (
                h < self.dash_min_vertical_span_px
                or area < self.dash_min_component_pixels
            ):
                # 짧은 조각은 단독 중앙선으로 채택하지 않고, 아래에서 엄격한
                # 기준 조각과 같은 선상일 때만 보조 조각으로 연결한다.
                continue
            candidates.append(candidate)

        if not candidates:
            return None, []

        # 하단 점선 조각을 기준으로 잡고, y구간이 겹치지 않는 위쪽/아래쪽
        # 조각도 같은 트랙에 추가한다. 같은 높이의 평행 실선은 제외된다.
        primary = min(candidates, key=lambda candidate: candidate[0])
        selected = [primary]
        for candidate in sorted(support_candidates, key=lambda item: item[0]):
            if candidate is primary:
                continue
            _distance, bottom_x, _label, _x, y, _w, h = candidate
            # 첨부 코드의 last_line_x lock과 같은 원리로, 기준 조각을 하단까지
            # 연장했을 때의 x와 너무 먼 조각은 다른 흰 선으로 보고 제외한다.
            if (
                abs(bottom_x - primary[1])
                > self.dash_connect_max_horizontal_gap_px
            ):
                continue
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
        self.connect_selected_dashed_components(filtered, labels, selected)
        self.last_connected_dashed_mask = filtered.copy()
        return filtered, boxes

    def connect_selected_dashed_components(self, mask, labels, selected):
        """같은 중앙선의 위·아래 점선 조각 사이 공백을 선분으로 채운다."""
        geometries = []
        for candidate in selected:
            label = candidate[2]
            ys, xs = np.where(labels == label)
            if ys.size == 0:
                continue
            y_top = int(ys.min())
            y_bottom = int(ys.max())
            if len(np.unique(ys)) >= 3:
                slope, intercept = np.polyfit(ys, xs, 1)
                x_top = float(slope * y_top + intercept)
                x_bottom = float(slope * y_bottom + intercept)
            else:
                x_top = x_bottom = float(xs.mean())
            geometries.append({
                'top': (int(round(x_top)), y_top),
                'bottom': (int(round(x_bottom)), y_bottom),
            })

        geometries.sort(key=lambda item: item['top'][1])
        for upper, lower in zip(geometries, geometries[1:]):
            start = upper['bottom']
            end = lower['top']
            vertical_gap = end[1] - start[1]
            horizontal_gap = abs(end[0] - start[0])
            if vertical_gap <= 0:
                continue
            if vertical_gap > self.dash_connect_max_vertical_gap_px:
                continue
            if horizontal_gap > self.dash_connect_max_horizontal_gap_px:
                continue
            cv2.line(
                mask,
                start,
                end,
                255,
                self.dash_connect_thickness_px,
                cv2.LINE_8,
            )
            self.last_dashed_connections.append((start, end))

    def show_connected_dashed_view(self):
        """원본 점선 조각은 흰색, 이어 붙인 구간은 노란색으로 표시한다."""
        if not self.debug_view or self.last_connected_dashed_mask is None:
            return
        debug = cv2.cvtColor(
            self.last_connected_dashed_mask, cv2.COLOR_GRAY2BGR
        )
        for start, end in self.last_dashed_connections:
            cv2.line(
                debug,
                start,
                end,
                (0, 255, 255),
                self.dash_connect_thickness_px,
                cv2.LINE_AA,
            )
        cv2.imshow(self.connected_dash_window_name, debug)

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
        # 바깥 실선으로 제외한 덩어리. 색으로 확인=하늘색 실선 박스,
        # 기억으로 유지=하늘색 점선 느낌(얇은 박스)으로 구분해 그린다.
        for x, y, w, h, side, source in self.last_outer_boxes:
            thickness = 2 if source == 'color' else 1
            cv2.rectangle(
                debug, (x, y), (x + w - 1, y + h - 1), (255, 255, 0), thickness
            )
            cv2.putText(
                debug, f'{side}-outer({source})', (x, max(12, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA,
            )
        # 바깥 실선 기억 위치(하늘색 세로 점선)
        for side, memory in self.outer_memory.items():
            if memory is None:
                continue
            self.draw_dashed_vline(
                debug, int(round(memory['x'])), self.roi_top, self.roi_bottom,
                (255, 255, 0), dash=4,
            )

        # 중앙 점선으로 선택한 connected component(자홍).
        if dashed_components:
            for x, y, w, h in dashed_components:
                cv2.rectangle(
                    debug, (x, y), (x + w - 1, y + h - 1), (255, 0, 255), 2
                )
        # 같은 중앙선으로 판정해 점선 공백을 이어 붙인 선분(굵은 노랑).
        for start, end in self.last_dashed_connections:
            cv2.line(
                debug,
                (start[0], start[1] + self.roi_top),
                (end[0], end[1] + self.roi_top),
                (0, 255, 255),
                3,
                cv2.LINE_AA,
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
            f'mode: {self.driving_mode}, dashed_ref_x: '
            f'{self.dashed_reference_x_px:.1f} -> '
            f'{self.dashed_reference_target_x_px:.1f}',
            f'white_ratio: {near_white_ratio:.2f}',
            f'outer: {self.describe_outer_memory()}',
            f'dash_links: {len(self.last_dashed_connections)} (yellow)',
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

    def describe_outer_memory(self):
        """디버그 텍스트용 바깥 실선 기억 요약."""
        parts = []
        for side, memory in self.outer_memory.items():
            if memory is None:
                parts.append(f'{side}=--')
            else:
                parts.append(
                    f'{side}={memory["x"]:.0f}(age{memory["age"]})'
                )
        return ' '.join(parts)

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
