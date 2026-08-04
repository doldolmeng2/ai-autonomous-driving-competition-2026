"""Track mission-lane outer solid lines and publish steering offsets.

2lane follows the white solid line beside the green road edge. 1lane follows
the white solid line beside the light-gray road edge. During a lane change the
node steers toward the destination lane until its corresponding solid line is
visible, then resumes line-based offset control.
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

######################## 구독/발행 토픽 ########################
IMAGE_TOPIC = '/camera/high/image_raw'
LANE_OFFSET_TOPIC = '/lane_offset'
LANE_INFO_TOPIC = '/lane_info'
LANE_CHANGE_COMPLETE_TOPIC = '/lane_change_complete'
################################################################


####################### 바깥 차선 필터/기억 #######################
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
# 회색 도로 경계는 BEV 왼쪽 절반에서만 근거로 사용한다.
LIGHT_GRAY_MASK_MAX_X_RATIO = 0.5
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
################################################################


######################## BEV/ROI 전처리 ########################
# timed 노드와 같이 원본 영상 높이의 40%부터 최하단까지 BEV로
# 펼친 뒤, BEV 전체를 추가 crop/사다리꼴 마스크 없이 탐색 ROI로 쓴다.
BEV_Y_TOP_RATIO = 0.4
BEV_Y_BOTTOM_RATIO = 1.0
BEV_TOP_WIDTH_RATIO = 1.0
BEV_BOTTOM_WIDTH_RATIO = 0.7
BEV_OUTPUT_WIDTH = 640
BEV_OUTPUT_HEIGHT = 0

# 차선보다 훨씬 긴 가로 성분(정지선/횡단보도)을 제거한다.
HORIZONTAL_RUN_MIN_PX = 50
HORIZONTAL_ERASE_HALF_BAND_PX = 3

# 흰색 과검출 검사에 쓰는 근접(BEV 하단) 밴드 높이
NEAR_FIELD_ROWS = 100
# 근접 밴드에서 흰색 비율이 이 값을 넘으면 횡단보도 등으로 판단하고 무시
WHITE_OVERLOAD_RATIO = 0.15
################################################################


######################## 차선/조향 기준 ########################
# 시작 차선은 이 노드에서 정하지 않는다. mission_lane_main_node가 발행하는
# /lane_info를 단일 기준으로 사용한다.
# 640px BEV에서 차가 정상 위치일 때 보이는 중앙 점선의 x좌표.
# 1차선은 화면 오른쪽 점선, 2차선은 화면 왼쪽 점선을 각각 기준으로 삼는다.
DASHED_REFERENCE_X_PX_2LANE = 190
DASHED_REFERENCE_X_PX_1LANE = 510
# 기준선과 이만큼 차이 나면 lane_offset의 최대/최소값(+/-45)에 도달한다.
OFFSET_ERROR_LIMIT_PX = 195
LANE_OFFSET_LIMIT = 45
# 한 프레임 사이 offset이 이 값보다 더 튀면 오검출로 보고 이전 값 유지
MAX_OFFSET_JUMP_PX = 80
# 중앙 점선 미검출 시 현재 차선 안쪽으로 꺾는 offset 누적량.
# 2차선은 좌회전(-), 1차선은 우회전(+) 방향이다.
NO_DASH_RECOVERY_STEP = 3
# 차로 전환 시 조향 기준을 한 프레임에 이동시킬 최대 픽셀 수. 검출 시작점은
# 이 값과 무관하게 기존에 추적하던 물리적 중앙선 위치를 계속 사용한다.
LANE_REFERENCE_TRANSITION_STEP_PX = 5.0

# 색상 경계 실선 주행 기준. 2차선은 timed 노드의 오른쪽 실선
# 기준을 공유하고, 1차선 왼쪽 실선 기준은 임시로 100px을 쓴다.
SOLID_REFERENCE_X_PX_2LANE = 540
SOLID_REFERENCE_X_PX_1LANE = 100
# 2->1은 왼조향, 1->2는 우조향을 최대 offset으로 유지한다.
LANE_CHANGE_STEER_OFFSET = 45
# 색상 근거가 붙은 흰 실선 후보 필터와 근접 측정 영역.
SOLID_MIN_COMPONENT_AREA = 50
SOLID_MIN_LINE_HEIGHT_PX = 25
SOLID_SMALL_COMPACT_MAX_AREA = 1800
SOLID_SMALL_COMPACT_MAX_SIDE_PX = 90
SOLID_SMALL_COMPACT_MIN_ELONGATION = 2.0
SOLID_NEAR_ROWS = 80
SOLID_NEAR_MIN_PIXELS = 20
################################################################


######################## 점선 추적/연결 ########################
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
################################################################


######################## 디버그 시각화 ########################
# 디버그 시각화: ROI/차선/슬라이딩 윈도우를 그린 화면을 바로 OpenCV 창으로 띄운다.
# (bag/카메라 토픽만 켜져 있으면, 이 노드 실행만으로 인식 화면이 뜬다.)
# 실차 대회 주행 시에는 CPU 절약을 위해 False로 끄는 것을 권장.
DEBUG_VIEW = True
WINDOW_NAME = 'mission_lane_offset_debug'
WHITE_MASK_WINDOW_NAME = 'mission_lane_offset_white_mask'
CONNECTED_DASH_WINDOW_NAME = 'mission_lane_offset_connected_dash'
DEBUG_IMAGE_TOPIC = '/lane_offset/debug_image'
################################################################


def class_mask(hsv, ycrcb, color_class):
    """light_gray는 YCrCb만, 나머지 색은 설정된 색 공간을 쓴다."""
    ycrcb_range = color_class['ycrcb']
    if color_class['name'] == 'light_gray':
        y_lo, y_hi = ycrcb_range['y']
        cr_lo, cr_hi = ycrcb_range['cr']
        cb_lo, cb_hi = ycrcb_range['cb']
        return cv2.inRange(
            ycrcb, (y_lo, cr_lo, cb_lo), (y_hi, cr_hi, cb_hi)
        )

    h_lo, h_hi = color_class['hsv']['h']
    s_lo, s_hi = color_class['hsv']['s']
    v_lo, v_hi = color_class['hsv']['v']
    mask = cv2.inRange(hsv, (h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi))

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
    """/camera/high/image_raw -> 색상 경계 실선 기준 /lane_offset 발행."""

    def __init__(self):
        super().__init__('mission_lane_offset_node')

        color_classes, self.color_profile_name, profile_path = load_color_classes([
            {'name': 'white', 'color_bgr': (255, 255, 255)},
            {
                'name': 'green',
                'side': 'right',
                'color_bgr': (0, 255, 0),
            },
            {
                'name': 'light_gray',
                'side': 'left',
                'color_bgr': (211, 211, 211),
            },
        ])
        self.color_classes = [
            item for item in color_classes if item['name'] == 'white'
        ]
        self.outer_color_classes = [
            item for item in color_classes if item['name'] != 'white'
        ]
        self.get_logger().info(
            f'Color profile={self.color_profile_name} ({profile_path})'
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
        self.declare_parameter(
            'solid_reference_x_px_2lane', SOLID_REFERENCE_X_PX_2LANE
        )
        self.declare_parameter(
            'solid_reference_x_px_1lane', SOLID_REFERENCE_X_PX_1LANE
        )
        self.declare_parameter(
            'lane_change_steer_offset', LANE_CHANGE_STEER_OFFSET
        )
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
        self.solid_reference_x = {
            '2lane': int(self.get_parameter('solid_reference_x_px_2lane').value),
            '1lane': int(self.get_parameter('solid_reference_x_px_1lane').value),
        }
        self.lane_change_steer_offset = max(
            1, int(self.get_parameter('lane_change_steer_offset').value)
        )
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
        # /lane_info 변경을 받은 뒤 목표 실선을 처음 볼 때까지 유지할
        # 차선 변경 상태. 각 방향의 횟수 제한은 mission_lane_main이 담당한다.
        self.lane_change_state = None
        self.last_solid_line_x = {'left': None, 'right': None}
        self.active_solid_side = None
        self.active_solid_mask = None

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.offset_pub = self.create_publisher(Int16, self.lane_offset_topic, 10)
        self.lane_change_complete_pub = self.create_publisher(
            Int16, LANE_CHANGE_COMPLETE_TOPIC, 10
        )
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, qos)
        self.create_subscription(Int16, LANE_INFO_TOPIC, self.lane_info_callback, 10)
        self.create_subscription(Image, self.image_topic, self.image_callback, qos)

        self.get_logger().info(
            f'Subscribing {self.image_topic}, publishing {self.lane_offset_topic}, '
            f'waiting for {LANE_INFO_TOPIC} to select the starting lane, '
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
        """Detect /lane_info transitions and arm destination-line acquisition."""
        mode = str(mode).lower()
        if mode not in ('1lane', '2lane'):
            return
        if not force and mode == self.driving_mode:
            return
        if self.lane_change_state is not None and mode != self.driving_mode:
            self.get_logger().warn(
                f'Ignoring mode change to {mode}; '
                f'lane change {self.lane_change_state} is still in progress',
                throttle_duration_sec=1.0,
            )
            return

        previous_mode = self.driving_mode
        self.driving_mode = mode
        reference_x = float(self.solid_reference_x[mode])
        # 아래 기존 디버그/보조 함수가 참조하는 필드도 새 실선 기준으로 맞춘다.
        self.dashed_reference_x_px = reference_x
        self.dashed_reference_target_x_px = reference_x
        self.lane_reference_transition_active = False

        if previous_mode is None or force:
            self.lane_change_state = None
        elif previous_mode == '2lane' and mode == '1lane':
            self.lane_change_state = '2->1'
            self.last_solid_line_x['left'] = None
        elif previous_mode == '1lane' and mode == '2lane':
            self.lane_change_state = '1->2'
            self.last_solid_line_x['right'] = None

        self.get_logger().info(
            f'Driving mode changed {previous_mode or "startup"} -> {mode}; '
            f'solid_reference_x={reference_x:.1f}, '
            f'lane_change_state={self.lane_change_state or "tracking"}'
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
        # 시작 차선과 차선 변경 명령은 mission_lane_main의 /lane_info가
        # 단일 기준이다. 신호등 색상 판정은 main 노드에서 독립적으로 수행된다.
        if self.driving_mode is None:
            self.get_logger().info(
                f'Waiting for {LANE_INFO_TOPIC}; skipping image',
                throttle_duration_sec=1.0,
            )
            return

        frame = self.to_bgr(msg)
        if frame is None:
            return

        segmented = self.segment_colors(frame)
        frame = self.warp_to_bev(segmented)
        if frame.size == 0:
            return

        # timed 노드와 동일하게 BEV 전체를 바로 마스크로 만든다.
        white_mask = self.make_white_mask(frame)
        outer_masks = self.make_outer_masks(frame)
        self.show_debug_mask(white_mask, outer_masks)

        near = white_mask[-self.near_field_rows:, :]
        near_white_ratio = float((near > 0).mean()) if near.size else 0.0

        side = 'left' if self.driving_mode == '1lane' else 'right'
        color_mask = outer_masks.get(side)
        line_mask, line_x, detect_reason = self.find_color_backed_solid_line(
            white_mask,
            color_mask,
            side,
            self.last_solid_line_x[side],
        )
        if near_white_ratio > self.white_overload_ratio:
            line_mask, line_x = None, None
            detect_reason = f'white overload {near_white_ratio:.2f}'

        self.active_solid_side = side
        self.active_solid_mask = line_mask
        reference_x = float(self.solid_reference_x[self.driving_mode])

        if line_x is None:
            if self.lane_change_state == '2->1':
                output = -self.lane_change_steer_offset
                status = 'CHANGING 2->1: STEER LEFT'
            elif self.lane_change_state == '1->2':
                output = self.lane_change_steer_offset
                status = 'CHANGING 1->2: STEER RIGHT'
            else:
                output = self.last_offset
                status = f'NO {side.upper()} COLOR-BACKED LINE'
            self.last_offset = int(np.clip(
                output, -self.lane_offset_limit, self.lane_offset_limit
            ))
            self.publish_offset(self.last_offset)
            self.get_logger().warn(
                f'{status}; {detect_reason}; offset={self.last_offset}',
                throttle_duration_sec=1.0,
            )
            self.publish_debug(
                msg, frame, status, near_white_ratio,
                base_x=reference_x, lane_offset=self.last_offset,
                solid_mask=line_mask, solid_side=side,
            )
            return

        transition_completed = self.lane_change_state is not None
        if transition_completed:
            completed_transition = self.lane_change_state
            self.get_logger().info(
                f'Lane change {completed_transition} complete: '
                f'{side} color-backed white line acquired at x={line_x:.1f}'
            )
            self.lane_change_state = None
            self.publish_lane_change_complete(
                1 if self.driving_mode == '1lane' else 2
            )

        lane_offset = self.map_solid_line_x_to_offset(line_x, reference_x)
        if (
            not transition_completed
            and abs(lane_offset - self.last_offset) > self.max_offset_jump_px
        ):
            self.get_logger().warn(
                f'Offset jump too large ({self.last_offset} -> {lane_offset}), '
                'holding last offset',
                throttle_duration_sec=1.0,
            )
            self.publish_offset(self.last_offset)
            self.publish_debug(
                msg, frame, 'JUMP REJECTED', near_white_ratio,
                base_x=reference_x, line_x=line_x, lane_offset=lane_offset,
                solid_mask=line_mask, solid_side=side,
            )
            return

        self.last_offset = lane_offset
        self.last_solid_line_x[side] = line_x
        self.publish_offset(lane_offset)
        self.publish_debug(
            msg, frame, 'OK', near_white_ratio,
            base_x=reference_x, line_x=line_x, lane_offset=lane_offset,
            solid_mask=line_mask, solid_side=side,
        )

    def publish_offset(self, value):
        msg = Int16()
        msg.data = int(np.clip(value, -self.lane_offset_limit, self.lane_offset_limit))
        self.offset_pub.publish(msg)

    def publish_lane_change_complete(self, lane_number):
        msg = Int16()
        msg.data = int(lane_number)
        self.lane_change_complete_pub.publish(msg)

    def apply_no_dash_recovery(self):
        """중앙선 미검출 동안 차선별 복구 방향으로 offset을 누적한다."""
        direction = -1 if self.driving_mode == '2lane' else 1
        self.last_offset = int(np.clip(
            self.last_offset + direction * NO_DASH_RECOVERY_STEP,
            -self.lane_offset_limit,
            self.lane_offset_limit,
        ))
        return self.last_offset

    def map_solid_line_x_to_offset(self, detected_lane_x, reference_x):
        """색상 경계 실선 오차를 -45~45 offset으로 매핑한다."""
        error_px = float(detected_lane_x) - float(reference_x)
        normalized = np.clip(
            error_px / self.offset_error_limit_px,
            -1.0,
            1.0,
        )
        scaled_offset = normalized * self.lane_offset_limit
        return int(round(np.clip(
            scaled_offset, -self.lane_offset_limit, self.lane_offset_limit
        )))

    def find_color_backed_solid_line(
        self, white_mask, color_mask, side, previous_x
    ):
        """Find a white solid line with gray on its left or green on its right."""
        if color_mask is None or not np.any(color_mask):
            return None, None, f'no {side} boundary color'

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            white_mask, connectivity=8
        )
        if num_labels <= 1:
            return None, None, 'no white component'

        pad = self.outer_near_distance_px
        kernel_size = pad * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        height, width = white_mask.shape
        best = None
        shape_pass = 0

        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            if area < SOLID_MIN_COMPONENT_AREA or h < SOLID_MIN_LINE_HEIGHT_PX:
                continue

            component = labels == label
            if (
                area <= SOLID_SMALL_COMPACT_MAX_AREA
                and max(w, h) <= SOLID_SMALL_COMPACT_MAX_SIDE_PX
                and self.component_elongation(component)
                < SOLID_SMALL_COMPACT_MIN_ELONGATION
            ):
                continue
            shape_pass += 1

            right = x + w - 1
            bottom = y + h - 1
            x0, x1 = max(0, x - pad), min(width, right + pad + 1)
            y0, y1 = max(0, y - pad), min(height, bottom + pad + 1)
            local_component = component[y0:y1, x0:x1].astype(np.uint8)
            neighborhood = cv2.dilate(local_component, kernel)

            if side == 'left':
                band = slice(x0, max(x0, x - self.outer_side_margin_px))
            else:
                band = slice(
                    min(x1, right + 1 + self.outer_side_margin_px), x1
                )
            if band.stop <= band.start:
                continue

            local_color = color_mask[y0:y1, band]
            local_neighborhood = neighborhood[
                :, band.start - x0:band.stop - x0
            ]
            near_color = cv2.bitwise_and(local_color, local_neighborhood)
            if int(cv2.countNonZero(near_color)) < self.outer_min_pixels:
                continue

            component_x = float(centroids[label][0])
            score = (
                -abs(component_x - previous_x)
                if previous_x is not None
                else float(area)
            )
            if best is None or score > best[0]:
                best = (score, label)

        if best is None:
            reason = (
                f'no white line backed by {side} color'
                if shape_pass
                else 'no line-shaped white component'
            )
            return None, None, reason

        line_mask = (labels == best[1]).astype(np.uint8) * 255
        measured_x, measure_mode = self.measure_solid_near_x(line_mask)
        if measured_x is None:
            return line_mask, None, f'{measure_mode}: insufficient near pixels'
        return line_mask, measured_x, measure_mode

    @staticmethod
    def component_elongation(component_mask):
        ys, xs = np.nonzero(component_mask)
        if len(xs) < 3:
            return 1.0
        points = np.column_stack((xs, ys)).astype(np.float32)
        side_a, side_b = cv2.minAreaRect(points)[1]
        long_side = max(float(side_a), float(side_b))
        short_side = min(float(side_a), float(side_b))
        return long_side if short_side < 1.0 else long_side / short_side

    @staticmethod
    def measure_solid_near_x(line_mask):
        """Measure the median x from the vehicle-near part of a solid line."""
        height = line_mask.shape[0]
        ys, xs = np.nonzero(line_mask)
        if ys.size == 0:
            return None, 'empty line'

        near = ys >= height - SOLID_NEAR_ROWS
        if int(near.sum()) >= SOLID_NEAR_MIN_PIXELS:
            return float(np.median(xs[near])), 'near band'

        bottom = ys >= ys.max() - SOLID_NEAR_ROWS
        if int(bottom.sum()) >= SOLID_NEAR_MIN_PIXELS:
            return float(np.median(xs[bottom])), 'line bottom'
        return None, 'near band empty'

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
        segmented = np.zeros_like(frame, dtype=np.uint8)
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
            if color_class['name'] == 'light_gray':
                cutoff_x = int(round(mask.shape[1] * LIGHT_GRAY_MASK_MAX_X_RATIO))
                mask[:, cutoff_x:] = 0
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

    def make_white_mask(self, segmented_bev):
        """timed 노드와 같이 BEV 분류 결과의 정확한 흰색만 꺼낸다."""
        white_mask = (
            np.all(segmented_bev == (255, 255, 255), axis=2).astype(np.uint8)
            * 255
        )
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
        dashed_components=None, solid_mask=None, solid_side=None,
    ):
        if not (self.debug_view or self.publish_debug_image):
            return

        debug = frame.copy()
        height, width = debug.shape[:2]
        roi_bottom_y = height - 1
        near_top_y = max(0, height - SOLID_NEAR_ROWS)
        cv2.rectangle(
            debug, (0, near_top_y), (width - 1, roi_bottom_y), (0, 220, 0), 1
        )

        if solid_mask is not None and solid_mask.any():
            ys, xs = np.nonzero(solid_mask)
            debug[ys, xs] = (0, 0, 255)

        if base_x is not None:
            self.draw_dashed_vline(
                debug, int(round(base_x)), 0, roi_bottom_y, (0, 165, 255)
            )
        if line_x is not None:
            measured_x = int(round(line_x))
            cv2.circle(debug, (measured_x, roi_bottom_y - 4), 7, (0, 255, 255), -1)
            self.draw_dashed_vline(
                debug, measured_x, near_top_y, roi_bottom_y, (0, 255, 255), dash=5
            )

        color = (0, 255, 0) if status == 'OK' else (0, 0, 255)
        lines = [
            f'status: {status}',
            f'lane_offset: {lane_offset if lane_offset is not None else self.last_offset}',
            f'mode: {self.driving_mode}, transition: '
            f'{self.lane_change_state or "none"}',
            f'solid: {solid_side or "--"}, reference_x: '
            f'{base_x if base_x is not None else "--"}, '
            f'measured_x: {line_x if line_x is not None else "--"}',
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
