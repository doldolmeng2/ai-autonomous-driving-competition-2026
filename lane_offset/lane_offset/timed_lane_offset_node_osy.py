"""timed_lane_offset_node.

역할:
    시간주행(2차로 고정 주행) 미션에서 /camera/high/image_raw 를 받아
    "실선-점선-실선" 차선 구성에서 중앙 점선의 위치를 계산해 /lane_offset 으로
    발행한다.

트랙 특징:
    - 화면의 차선 배치는 왼쪽 실선 - 중앙 점선 - 오른쪽 실선이다.
    - 왼쪽 실선의 왼쪽에는 회색 매트, 오른쪽 실선의 오른쪽에는 초록 매트가 있다.
    - offset 계산은 두 실선 사이의 중앙 점선만 사용한다. 따라서 양쪽 경계 실선과
      그 바깥 매트 색이 모두 확인될 때만 점선 후보를 인정한다.

기준값(dashed_reference_x_px):
    주행 모드별로 640px 너비 영상에서 정상 위치의 중앙 점선 x좌표를 하드코딩한다.
    현재 검출한 점선 x좌표와 이 기준의 차이를 -45~45로 매핑해 /lane_offset 으로
    발행한다. 1차선 모드 전환 때는 dashed_reference_x_px_1lane만 바꾸면 된다.

노이즈 대응:
    - ROI 상/하단을 크롭해 차량 후드와 배경(천장/바닥 반사)을 제외한다.
    - 횡단보도 등으로 흰색 픽셀이 비정상적으로 많아지면(near-field 흰색 비율 급증)
      이번 프레임 측정을 버리고 마지막으로 유효했던 offset을 그대로 재발행한다.
    - 한 프레임 사이에 offset이 비정상적으로 크게 튀는 경우도 같은 방식으로 무시한다.
    - 초록 매트 위 흰색 꽃 그림 등은 색은 차선과 비슷해도 작고 동글동글한
      덩어리다. near-field(차선 시작점 탐색) 단계에서는 connected components로
      "밴드 높이 대부분을 채우는 덩어리(=곡선에서도 안 끊기는 실선/점선)" 또는
      "세로로 길고 가로로 짧은 덩어리(=짧은 점선 조각)"만 차선 후보로 인정해
      꽃 그림을 걸러낸다.

점선 공백 대응:
    - 슬라이딩 윈도우로 아래에서 위로 올라가며 차선 위치를 추적하고,
      수집한 픽셀에 1차 직선을 피팅해 점선 공백도 하나의 직선처럼 연결한다.
    - 윈도우 한 단은 높이가 짧아 모양 필터가 잘 안 통하므로, 대신 윈도우 폭을
      좁게 잡고 윈도우 안에 여러 덩어리가 있으면 지금 추적 중인 x와 가장 가까운
      덩어리만 사용해 옆에 있는 꽃 그림 등이 평균에 섞이지 않게 한다.
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

# ROI: 이미지 상단(배경)과 하단(차량 후드)을 잘라낸다. (640x480 기준)
ROI_TOP = 250
ROI_BOTTOM = 450
# ROI 안에서 사용할 사다리꼴의 좌/우 inset. 아래쪽은 경계 차선을 보존하기 위해
# 거의 자르지 않고, 위쪽만 좁혀 먼 거리의 양옆 잡음을 제외한다. (640px 폭 기준)
ROI_TRAPEZOID_TOP_INSET_PX = 150
ROI_TRAPEZOID_BOTTOM_INSET_PX = 0

# 흰색(차선/횡단보도) HSV 임계값
WHITE_S_MAX = 25
WHITE_V_MIN = 160
# 초록색(오른쪽 차선 바깥 매트) HSV 임계값
GREEN_H_MIN = 30
GREEN_H_MAX = 90
GREEN_S_MIN = 40
GREEN_V_MIN = 70
# 회색(왼쪽 실선 바깥쪽 매트) HSV 임계값.
# 요청 튜닝값: S >= 25, V >= 150. 회색은 Hue 조건을 사용하지 않는다.
GRAY_S_MIN = 25
GRAY_V_MIN = 150

# 오른쪽 차선 탐색에 쓰는 근접(ROI 하단) 밴드 높이
NEAR_FIELD_ROWS = 100
# 근접 밴드에서 흰색 비율이 이 값을 넘으면 횡단보도 등으로 판단하고 무시
WHITE_OVERLOAD_RATIO = 0.15

# 주행 모드. '2lane'은 중앙 점선의 오른쪽 차로, '1lane'은 필요 시 기준값만 바꿔
# 재사용한다. (ROS 실행 시 -p driving_mode:=1lane 으로 변경 가능)
DRIVING_MODE = '2lane' # 2lane
# 640px 너비 영상에서 차가 정상 위치일 때의 중앙 점선 x좌표.
# 사진 기준 640px 영상에서의 초기값이다. 1차선/2차선은 중앙 점선을 서로 다른
# 위치에서 보기 때문에, 이 값은 각각 점선 슬라이딩 윈도우의 시작점이기도 하다.
DASHED_REFERENCE_X_PX_2LANE = 160
DASHED_REFERENCE_X_PX_1LANE = 510
# 기준선과 이만큼 차이 나면 lane_offset의 최대/최소값(+/-45)에 도달한다.
OFFSET_ERROR_LIMIT_PX = 195
LANE_OFFSET_LIMIT = 45
# 기준선 오차를 offset으로 바꾼 뒤 적용할 비례 이득. 최종값은 +/-45로 제한한다.
OFFSET_KP = 10.0
# 한 프레임 사이 offset이 이 값보다 더 튀면 오검출로 보고 이전 값 유지
MAX_OFFSET_JUMP_PX = 80

# 슬라이딩 윈도우 (범위를 좁게 잡아서 옆에 있는 꽃 그림 등을 덜 건드리게 함)
NUM_WINDOWS = 10
WINDOW_MARGIN = 27
WINDOW_MINPIX = 50
# 점선은 빈 구간을 넘어 다음 조각을 잡아야 하므로 실선보다 넓게 탐색한다.
DASHED_WINDOW_MARGIN = 130
WINDOW_MIN_COMPONENT_PIXELS = 30
# 중앙 점선과 1/2차선 실선의 하단 x가 이 거리보다 가까우면 같은 선을 추적한
# 것으로 판단한다. 이 프레임은 무효 처리하고 이전 offset을 유지한다.
CENTER_LINE_OVERLAP_DISTANCE_PX = 45
# 이전 프레임의 검출 위치를 다음 프레임 윈도우 시작점에 반영하는 비율.
# 0.20이면 한 프레임에 차이의 20%만 움직여 급격한 점프를 막는다.
WINDOW_START_ADAPT_RATE = 0.5
# 새 검출 위치가 이전 박스 시작점에서 이 거리보다 크게 튀면 오검출로 보고
# 시작점을 갱신하지 않는다.
MAX_WINDOW_START_JUMP_PX = 100
# 곡선에서는 색 매트와 흰 실선이 같은 x strip에 정확히 겹치지 않는다. 이 거리
# 이내면 "색 덩어리 근처의 흰 실선"으로 보고 슬라이딩 윈도우 시작점으로 사용한다.
BOUNDARY_COLOR_NEAR_DISTANCE_PX = 100
# 색 매트가 흰 실선 주변에 이 픽셀 수 이상 있어야 경계로 인정한다.
# 색 노이즈 한두 점을 배제하기 위해 200px 이상을 요구한다.
BOUNDARY_COLOR_MIN_PIXELS = 100
# True면 모드별 필수 경계 실선(1차선=왼쪽, 2차선=오른쪽)을 반드시 찾아야 한다.
# 점선만으로 주행을 허용하려면 False로 둔다.
REQUIRE_BOUNDARY_LINE = False

# 모양 필터(near-field 차선 시작점 탐색용): 초록 매트 위 흰 꽃 그림 등은
# 색은 흰색이지만 작고 동글동글한 덩어리다. 아래 둘 중 하나를 만족해야
# 차선 후보로 인정한다.
#   1) span: 밴드 높이의 대부분을 채움 -> 실선/점선은 곡선에서 옆으로
#      휘어져도(가로 폭이 넓어져도) 밴드를 처음부터 끝까지 관통하지만,
#      꽃 그림은 작은 덩어리라 밴드 높이를 거의 못 채운다.
#   2) aspect: 세로로 길고 가로로 짧음 -> 밴드를 다 못 채우는 짧은
#      점선 조각이라도 모양 자체가 길쭉하면 인정.
NEAR_FIELD_FULL_HEIGHT_RATIO = 0.8
MIN_LINE_ASPECT_RATIO = 1.5
MIN_LINE_HEIGHT_PX = 20

# 디버그 시각화: ROI/차선/슬라이딩 윈도우를 그린 화면을 바로 OpenCV 창으로 띄운다.
# (bag/카메라 토픽만 켜져 있으면, 이 노드 실행만으로 인식 화면이 뜬다.)
# 실차 대회 주행 시에는 CPU 절약을 위해 False로 끄는 것을 권장.
DEBUG_VIEW = True
WINDOW_NAME = 'timed_lane_offset_debug'
WHITE_MASK_WINDOW_NAME = 'timed_lane_offset_white_mask_osy'
GREEN_MASK_WINDOW_NAME = 'timed_lane_offset_green_mask_osy'
GRAY_MASK_WINDOW_NAME = 'timed_lane_offset_gray_mask_osy'
DEBUG_IMAGE_TOPIC = '/lane_offset/debug_image'


class TimedLaneOffsetNode(Node):
    """/camera/high/image_raw -> 중앙 점선 기준 offset을 계산해 /lane_offset 발행."""

    def __init__(self):
        super().__init__('timed_lane_offset_node_osy')

        # ---- 파라미터 ------------------------------------------------------
        self.declare_parameter('roi_top', ROI_TOP)
        self.declare_parameter('roi_bottom', ROI_BOTTOM)
        self.declare_parameter('roi_trapezoid_top_inset_px', ROI_TRAPEZOID_TOP_INSET_PX)
        self.declare_parameter(
            'roi_trapezoid_bottom_inset_px', ROI_TRAPEZOID_BOTTOM_INSET_PX
        )
        self.declare_parameter('white_s_max', WHITE_S_MAX)
        self.declare_parameter('white_v_min', WHITE_V_MIN)
        self.declare_parameter('green_h_min', GREEN_H_MIN)
        self.declare_parameter('green_h_max', GREEN_H_MAX)
        self.declare_parameter('green_s_min', GREEN_S_MIN)
        self.declare_parameter('green_v_min', GREEN_V_MIN)
        self.declare_parameter('gray_s_min', GRAY_S_MIN)
        self.declare_parameter('gray_v_min', GRAY_V_MIN)
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
        self.declare_parameter('window_margin', WINDOW_MARGIN)
        self.declare_parameter('window_minpix', WINDOW_MINPIX)
        self.declare_parameter('dashed_window_margin', DASHED_WINDOW_MARGIN)
        self.declare_parameter(
            'center_line_overlap_distance_px', CENTER_LINE_OVERLAP_DISTANCE_PX
        )
        self.declare_parameter('window_min_component_pixels', WINDOW_MIN_COMPONENT_PIXELS)
        self.declare_parameter('window_start_adapt_rate', WINDOW_START_ADAPT_RATE)
        self.declare_parameter('max_window_start_jump_px', MAX_WINDOW_START_JUMP_PX)
        self.declare_parameter(
            'boundary_color_near_distance_px', BOUNDARY_COLOR_NEAR_DISTANCE_PX
        )
        self.declare_parameter('boundary_color_min_pixels', BOUNDARY_COLOR_MIN_PIXELS)
        self.declare_parameter('require_boundary_line', REQUIRE_BOUNDARY_LINE)
        self.declare_parameter(
            'near_field_full_height_ratio', NEAR_FIELD_FULL_HEIGHT_RATIO
        )
        self.declare_parameter('min_line_aspect_ratio', MIN_LINE_ASPECT_RATIO)
        self.declare_parameter('min_line_height_px', MIN_LINE_HEIGHT_PX)
        self.declare_parameter('debug_view', DEBUG_VIEW)

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
        self.white_s_max = int(self.get_parameter('white_s_max').value)
        self.white_v_min = int(self.get_parameter('white_v_min').value)
        self.green_h_min = int(self.get_parameter('green_h_min').value)
        self.green_h_max = int(self.get_parameter('green_h_max').value)
        self.green_s_min = int(self.get_parameter('green_s_min').value)
        self.green_v_min = int(self.get_parameter('green_v_min').value)
        self.gray_s_min = int(self.get_parameter('gray_s_min').value)
        self.gray_v_min = int(self.get_parameter('gray_v_min').value)
        self.near_field_rows = int(self.get_parameter('near_field_rows').value)
        self.white_overload_ratio = float(
            self.get_parameter('white_overload_ratio').value
        )
        self.driving_mode = str(self.get_parameter('driving_mode').value).lower()
        if self.driving_mode not in ('1lane', '2lane'):
            self.get_logger().warn(
                f"Unknown driving_mode='{self.driving_mode}'; using '2lane'"
            )
            self.driving_mode = '2lane'
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
        self.window_margin = int(self.get_parameter('window_margin').value)
        self.window_minpix = int(self.get_parameter('window_minpix').value)
        self.dashed_window_margin = int(
            self.get_parameter('dashed_window_margin').value
        )
        self.center_line_overlap_distance_px = max(0, int(
            self.get_parameter('center_line_overlap_distance_px').value
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
        self.boundary_color_near_distance_px = int(
            self.get_parameter('boundary_color_near_distance_px').value
        )
        self.boundary_color_min_pixels = int(
            self.get_parameter('boundary_color_min_pixels').value
        )
        self.require_boundary_line = bool(
            self.get_parameter('require_boundary_line').value
        )
        self.near_field_full_height_ratio = float(
            self.get_parameter('near_field_full_height_ratio').value
        )
        self.min_line_aspect_ratio = float(
            self.get_parameter('min_line_aspect_ratio').value
        )
        self.min_line_height_px = int(self.get_parameter('min_line_height_px').value)
        self.debug_view = bool(self.get_parameter('debug_view').value)
        self.window_name = WINDOW_NAME
        self.white_mask_window_name = WHITE_MASK_WINDOW_NAME
        self.green_mask_window_name = GREEN_MASK_WINDOW_NAME
        self.gray_mask_window_name = GRAY_MASK_WINDOW_NAME
        self.publish_debug_image = False
        self.debug_image_topic = DEBUG_IMAGE_TOPIC

        # 마지막으로 발행한(유효했던) offset. 오검출 프레임에서는 이 값을 그대로 재사용.
        self.last_offset = 0
        # 다음 프레임의 각 차선 슬라이딩 윈도우 시작점. 검출 결과를 저역통과
        # 반영해 차선을 따라 서서히 이동한다.
        self.window_start_x = {
            'left': None,
            'dashed': float(self.dashed_reference_x_px),
            'right': None,
        }
        # 마지막으로 유효했던 세 차선의 슬라이딩 윈도우. 인식 공백에서도 디버그
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
    def image_callback(self, msg):
        frame = self.to_bgr(msg)
        if frame is None:
            return

        roi = frame[self.roi_top:self.roi_bottom, :]
        if roi.size == 0:
            return

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        trapezoid_mask = self.make_roi_trapezoid_mask(roi.shape[:2])
        white_mask = cv2.bitwise_and(self.make_white_mask(hsv), trapezoid_mask)
        green_mask = cv2.bitwise_and(self.make_green_mask(hsv), trapezoid_mask)
        gray_mask = cv2.bitwise_and(self.make_gray_mask(hsv), trapezoid_mask)
        self.show_debug_masks(white_mask, green_mask, gray_mask)

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

        left_base_x, right_base_x = self.find_boundary_line_bases(
            white_mask, green_mask, gray_mask
        )
        # 2차선 화면에서는 좌측 외곽 실선이 ROI 밖으로 나갈 수 있다. 이때는
        # 초록색으로 검증되는 오른쪽 실선만 필수 경계로 사용한다.
        required_boundary_missing = (
            right_base_x is None if self.driving_mode == '2lane'
            else left_base_x is None
        )
        if self.require_boundary_line and required_boundary_missing:
            self.get_logger().warn(
                'Required boundary solid line not found, holding last offset',
                throttle_duration_sec=1.0,
            )
            self.publish_offset(self.last_offset)
            self.publish_debug(msg, frame, 'NO BOUNDARY LINES', near_white_ratio)
            return

        # 세 차선을 독립 슬라이딩 윈도우로 추적한다. 모드별 중앙 점선 기준값은
        # 점선 트래커의 시작점이며, 점선 창은 빈 구간에서 진행 방향을 예측해 연결한다.
        left_start_x = self.get_window_start_x('left', left_base_x)
        dashed_start_x = self.get_window_start_x(
            'dashed', self.dashed_reference_x_px
        )
        right_start_x = self.get_window_start_x('right', right_base_x)
        left_track = (
            self.track_lane_with_sliding_window(
                white_mask, left_start_x, self.window_margin, allow_gaps=False
            )
            if left_start_x is not None
            else (None, [], [], [])
        )
        dashed_track = self.track_lane_with_sliding_window(
            white_mask, dashed_start_x, self.dashed_window_margin,
            allow_gaps=True,
        )
        right_track = (
            self.track_lane_with_sliding_window(
                white_mask, right_start_x, self.window_margin, allow_gaps=False
            )
            if right_start_x is not None
            else (None, [], [], [])
        )
        left_x, _left_windows, _left_points_x, _left_points_y = left_track
        dashed_x, windows, points_x, points_y = dashed_track
        right_x, _right_windows, _right_points_x, _right_points_y = right_track
        lane_tracks = {
            'left': left_track,
            'dashed': dashed_track,
            'right': right_track,
        }

        # 중앙 점선 트래커가 1차선 또는 2차선 실선을 같은 x에서 잡으면 중앙선
        # 인식을 취소한다. 이전 offset/윈도우 시작점을 그대로 유지해 오검출로
        # 조향값이 바뀌는 것을 막는다.
        center_overlaps_lane = (
            dashed_x is not None
            and (
                (
                    left_x is not None
                    and abs(dashed_x - left_x) <= self.center_line_overlap_distance_px
                )
                or (
                    right_x is not None
                    and abs(dashed_x - right_x) <= self.center_line_overlap_distance_px
                )
            )
        )
        if center_overlaps_lane:
            self.get_logger().warn(
                'Center dashed overlaps a lane line; holding last offset',
                throttle_duration_sec=1.0,
            )
            self.publish_offset(self.last_offset)
            self.publish_debug(
                msg, frame, 'CENTER OVERLAP', near_white_ratio,
                base_x=dashed_start_x, line_x=dashed_x, windows=windows,
                points_x=points_x, points_y=points_y, lane_tracks=lane_tracks,
            )
            return

        if self.require_boundary_line and self.driving_mode == '2lane':
            lane_order_valid = (
                dashed_x is not None
                and right_x is not None
                and dashed_x < right_x
                and (left_x is None or left_x < dashed_x)
            )
        elif self.require_boundary_line:
            lane_order_valid = (
                dashed_x is not None
                and left_x is not None
                and left_x < dashed_x
                and (right_x is None or dashed_x < right_x)
            )
        else:
            # 경계선 필수 조건이 꺼진 경우 중앙 점선만 필수다. 경계선이 검출된
            # 경우에만 중앙선과의 명백한 역순 관계를 보조 검증한다.
            lane_order_valid = (
                dashed_x is not None
                and (left_x is None or left_x < dashed_x)
                and (right_x is None or dashed_x < right_x)
            )
        if not lane_order_valid:
            self.get_logger().warn(
                'Sliding-window lane order invalid, holding last offset',
                throttle_duration_sec=1.0,
            )
            self.publish_offset(self.last_offset)
            self.publish_debug(
                msg, frame, 'INVALID LANE ORDER', near_white_ratio,
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
        self.update_window_start_x('left', left_x, white_mask.shape[1])
        self.update_window_start_x('dashed', dashed_x, white_mask.shape[1])
        self.update_window_start_x('right', right_x, white_mask.shape[1])
        self.last_lane_tracks = lane_tracks
        self.publish_offset(lane_offset)
        self.publish_debug(
            msg, frame, 'OK', near_white_ratio,
            base_x=dashed_start_x, line_x=line_x, windows=windows,
            points_x=points_x, points_y=points_y, lane_offset=lane_offset,
            lane_tracks=lane_tracks,
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
    def make_white_mask(self, hsv):
        _, s, v = cv2.split(hsv)
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

    def make_gray_mask(self, hsv):
        """왼쪽 실선 바깥쪽 회색 매트 마스크를 만든다."""
        _h, s, v = cv2.split(hsv)
        mask = (
            (s >= self.gray_s_min)
            & (v >= self.gray_v_min)
        )
        return (mask.astype(np.uint8)) * 255

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

    def show_debug_masks(self, white_mask, green_mask, gray_mask):
        """ROI에서 만든 흰색/초록색/회색 마스크를 별도 디버그 창으로 표시한다."""
        if not self.debug_view:
            return

        white_debug = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        green_debug = np.zeros((*green_mask.shape, 3), dtype=np.uint8)
        green_debug[green_mask > 0] = (0, 255, 0)
        gray_debug = cv2.cvtColor(gray_mask, cv2.COLOR_GRAY2BGR)
        cv2.imshow(self.white_mask_window_name, white_debug)
        cv2.imshow(self.green_mask_window_name, green_debug)
        cv2.imshow(self.gray_mask_window_name, gray_debug)

    # ======================================================================
    # 모양 필터: 차선(세로로 길고 가로로 짧음) vs 꽃 그림 등 (동글동글한 덩어리)
    # ======================================================================
    def find_lane_shaped_components(self, mask, min_height, min_aspect_ratio, full_height_ratio):
        """mask에서 차선처럼 생긴 픽셀 뭉치만 골라 (x, y, w, h, label) bbox 리스트로 반환.

        아래 둘 중 하나를 만족해야 통과:
          - span: 높이가 mask 전체 높이의 full_height_ratio 이상
            (곡선에서 실선이 옆으로 휘어 폭이 넓어져도, 밴드를 처음부터
            끝까지 관통하는 건 변하지 않는다)
          - aspect: 세로 길이가 min_height 이상이면서 세로/가로 비율이
            min_aspect_ratio 이상 (밴드를 다 못 채우는 짧은 점선 조각도 인정)

        꽃 그림처럼 작고 동글동글한 덩어리는 둘 다 만족 못 해서 걸러진다.
        """
        band_height = mask.shape[0]
        full_height_threshold = band_height * full_height_ratio
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes = []
        for label in range(1, num_labels):  # label 0 = 배경
            x, y, w, h, _area = stats[label]
            if w <= 0 or h <= 0:
                continue
            spans_band = h >= full_height_threshold
            is_tall_narrow = h >= min_height and (h / float(w)) >= min_aspect_ratio
            if spans_band or is_tall_narrow:
                boxes.append((x, y, w, h, label))
        return boxes, labels

    # ======================================================================
    # 중앙 점선 위치 찾기
    # ======================================================================
    def find_center_dashed_base(self, white_mask, green_mask, gray_mask):
        """두 외곽 실선 사이에 있는 중앙 점선 조각을 찾는다.

        점선은 밴드를 끝까지 잇지 못하는 짧고 세로로 긴 connected component,
        실선은 밴드 대부분을 잇는 component로 분류한다. 왼쪽에 회색 매트가 있는
        실선과 오른쪽에 초록 매트가 있는 실선을 각각 확인하고, 그 사이의 짧은
        흰 성분만 중앙 점선으로 인정한다.
        """
        band_white = white_mask[-self.near_field_rows:, :]
        band_green = green_mask[-self.near_field_rows:, :]
        band_gray = gray_mask[-self.near_field_rows:, :]
        boxes, labels = self.find_lane_shaped_components(
            band_white,
            self.min_line_height_px,
            self.min_line_aspect_ratio,
            self.near_field_full_height_ratio,
        )
        if not boxes:
            return None, []

        full_height_threshold = band_white.shape[0] * self.near_field_full_height_ratio
        solid_boxes = []
        dashed_boxes = []
        for x, _y, w, h, _label in boxes:
            if h >= full_height_threshold:
                solid_boxes.append((x, _y, w, h))
            else:
                dashed_boxes.append((x, _y, w, h))

        solid_xs = [x + w // 2 for x, _y, w, _h in solid_boxes]
        if not solid_boxes or not dashed_boxes:
            return None, solid_xs

        # 초록 매트가 바로 오른쪽에 붙은 실선 = 트랙의 오른쪽 외곽 실선.
        green_backed_solids = []
        # 회색 매트가 바로 왼쪽에 붙은 실선 = 트랙의 왼쪽 외곽 실선.
        gray_backed_solids = []
        for x, _y, w, _h in solid_boxes:
            right_x0 = min(x + w + 10, width - 1)
            right_x1 = min(x + w + 40, width)
            right_strip = band_green[:, right_x0:right_x1]
            green_ratio = float((right_strip > 0).mean()) if right_strip.size else 0.0
            if green_ratio > 0.4:
                green_backed_solids.append(x + w // 2)

            left_x0 = max(0, x - 40)
            left_x1 = max(0, x - 10)
            left_strip = band_gray[:, left_x0:left_x1]
            gray_ratio = float((left_strip > 0).mean()) if left_strip.size else 0.0
            if gray_ratio > 0.4:
                gray_backed_solids.append(x + w // 2)

        # 두 경계 실선이 모두 확인되지 않으면 점선 후보를 신뢰하지 않는다.
        if not green_backed_solids or not gray_backed_solids:
            return None, solid_xs
        left_solid_x = min(gray_backed_solids)
        right_solid_x = max(green_backed_solids)
        if left_solid_x >= right_solid_x:
            return None, solid_xs

        # 검증된 왼쪽/오른쪽 실선 사이의 짧은 성분만 중앙 점선 후보로 본다.
        # 아래쪽에 있는 조각이 현재 차량에 가장 가까워 offset에 더 적합하다.
        dashed_candidates = [
            (x + w // 2, y, w, h)
            for x, y, w, h in dashed_boxes
            if left_solid_x + 10 < x + w // 2 < right_solid_x - 10
        ]
        if not dashed_candidates:
            return None, solid_xs

        if self.last_dashed_x is not None:
            # 직전 프레임과 가장 가까운 조각을 우선해 꽃/노이즈로의 전환을 막는다.
            dashed_x, _y, _w, _h = min(
                dashed_candidates, key=lambda c: abs(c[0] - self.last_dashed_x)
            )
        else:
            # 시작 프레임에서는 가장 아래(차량에 가장 가까운) 점선 조각을 사용한다.
            dashed_x, _y, _w, _h = max(dashed_candidates, key=lambda c: c[1] + c[3])
        return float(dashed_x), solid_xs

    # ======================================================================
    # 슬라이딩 윈도우로 차선 추적
    # ======================================================================
    def track_line_with_sliding_window(self, white_mask, base_x):
        """오른쪽 차선 x좌표(ROI 하단 기준)와 함께, 그린 윈도우/사용된 픽셀도 반환(디버그용).

        윈도우 한 단(~20px)은 근접 밴드(70px)와 달리 높이가 짧아서 "세로로
        길다"는 모양 기준이 잘 안 통한다(꽃 그림도 짧은 윈도우 하나를 그냥
        관통해버림). 대신 윈도우 폭(window_margin)을 좁게 잡고, 윈도우 안에
        여러 덩어리가 있으면 지금 추적 중인 x와 가장 가까운 덩어리만 골라서
        옆에 있는 꽃 그림 등이 평균에 섞여 들어가지 않게 한다.
        """
        height, width = white_mask.shape
        window_height = max(1, height // self.num_windows)
        x_current = base_x

        windows = []
        collected_x = []
        collected_y = []
        for i in range(self.num_windows):
            y_high = height - i * window_height
            y_low = max(0, height - (i + 1) * window_height)
            # x_current가 윈도우의 가운데가 아니라 오른쪽 경계에 오게 해서
            # 차선 오른쪽의 초록 매트/꽃 그림이 윈도우에 덜 들어오게 한다.
            x_right = min(width - 1, x_current)
            x_low = max(0, x_right - self.window_margin * 2)
            x_high = min(width, x_right + 1)
            windows.append((x_low, x_right, y_low, y_high))

            if x_high <= x_low or y_high <= y_low:
                continue

            sub_mask = white_mask[y_low:y_high, x_low:x_high]
            num_labels, labels, _stats, centroids = cv2.connectedComponentsWithStats(
                sub_mask, connectivity=8
            )
            if num_labels <= 1:
                continue

            best_label = min(
                range(1, num_labels),
                key=lambda label: abs((centroids[label][0] + x_low) - x_current),
            )
            local_ys, local_xs = np.where(labels == best_label)
            xs = local_xs + x_low
            ys = local_ys + y_low

            if xs.size >= self.window_minpix:
                x_current = int(round(float(xs.mean())))
                collected_x.extend(xs.tolist())
                collected_y.extend(ys.tolist())

        if len(collected_x) < self.window_minpix:
            # 위쪽 윈도우들에서 거의 못 찾았으면 근접 기준점만 사용
            return float(base_x), windows, collected_x, collected_y

        # 수집된 점을 1차 직선으로 피팅하고 ROI 하단에서의 x를 사용한다.
        if len(set(collected_y)) >= 3:
            coeffs = np.polyfit(collected_y, collected_x, 1)
            line_x = float(np.polyval(coeffs, height - 1))
        else:
            line_x = float(np.mean(collected_x))

        return line_x, windows, collected_x, collected_y

    def find_boundary_line_bases(self, white_mask, green_mask, gray_mask):
        """회색 오른쪽의 왼 실선과 초록색 왼쪽의 오른 실선 시작점을 찾는다."""
        band_white = white_mask[-self.near_field_rows:, :]
        band_green = green_mask[-self.near_field_rows:, :]
        band_gray = gray_mask[-self.near_field_rows:, :]
        boxes, labels = self.find_lane_shaped_components(
            band_white,
            self.min_line_height_px,
            self.min_line_aspect_ratio,
            self.near_field_full_height_ratio,
        )
        full_height = band_white.shape[0] * self.near_field_full_height_ratio
        solid_boxes = [
            (x, y, w, h, label) for x, y, w, h, label in boxes if h >= full_height
        ]
        if not solid_boxes:
            return None, None

        # 각 흰 실선 성분 픽셀에서 가장 가까운 색 덩어리까지의 거리. 고정된
        # 좌/우 strip 대신 이 값을 쓰면 곡선에서도 색 매트 근처의 선을 찾는다.
        green_distance = cv2.distanceTransform(
            (band_green == 0).astype(np.uint8), cv2.DIST_L2, 3
        )
        gray_distance = cv2.distanceTransform(
            (band_gray == 0).astype(np.uint8), cv2.DIST_L2, 3
        )
        left_candidates = []
        right_candidates = []
        for x, _y, w, _h, label in solid_boxes:
            center_x = x + w // 2
            line_pixels = labels == label
            # 해당 흰 실선 주변의 색 픽셀 수까지 확인해 점 형태의 색 노이즈는 제외한다.
            kernel_size = self.boundary_color_near_distance_px * 2 + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            line_neighborhood = cv2.dilate(line_pixels.astype(np.uint8), kernel) > 0
            gray_near_pixels = int(np.count_nonzero(band_gray[line_neighborhood]))
            green_near_pixels = int(np.count_nonzero(band_green[line_neighborhood]))
            min_gray_distance = float(gray_distance[line_pixels].min())
            min_green_distance = float(green_distance[line_pixels].min())
            if (
                min_gray_distance <= self.boundary_color_near_distance_px
                and gray_near_pixels >= self.boundary_color_min_pixels
            ):
                left_candidates.append(center_x)
            if (
                min_green_distance <= self.boundary_color_near_distance_px
                and green_near_pixels >= self.boundary_color_min_pixels
            ):
                right_candidates.append(center_x)

        # 곡선/모드에 따라 한쪽 외곽선이 근접 ROI 밖으로 나갈 수 있으므로 각
        # 시작점을 독립적으로 반환한다. 모드별 필수 경계 검증은 콜백에서 한다.
        right_x = float(max(right_candidates)) if right_candidates else None
        # Hue를 쓰지 않는 회색 마스크에는 초록 매트도 일부 포함될 수 있다. 따라서
        # 왼 경계 후보는 오른쪽 초록 경계보다 충분히 왼쪽에 있는 것만 채택한다.
        if right_x is not None:
            left_candidates = [x for x in left_candidates if x < right_x - 20]
        left_x = float(min(left_candidates)) if left_candidates else None
        return left_x, right_x

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
    node = TimedLaneOffsetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
