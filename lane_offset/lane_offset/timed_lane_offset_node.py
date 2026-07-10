"""timed_lane_offset_node.

역할:
    시간주행(2차로 고정 주행) 미션에서 /camera/high/image_raw 를 받아
    차와 차선 사이의 간격을 계산해 /lane_offset 으로 발행한다.

트랙 특징(2차로 = 오른쪽 차로 기준):
    - 오른쪽 차선(실선) 바깥쪽은 초록색 매트
    - 왼쪽 차선(점선, 1차로와의 중앙선) 양옆은 회색 아스팔트 도로
    - 오른쪽은 "바로 오른쪽이 초록색인가"로, 왼쪽은 "양옆이 회색 도로
      색인가"로 서로 독립적으로 판별한다.

기준선 결정(오른쪽/왼쪽 독립 탐지 후 결합):
    - 근접 밴드에서 차선 모양 후보 중 green_backed(오른쪽이 초록색)인 것을
      오른쪽 차선으로, road_flanked(양옆이 회색 도로)인 것을 왼쪽 차선으로
      각각 독립적으로 찾는다(둘 다 동시에 검출될 수 있음).
    - 둘 다 보이면: 각자의 target_offset_px를 뺀 offset을 평균해서 최종
      lane_offset으로 쓴다(두 측정이 서로를 검증해줘서 더 안정적).
    - 하나만 보이면: 보이는 쪽의 offset만 그대로 쓴다.
    - 둘 다 안 보이면: 마지막 유효 offset을 유지한다.

기준값(target_offset_px / left_target_offset_px):
    rosbag2_2026_07_01-15_30_56 (2차선 주행 녹화본)을 분석해 직선 구간에서
    "오른쪽 차선 x좌표 - 이미지 중심" 값이 평균 195px 근처로 유지됨을 확인했다.
    이 값을 기준(0)으로 삼고, 현재 프레임에서 측정한 값과의 차이를
    /lane_offset 으로 발행한다. (양수 = 차가 왼쪽으로 치우쳐서 우회전 필요)
    left_target_offset_px(-212)는 scripts/calibrate_left_offset.py 로 같은
    bag에서 "오른쪽 기준 offset이 거의 0으로 잘 맞아떨어지는 프레임들"만 골라
    그때의 왼쪽 차선 x좌표 중앙값으로 구했다(오른쪽 캘리브레이션과 같은 기준을
    공유). left 토픽을 camera_high로 remap해 측정한 값이라 실제 카메라로는
    재검증이 필요할 수 있다.

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

곡선 대응:
    - 슬라이딩 윈도우로 아래에서 위로 올라가며 차선 위치를 추적하고,
      수집한 픽셀에 2차 다항식을 피팅해 곡선 구간에서도 차선을 놓치지 않게 한다.
    - 윈도우 한 단은 높이가 짧아 모양 필터가 잘 안 통하고, 근접 밴드에서 쓰는
      "이미지 오른쪽/양옆" 방향 기준 색 검사도 급커브에서는 깨진다(차선의 로컬
      방향이 카메라 좌/우 축과 어긋나기 때문). 그래서 방향 가정이 없는 패치
      (부분공간) 기반 검사를 쓴다: 각 윈도우 단의 모든 컬럼 x에 대해 그 x를
      중심으로 한 정사각 패치(patch_width_px)를 만들고, 그 안의 흰색/초록/회색
      비율만으로 "이 패치가 lane_kind 경계선 조각인가"를 판정한다
      (오른쪽=초록+회색 혼합, 왼쪽=회색). 패치가 등방적이라 차선이 어느 각도로
      놓여도 그대로 적용되고, 여러 패치가 조건을 만족하면 이전 단의 x_current에서
      window_max_step_px 이내로 가장 가까운 것만 이어붙인다(연속성 제약). 조건을
      만족하는 패치가 하나도 없으면 그 단은 건너뛰고 이전 위치를 유지한다.
"""

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16

import cv2

# ============================================================================
# 파라미터 기본값 - 튜닝은 대부분 여기서만 하면 된다.
# (전부 ROS 파라미터로도 선언되므로 --ros-args -p 로 실행 중 덮어쓰기도 가능)
# ============================================================================

# ---- 토픽 -------------------------------------------------------------
IMAGE_TOPIC = '/camera/high/image_raw'  # 구독: 원본 카메라 이미지
LANE_OFFSET_TOPIC = '/lane_offset'      # 발행: 계산된 차선 offset(px, Int16)

# ---- ROI ---------------------------------------------------------------
# 이미지 상단(배경/천장·바닥 반사)과 하단(차량 후드)을 잘라낸다. (640x360 기준)
ROI_TOP = 90     # ROI 상단 y좌표: 이보다 위는 크롭
ROI_BOTTOM = 280  # ROI 하단 y좌표: 이보다 아래(차량 후드)는 크롭

# ---- 흰색(차선/횡단보도) HSV 임계값 --------------------------------------
# H 범위를 좁혀서 신호등 트러스 등 다른 흰색 구조물을 배제한다.
WHITE_H_MIN = 25   # 흰색 판정 H 하한
WHITE_H_MAX = 115  # 흰색 판정 H 상한
WHITE_S_MAX = 25   # 흰색 판정 S 상한(채도가 낮아야 흰색)
WHITE_V_MIN = 173  # 흰색 판정 V 하한(밝아야 흰색)
# H 범위를 좁힌 부작용으로 차선 픽셀이 군데군데 끊기는 것을 메우는
# morphological closing 커널 크기(px). 노이즈 자체를 다시 허용하는 대신
# 이미 검출된 흰색 조각들끼리만 이어붙인다.
WHITE_CLOSE_KERNEL_PX = 7

# ---- 초록색(오른쪽 차선 바깥 매트) HSV 임계값 -----------------------------
GREEN_H_MIN = 30  # 초록 판정 H 하한
GREEN_H_MAX = 90  # 초록 판정 H 상한
GREEN_S_MIN = 40  # 초록 판정 S 하한
GREEN_V_MIN = 70  # 초록 판정 V 하한

# ---- 회색 도로(왼쪽 차선 양옆) HSV 임계값 ---------------------------------
# hsv_tuner로 실측한 값.
ROAD_H_MIN = 31    # 도로색 판정 H 하한
ROAD_H_MAX = 179   # 도로색 판정 H 상한
ROAD_S_MIN = 0     # 도로색 판정 S 하한
ROAD_S_MAX = 54    # 도로색 판정 S 상한(채도가 낮은 무채색 계열)
ROAD_V_MIN = 0     # 도로색 판정 V 하한
ROAD_V_MAX = 92   # 도로색 판정 V 상한(너무 밝지 않은 아스팔트 톤)

# ---- 근접 밴드(기준선 탐색) ----------------------------------------------
NEAR_FIELD_ROWS = 70        # 오른쪽/왼쪽 차선 시작점을 찾는 ROI 하단 밴드 높이(px)
WHITE_OVERLOAD_RATIO = 0.30  # 근접 밴드 흰색 비율이 이 값을 넘으면 횡단보도 등으로 보고 이번 측정을 버림

# ---- 기준 offset(px): 차량이 차로 중앙에 있을 때의 기대 x좌표 - 이미지 중심 ---
# 직선 구간 rosbag 분석으로 얻은 오른쪽 차선 기준 offset. 실차 재조정 시 수정.
TARGET_OFFSET_PX = 195
# 왼쪽 차선(점선) 전용 기준 offset(px).
# scripts/calibrate_left_offset.py 로 실측(중앙값, n=375, stdev=94px).
# 오른쪽 기준(+195)과 부호만 반대로 거의 대칭이라(차로 폭 ≈ 407px) 물리적으로도
# 타당하다. bag의 left 토픽을 camera_high로 remap해 측정한 값이라 실제
# camera_high 하드웨어로 재측정하면 미세 조정이 필요할 수 있음.
LEFT_TARGET_OFFSET_PX = -212
# 한 프레임 사이 offset이 이 값(px)보다 더 튀면 오검출로 보고 이전 값 유지.
# None으로 두면 점프 검사 자체를 꺼서(제한 없음) 항상 이번 프레임 값을 그대로 씀.
MAX_OFFSET_JUMP_PX = None

# ---- 슬라이딩 윈도우(곡선 추적) -------------------------------------------
# 범위를 좁게 잡아서 옆에 있는 꽃 그림 등을 덜 건드리게 함.
NUM_WINDOWS = 7      # ROI를 세로로 나누는 윈도우 개수(아래->위로 순차 추적)
WINDOW_MARGIN = 27   # 윈도우 폭의 절반(px): 실제 폭은 x_current 기준 좌우 margin*2
WINDOW_MINPIX = 50   # 윈도우 안에서 이 픽셀 수 이상 모여야 x_current를 갱신

# ---- 근접 밴드 기준선 판정용 색상 스트립(green_backed/road_flanked) ---------
# find_lane_bases(근접 밴드, 차선이 거의 수직이라 좌/우 방향 가정이 유효한
# 구간)에서만 쓰는 방향 기준 색 검사. 슬라이딩 윈도우 곡선 추적에는 쓰지
# 않는다(아래 패치 기반 검사 참고).
GREEN_BACKED_RATIO_MIN = 0.30   # 오른쪽 판정: 덩어리 바로 오른쪽 초록 비율 임계값
ROAD_FLANKED_RATIO_MIN = 0.40   # 왼쪽 판정: 덩어리 양옆 회색 도로 비율 임계값
COLOR_CHECK_NEAR_PX = 10        # 색 비교 스트립 시작 위치(덩어리 경계에서부터)
COLOR_CHECK_FAR_PX = 40         # 색 비교 스트립 끝 위치(덩어리 경계에서부터)

# ---- 패치(부분공간) 기반 경계선 인식 + 연속성 제약(곡선 대응) ----------------
# 방향(이미지 좌/우) 가정 없이, 작은 정사각형 패치 하나하나에 대해 "그 안에
# 초록/회색/흰색이 각각 일정 비율 이상 섞여 있는가"만으로 경계선 조각인지
# 판정한다. 오른쪽 차선(실선)은 초록-회색 경계 위에 있으므로 패치 안에
# 초록+회색이 모두 일정 비율 이상 있으면 인정하고, 왼쪽 차선(점선)은 양옆이
# 모두 회색이므로 회색 비율만 본다. 방향을 안 따지기 때문에 차선이 급커브로
# 휘어 로컬 방향이 수평에 가까워져도 깨지지 않는다.
# 패치들을 연결할 때도 "이전 단에서 얼마나 떨어졌는지"(연속성)만 보고,
# 좌/우 전역 분리는 하지 않는다.
PATCH_WIDTH_PX = 30          # 패치 폭(px). 높이는 슬라이딩 윈도우 한 단의 높이를 그대로 씀
MIN_PATCH_WHITE_RATIO = 0.05  # 패치 안에 흰색(차선 픽셀)이 최소 이만큼은 있어야 후보로 인정
# 패치 전용 초록/회색 비율 임계값. 근접 밴드의 GREEN_BACKED_RATIO_MIN(0.30)/
# ROAD_FLANKED_RATIO_MIN(0.40)을 그대로 재사용하면 안 된다: 그 값들은 "차선
# 경계에서 떨어진 순수 초록/회색 스트립"을 보는 기준이라 초록+회색 합이
# 0.70을 넘어야 하는데, 패치 방식은 흰 차선 자체가 패치 폭 일부를 차지하고
# 있어서(polyfit 실험 결과 경계 부근 흰색 비율이 최대 ~0.35~0.4까지 남음)
# 초록+회색+흰색 합이 1이 되는 한 두 임계값의 합이 0.70을 넘으면 경계 부근
# 어떤 컬럼에서도 동시에 만족시킬 수 없다. 그래서 패치용은 합이 충분히
# 작은 값으로 따로 둔다.
PATCH_GREEN_RATIO_MIN = 0.15  # 오른쪽 판정(패치): 패치 안 초록 비율 임계값
PATCH_ROAD_RATIO_MIN = 0.15   # 왼쪽/오른쪽 판정(패치): 패치 안 회색 도로 비율 임계값
# 연속성 제약: 이전 윈도우의 x_current 대비 이 거리(px)보다 멀리 떨어진
# 패치는(색이 맞아도) 후보에서 제외한다. 급커브 안쪽의 반대쪽 차선처럼
# 색 조건은 우연히 맞아도 물리적으로 이어질 수 없는 패치를 걸러낸다.
WINDOW_MAX_STEP_PX = 40

# ---- 모양 필터(near-field 차선 시작점 탐색용) ------------------------------
# 초록 매트 위 흰 꽃 그림 등은 색은 흰색이지만 작고 동글동글한 덩어리다.
# 아래 둘 중 하나를 만족해야 차선 후보로 인정한다.
#   1) span: 밴드 높이의 대부분을 채움 -> 실선/점선은 곡선에서 옆으로
#      휘어져도(가로 폭이 넓어져도) 밴드를 처음부터 끝까지 관통하지만,
#      꽃 그림은 작은 덩어리라 밴드 높이를 거의 못 채운다.
#   2) aspect: 세로로 길고 가로로 짧음 -> 밴드를 다 못 채우는 짧은
#      점선 조각이라도 모양 자체가 길쭉하면 인정.
NEAR_FIELD_FULL_HEIGHT_RATIO = 0.8  # span 기준: 밴드 높이 대비 덩어리 높이 비율 임계값
MIN_LINE_ASPECT_RATIO = 1.5         # aspect 기준: 세로/가로 비율 임계값
MIN_LINE_HEIGHT_PX = 20             # aspect 기준: 최소 세로 길이(px)
# 도로 이음새/실금처럼 실제 차선보다 훨씬 가느다란 흰 줄을 걸러내기 위한
# 최소 평균 폭(px). bbox width 대신 area/height(=평균 두께)를 쓴다 - 근접
# 밴드 안에서 살짝 대각선으로 지나가는 덩어리는 bbox 폭이 실제 두께보다
# 커져서, bbox 폭만 보면 얇은 실금도 통과할 수 있기 때문이다.
MIN_LINE_AVG_WIDTH_PX = 4

# ---- 디버그 시각화 --------------------------------------------------------
# ROI/차선/슬라이딩 윈도우를 그린 화면을 바로 OpenCV 창으로 띄운다.
# (bag/카메라 토픽만 켜져 있으면, 이 노드 실행만으로 인식 화면이 뜬다.)
# 실차 대회 주행 시에는 CPU 절약을 위해 False로 끄는 것을 권장.
DEBUG_VIEW = True
PUBLISH_DEBUG_IMAGE = False  # True로 켜면 OpenCV 창 없이도 디버그 이미지를 토픽으로 발행
WINDOW_NAME = 'timed_lane_offset_debug'          # OpenCV 디버그 창 이름
DEBUG_IMAGE_TOPIC = '/lane_offset/debug_image'    # 디버그 이미지를 토픽으로도 발행할 때 쓰는 토픽명


class TimedLaneOffsetNode(Node):
    """/camera/high/image_raw -> 오른쪽(주)/왼쪽(보조) 차선 기준 offset을 계산해 /lane_offset 발행."""

    def __init__(self):
        super().__init__('timed_lane_offset_node')

        # ---- 파라미터 ------------------------------------------------------
        self.declare_parameter('roi_top', ROI_TOP)
        self.declare_parameter('roi_bottom', ROI_BOTTOM)
        self.declare_parameter('white_h_min', WHITE_H_MIN)
        self.declare_parameter('white_h_max', WHITE_H_MAX)
        self.declare_parameter('white_s_max', WHITE_S_MAX)
        self.declare_parameter('white_v_min', WHITE_V_MIN)
        self.declare_parameter('white_close_kernel_px', WHITE_CLOSE_KERNEL_PX)
        self.declare_parameter('green_h_min', GREEN_H_MIN)
        self.declare_parameter('green_h_max', GREEN_H_MAX)
        self.declare_parameter('green_s_min', GREEN_S_MIN)
        self.declare_parameter('green_v_min', GREEN_V_MIN)
        self.declare_parameter('road_h_min', ROAD_H_MIN)
        self.declare_parameter('road_h_max', ROAD_H_MAX)
        self.declare_parameter('road_s_min', ROAD_S_MIN)
        self.declare_parameter('road_s_max', ROAD_S_MAX)
        self.declare_parameter('road_v_min', ROAD_V_MIN)
        self.declare_parameter('road_v_max', ROAD_V_MAX)
        self.declare_parameter('near_field_rows', NEAR_FIELD_ROWS)
        self.declare_parameter('white_overload_ratio', WHITE_OVERLOAD_RATIO)
        self.declare_parameter('target_offset_px', TARGET_OFFSET_PX)
        self.declare_parameter('left_target_offset_px', LEFT_TARGET_OFFSET_PX)
        # dynamic_typing=True: MAX_OFFSET_JUMP_PX(또는 이 파라미터 오버라이드)가
        # None이어도(점프 검사 끔) 타입 에러 없이 선언할 수 있게 함.
        self.declare_parameter(
            'max_offset_jump_px',
            MAX_OFFSET_JUMP_PX,
            descriptor=ParameterDescriptor(dynamic_typing=True),
        )
        self.declare_parameter('num_windows', NUM_WINDOWS)
        self.declare_parameter('window_margin', WINDOW_MARGIN)
        self.declare_parameter('window_minpix', WINDOW_MINPIX)
        self.declare_parameter('green_backed_ratio_min', GREEN_BACKED_RATIO_MIN)
        self.declare_parameter('road_flanked_ratio_min', ROAD_FLANKED_RATIO_MIN)
        self.declare_parameter('color_check_near_px', COLOR_CHECK_NEAR_PX)
        self.declare_parameter('color_check_far_px', COLOR_CHECK_FAR_PX)
        self.declare_parameter('patch_width_px', PATCH_WIDTH_PX)
        self.declare_parameter('min_patch_white_ratio', MIN_PATCH_WHITE_RATIO)
        self.declare_parameter('patch_green_ratio_min', PATCH_GREEN_RATIO_MIN)
        self.declare_parameter('patch_road_ratio_min', PATCH_ROAD_RATIO_MIN)
        self.declare_parameter('window_max_step_px', WINDOW_MAX_STEP_PX)
        self.declare_parameter(
            'near_field_full_height_ratio', NEAR_FIELD_FULL_HEIGHT_RATIO
        )
        self.declare_parameter('min_line_aspect_ratio', MIN_LINE_ASPECT_RATIO)
        self.declare_parameter('min_line_height_px', MIN_LINE_HEIGHT_PX)
        self.declare_parameter('min_line_avg_width_px', MIN_LINE_AVG_WIDTH_PX)
        self.declare_parameter('debug_view', DEBUG_VIEW)
        self.declare_parameter('publish_debug_image', PUBLISH_DEBUG_IMAGE)

        self.image_topic = IMAGE_TOPIC
        self.lane_offset_topic = LANE_OFFSET_TOPIC
        self.roi_top = int(self.get_parameter('roi_top').value)
        self.roi_bottom = int(self.get_parameter('roi_bottom').value)
        self.white_h_min = int(self.get_parameter('white_h_min').value)
        self.white_h_max = int(self.get_parameter('white_h_max').value)
        self.white_s_max = int(self.get_parameter('white_s_max').value)
        self.white_v_min = int(self.get_parameter('white_v_min').value)
        white_close_kernel_px = int(self.get_parameter('white_close_kernel_px').value)
        self.white_close_kernel = None
        if white_close_kernel_px > 1:
            self.white_close_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (white_close_kernel_px, white_close_kernel_px)
            )
        self.green_h_min = int(self.get_parameter('green_h_min').value)
        self.green_h_max = int(self.get_parameter('green_h_max').value)
        self.green_s_min = int(self.get_parameter('green_s_min').value)
        self.green_v_min = int(self.get_parameter('green_v_min').value)
        self.road_h_min = int(self.get_parameter('road_h_min').value)
        self.road_h_max = int(self.get_parameter('road_h_max').value)
        self.road_s_min = int(self.get_parameter('road_s_min').value)
        self.road_s_max = int(self.get_parameter('road_s_max').value)
        self.road_v_min = int(self.get_parameter('road_v_min').value)
        self.road_v_max = int(self.get_parameter('road_v_max').value)
        self.near_field_rows = int(self.get_parameter('near_field_rows').value)
        self.white_overload_ratio = float(
            self.get_parameter('white_overload_ratio').value
        )
        self.target_offset_px = int(self.get_parameter('target_offset_px').value)
        self.left_target_offset_px = int(
            self.get_parameter('left_target_offset_px').value
        )
        _max_offset_jump_px = self.get_parameter('max_offset_jump_px').value
        self.max_offset_jump_px = (
            None if _max_offset_jump_px is None else int(_max_offset_jump_px)
        )
        self.num_windows = int(self.get_parameter('num_windows').value)
        self.window_margin = int(self.get_parameter('window_margin').value)
        self.window_minpix = int(self.get_parameter('window_minpix').value)
        self.green_backed_ratio_min = float(
            self.get_parameter('green_backed_ratio_min').value
        )
        self.road_flanked_ratio_min = float(
            self.get_parameter('road_flanked_ratio_min').value
        )
        self.color_check_near_px = int(self.get_parameter('color_check_near_px').value)
        self.color_check_far_px = int(self.get_parameter('color_check_far_px').value)
        self.patch_width_px = int(self.get_parameter('patch_width_px').value)
        self.min_patch_white_ratio = float(
            self.get_parameter('min_patch_white_ratio').value
        )
        self.patch_green_ratio_min = float(
            self.get_parameter('patch_green_ratio_min').value
        )
        self.patch_road_ratio_min = float(
            self.get_parameter('patch_road_ratio_min').value
        )
        self.window_max_step_px = int(self.get_parameter('window_max_step_px').value)
        self.near_field_full_height_ratio = float(
            self.get_parameter('near_field_full_height_ratio').value
        )
        self.min_line_aspect_ratio = float(
            self.get_parameter('min_line_aspect_ratio').value
        )
        self.min_line_height_px = int(self.get_parameter('min_line_height_px').value)
        self.min_line_avg_width_px = float(
            self.get_parameter('min_line_avg_width_px').value
        )
        self.debug_view = bool(self.get_parameter('debug_view').value)
        self.window_name = WINDOW_NAME
        self.publish_debug_image = bool(
            self.get_parameter('publish_debug_image').value
        )
        self.debug_image_topic = DEBUG_IMAGE_TOPIC

        # 마지막으로 발행한(유효했던) offset. 오검출 프레임에서는 이 값을 그대로 재사용.
        self.last_offset = 0
        # 근접 밴드에서 마지막으로 채택한 오른쪽/왼쪽 기준점 x좌표. 후보가 여럿일 때
        # "가장 오른쪽"이 아니라 "이전 프레임과 가장 가까운 것"을 고르는 연속성
        # 기준으로 쓴다(도로 위 실금 등 노이즈가 필터를 뚫고 들어와도 프레임마다
        # 다른 걸 골라 값이 튀는 것을 막기 위함). 탐지가 안 된 프레임에는 갱신하지
        # 않고 유지해서, 점선처럼 잠깐 끊기는 경우에도 연속성이 이어지게 한다.
        self.prev_right_x = None
        self.prev_left_x = None

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
            f'target_offset_px={self.target_offset_px}, '
            f'left_target_offset_px={self.left_target_offset_px}'
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
        white_mask = self.make_white_mask(hsv)
        green_mask = self.make_green_mask(hsv)
        road_mask = self.make_road_mask(hsv)

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

        right_x, left_x = self.find_lane_bases(white_mask, green_mask, road_mask)
        if right_x is None and left_x is None:
            self.get_logger().warn(
                'No lane pixels found, holding last offset', throttle_duration_sec=1.0
            )
            self.publish_offset(self.last_offset)
            self.publish_debug(msg, frame, 'NO LANE FOUND', near_white_ratio)
            return

        # 오른쪽/왼쪽 중 보이는 쪽마다 각자의 기준 offset을 뺀 값을 구하고,
        # 둘 다 보이면 평균해서 서로가 서로를 검증하도록 한다.
        center_x = frame.shape[1] / 2.0
        offsets = []
        base_points = []
        windows_all = []
        points_x_all = []
        points_y_all = []

        if right_x is not None:
            line_x_r, windows_r, px_r, py_r = self.track_line_with_sliding_window(
                white_mask, green_mask, road_mask, right_x, 'right'
            )
            offsets.append((line_x_r - center_x) - self.target_offset_px)
            base_points.append(right_x)
            windows_all.extend(windows_r)
            points_x_all.extend(px_r)
            points_y_all.extend(py_r)

        if left_x is not None:
            line_x_l, windows_l, px_l, py_l = self.track_line_with_sliding_window(
                white_mask, green_mask, road_mask, left_x, 'left'
            )
            offsets.append((line_x_l - center_x) - self.left_target_offset_px)
            base_points.append(left_x)
            windows_all.extend(windows_l)
            points_x_all.extend(px_l)
            points_y_all.extend(py_l)

        lane_offset = int(round(sum(offsets) / len(offsets)))
        if right_x is not None and left_x is not None:
            mode = 'both'
        elif right_x is not None:
            mode = 'right'
        else:
            mode = 'left'

        jump_too_large = (
            self.max_offset_jump_px is not None
            and abs(lane_offset - self.last_offset) > self.max_offset_jump_px
        )
        if jump_too_large:
            # 한 프레임 만에 비정상적으로 튀면 오검출로 보고 이전 값 유지
            # (max_offset_jump_px가 None이면 이 검사 자체를 건너뛴다)
            self.get_logger().warn(
                f'Offset jump too large ({self.last_offset} -> {lane_offset}), '
                'holding last offset',
                throttle_duration_sec=1.0,
            )
            self.publish_offset(self.last_offset)
            self.publish_debug(
                msg, frame, 'JUMP REJECTED', near_white_ratio,
                base_x=base_points, windows=windows_all,
                points_x=points_x_all, points_y=points_y_all, lane_offset=lane_offset,
                mode=mode,
            )
            return

        self.last_offset = lane_offset
        self.publish_offset(lane_offset)
        self.publish_debug(
            msg, frame, 'OK', near_white_ratio,
            base_x=base_points, windows=windows_all,
            points_x=points_x_all, points_y=points_y_all, lane_offset=lane_offset,
            mode=mode,
        )

    def publish_offset(self, value):
        msg = Int16()
        msg.data = int(value)
        self.offset_pub.publish(msg)

    # ======================================================================
    # 색상 마스크
    # ======================================================================
    def make_white_mask(self, hsv):
        h, s, v = cv2.split(hsv)
        mask = (
            (h > self.white_h_min)
            & (h < self.white_h_max)
            & (s < self.white_s_max)
            & (v > self.white_v_min)
        )
        mask = (mask.astype(np.uint8)) * 255
        if self.white_close_kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.white_close_kernel)
        return mask

    def make_green_mask(self, hsv):
        h, s, v = cv2.split(hsv)
        mask = (
            (h > self.green_h_min)
            & (h < self.green_h_max)
            & (s > self.green_s_min)
            & (v > self.green_v_min)
        )
        return (mask.astype(np.uint8)) * 255

    def make_road_mask(self, hsv):
        lower = (self.road_h_min, self.road_s_min, self.road_v_min)
        upper = (self.road_h_max, self.road_s_max, self.road_v_max)
        return cv2.inRange(hsv, lower, upper)

    # ======================================================================
    # 모양 필터: 차선(세로로 길고 가로로 짧음) vs 꽃 그림 등 (동글동글한 덩어리)
    # ======================================================================
    def find_lane_shaped_components(
        self, mask, min_height, min_aspect_ratio, full_height_ratio, min_avg_width
    ):
        """mask에서 차선처럼 생긴 픽셀 뭉치만 골라 (x, y, w, h, label) bbox 리스트로 반환.

        먼저 평균 폭(area/h, 두께 근사치)이 min_avg_width 이상이어야 한다.
        도로 이음새/실금처럼 실제 차선보다 훨씬 가느다란 흰 줄을 걸러내기 위함.
        bbox 폭(w) 대신 area/h를 쓰는 이유: 근접 밴드 안에서 살짝 대각선으로
        지나가는 덩어리는 bbox 폭이 실제 두께보다 커지므로, bbox 폭만 보면
        얇은 실금도 통과할 수 있다.

        그다음 아래 둘 중 하나를 만족해야 통과:
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
            x, y, w, h, area = stats[label]
            if w <= 0 or h <= 0:
                continue
            avg_width = area / float(h)
            if avg_width < min_avg_width:
                continue
            spans_band = h >= full_height_threshold
            is_tall_narrow = h >= min_height and (h / float(w)) >= min_aspect_ratio
            if spans_band or is_tall_narrow:
                boxes.append((x, y, w, h, label))
        return boxes, labels

    # ======================================================================
    # 색상 일관성 체크 (근접 밴드 전용 - 방향 기준)
    # ======================================================================
    def check_color_consistency(self, green_band, road_band, left_edge, right_edge, lane_kind):
        """덩어리(left_edge~right_edge, band 로컬 x좌표)가 lane_kind 특징에 맞는지 검사.

        - lane_kind == 'right': 덩어리 바로 오른쪽이 초록 매트인가(green_backed)
        - lane_kind == 'left' : 덩어리 양옆이 회색 도로인가(road_flanked)

        근접 밴드(find_lane_bases)는 차량 바로 앞이라 차선이 거의 수직에
        가까우므로 "이미지 오른쪽/양옆"이라는 방향 기준이 유효하다. 슬라이딩
        윈도우 곡선 추적에는 이 방식 대신 방향 가정이 없는 패치 기반 검사
        (compute_patch_ratio_profile / patch_is_lane_boundary)를 쓴다.

        green_band/road_band는 검사 대상과 같은 y범위(행)를 공유하는 마스크여야 한다.
        """
        width = (green_band if green_band is not None else road_band).shape[1]
        near = self.color_check_near_px
        far = self.color_check_far_px

        if lane_kind == 'right':
            rx0 = min(right_edge + near, width - 1)
            rx1 = min(right_edge + far, width)
            if rx1 <= rx0:
                return False
            strip = green_band[:, rx0:rx1]
            green_ratio = float((strip > 0).mean()) if strip.size else 0.0
            return green_ratio > self.green_backed_ratio_min

        rx0 = min(right_edge + near, width - 1)
        rx1 = min(right_edge + far, width)
        lx0 = max(left_edge - far, 0)
        lx1 = max(left_edge - near, 0)
        left_ratio = 0.0
        if lx1 > lx0:
            strip = road_band[:, lx0:lx1]
            left_ratio = float((strip > 0).mean()) if strip.size else 0.0
        right_ratio = 0.0
        if rx1 > rx0:
            strip = road_band[:, rx0:rx1]
            right_ratio = float((strip > 0).mean()) if strip.size else 0.0
        return left_ratio > self.road_flanked_ratio_min and right_ratio > self.road_flanked_ratio_min

    # ======================================================================
    # 패치(부분공간) 기반 경계선 인식 (슬라이딩 윈도우 곡선 추적 전용 - 방향 무관)
    # ======================================================================
    def compute_patch_ratio_profile(self, band_mask):
        """band_mask의 각 컬럼 x를 중심으로 patch_width_px 폭 패치의 on-비율 프로파일을 반환.

        patch_width_px x band_mask.shape[0] 크기 정사각형에 가까운 패치를 한 칸씩
        옆으로 밀면서(컬럼마다) 그 안의 on-픽셀 비율을 계산한다. 좌/우를 구분하지
        않는 등방적(isotropic) 검사라서, 차선이 어느 각도로 놓여 있어도(급커브로
        로컬 방향이 수평에 가까워져도) 그대로 적용된다.
        """
        height, width = band_mask.shape
        binary = (band_mask > 0).astype(np.float32)
        col_sum = binary.sum(axis=0)
        cumsum = np.concatenate(([0.0], np.cumsum(col_sum)))
        half = self.patch_width_px // 2
        xs = np.arange(width)
        x0 = np.clip(xs - half, 0, width)
        x1 = np.clip(xs + half + 1, 0, width)
        areas = (x1 - x0) * height
        sums = cumsum[x1] - cumsum[x0]
        return np.divide(sums, areas, out=np.zeros_like(sums), where=areas > 0)

    def patch_is_lane_boundary(self, white_ratio, green_ratio, road_ratio, lane_kind):
        """컬럼(패치)이 lane_kind 경계선 조각인지, 색 혼합 비율만으로 판정.

        스칼라와 numpy 배열(컬럼별 프로파일 전체) 양쪽 다 그대로 넣을 수 있게
        비교 연산(& / >)만 사용한다(and/or, 조기 return 금지 - 배열이면 진리값이
        모호해져 에러가 난다).

        - 흰색(차선 픽셀) 비율이 min_patch_white_ratio 이상이어야 함(둘 다 공통).
        - lane_kind == 'right': 초록(patch_green_ratio_min)과 회색(patch_road_ratio_min)이
          "둘 다" 일정 비율 이상이어야 함 -> 패치가 초록-회색 경계에 걸쳐 있다는 뜻.
        - lane_kind == 'left' : 양옆이 전부 회색이라 초록은 안 나온다. 회색 비율만 검사.

        패치 임계값(patch_green_ratio_min/patch_road_ratio_min)은 근접 밴드용
        green_backed_ratio_min/road_flanked_ratio_min과 다른 값을 쓴다. 패치
        방식은 흰 차선 자체가 패치 폭 일부를 차지하므로 초록+회색+흰색 비율의
        합이 대략 1이 되고, 경계 부근에서는 흰색 비율만으로도 상당 부분을
        차지해 초록+회색 임계값의 합이 크면(예: 0.70) 어떤 컬럼도 동시에
        만족시킬 수 없다.
        """
        white_ok = white_ratio > self.min_patch_white_ratio
        if lane_kind == 'right':
            color_ok = (green_ratio > self.patch_green_ratio_min) & (
                road_ratio > self.patch_road_ratio_min
            )
        else:
            color_ok = road_ratio > self.patch_road_ratio_min
        return white_ok & color_ok

    # ======================================================================
    # 기준선 시작 위치 찾기 (오른쪽 실선 / 왼쪽 점선을 각각 독립적으로 탐지)
    # ======================================================================
    def find_lane_bases(self, white_mask, green_mask, road_mask):
        """근접 밴드에서 오른쪽/왼쪽 후보를 독립적으로 찾아 (right_x, left_x)로 반환.

        같은 후보 목록(차선 모양 필터를 통과한 흰 덩어리)에 대해 서로 다른
        기준으로 각각 검사한다. 한 후보가 두 조건을 동시에 만족하는 일은
        거의 없다(초록으로 확인되면 road_ratio가 낮게 나온다).

        - 오른쪽: 덩어리 바로 오른쪽이 초록색 매트인가 (green_backed)
        - 왼쪽: 덩어리 양옆(왼쪽 스트립 + 오른쪽 스트립) 모두 회색 도로색인가
          (road_flanked). 소거법이 아니라 hsv_tuner로 실측한 도로색을 직접
          검사하므로, 크로스워크나 주차구획선처럼 green_backed는 아니지만
          왼쪽 차선도 아닌 흰 마킹을 잘못 왼쪽 차선으로 잡는 걸 막아준다.

        각각 후보가 여럿이면 이전 프레임에서 채택했던 위치(prev_right_x/
        prev_left_x)와 가장 가까운 것을 사용한다(select_candidate_with_continuity).
        도로 이음새 등 폭 필터를 뚫고 들어온 노이즈가 매 프레임 다른 후보로
        잘못 선택되어 값이 튀는 것을 막기 위함이다. 이전 프레임 기록이 없으면
        (첫 프레임 등) 가장 오른쪽 것을 쓴다. 후보가 없으면 None.
        """
        band_white = white_mask[-self.near_field_rows:, :]
        band_green = green_mask[-self.near_field_rows:, :]
        band_road = road_mask[-self.near_field_rows:, :]
        width = band_white.shape[1]

        boxes, _labels = self.find_lane_shaped_components(
            band_white,
            self.min_line_height_px,
            self.min_line_aspect_ratio,
            self.near_field_full_height_ratio,
            self.min_line_avg_width_px,
        )
        if not boxes:
            return None, None

        right_candidates = []
        left_candidates = []
        for x, _y, w, _h, _label in boxes:
            center_x = x + w // 2
            left_edge = x
            right_edge = x + w

            if self.check_color_consistency(band_green, band_road, left_edge, right_edge, 'right'):
                right_candidates.append(center_x)
            if self.check_color_consistency(band_green, band_road, left_edge, right_edge, 'left'):
                left_candidates.append(center_x)

        right_x = self.select_candidate_with_continuity(right_candidates, self.prev_right_x)
        left_x = self.select_candidate_with_continuity(left_candidates, self.prev_left_x)
        if right_x is not None:
            self.prev_right_x = right_x
        if left_x is not None:
            self.prev_left_x = left_x
        return right_x, left_x

    def select_candidate_with_continuity(self, candidates, prev_x):
        """후보 목록에서 이전 프레임 위치(prev_x)와 가장 가까운 것을 고른다.

        prev_x가 없으면(아직 한 번도 탐지된 적 없음) 가장 오른쪽 것을 쓴다
        (기존 동작과 동일한 초기값 규칙).
        """
        if not candidates:
            return None
        if prev_x is not None:
            return min(candidates, key=lambda c: abs(c - prev_x))
        return max(candidates)

    # ======================================================================
    # 슬라이딩 윈도우로 곡선 추적
    # ======================================================================
    def track_line_with_sliding_window(self, white_mask, green_mask, road_mask, base_x, lane_kind):
        """차선 x좌표(ROI 하단 기준)와 함께, 그린 윈도우/사용된 픽셀도 반환(디버그용).

        아래에서 위로 올라가며(윈도우 한 단씩) 차선을 추적하되, 각 단에서
        "어느 컬럼이 lane_kind 경계선 조각인가"를 방향(좌/우) 가정 없이 패치
        단위로 검사한다:

        1) 색 혼합 비율: 이 단의 각 컬럼 x를 중심으로 patch_width_px 폭 정사각
           패치를 만들고, 그 안의 흰색/초록/회색 비율을 계산한다
           (compute_patch_ratio_profile). patch_is_lane_boundary로 "이 패치가
           lane_kind 특징(오른쪽=초록+회색 혼합, 왼쪽=회색)에 맞는가"를 판정한다.
           패치가 등방적(정사각형)이라 차선이 이미지 안에서 어느 각도로 놓여
           있어도(급커브로 로컬 방향이 수평에 가까워져도) 그대로 적용된다.
        2) 연속성 제약(window_max_step_px): 색이 맞는 패치가 여럿이면 이전
           윈도우의 x_current에서 가장 가까운 것만 인정한다. 너무 멀리 떨어진
           패치는(색이 맞아도) 물리적으로 이어질 수 없다고 보고 후보에서 뺀다
           (급커브 안쪽의 반대쪽 차선 등).

        색 조건을 만족하는 컬럼이 있으면 그중 x_current와 가장 가까운 것을 쓰고,
        없으면(그림자 등으로 색 판정이 애매한 구간) 이번 윈도우는 건너뛴다(추적 유지).
        """
        height, width = white_mask.shape
        window_height = max(1, height // self.num_windows)
        half_patch = self.patch_width_px // 2
        x_current = base_x

        windows = []
        collected_x = []
        collected_y = []
        for i in range(self.num_windows):
            y_high = height - i * window_height
            y_low = max(0, height - (i + 1) * window_height)

            if y_high <= y_low:
                windows.append((x_current, x_current, y_low, y_high))
                continue

            white_band = white_mask[y_low:y_high, :]
            green_band = green_mask[y_low:y_high, :]
            road_band = road_mask[y_low:y_high, :]

            white_ratio_profile = self.compute_patch_ratio_profile(white_band)
            green_ratio_profile = self.compute_patch_ratio_profile(green_band)
            road_ratio_profile = self.compute_patch_ratio_profile(road_band)

            is_boundary = self.patch_is_lane_boundary(
                white_ratio_profile, green_ratio_profile, road_ratio_profile, lane_kind
            )
            candidate_xs = np.where(is_boundary)[0]

            if candidate_xs.size == 0:
                windows.append((x_current, x_current, y_low, y_high))
                continue

            steps = np.abs(candidate_xs - x_current)
            within_step = steps <= self.window_max_step_px
            if not np.any(within_step):
                windows.append((x_current, x_current, y_low, y_high))
                continue
            candidate_xs = candidate_xs[within_step]
            steps = steps[within_step]

            x_new = int(candidate_xs[np.argmin(steps)])
            x_low_patch = max(0, x_new - half_patch)
            x_high_patch = min(width, x_new + half_patch + 1)
            windows.append((x_low_patch, x_high_patch, y_low, y_high))

            local_ys, local_xs = np.where(white_band[:, x_low_patch:x_high_patch] > 0)
            xs = local_xs + x_low_patch
            ys = local_ys + y_low

            if xs.size >= self.window_minpix:
                x_current = int(round(float(xs.mean())))
                collected_x.extend(xs.tolist())
                collected_y.extend(ys.tolist())

        if len(collected_x) < self.window_minpix:
            # 위쪽 윈도우들에서 거의 못 찾았으면 근접 기준점만 사용
            return float(base_x), windows, collected_x, collected_y

        # y가 최소 3개 구간 이상 걸쳐 있어야 2차 다항식 피팅 의미가 있음
        if len(set(collected_y)) >= 3:
            coeffs = np.polyfit(collected_y, collected_x, 2)
            line_x = float(np.polyval(coeffs, height - 1))
        else:
            line_x = float(np.mean(collected_x))

        return line_x, windows, collected_x, collected_y

    # ======================================================================
    # 디버그 시각화
    # ======================================================================
    def publish_debug(
        self, src_msg, frame, status, near_white_ratio,
        base_x=None, windows=None,
        points_x=None, points_y=None, lane_offset=None, mode=None,
    ):
        if not (self.debug_view or self.publish_debug_image):
            return

        debug = frame.copy()
        height, width = debug.shape[:2]
        center_x = width // 2

        # ROI 영역 표시
        cv2.rectangle(
            debug, (0, self.roi_top), (width - 1, self.roi_bottom - 1), (0, 255, 255), 1
        )
        # 차량 중심선(흰색)과 오른쪽/왼쪽 기준 offset 위치(각각 주황/청록 점선,
        # "여기 있어야 그 기준으로는 offset=0")를 둘 다 그려서 비교할 수 있게 한다.
        cv2.line(debug, (center_x, self.roi_top), (center_x, self.roi_bottom), (255, 255, 255), 1)
        self.draw_dashed_vline(
            debug, center_x + self.target_offset_px, self.roi_top, self.roi_bottom, (0, 165, 255)
        )
        self.draw_dashed_vline(
            debug, center_x + self.left_target_offset_px, self.roi_top, self.roi_bottom, (255, 255, 0)
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

        # 근접 밴드에서 찾은 기준점(오른쪽/왼쪽 각각, 있는 만큼)
        if base_x:
            for bx in base_x:
                cv2.circle(debug, (int(bx), self.roi_bottom - 1), 5, (0, 0, 255), -1)

        color = (0, 255, 0) if status == 'OK' else (0, 0, 255)
        lines = [
            f'status: {status} ({mode or "-"})',
            f'lane_offset: {lane_offset if lane_offset is not None else self.last_offset}',
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
