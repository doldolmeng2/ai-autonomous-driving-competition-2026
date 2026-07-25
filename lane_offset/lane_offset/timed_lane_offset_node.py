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

# 구독/발행 토픽
IMAGE_TOPIC = '/camera/high/image_raw'
LANE_OFFSET_TOPIC = '/lane_offset'

# ROI: 이미지 상단(배경)과 하단(차량 후드)을 잘라낸다. (640x360 기준)
# 카메라가 640x480이 아니라 640x360(16:9)으로 설정되어 있어(sensor_topic/config/
# camera.yaml), 이전에 480 기준으로 잡혀있던 값(250~450)은 실제 프레임 높이(360)를
# 넘어가 아래쪽이 조용히 잘리는 문제가 있었다. 실측 프레임(640x360) 기준으로
# 다시 잡은 값이다.
ROI_TOP = 220
ROI_BOTTOM = 360
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

# 오른쪽 차선 탐색에 쓰는 근접(ROI 하단) 밴드 높이. ROI 높이(150px)의 절반 정도로,
# 이전 480 기준(100/200=50%) 비율을 그대로 유지한다.
NEAR_FIELD_ROWS = 75
# 근접 밴드에서 흰색 비율이 이 값을 넘으면 횡단보도 등으로 판단하고 무시
WHITE_OVERLOAD_RATIO = 0.15

# 주행 모드. '2lane'은 중앙 점선의 오른쪽 차로, '1lane'은 필요 시 기준값만 바꿔
# 재사용한다. (ROS 실행 시 -p driving_mode:=1lane 으로 변경 가능)
DRIVING_MODE = '2lane' # 2lane
# 640px 너비 영상에서 차가 정상 위치일 때의 중앙 점선 x좌표.
# 사진 기준 640px 영상에서의 초기값이다. 1차선/2차선은 중앙 점선을 서로 다른
# 위치에서 보기 때문에, 이 값은 각각 점선 슬라이딩 윈도우의 시작점이기도 하다.
DASHED_REFERENCE_X_PX_2LANE = 130
DASHED_REFERENCE_X_PX_1LANE = 510
# 기준선과 이만큼 차이 나면 lane_offset의 최대/최소값(+/-45)에 도달한다.
OFFSET_ERROR_LIMIT_PX = 120
LANE_OFFSET_LIMIT = 45
# 기준선 오차를 offset으로 바꾼 뒤 적용할 비례 이득. 최종값은 +/-45로 제한한다.
OFFSET_KP = 1.5

# 곡률 적응형 기준선(curvature-adaptive ref). 고정 dashed_ref는 곡률을 무시해,
# 좌커브에선 점선이 정상적으로 화면 왼쪽에 보여도 130으로 되돌리려다 차를 안쪽
# (중앙선)으로 민다. 피팅 곡선의 "휨(bend)"으로 목표 기준선을 커브 바깥쪽으로
# 이동시킨다: 좌커브(bend>0) -> ref 낮춤 -> 차가 더 오른쪽, 우커브 -> 반대.
# bend = (near x) - (near에서 lookahead_dy 만큼 위의 x). effective_ref = ref - gain*bend.
# gain=0 이면 기존 고정 기준선(현재 동작). 라이브로 올리며 튜닝한다.
CURVE_REF_GAIN = 0.4
CURVE_REF_LOOKAHEAD_DY = 35

# 한 프레임 사이 offset이 이 값보다 
# 더 튀면 오검출로 보고 이전 값 유지
MAX_OFFSET_JUMP_PX = 80
# 발행하는 lane_offset에 적용할 저역통과(EMA) 필터 계수.
# smoothed = alpha*이번 계산값 + (1-alpha)*직전 발행값. 1.0이면 필터 없음(그대로 발행),
# 작을수록 부드럽지만 반응이 느려진다.
OFFSET_SMOOTHING_ALPHA = 0.6

# ============================================================================
# 오른쪽 차선 이탈 보호(히스테리시스)
# ------------------------------------------------------기----------------------
# 평소에는 중앙 점선 기준으로 주행하지만, 오른쪽 실선이 화면에서 너무 왼쪽
# (=차가 오른쪽으로 치우침)으로 오면 중앙선 기준을 잠시 버리고 "오른쪽 실선을
# 다시 화면 오른쪽으로 밀어내는" 방향(=좌조향)을 최우선으로 한다. 보호모드에서는
# 중앙 점선 x를 맞추는 것이 우선이 아니다.
#
# 좌표 규약: 2차선 주행에서 오른쪽 실선의 near-field x는 차가 오른쪽으로 갈수록
# 작아진다(오른쪽 경계가 화면 중앙 쪽으로 다가옴). 따라서 x가 "임계값보다
# 작아지면" 오른쪽으로 이탈한 것으로 본다.
#   - enter_x  : right_x 가 이 값 "이하"로 내려가면 보호모드 진입
#   - exit_x   : right_x 가 이 값 "이상"으로 회복되면 보호모드 해제
#                (enter < exit 로 히스테리시스: 진입/해제 임계값을 벌려 채터링 방지)
#   - target_x : 보호모드에서 오른쪽 실선을 되돌릴 목표 x (여기에 오면 조향=0)
#   - error_limit_px : |right_x - target_x| 가 이 값이면 조향이 최대(+/-45)에 도달.
#                작을수록 진입 직후 바로 강하게 되돌린다(중앙선용 195와 별개).
# 오른쪽 실선은 중앙 점선과 마찬가지로 near-field(차 바로 앞) x만 사용하므로
# 멀리 있는 오른쪽 차선이 임계값을 넘었다고 오인식하지 않는다(offset_near_rows).
#
# 값 근거(rosbag2_2026_07_19-17_44_07 실측): 정상 right_x ~456~490. 이전 설정
# (target=460, 중앙선용 error_limit 195 공용)은 진입 지점(right_x~435)에서 오차가
# 겨우 ~25px라 조향이 -12밖에 안 나와, 보호모드가 켜졌는데도 중앙선 모드(-20)보다
# 오히려 약했다. 그래서 차가 못 돌아오고 오른쪽으로 나가버렸다. 이를 고치려고:
#   * target_x = exit_x = 470  : 해제 지점에서 조향이 0으로 매끄럽게 넘어가고,
#     진입~해제 구간 내내 오차가 커져 강하게 되돌린다.
#   * error_limit_px = 50 (전용) : 진입(440) 즉시 -45로 포화(진입 순간 error=-30,
#     normalized=-30/50=-0.6 -> -0.6*45*1.8=-48.6 -> -45). "켜지자마자 최대 좌조향".
#     (right_x<=~442 면 이미 -45라 진입선 440 에서는 확실히 포화된다.)
#   * enter=440 / exit=470 : 히스테리시스 폭 30px 로 채터링(진입/해제 반복) 제거.
RIGHT_GUARD_ENABLE = True
RIGHT_GUARD_ENTER_X = 485
RIGHT_GUARD_EXIT_X = 492
RIGHT_GUARD_TARGET_X = 470
# 보호모드 전용 조향 포화 거리(px). 중앙선용 offset_error_limit_px(195)와 별개로
# 작게 잡아, 진입하는 순간(right_x=enter_x=440) 바로 -45 최대 좌조향이 나오게 한다.
# 50 이면 진입 시점 error(-30)에서 normalized=-0.6 -> kp(1.8) 적용 후 -45로 포화.
RIGHT_GUARD_ERROR_LIMIT_PX = 50
# 보호모드 조향 비례 이득(중앙선 offset_kp 와 별개로 조금 더 단호하게 되돌린다).
RIGHT_GUARD_KP = 1.8
# 오른쪽 실선 x가 이 값보다 작으면(=트래커 붕괴/오검출로 화면 왼쪽 끝에 붙는
# 경우) 신뢰하지 않고 "안 보임"으로 처리한다. 커브에서 오른쪽 트래커가 픽셀을
# 놓쳐 right_x=None 이 되는 경우와 함께, 강제로 보호모드를 켜지 않고 중앙선
# 모드로 폴백한다.
RIGHT_GUARD_MIN_VALID_X = 80
# 오른쪽 이탈 보호가 오른쪽 실선을 "얼마나 멀리까지" 보고 켜질지 정하는 룩어헤드
# 밴드 높이(px, ROI 바닥 기준). 가드는 near x(차 바로 앞)만 보면, 좌커브에서 차
# 앞 x 는 아직 enter_x 위인데 위쪽에서 오른쪽 실선이 안쪽으로 굽어 들어오는 경우를
# 놓친다. 그래서 이 밴드 안 오른쪽 실선 점들의 median 을 함께 구해, near x 와 둘 중
# 더 안쪽(작은) 값으로 진입/조향을 판단한다. 밴드를 넓힐수록 굽은 위쪽 점이 섞여
# 값이 작아져 더 일찍 켜진다.
#   * 기본 60(=offset_near_rows 수준): 기존 near 중심 동작에 가깝다(안전).
#   * 좌커브에서 안 켜지면 90~120 으로 키운다. 단 원근 수렴 때문에 직선에서도
#     값이 다소 작아지므로, 너무 키우면 직선에서 오발동하거나 exit_x(470)에
#     도달 못 해 안 꺼질 수 있다. 그때는 enter_x/exit_x 를 같이 낮춰 맞춘다.
RIGHT_GUARD_LOOKAHEAD_ROWS = 62
# True면 보호모드가 켜진 동안 right_x 비례 계산 없이 "무조건 최대 좌조향(-45)"을
# 낸다. 커브에서 비례+EMA 램프가 약해 오른쪽으로 이탈하는 걸 막기 위한 단호한
# 모드로, EMA도 우회해 진입 즉시 -45로 스냅한다(비례 매핑용 target/error_limit/kp
# 는 이때 무시된다). False면 기존 비례 매핑(map_right_x_to_offset)을 쓴다.
# 주의: 이건 "켜졌을 때 얼마나 세게"이지 "언제 켜지나"가 아니다. 커브에서 애초에
# 가드가 안 켜져 이탈한 거라면 right_guard_lookahead_rows/enter_x 쪽을 조정해야 한다.
RIGHT_GUARD_FULL_STEER = True

# ----------------------------------------------------------------------------
# Soft 오른쪽 실선 회피(binary guard 대체)
# ----------------------------------------------------------------------------
# 임계값을 넘으면 무조건 -45로 스냅하던 binary 방식은, 원근 때문에 look-ahead x가
# 직선에서도 작아져 오발동/불안정했다. 대신 "평소엔 중앙선 추종, 오른쪽 실선에
# 가까워질수록 좌조향 가중치를 연속적으로" 부여한다. guard_x(=near/look-ahead min)
# 로 0~1 가중치 w를 만들어, 발행 offset = (1-w)*중앙선offset + w*(-45) 로 블렌딩한다.
#   - start_x : guard_x 가 이 값 "이상"이면 w=0 (평소, 순수 중앙선 추종)
#   - full_x  : guard_x 가 이 값 "이하"면 w=1 (완전 좌조향). start_x > full_x.
# 그 사이는 선형 램프라 binary 진입/해제(히스테리시스)나 채터링이 없다. 값은
# look-ahead x 스케일로 잡는다(예: 직선 la≈485~495, 커브 딥 la≈448 이면
# start=475/full=450 이면 직선 w=0, 커브에서만 램프). enter_x/exit_x/target_x/
# error_limit_px/kp/full_steer 등 binary 전용 값들은 이제 사용하지 않는다.
RIGHT_AVOID_START_X = 480
RIGHT_AVOID_FULL_X = 470

# 슬라이딩 윈도우 (범위를 좁게 잡아서 옆에 있는 꽃 그림 등을 덜 건드리게 함)
NUM_WINDOWS = 10
WINDOW_MARGIN = 27
WINDOW_MINPIX = 50
# 점선은 빈 구간을 넘어 다음 조각을 잡아야 하므로 실선보다 넓게 탐색한다.
DASHED_WINDOW_MARGIN = 150
WINDOW_MIN_COMPONENT_PIXELS = 30
# offset에 쓰는 차선 x는 "차 바로 앞(ROI 바닥에서 이 픽셀 수 이내)"의 점만으로
# 계산한다. 슬라이딩 윈도우 추적 자체는 ROI 전체(먼 곳 포함)를 쓰지만, 커브에서
# 먼 점선이 직선 피팅 기울기를 당겨 바닥 x를 왜곡하는 것을 막기 위해, 최종 x는
# 근접 밴드로 제한한다. 근접 점이 부족하면(점선 공백) 전체 수집점으로 폴백한다.
OFFSET_NEAR_ROWS = 45
# 근접 밴드의 "아래쪽" 컷오프(px, ROI 바닥 기준). ROI 맨 아래(차 후드 바로 앞)는
# 차선이 화면 폭을 크게 차지해 흰색이 뭉개져 다 차선처럼 잡히거나 각도가 왜곡된다.
# 그래서 offset x 계산 밴드를 [바닥에서 offset_near_rows] ~ [바닥에서
# offset_near_bottom_rows] 사이의 창으로 좁혀, 맨 아래 이 픽셀 수만큼은 제외한다.
# 0이면 기존처럼 바닥까지 사용. offset_near_rows 보다 작아야 한다(창이 비지 않게).
OFFSET_NEAR_BOTTOM_ROWS = 20
# 브리지 래치: 점선이 2개 이상 보이는(커버리지 충분) 프레임에서만 브리지 곡선을
# 새로 만들고, 점선 1개/gap 프레임에서는 저장된 곡선을 그대로 재사용한다. 가장
# 가까운 점선이 사라졌을 때 먼 점 하나로 억지 재피팅해 곡선이 깨지는 걸 막는다.
# 저장된 브리지를 갱신 없이 이 프레임 수 넘게 재사용하면(오래되면) 버린다.
DASHED_BRIDGE_MAX_AGE = 12
# 중앙 점선과 1/2차선 실선의 하단 x가 이 거리보다 가까우면 같은 선을 추적한
# 것으로 판단한다. 이 프레임은 무효 처리하고 이전 offset을 유지한다.
CENTER_LINE_OVERLAP_DISTANCE_PX = 45
# 이전 프레임의 검출 위치를 다음 프레임 윈도우 시작점에 반영하는 비율.
# 0.20이면 한 프레임에 차이의 20%만 움직여 급격한 점프를 막는다.
WINDOW_START_ADAPT_RATE = 0.5
# 새 검출 위치가 이전 박스 시작점에서 이 거리보다 크게 튀면 오검출로 보고
# 시작점을 갱신하지 않는다.
MAX_WINDOW_START_JUMP_PX = 130
# 중앙 점선 탐색 앵커가 기준선(dashed_reference_x_px)에서 벗어날 수 있는 최대
# 거리(px). 점선은 "148로 되돌릴 대상"이라 탐색 앵커가 기준 근처에 머물러야 한다.
# 이 값이 너무 크면 커브에서 앵커가 왼쪽 실선/잡음 쪽으로 자유 표류해, closest-
# component 규칙이 점선 대신 그 잡음을 계속 물어 dashed_x 가 기준보다 왼쪽에
# 고착(=좌조향 지속)된다. 0에 가까우면 항상 기준선에서만 탐색(=완전 고정).
# margin(dashed_window_margin=150)보다 충분히 작게 잡아 표류를 억제한다.
DASHED_ANCHOR_MAX_DRIFT_PX =15
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
DEBUG_VIEW = False
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
        self.declare_parameter('curve_ref_gain', CURVE_REF_GAIN)
        self.declare_parameter('curve_ref_lookahead_dy', CURVE_REF_LOOKAHEAD_DY)
        self.declare_parameter('max_offset_jump_px', MAX_OFFSET_JUMP_PX)
        self.declare_parameter('offset_smoothing_alpha', OFFSET_SMOOTHING_ALPHA)
        self.declare_parameter('right_guard_enable', RIGHT_GUARD_ENABLE)
        self.declare_parameter('right_guard_enter_x', RIGHT_GUARD_ENTER_X)
        self.declare_parameter('right_guard_exit_x', RIGHT_GUARD_EXIT_X)
        self.declare_parameter('right_guard_target_x', RIGHT_GUARD_TARGET_X)
        self.declare_parameter(
            'right_guard_error_limit_px', RIGHT_GUARD_ERROR_LIMIT_PX
        )
        self.declare_parameter('right_guard_kp', RIGHT_GUARD_KP)
        self.declare_parameter('right_guard_min_valid_x', RIGHT_GUARD_MIN_VALID_X)
        self.declare_parameter('right_guard_lookahead_rows', RIGHT_GUARD_LOOKAHEAD_ROWS)
        self.declare_parameter('right_guard_full_steer', RIGHT_GUARD_FULL_STEER)
        self.declare_parameter('right_avoid_start_x', RIGHT_AVOID_START_X)
        self.declare_parameter('right_avoid_full_x', RIGHT_AVOID_FULL_X)
        self.declare_parameter('num_windows', NUM_WINDOWS)
        self.declare_parameter('window_margin', WINDOW_MARGIN)
        self.declare_parameter('window_minpix', WINDOW_MINPIX)
        self.declare_parameter('dashed_window_margin', DASHED_WINDOW_MARGIN)
        self.declare_parameter('offset_near_rows', OFFSET_NEAR_ROWS)
        self.declare_parameter('offset_near_bottom_rows', OFFSET_NEAR_BOTTOM_ROWS)
        self.declare_parameter('dashed_bridge_max_age', DASHED_BRIDGE_MAX_AGE)
        self.declare_parameter(
            'center_line_overlap_distance_px', CENTER_LINE_OVERLAP_DISTANCE_PX
        )
        self.declare_parameter('window_min_component_pixels', WINDOW_MIN_COMPONENT_PIXELS)
        self.declare_parameter('window_start_adapt_rate', WINDOW_START_ADAPT_RATE)
        self.declare_parameter('max_window_start_jump_px', MAX_WINDOW_START_JUMP_PX)
        self.declare_parameter('dashed_anchor_max_drift_px', DASHED_ANCHOR_MAX_DRIFT_PX)
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
        # 선언한 파라미터를 self.* 로 적용. 아래 _load_parameters 는 파라미터 변경
        # 콜백(_on_set_parameters)과 공유되어, ros2 param set 으로 바꾼 값이 노드
        # 재시작/재빌드 없이 바로 반영되게 한다.
        self._load_parameters(lambda name: self.get_parameter(name).value)
        # 보호모드 on/off 는 파라미터가 아니라 런타임 상태다(파라미터 콜백에서
        # 리셋하면 안 되므로 여기서만 초기화한다).
        self.right_guard_active = False
        # soft 오른쪽 실선 회피 가중치(0~1). 발행 offset 블렌딩에 쓰인다.
        self.right_avoid_w = 0.0
        # 디버그 표시용: 오른쪽 실선의 near x / 룩어헤드 x / 실제 가드 입력 x(min).
        self.dbg_right_near_x = None
        self.dbg_right_lookahead_x = None
        self.dbg_right_guard_x = None
        self.dbg_published_offset = 0
        # 중앙 점선 피팅 곡선(가상의 연속 점선): (coeffs, y_lo, y_hi, mode).
        self.dbg_dashed_curve = None
        # 브리지 래치 상태: 저장된 곡선 (coeffs, y_lo, y_hi) 와 갱신 후 경과 프레임.
        self.dashed_bridge = None
        self.dashed_bridge_age = 0
        # 곡률 적응형 기준선용 점선 휨(bend). 매 프레임 트래커가 갱신.
        self.dashed_curve_bend = 0.0
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

        # ros2 param set 으로 바뀐 값을 재시작 없이 즉시 반영한다.
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.get_logger().info(
            f'Subscribing {self.image_topic}, publishing {self.lane_offset_topic}, '
            f'driving_mode={self.driving_mode}, '
            f'dashed_reference_x_px={self.dashed_reference_x_px}, '
            f'offset_kp={self.offset_kp:.2f}, '
            f'debug_view={self.debug_view}, '
            f'offset range=+/-{self.lane_offset_limit}, '
            f'right_guard={"on" if self.right_guard_enable else "off"} '
            f'(enter<={int(self.right_guard_enter_x)}, '
            f'exit>={int(self.right_guard_exit_x)}, '
            f'target={int(self.right_guard_target_x)})'
        )

    # ======================================================================
    # 파라미터 로드 / 런타임 갱신
    # ======================================================================
    def _load_parameters(self, get):
        """튜닝 파라미터를 self.* 에 적용한다. get(name) 이 값을 돌려준다.

        __init__ 과 파라미터 변경 콜백(_on_set_parameters)이 이 메서드를 공유하므로,
        `ros2 param set` 으로 바꾼 값이 노드 재시작/재빌드 없이 바로 반영된다.
        스칼라 튜닝값만 다루며, 토픽/퍼블리셔/런타임 상태는 여기서 건드리지 않는다.
        """
        self.roi_top = int(get('roi_top'))
        self.roi_bottom = int(get('roi_bottom'))
        self.roi_trapezoid_top_inset_px = int(get('roi_trapezoid_top_inset_px'))
        self.roi_trapezoid_bottom_inset_px = int(get('roi_trapezoid_bottom_inset_px'))
        self.white_s_max = int(get('white_s_max'))
        self.white_v_min = int(get('white_v_min'))
        self.green_h_min = int(get('green_h_min'))
        self.green_h_max = int(get('green_h_max'))
        self.green_s_min = int(get('green_s_min'))
        self.green_v_min = int(get('green_v_min'))
        self.gray_s_min = int(get('gray_s_min'))
        self.gray_v_min = int(get('gray_v_min'))
        self.near_field_rows = int(get('near_field_rows'))
        self.white_overload_ratio = float(get('white_overload_ratio'))
        self.driving_mode = str(get('driving_mode')).lower()
        if self.driving_mode not in ('1lane', '2lane'):
            self.get_logger().warn(
                f"Unknown driving_mode='{self.driving_mode}'; using '2lane'"
            )
            self.driving_mode = '2lane'
        self.dashed_reference_x_px_2lane = int(get('dashed_reference_x_px_2lane'))
        self.dashed_reference_x_px_1lane = int(get('dashed_reference_x_px_1lane'))
        self.dashed_reference_x_px = (
            self.dashed_reference_x_px_2lane
            if self.driving_mode == '2lane'
            else self.dashed_reference_x_px_1lane
        )
        self.offset_error_limit_px = max(1, int(get('offset_error_limit_px')))
        self.lane_offset_limit = max(1, int(get('lane_offset_limit')))
        self.offset_kp = max(0.0, float(get('offset_kp')))
        self.curve_ref_gain = float(get('curve_ref_gain'))
        self.curve_ref_lookahead_dy = max(1, int(get('curve_ref_lookahead_dy')))
        self.max_offset_jump_px = int(get('max_offset_jump_px'))
        self.offset_smoothing_alpha = float(np.clip(
            get('offset_smoothing_alpha'), 0.0, 1.0
        ))
        self.right_guard_enable = bool(get('right_guard_enable'))
        self.right_guard_enter_x = float(get('right_guard_enter_x'))
        self.right_guard_exit_x = float(get('right_guard_exit_x'))
        # 히스테리시스가 성립하려면 진입 임계값 < 해제 임계값이어야 한다.
        if self.right_guard_exit_x <= self.right_guard_enter_x:
            self.get_logger().warn(
                f'right_guard_exit_x({self.right_guard_exit_x:.0f}) must be > '
                f'enter_x({self.right_guard_enter_x:.0f}); '
                'forcing exit_x = enter_x + 50'
            )
            self.right_guard_exit_x = self.right_guard_enter_x + 50.0
        self.right_guard_target_x = float(get('right_guard_target_x'))
        self.right_guard_error_limit_px = max(
            1.0, float(get('right_guard_error_limit_px'))
        )
        self.right_guard_kp = max(0.0, float(get('right_guard_kp')))
        self.right_guard_min_valid_x = float(get('right_guard_min_valid_x'))
        self.right_guard_lookahead_rows = max(
            1, int(get('right_guard_lookahead_rows'))
        )
        self.right_guard_full_steer = bool(get('right_guard_full_steer'))
        self.right_avoid_start_x = float(get('right_avoid_start_x'))
        self.right_avoid_full_x = float(get('right_avoid_full_x'))
        self.num_windows = int(get('num_windows'))
        self.window_margin = int(get('window_margin'))
        self.window_minpix = int(get('window_minpix'))
        self.dashed_window_margin = int(get('dashed_window_margin'))
        self.offset_near_rows = max(1, int(get('offset_near_rows')))
        # 아래쪽 컷오프는 창이 비지 않도록 [0, offset_near_rows-1] 로 제한한다.
        self.offset_near_bottom_rows = int(np.clip(
            get('offset_near_bottom_rows'), 0, self.offset_near_rows - 1
        ))
        self.dashed_bridge_max_age = max(0, int(get('dashed_bridge_max_age')))
        self.center_line_overlap_distance_px = max(
            0, int(get('center_line_overlap_distance_px'))
        )
        self.window_min_component_pixels = int(get('window_min_component_pixels'))
        self.window_start_adapt_rate = float(np.clip(
            get('window_start_adapt_rate'), 0.0, 1.0
        ))
        self.max_window_start_jump_px = max(0, int(get('max_window_start_jump_px')))
        self.dashed_anchor_max_drift_px = max(
            0.0, float(get('dashed_anchor_max_drift_px'))
        )
        self.boundary_color_near_distance_px = int(
            get('boundary_color_near_distance_px')
        )
        self.boundary_color_min_pixels = int(get('boundary_color_min_pixels'))
        self.require_boundary_line = bool(get('require_boundary_line'))
        self.near_field_full_height_ratio = float(get('near_field_full_height_ratio'))
        self.min_line_aspect_ratio = float(get('min_line_aspect_ratio'))
        self.min_line_height_px = int(get('min_line_height_px'))
        self.debug_view = bool(get('debug_view'))

    def _on_set_parameters(self, params):
        """`ros2 param set` 콜백: 바뀐 값으로 튜닝값을 즉시 다시 적용한다.

        아직 적용 전인 새 값들(params)을 우선 사용하고, 나머지는 기존 파라미터
        저장소에서 읽어 _load_parameters 를 통째로 다시 돌린다. 잘못된 값이면
        거부(successful=False)만 하고 노드는 계속 유지한다.
        """
        incoming = {p.name: p.value for p in params}

        def get(name):
            return incoming[name] if name in incoming else self.get_parameter(name).value

        try:
            self._load_parameters(get)
        except Exception as exc:  # noqa: BLE001 - 잘못된 값은 거부만, 노드는 유지
            return SetParametersResult(successful=False, reason=str(exc))
        self.get_logger().info(
            'Params updated live: '
            + ', '.join(f'{p.name}={p.value}' for p in params)
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
        # 모드별로 화면 밖으로 나가는 외곽 실선(2차로=왼쪽, 1차로=오른쪽)은 아예
        # 추적하지 않는다. 존재하지 않는 선을 억지로 좇다가 반대편 실선을 물어
        # left_x>=dashed_x 같은 거짓 INVALID LANE ORDER를 만드는 것을 막는다.
        track_left = self.driving_mode != '2lane'
        track_right = self.driving_mode != '1lane'
        left_start_x = self.get_window_start_x('left', left_base_x)
        dashed_start_x = self.get_window_start_x(
            'dashed', self.dashed_reference_x_px
        )
        right_start_x = self.get_window_start_x('right', right_base_x)
        left_track = (
            self.track_lane_with_sliding_window(
                white_mask, left_start_x, self.window_margin, allow_gaps=False
            )
            if track_left and left_start_x is not None
            else (None, [], [], [])
        )
        dashed_track = self.track_lane_with_sliding_window(
            white_mask, dashed_start_x, self.dashed_window_margin,
            allow_gaps=True, near_only=True,
        )
        right_track = (
            self.track_lane_with_sliding_window(
                white_mask, right_start_x, self.window_margin, allow_gaps=False
            )
            if track_right and right_start_x is not None
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

        # ------------------------------------------------------------------
        # 오른쪽 차선 이탈 보호(히스테리시스). 오른쪽 실선의 near-field x가
        # enter_x 이하로 내려가면(차가 우측으로 치우침) 중앙선 기준을 잠시 버리고
        # 오른쪽 실선을 다시 안쪽으로 되돌리는 좌조향을 최우선으로 낸다. exit_x
        # 이상으로 회복되면 해제한다. 오른쪽 실선이 안 보이거나(커브에서 트래커가
        # 픽셀을 놓침) 신뢰할 수 없이 작은 x면 강제로 켜지 않고 중앙선 모드로
        # 폴백한다.
        # 가드 입력 x: near x(차 앞)와 룩어헤드 밴드의 오른쪽 실선 median 중 더
        # 안쪽(작은) 값. 좌커브에서 위쪽 실선이 안쪽으로 굽어 들어오면 near x 가
        # 아직 임계 위여도 룩어헤드 값이 먼저 내려가 가드가 일찍 켜진다. 진입/조향
        # 판단에만 쓰고, 아래 lane order/overlap 검증은 계속 near right_x 를 쓴다.
        right_lookahead_x = self.compute_right_guard_x(
            _right_points_x, _right_points_y, white_mask.shape[0]
        )
        if right_x is None:
            right_guard_x = right_lookahead_x
        elif right_lookahead_x is None:
            right_guard_x = right_x
        else:
            right_guard_x = min(right_x, right_lookahead_x)
        # 디버그 창 표시용으로 세 값을 보관한다.
        self.dbg_right_near_x = right_x
        self.dbg_right_lookahead_x = right_lookahead_x
        self.dbg_right_guard_x = right_guard_x
        # 오른쪽 실선 근접 가중치(soft guard). binary 진입/해제(무조건 -45) 대신,
        # guard_x가 start_x 이하로 내려올수록 0->1로 커지는 연속 가중치를 만든다.
        # 발행 단계(publish_offset)에서 중앙선 offset과 -45를 이 w로 블렌딩하므로,
        # 평소(오른쪽 실선이 멀면 w=0)에는 순수 중앙선 추종이고, 오른쪽 실선이
        # 다가올수록(w->1) 좌조향이 매끄럽게 섞인다. 유효성은 near x로 판단해
        # 룩어헤드 이상치가 잘못 개입시키지 않게 하고, w는 EMA로 완만히 바꾼다.
        right_valid = (
            track_right
            and right_x is not None
            and right_x >= self.right_guard_min_valid_x
        )
        target_w = self.right_avoid_weight(right_guard_x) if right_valid else 0.0
        self.right_avoid_w = (
            self.offset_smoothing_alpha * target_w
            + (1.0 - self.offset_smoothing_alpha) * self.right_avoid_w
        )

        # 차 바로 앞에 중앙 점선 조각이 없으면(near_only gap) dashed_x가 None이다.
        # 이때는 슬라이딩 윈도우의 먼 점 외삽으로 조향하지 않고 직전 offset을
        # 유지한다("점선을 148로 맞추기"가 주 목표, 윈도우는 보조). 앵커는 계속
        # 갱신해 다음 프레임 탐색창 위치를 유지한다.
        if dashed_x is None:
            self.get_logger().info(
                'No near center-dash (gap); holding last offset',
                throttle_duration_sec=1.0,
            )
            self.update_tracker_anchors(left_x, dashed_x, right_x, white_mask.shape[1])
            self.last_lane_tracks = lane_tracks
            self.publish_offset(self.last_offset)
            self.publish_debug(
                msg, frame, 'NO NEAR DASH', near_white_ratio, lane_tracks=lane_tracks,
            )
            return

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
            # offset 발행은 보류하되, 앵커는 실제 관측치를 계속 따라가게 해서
            # 급커브 등에서 다음 프레임에 스스로 회복할 수 있게 한다.
            self.update_tracker_anchors(left_x, dashed_x, right_x, white_mask.shape[1])
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
            # offset 발행은 보류하되, 앵커는 실제 관측치를 계속 따라가게 해서
            # 급커브 등에서 다음 프레임에 스스로 회복할 수 있게 한다.
            self.update_tracker_anchors(left_x, dashed_x, right_x, white_mask.shape[1])
            self.publish_offset(self.last_offset)
            self.publish_debug(
                msg, frame, 'JUMP REJECTED', near_white_ratio,
                base_x=dashed_start_x, line_x=line_x, windows=windows,
                points_x=points_x, points_y=points_y, lane_offset=lane_offset,
                lane_tracks=lane_tracks,
            )
            return

        # 검출 노이즈로 매 프레임 값이 튀는 걸 줄이기 위해 저역통과(EMA) 필터를
        # 적용해서 발행한다. 급격한 오검출은 위 max_offset_jump_px 체크가 이미
        # 걸러내므로, 여기서는 정상 범위 내의 미세한 흔들림만 완만하게 만든다.
        self.last_offset = (
            self.offset_smoothing_alpha * lane_offset
            + (1.0 - self.offset_smoothing_alpha) * self.last_offset
        )
        self.update_tracker_anchors(left_x, dashed_x, right_x, white_mask.shape[1])
        self.last_lane_tracks = lane_tracks
        self.publish_offset(self.last_offset)
        self.publish_debug(
            msg, frame, 'OK', near_white_ratio,
            base_x=dashed_start_x, line_x=line_x, windows=windows,
            points_x=points_x, points_y=points_y, lane_offset=lane_offset,
            lane_tracks=lane_tracks,
        )

    def publish_offset(self, value):
        # soft 오른쪽 실선 회피: 중앙선 offset(value)과 최대 좌조향(-45)을 근접
        # 가중치 w로 블렌딩한다. w=0이면 순수 중앙선, w=1이면 완전 좌조향.
        w = float(np.clip(self.right_avoid_w, 0.0, 1.0))
        blended = (1.0 - w) * float(value) + w * (-float(self.lane_offset_limit))
        self.dbg_published_offset = int(np.clip(
            blended, -self.lane_offset_limit, self.lane_offset_limit
        ))
        msg = Int16()
        msg.data = self.dbg_published_offset
        self.offset_pub.publish(msg)

    def right_avoid_weight(self, guard_x):
        """오른쪽 실선 근접 가중치 w(0~1)를 구한다.

        guard_x(= near/look-ahead min x)가 start_x 이상이면 0(평소, 순수 중앙선
        추종), full_x 이하면 1(완전 좌조향), 그 사이는 선형 램프. 트래커 붕괴로
        화면 왼쪽 끝에 붙은 신뢰 못 할 작은 x(min_valid_x 미만)는 0으로 무시한다.
        """
        if not self.right_guard_enable or guard_x is None:
            return 0.0
        if guard_x < self.right_guard_min_valid_x:
            return 0.0
        start = float(self.right_avoid_start_x)
        full = float(self.right_avoid_full_x)
        if start <= full:
            return 1.0 if guard_x <= full else 0.0
        return float(np.clip((start - guard_x) / (start - full), 0.0, 1.0))

    def map_lane_x_to_offset(self, detected_lane_x):
        """점선 오차에 Kp를 적용해 -45~45 offset으로 매핑한다.

        곡률 적응형 기준선: 고정 dashed_ref 대신, 피팅 곡선의 휨(bend)만큼 목표를
        커브 바깥쪽으로 옮긴다. 좌커브(bend>0)면 기준선을 낮춰 차가 더 오른쪽에,
        우커브(bend<0)면 높여 더 왼쪽에 서게 해 안쪽 선에서 여유를 둔다.
        curve_ref_gain=0 이면 기존 고정 기준선.
        """
        effective_ref = (
            self.dashed_reference_x_px + self.curve_ref_gain * self.dashed_curve_bend
        )
        error_px = float(detected_lane_x) - effective_ref
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
    # 오른쪽 차선 이탈 보호
    # ======================================================================
    def update_right_guard_state(self, right_valid, right_x):
        """히스테리시스로 보호모드 on/off 상태를 갱신한다.

        - 비활성 -> 활성: right_x <= enter_x (오른쪽으로 치우쳐 우측 실선이
          화면 중앙 쪽으로 다가옴)
        - 활성 -> 비활성: right_x >= exit_x (충분히 복귀) — enter<exit 로 진입/해제
          임계값을 벌려 경계 근처에서 껐다 켰다 하는 채터링을 막는다.
        - 오른쪽 실선이 안 보이거나(right_valid=False) 신뢰할 수 없으면 강제로
          켤 수 없으므로 보호모드를 끄고 중앙선 모드로 폴백한다.
        """
        if not self.right_guard_enable:
            self.right_guard_active = False
            return
        if not right_valid:
            if self.right_guard_active:
                self.get_logger().info(
                    'Right guard OFF (right line lost/untrusted) -> center mode',
                    throttle_duration_sec=1.0,
                )
            self.right_guard_active = False
            return
        if self.right_guard_active:
            if right_x >= self.right_guard_exit_x:
                self.right_guard_active = False
                self.get_logger().info(
                    f'Right guard OFF (right_x={right_x:.0f} >= '
                    f'exit_x={self.right_guard_exit_x:.0f})'
                )
        else:
            if right_x <= self.right_guard_enter_x:
                self.right_guard_active = True
                self.get_logger().warn(
                    f'Right guard ON (right_x={right_x:.0f} <= '
                    f'enter_x={self.right_guard_enter_x:.0f}); '
                    'steering to push right line back inward'
                )

    def map_right_x_to_offset(self, right_x):
        """보호모드: 오른쪽 실선 x를 목표(target_x)로 되돌리는 offset을 만든다.

        right_x < target_x(우측 치우침)이면 error<0 -> offset<0 -> 좌조향으로,
        중앙선 매핑과 동일한 부호 규약을 그대로 따른다. 목표에 도달하면 0.
        """
        error_px = float(right_x) - self.right_guard_target_x
        normalized = np.clip(
            error_px / self.right_guard_error_limit_px,
            -1.0,
            1.0,
        )
        scaled_offset = normalized * self.lane_offset_limit * self.right_guard_kp
        return int(round(np.clip(
            scaled_offset, -self.lane_offset_limit, self.lane_offset_limit
        )))

    def compute_right_guard_x(self, points_x, points_y, height):
        """오른쪽 이탈 보호가 볼 오른쪽 실선 x를 룩어헤드 밴드에서 구한다.

        near-field(차 앞) x만 보면, 좌커브에서 오른쪽 실선이 위쪽에서 안쪽으로
        굽어 들어와도 차 앞 x는 아직 enter_x 위라 가드가 안 켜진다. 그래서 ROI
        바닥에서 right_guard_lookahead_rows 높이까지의 오른쪽 실선 점들 x의
        median을 함께 본다. 밴드를 넓힐수록 굽은 위쪽 점이 섞여 값이 작아져
        가드가 더 일찍 켜진다. 점이 부족하면 None(호출부가 near x로 폴백).

        points_x/points_y는 오른쪽 트래커가 수집한 실선 픽셀(ROI 로컬 y, 프레임
        x)이라, 별도 마스킹 없이 이 점들만으로 실선 위치를 구할 수 있다.
        """
        if not points_x:
            return None
        px = np.asarray(points_x, dtype=float)
        py = np.asarray(points_y, dtype=float)
        band = py >= (height - self.right_guard_lookahead_rows)
        if int(band.sum()) < 8:
            return None
        return float(np.median(px[band]))

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

    def update_tracker_anchors(self, left_x, dashed_x, right_x, image_width):
        """세 차선의 탐색 앵커를 갱신한다.

        이번 프레임의 offset을 신뢰해서 발행할지와, 다음 프레임 탐색창을 어디서
        열지는 별개다. INVALID LANE ORDER나 JUMP REJECTED로 offset 발행은
        보류하더라도 앵커는 계속 실제 관측치를 따라가야, 급커브처럼 화면상
        차선이 빠르게 움직이는 구간에서 앵커가 옛 위치에 멈춰 계속 같은
        실패를 반복하는 걸 막을 수 있다. (MAX_WINDOW_START_JUMP_PX/
        WINDOW_START_ADAPT_RATE 제한은 그대로 적용되어 여전히 완만하게만 움직인다.)
        """
        self.update_window_start_x('left', left_x, image_width)
        self.update_window_start_x('dashed', dashed_x, image_width)
        self.update_window_start_x('right', right_x, image_width)

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
        # 중앙 점선 탐색 앵커는 기준선(dashed_reference_x_px)에서 dashed_anchor_
        # max_drift_px 이상 벗어나지 못하게 묶는다. 앵커가 왼쪽 실선/잡음 쪽으로
        # 자유 표류하면 closest-component 규칙이 점선 대신 그 잡음을 계속 물어
        # dashed_x 가 기준보다 왼쪽에 고착(→ 좌조향 지속)되기 때문이다. 슬라이딩
        # 추적은 여전히 창을 따라 커브 위쪽을 그리지만(보조 역할), 다음 프레임의
        # 탐색 시작점만은 항상 148 근처로 되돌아온다.
        if lane_name == 'dashed':
            ref = float(self.dashed_reference_x_px)
            next_x = float(np.clip(
                next_x,
                ref - self.dashed_anchor_max_drift_px,
                ref + self.dashed_anchor_max_drift_px,
            ))
        self.window_start_x[lane_name] = float(
            np.clip(next_x, 0, max(0, image_width - 1))
        )

    def _count_dash_segments(self, ys):
        """수집된 점선 점들의 y를 훑어 서로 떨어진 조각(=보이는 점선 개수)을 센다.

        인접한 행들이 8px보다 크게 벌어지면 다른 점선 조각으로 본다. 브리지를
        "점선 2개 이상 보일 때만" 갱신할지 판단하는 데 쓴다.
        """
        rows = sorted(set(int(v) for v in ys.tolist()))
        if not rows:
            return 0
        segments = 1
        for a, b in zip(rows, rows[1:]):
            if b - a > 8:
                segments += 1
        return segments

    def track_lane_with_sliding_window(
        self, white_mask, base_x, margin, allow_gaps, near_only=False
    ):
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
            if near_only:
                self.dbg_dashed_curve = None
                self.dashed_bridge_age += 1  # 완전 소실 -> 저장 브리지도 나이 먹음
            return None, windows, collected_x, collected_y
        # offset에 쓰는 x는 근접 밴드(바닥에서 offset_near_rows 이내)의 점만으로
        # 계산해 먼 커브가 바닥 x를 당기는 것을 막는다. 근접 점이 부족하면(점선
        # 공백 등) 전체 수집점으로 폴백한다. 추적/디버그용 collected_*는 그대로 반환.
        cy = np.asarray(collected_y, dtype=float)
        cx = np.asarray(collected_x, dtype=float)
        # 근접 밴드: [바닥에서 offset_near_rows] ~ [바닥에서 offset_near_bottom_rows]
        # 사이의 창. 위쪽(먼 커브)뿐 아니라 맨 아래(차 후드 앞, 흰색이 뭉개지는
        # 구간)도 잘라내 offset x가 왜곡되지 않게 한다.
        near_top = height - self.offset_near_rows
        near_bottom_excl = height - self.offset_near_bottom_rows  # 이 행부터는 제외
        near = (cy >= near_top) & (cy < near_bottom_excl)
        eval_y = float(near_bottom_excl - 1)  # 신뢰 구간 맨 아래(차에 가장 가까운 행)
        # 디버그 시각화용 점선 피팅 곡선: (coeffs, y_lo, y_hi, mode). near_only(중앙
        # 점선)일 때만 채우고, publish_debug 가 이걸 곡선으로 그린다.
        fit_curve = None

        # 브리지 래치: 점선이 2개 이상 보이고(커버리지 충분) y범위가 넓을 때만
        # 곡선을 새로 만들어 저장한다. 그렇지 않은 프레임(점선 1개/gap)에서는
        # 갱신하지 않고 나이만 먹인다. -> 가까운 점선이 사라져도 남은 먼 점 하나로
        # 억지 재피팅하지 않고, 마지막으로 잘 만든 곡선을 유지한다.
        usable = cy < near_bottom_excl
        uy, ux = cy[usable], cx[usable]
        bridge_fresh = False
        if near_only:
            if (
                uy.size >= 15
                and self._count_dash_segments(uy) >= 2
                and float(uy.max() - uy.min()) >= 40
            ):
                self.dashed_bridge = (np.polyfit(uy, ux, 2), float(uy.min()), eval_y)
                self.dashed_bridge_age = 0
                bridge_fresh = True
            elif self.dashed_bridge is not None:
                self.dashed_bridge_age += 1

        if int(near.sum()) >= 8:
            # 가장 가까운 점선이 근접 밴드에 보임 = 가장 정확. 근접 점만 1차
            # 피팅해 신뢰 구간 맨 아래(eval_y)에서 평가한다. 행이 부족하면 median.
            ny, nx = cy[near], cx[near]
            if len(set(ny.tolist())) >= 3:
                coeffs = np.polyfit(ny, nx, 1)
                line_x = float(np.polyval(coeffs, eval_y))
                fit_curve = (coeffs, float(ny.min()), eval_y, 'near')
            else:
                line_x = float(np.median(nx))
        elif (
            near_only
            and self.dashed_bridge is not None
            and self.dashed_bridge_age <= self.dashed_bridge_max_age
        ):
            # 근접 gap: 저장된 브리지 곡선으로 잇는다. 이번 프레임에 새로 만들었으면
            # 'bridge'(주황), 예전 곡선을 유지 중이면 'hold'(빨강). 점선 1개여도 유지.
            coeffs, y_lo, y_hi = self.dashed_bridge
            line_x = float(np.polyval(coeffs, eval_y))
            fit_curve = (coeffs, y_lo, y_hi, 'bridge' if bridge_fresh else 'hold')
        elif near_only:
            # 저장된 브리지도 없거나 너무 오래됨 + 커버리지도 부족 -> 최소 폴백.
            # 점/행이 조금이라도 되면 1차로 잇고, 아니면 None(→ 콜백이 hold).
            if uy.size >= 12 and len(set(uy.tolist())) >= 3:
                coeffs = np.polyfit(uy, ux, 1)
                line_x = float(np.polyval(coeffs, eval_y))
                fit_curve = (coeffs, float(uy.min()), eval_y, 'bridge')
            else:
                line_x = None
        elif len(set(collected_y)) >= 3 and len(collected_x) >= 12:
            # 근접 점이 부족(점선 공백 등)하면 전체 수집점 직선 피팅으로 폴백.
            coeffs = np.polyfit(cy, cx, 1)
            line_x = float(np.polyval(coeffs, height - 1))
        else:
            line_x = float(cx.mean())
        # 중앙 점선 트래커일 때만 피팅 곡선을 디버그용으로 저장하고, 곡률 적응형
        # 기준선에 쓸 휨(bend)을 계산한다. bend = near x - (위로 lookahead_dy 만큼의
        # x). 좌커브면 위로 갈수록 왼쪽이라 bend>0, 우커브면 bend<0. 피팅 없으면 0.
        if near_only:
            self.dbg_dashed_curve = fit_curve
            if fit_curve is not None:
                c = fit_curve[0]
                self.dashed_curve_bend = float(
                    np.polyval(c, eval_y)
                    - np.polyval(c, eval_y - self.curve_ref_lookahead_dy)
                )
            else:
                self.dashed_curve_bend = 0.0
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
        # offset 계산에 실제로 쓰는 근접 밴드(창): 위=바닥에서 offset_near_rows,
        # 아래=바닥에서 offset_near_bottom_rows. 반투명 초록으로 칠해 "이 띠의 점만
        # offset x로 쓴다"를 눈으로 확인. 맨 아래 컷오프 아래는 칠하지 않는다.
        near_top_y = int(np.clip(self.roi_bottom - self.offset_near_rows, 0, height - 1))
        near_bottom_y = int(np.clip(
            self.roi_bottom - self.offset_near_bottom_rows, 0, height - 1
        ))
        overlay = debug.copy()
        cv2.rectangle(
            overlay, (0, near_top_y), (width - 1, near_bottom_y), (0, 220, 0), -1
        )
        cv2.addWeighted(overlay, 0.3, debug, 0.7, 0, debug)
        cv2.line(debug, (0, near_top_y), (width - 1, near_top_y), (0, 220, 0), 1)
        cv2.line(debug, (0, near_bottom_y), (width - 1, near_bottom_y), (0, 220, 0), 1)
        cv2.putText(
            debug,
            f'offset_near {self.offset_near_rows}..{self.offset_near_bottom_rows}',
            (width - 260, near_top_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA,
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
        # Soft 회피 임계선: start_x(자홍, 여기부터 좌조향 섞이기 시작) /
        # full_x(청록, 여기 도달하면 w=1 완전 좌조향). guard_x(보라 채운 원)가
        # start_x 왼쪽으로 오면 w>0, full_x 왼쪽이면 w=1.
        if self.right_guard_enable:
            self.draw_dashed_vline(
                debug, int(self.right_avoid_start_x),
                self.roi_top, self.roi_bottom, (255, 0, 255)
            )
            self.draw_dashed_vline(
                debug, int(self.right_avoid_full_x),
                self.roi_top, self.roi_bottom, (255, 255, 0)
            )
            # 가드 룩어헤드: 오른쪽 실선을 이 높이(보라 가로선)까지 보고, 그 밴드
            # 안 median 을 look-ahead x(보라 원+세로선)로 쓴다. 실제 가드 입력은
            # near x 와의 min(=보라 채운 원, ROI 바닥). near/la/guard 수치는 좌상단
            # 텍스트에도 표시.
            la_purple = (255, 0, 128)
            la_top_y = int(np.clip(
                self.roi_bottom - self.right_guard_lookahead_rows, 0, height - 1
            ))
            cv2.line(debug, (0, la_top_y), (width - 1, la_top_y), la_purple, 1)
            cv2.putText(
                debug, 'guard lookahead top', (8, la_top_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, la_purple, 1, cv2.LINE_AA,
            )
            if self.dbg_right_lookahead_x is not None:
                lax = int(round(self.dbg_right_lookahead_x))
                cv2.circle(debug, (lax, la_top_y), 6, la_purple, 2)
                self.draw_dashed_vline(
                    debug, lax, la_top_y, self.roi_bottom, la_purple
                )
            if self.dbg_right_guard_x is not None:
                cv2.circle(
                    debug, (int(round(self.dbg_right_guard_x)), roi_bottom_y),
                    7, la_purple, -1,
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

        # 중앙 점선 피팅 곡선(가상의 연속 점선)을 그린다. 여러 점선 조각에 피팅한
        # 곡선을 신뢰 구간 위쪽부터 near 지점까지 이어 그려, gap을 어떻게 잇는지
        # 눈으로 확인한다. gap을 곡률(2차)로 잇는 브리지 모드는 주황 굵게 + 'DASH
        # BRIDGE' 라벨, 근접 밴드가 충분한 평소(near 1차)는 흰색.
        if self.dbg_dashed_curve is not None:
            coeffs, y_lo, y_hi, mode = self.dbg_dashed_curve
            ys = np.linspace(float(y_lo), float(y_hi), 24)
            xs = np.polyval(coeffs, ys)
            pts = []
            for cxv, cyv in zip(xs, ys):
                px = int(round(float(cxv)))
                py = int(round(float(cyv))) + self.roi_top
                if 0 <= px < width and roi_top_y <= py <= roi_bottom_y:
                    pts.append((px, py))
            if len(pts) >= 2:
                # near=흰색(근접 1차), bridge=주황(2개 보여 새로 만든 곡선),
                # hold=빨강(점선 1개/gap이라 저장된 곡선을 유지 중).
                curve_style = {
                    'near': ((255, 255, 255), 2, None),
                    'bridge': ((0, 165, 255), 3, 'DASH BRIDGE'),
                    'hold': ((0, 0, 255), 3, 'DASH HOLD'),
                }
                curve_color, thick, label = curve_style.get(
                    mode, ((255, 255, 255), 2, None)
                )
                cv2.polylines(
                    debug, [np.array(pts, dtype=np.int32)], False,
                    curve_color, thick, cv2.LINE_AA,
                )
                if label:
                    cv2.putText(
                        debug, label, (pts[0][0] + 4, pts[0][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, curve_color, 2, cv2.LINE_AA,
                    )

        if status == 'OK':
            color = (0, 255, 0)
        elif status == 'RIGHT GUARD':
            color = (255, 0, 255)
        else:
            color = (0, 0, 255)
        def _fmt_x(value):
            return '--' if value is None else f'{value:.0f}'

        lines = [
            f'status: {status}',
            f'center_offset: {lane_offset if lane_offset is not None else round(self.last_offset)} '
            f'(smoothed: {self.last_offset:.1f}) -> published: {self.dbg_published_offset}',
            f'mode: {self.driving_mode}, dashed_ref_x: {self.dashed_reference_x_px}',
            f'curve: bend={self.dashed_curve_bend:.0f} gain={self.curve_ref_gain:.1f} '
            f'eff_ref={self.dashed_reference_x_px + self.curve_ref_gain * self.dashed_curve_bend:.0f}',
            f'white_ratio: {near_white_ratio:.2f}',
            f'right_avoid: w={self.right_avoid_w:.2f} '
            f'(start<={int(self.right_avoid_start_x)} full<={int(self.right_avoid_full_x)})',
            f'right x: near={_fmt_x(self.dbg_right_near_x)} '
            f'la={_fmt_x(self.dbg_right_lookahead_x)} '
            f'guard={_fmt_x(self.dbg_right_guard_x)} '
            f'(la_rows={self.right_guard_lookahead_rows})',
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
