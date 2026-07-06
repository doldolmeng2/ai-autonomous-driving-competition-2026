"""timed_lane_offset_node.

역할:
    시간주행(2차로 고정 주행) 미션에서 /camera/high/image_raw 를 받아
    "오른쪽 차선(실선)과 차 사이의 간격"을 계산해 /lane_offset 으로 발행한다.

트랙 특징(2차로 = 오른쪽 차로 기준):
    - 오른쪽 차선(실선) 바깥쪽은 초록색 매트
    - 왼쪽 차선(점선, 1차로와의 중앙선) 바깥쪽은 그냥 검은 도로(색으로 구분 안 됨)
    - 따라서 점선이 끊겨서 좌/우 차선 중 어느 쪽인지 애매할 때는,
      "차선 바로 오른쪽이 초록색인가"로 오른쪽(실선) 차선을 특정한다.

기준값(target_offset_px):
    rosbag2_2026_07_01-15_30_56 (2차선 주행 녹화본)을 분석해 직선 구간에서
    "오른쪽 차선 x좌표 - 이미지 중심" 값이 평균 195px 근처로 유지됨을 확인했다.
    이 값을 기준(0)으로 삼고, 현재 프레임에서 측정한 값과의 차이를
    /lane_offset 으로 발행한다. (양수 = 차가 왼쪽으로 치우쳐서 우회전 필요)

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

# ROI: 이미지 상단(배경)과 하단(차량 후드)을 잘라낸다. (640x360 기준)
ROI_TOP = 100
ROI_BOTTOM = 280

# 흰색(차선/횡단보도) HSV 임계값
WHITE_S_MAX = 60
WHITE_V_MIN = 140
# 초록색(오른쪽 차선 바깥 매트) HSV 임계값
GREEN_H_MIN = 30
GREEN_H_MAX = 90
GREEN_S_MIN = 40
GREEN_V_MIN = 70

# 오른쪽 차선 탐색에 쓰는 근접(ROI 하단) 밴드 높이
NEAR_FIELD_ROWS = 70
# 근접 밴드에서 흰색 비율이 이 값을 넘으면 횡단보도 등으로 판단하고 무시
WHITE_OVERLOAD_RATIO = 0.15

# 직선 구간 rosbag 분석으로 얻은 기준 offset(px). 실차 재조정 시 수정.
TARGET_OFFSET_PX = 195
# 한 프레임 사이 offset이 이 값보다 더 튀면 오검출로 보고 이전 값 유지
MAX_OFFSET_JUMP_PX = 80

# 슬라이딩 윈도우 (범위를 좁게 잡아서 옆에 있는 꽃 그림 등을 덜 건드리게 함)
NUM_WINDOWS = 7
WINDOW_MARGIN = 27
WINDOW_MINPIX = 50

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
# 원격 모니터링 등을 위해 같은 화면을 토픽으로도 발행하고 싶을 때만 True
PUBLISH_DEBUG_IMAGE = False
DEBUG_IMAGE_TOPIC = '/lane_offset/debug_image'


class TimedLaneOffsetNode(Node):
    """/camera/high/image_raw -> 오른쪽 차선 기준 offset을 계산해 /lane_offset 발행."""

    def __init__(self):
        super().__init__('timed_lane_offset_node')

        # ---- 파라미터 ------------------------------------------------------
        self.declare_parameter('image_topic', IMAGE_TOPIC)
        self.declare_parameter('lane_offset_topic', LANE_OFFSET_TOPIC)
        self.declare_parameter('roi_top', ROI_TOP)
        self.declare_parameter('roi_bottom', ROI_BOTTOM)
        self.declare_parameter('white_s_max', WHITE_S_MAX)
        self.declare_parameter('white_v_min', WHITE_V_MIN)
        self.declare_parameter('green_h_min', GREEN_H_MIN)
        self.declare_parameter('green_h_max', GREEN_H_MAX)
        self.declare_parameter('green_s_min', GREEN_S_MIN)
        self.declare_parameter('green_v_min', GREEN_V_MIN)
        self.declare_parameter('near_field_rows', NEAR_FIELD_ROWS)
        self.declare_parameter('white_overload_ratio', WHITE_OVERLOAD_RATIO)
        self.declare_parameter('target_offset_px', TARGET_OFFSET_PX)
        self.declare_parameter('max_offset_jump_px', MAX_OFFSET_JUMP_PX)
        self.declare_parameter('num_windows', NUM_WINDOWS)
        self.declare_parameter('window_margin', WINDOW_MARGIN)
        self.declare_parameter('window_minpix', WINDOW_MINPIX)
        self.declare_parameter(
            'near_field_full_height_ratio', NEAR_FIELD_FULL_HEIGHT_RATIO
        )
        self.declare_parameter('min_line_aspect_ratio', MIN_LINE_ASPECT_RATIO)
        self.declare_parameter('min_line_height_px', MIN_LINE_HEIGHT_PX)
        self.declare_parameter('debug_view', DEBUG_VIEW)
        self.declare_parameter('window_name', WINDOW_NAME)
        self.declare_parameter('publish_debug_image', PUBLISH_DEBUG_IMAGE)
        self.declare_parameter('debug_image_topic', DEBUG_IMAGE_TOPIC)

        self.image_topic = self.get_parameter('image_topic').value
        self.lane_offset_topic = self.get_parameter('lane_offset_topic').value
        self.roi_top = int(self.get_parameter('roi_top').value)
        self.roi_bottom = int(self.get_parameter('roi_bottom').value)
        self.white_s_max = int(self.get_parameter('white_s_max').value)
        self.white_v_min = int(self.get_parameter('white_v_min').value)
        self.green_h_min = int(self.get_parameter('green_h_min').value)
        self.green_h_max = int(self.get_parameter('green_h_max').value)
        self.green_s_min = int(self.get_parameter('green_s_min').value)
        self.green_v_min = int(self.get_parameter('green_v_min').value)
        self.near_field_rows = int(self.get_parameter('near_field_rows').value)
        self.white_overload_ratio = float(
            self.get_parameter('white_overload_ratio').value
        )
        self.target_offset_px = int(self.get_parameter('target_offset_px').value)
        self.max_offset_jump_px = int(
            self.get_parameter('max_offset_jump_px').value
        )
        self.num_windows = int(self.get_parameter('num_windows').value)
        self.window_margin = int(self.get_parameter('window_margin').value)
        self.window_minpix = int(self.get_parameter('window_minpix').value)
        self.near_field_full_height_ratio = float(
            self.get_parameter('near_field_full_height_ratio').value
        )
        self.min_line_aspect_ratio = float(
            self.get_parameter('min_line_aspect_ratio').value
        )
        self.min_line_height_px = int(self.get_parameter('min_line_height_px').value)
        self.debug_view = bool(self.get_parameter('debug_view').value)
        self.window_name = self.get_parameter('window_name').value
        self.publish_debug_image = bool(
            self.get_parameter('publish_debug_image').value
        )
        self.debug_image_topic = self.get_parameter('debug_image_topic').value

        # 마지막으로 발행한(유효했던) offset. 오검출 프레임에서는 이 값을 그대로 재사용.
        self.last_offset = 0

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
            f'target_offset_px={self.target_offset_px}'
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

        base_x = self.find_right_line_base(white_mask, green_mask)
        if base_x is None:
            self.get_logger().warn(
                'No lane pixels found, holding last offset', throttle_duration_sec=1.0
            )
            self.publish_offset(self.last_offset)
            self.publish_debug(msg, frame, 'NO LANE FOUND', near_white_ratio)
            return

        line_x, windows, points_x, points_y = self.track_line_with_sliding_window(
            white_mask, base_x
        )
        center_x = frame.shape[1] / 2.0
        raw_offset = line_x - center_x
        lane_offset = int(round(raw_offset - self.target_offset_px))

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
                base_x=base_x, line_x=line_x, windows=windows,
                points_x=points_x, points_y=points_y, lane_offset=lane_offset,
            )
            return

        self.last_offset = lane_offset
        self.publish_offset(lane_offset)
        self.publish_debug(
            msg, frame, 'OK', near_white_ratio,
            base_x=base_x, line_x=line_x, windows=windows,
            points_x=points_x, points_y=points_y, lane_offset=lane_offset,
        )

    def publish_offset(self, value):
        msg = Int16()
        msg.data = int(value)
        self.offset_pub.publish(msg)

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
    # 오른쪽 차선(실선) 시작 위치 찾기
    # ======================================================================
    def find_right_line_base(self, white_mask, green_mask):
        """근접 밴드에서 차선 모양 후보를 찾고, 오른쪽이 초록색인 것을 오른쪽 차선으로 본다."""
        band_white = white_mask[-self.near_field_rows:, :]
        band_green = green_mask[-self.near_field_rows:, :]
        width = band_white.shape[1]

        boxes, _labels = self.find_lane_shaped_components(
            band_white,
            self.min_line_height_px,
            self.min_line_aspect_ratio,
            self.near_field_full_height_ratio,
        )
        if not boxes:
            return None

        green_backed = []
        for x, _y, w, _h, _label in boxes:
            right_edge = x + w
            x0 = min(right_edge + 10, width - 1)
            x1 = min(right_edge + 40, width)
            if x1 <= x0:
                continue
            strip = band_green[:, x0:x1]
            green_ratio = float((strip > 0).mean()) if strip.size else 0.0
            if green_ratio > 0.4:
                green_backed.append(x + w // 2)

        if green_backed:
            # 오른쪽 바깥이 초록색으로 확인된 것 중 가장 오른쪽 = 오른쪽 실선
            return max(green_backed)
        # 점선/실선 둘 다 초록 확인이 안 되면(가려짐 등) 가장 오른쪽 후보로 대체
        x, _y, w, _h, _label = max(boxes, key=lambda b: b[0] + b[2])
        return x + w // 2

    # ======================================================================
    # 슬라이딩 윈도우로 곡선 추적
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
        base_x=None, line_x=None, windows=None,
        points_x=None, points_y=None, lane_offset=None,
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
        # 차량 중심선(흰색)과 기준 offset 위치(주황, "여기 있어야 lane_offset=0")
        cv2.line(debug, (center_x, self.roi_top), (center_x, self.roi_bottom), (255, 255, 255), 1)
        self.draw_dashed_vline(
            debug, center_x + self.target_offset_px, self.roi_top, self.roi_bottom, (0, 165, 255)
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

        color = (0, 255, 0) if status == 'OK' else (0, 0, 255)
        lines = [
            f'status: {status}',
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
