import math

import cv2
import numpy as np
import rclpy
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.node import Node
from sensor_msgs.msg import Image, Range
from std_msgs.msg import Int16, Int16MultiArray


LOW_IMAGE_TOPIC = '/camera/low/image_raw'
ULTRASONIC_TOPICS = [f'/ultrasonic/range_{index}' for index in range(1, 7)]
LANE_OFFSET_TOPIC = '/lane_offset'
LANE_INFO_TOPIC = '/lane_info'
MOTOR_CONTROL_TOPIC = '/motor_control'
BASE_SPEED = 120
MAX_STEER = 45
# 미션 시작 차선의 단일 설정값. lane_offset 노드는 /lane_info를 받아 이 값을
# 따라가므로 시작 차선을 바꿀 때는 이 상수만 수정한다.
DRIVING_MODE = '2lane'
LANE_CHANGE_CLOSE_DISTANCE = 0.6
LANE_CHANGE_CLEAR_DISTANCE = 1.2
LANE_CHANGE_CLOSE_CONFIRM_SAMPLES = 3
LANE_CHANGE_CLEAR_CONFIRM_SAMPLES = 3
# 차선 변경을 한 번 확정한 뒤 다음 변경 감지를 잠그는 시간.
LANE_CHANGE_COOLDOWN_SEC = 7.0
# 한 번의 노드 실행에서 각 방향(1->2, 2->1)으로 허용할 최대 변경 횟수.
LANE_CHANGE_MAX_PER_DIRECTION = 1
DEBUG_VIEW = True
DEBUG_WINDOW_NAME = 'mission_lane_main_debug'
TRAFFIC_LIGHT_DEBUG_WINDOW_NAME = 'mission_lane_traffic_light_mask'
TRAFFIC_LIGHT_ROI_HEIGHT_RATIO = 0.55
TRAFFIC_LIGHT_ROI_X_MIN_RATIO = 0.0
TRAFFIC_LIGHT_ROI_X_MAX_RATIO = 0.70
TRAFFIC_LIGHT_MIN_COMPONENT_PIXELS = 15
TRAFFIC_LIGHT_MAX_ASPECT_RATIO = 3.0
TRAFFIC_LIGHT_MIN_FILL_RATIO = 0.05
TRAFFIC_LIGHT_BRIGHT_CORE_MIN_PIXELS = 10
TRAFFIC_LIGHT_CONFIRM_FRAMES = 3
# 실제 주행에서 3,000픽셀 조건에 도달하지 못한 경우가 있어 정지 조건을
# 완화한다. 색상/위치/형태/밝은 LED 중심과 연속 프레임 검증은 그대로 유지한다.
RED_STOP_PIXEL_COUNT = 4000
RED_RESUME_PIXEL_COUNT = 700
GREEN_GO_PIXEL_COUNT = 200
GREEN_STRAIGHT_DURATION_SEC = 3.0
# 빨강은 HSV 색상환 양 끝만 사용한다. 6~34 영역의 주황/노랑은 제외한다.
RED_HUE_LOW_MAX = 5
RED_HUE_HIGH_MIN = 175
RED_SATURATION_MIN = 100
RED_VALUE_MIN = 100
RED_HLS_SATURATION_MIN = 80
RED_YCRCB_CR_MIN = 150
GREEN_HUE_MIN = 35
GREEN_HUE_MAX = 95
GREEN_SATURATION_MIN = 60
GREEN_VALUE_MIN = 80
GREEN_HLS_SATURATION_MIN = 60
GREEN_YCRCB_CR_MAX = 135


class MissionLaneMainNode(Node):
    """Mission lane driving node.

    PDF flow:
        /camera/low/image_raw
        /ultrasonic/range_1 ... /ultrasonic/range_6
        /lane_offset
            -> mission_lane_main_node
            -> /lane_info, /motor_control
    """

    def __init__(self):
        super().__init__('mission_lane_main_node')

        self.declare_parameter('base_speed', BASE_SPEED)
        self.declare_parameter('max_steer', MAX_STEER)
        self.declare_parameter('driving_mode', DRIVING_MODE)
        self.declare_parameter(
            'lane_change_close_distance', LANE_CHANGE_CLOSE_DISTANCE
        )
        self.declare_parameter(
            'lane_change_clear_distance', LANE_CHANGE_CLEAR_DISTANCE
        )
        self.declare_parameter(
            'lane_change_close_confirm_samples',
            LANE_CHANGE_CLOSE_CONFIRM_SAMPLES,
        )
        self.declare_parameter(
            'lane_change_clear_confirm_samples',
            LANE_CHANGE_CLEAR_CONFIRM_SAMPLES,
        )
        self.declare_parameter(
            'lane_change_cooldown_sec', LANE_CHANGE_COOLDOWN_SEC
        )
        self.declare_parameter(
            'lane_change_max_per_direction',
            LANE_CHANGE_MAX_PER_DIRECTION,
        )
        self.declare_parameter('debug_view', DEBUG_VIEW)
        self.declare_parameter('debug_window_name', DEBUG_WINDOW_NAME)
        self.declare_parameter(
            'traffic_light_debug_window_name', TRAFFIC_LIGHT_DEBUG_WINDOW_NAME
        )
        self.declare_parameter(
            'traffic_light_roi_height_ratio', TRAFFIC_LIGHT_ROI_HEIGHT_RATIO
        )
        self.declare_parameter(
            'traffic_light_roi_x_min_ratio', TRAFFIC_LIGHT_ROI_X_MIN_RATIO
        )
        self.declare_parameter(
            'traffic_light_roi_x_max_ratio', TRAFFIC_LIGHT_ROI_X_MAX_RATIO
        )
        self.declare_parameter(
            'traffic_light_min_component_pixels',
            TRAFFIC_LIGHT_MIN_COMPONENT_PIXELS,
        )
        self.declare_parameter(
            'traffic_light_max_aspect_ratio', TRAFFIC_LIGHT_MAX_ASPECT_RATIO
        )
        self.declare_parameter(
            'traffic_light_min_fill_ratio', TRAFFIC_LIGHT_MIN_FILL_RATIO
        )
        self.declare_parameter(
            'traffic_light_bright_core_min_pixels',
            TRAFFIC_LIGHT_BRIGHT_CORE_MIN_PIXELS,
        )
        self.declare_parameter(
            'traffic_light_confirm_frames', TRAFFIC_LIGHT_CONFIRM_FRAMES
        )
        self.declare_parameter('red_stop_pixel_count', RED_STOP_PIXEL_COUNT)
        self.declare_parameter('red_resume_pixel_count', RED_RESUME_PIXEL_COUNT)
        self.declare_parameter('green_go_pixel_count', GREEN_GO_PIXEL_COUNT)
        self.declare_parameter(
            'green_straight_duration_sec', GREEN_STRAIGHT_DURATION_SEC
        )
        self.declare_parameter('red_hue_low_max', RED_HUE_LOW_MAX)
        self.declare_parameter('red_hue_high_min', RED_HUE_HIGH_MIN)
        self.declare_parameter('red_saturation_min', RED_SATURATION_MIN)
        self.declare_parameter('red_value_min', RED_VALUE_MIN)
        self.declare_parameter(
            'red_hls_saturation_min', RED_HLS_SATURATION_MIN
        )
        self.declare_parameter('red_ycrcb_cr_min', RED_YCRCB_CR_MIN)
        self.declare_parameter('green_hue_min', GREEN_HUE_MIN)
        self.declare_parameter('green_hue_max', GREEN_HUE_MAX)
        self.declare_parameter('green_saturation_min', GREEN_SATURATION_MIN)
        self.declare_parameter('green_value_min', GREEN_VALUE_MIN)
        self.declare_parameter(
            'green_hls_saturation_min', GREEN_HLS_SATURATION_MIN
        )
        self.declare_parameter('green_ycrcb_cr_max', GREEN_YCRCB_CR_MAX)

        self.base_speed = int(self.get_parameter('base_speed').value)
        self.max_steer = int(self.get_parameter('max_steer').value)
        driving_mode = str(self.get_parameter('driving_mode').value).lower()
        if driving_mode not in ('1lane', '2lane'):
            self.get_logger().warn(
                f"Unknown driving_mode='{driving_mode}'; using '2lane'"
            )
            driving_mode = '2lane'
        self.lane_number = 1 if driving_mode == '1lane' else 2
        self.lane_change_close_distance = max(
            0.0,
            float(self.get_parameter('lane_change_close_distance').value),
        )
        self.lane_change_clear_distance = max(
            self.lane_change_close_distance,
            float(self.get_parameter('lane_change_clear_distance').value),
        )
        self.lane_change_close_confirm_samples = max(
            1,
            int(self.get_parameter('lane_change_close_confirm_samples').value),
        )
        self.lane_change_clear_confirm_samples = max(
            1,
            int(self.get_parameter('lane_change_clear_confirm_samples').value),
        )
        self.lane_change_cooldown_sec = max(
            0.0,
            float(self.get_parameter('lane_change_cooldown_sec').value),
        )
        self.lane_change_max_per_direction = max(
            0,
            int(self.get_parameter('lane_change_max_per_direction').value),
        )
        self.debug_view = bool(self.get_parameter('debug_view').value)
        self.debug_window_name = str(
            self.get_parameter('debug_window_name').value
        )
        self.traffic_light_debug_window_name = str(
            self.get_parameter('traffic_light_debug_window_name').value
        )
        self.traffic_light_roi_height_ratio = float(np.clip(
            self.get_parameter('traffic_light_roi_height_ratio').value,
            0.05,
            1.0,
        ))
        self.traffic_light_roi_x_min_ratio = float(np.clip(
            self.get_parameter('traffic_light_roi_x_min_ratio').value,
            0.0,
            0.95,
        ))
        self.traffic_light_roi_x_max_ratio = float(np.clip(
            self.get_parameter('traffic_light_roi_x_max_ratio').value,
            self.traffic_light_roi_x_min_ratio + 0.05,
            1.0,
        ))
        self.traffic_light_min_component_pixels = max(
            1,
            int(self.get_parameter('traffic_light_min_component_pixels').value),
        )
        self.traffic_light_max_aspect_ratio = max(
            1.0,
            float(self.get_parameter('traffic_light_max_aspect_ratio').value),
        )
        self.traffic_light_min_fill_ratio = float(np.clip(
            self.get_parameter('traffic_light_min_fill_ratio').value,
            0.0,
            1.0,
        ))
        self.traffic_light_bright_core_min_pixels = max(
            1,
            int(
                self.get_parameter(
                    'traffic_light_bright_core_min_pixels'
                ).value
            ),
        )
        self.traffic_light_confirm_frames = max(
            1, int(self.get_parameter('traffic_light_confirm_frames').value)
        )
        self.red_stop_pixel_count = max(
            1, int(self.get_parameter('red_stop_pixel_count').value)
        )
        self.red_resume_pixel_count = max(
            0, int(self.get_parameter('red_resume_pixel_count').value)
        )
        if self.red_resume_pixel_count >= self.red_stop_pixel_count:
            self.get_logger().warn(
                'red_resume_pixel_count must be smaller than '
                'red_stop_pixel_count; using stop threshold - 1'
            )
            self.red_resume_pixel_count = self.red_stop_pixel_count - 1
        self.green_go_pixel_count = max(
            1, int(self.get_parameter('green_go_pixel_count').value)
        )
        self.green_straight_duration_sec = max(
            0.0,
            float(self.get_parameter('green_straight_duration_sec').value),
        )
        self.red_hue_low_max = int(np.clip(
            self.get_parameter('red_hue_low_max').value, 0, 179
        ))
        self.red_hue_high_min = int(np.clip(
            self.get_parameter('red_hue_high_min').value, 0, 179
        ))
        self.red_saturation_min = int(np.clip(
            self.get_parameter('red_saturation_min').value, 0, 255
        ))
        self.red_value_min = int(np.clip(
            self.get_parameter('red_value_min').value, 0, 255
        ))
        self.red_hls_saturation_min = int(np.clip(
            self.get_parameter('red_hls_saturation_min').value, 0, 255
        ))
        self.red_ycrcb_cr_min = int(np.clip(
            self.get_parameter('red_ycrcb_cr_min').value, 0, 255
        ))
        self.green_hue_min = int(np.clip(
            self.get_parameter('green_hue_min').value, 0, 179
        ))
        self.green_hue_max = int(np.clip(
            self.get_parameter('green_hue_max').value,
            self.green_hue_min,
            179,
        ))
        self.green_saturation_min = int(np.clip(
            self.get_parameter('green_saturation_min').value, 0, 255
        ))
        self.green_value_min = int(np.clip(
            self.get_parameter('green_value_min').value, 0, 255
        ))
        self.green_hls_saturation_min = int(np.clip(
            self.get_parameter('green_hls_saturation_min').value, 0, 255
        ))
        self.green_ycrcb_cr_max = int(np.clip(
            self.get_parameter('green_ycrcb_cr_max').value, 0, 255
        ))

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.motor_pub = self.create_publisher(
            Int16MultiArray, MOTOR_CONTROL_TOPIC, 10
        )
        self.lane_info_pub = self.create_publisher(Int16, LANE_INFO_TOPIC, 10)
        self.create_subscription(Image, LOW_IMAGE_TOPIC, self.low_image_callback, qos)
        for sensor_index, topic in enumerate(ULTRASONIC_TOPICS, start=1):
            self.create_subscription(
                Range,
                topic,
                lambda msg, index=sensor_index: self.ultrasonic_callback(
                    msg, index
                ),
                10,
            )
        self.create_subscription(
            Int16, LANE_OFFSET_TOPIC, self.lane_offset_callback, 10
        )

        self.low_image = None
        self.red_pixel_count = 0
        self.green_pixel_count = 0
        self.red_confirm_count = 0
        self.green_confirm_count = 0
        self.red_stop_active = False
        self.green_straight_active = False
        self.green_straight_deadline_ns = None
        self.last_lane_offset = None
        self.ultrasonic_ranges = {}
        self.lane_change_armed = False
        self.close_sample_count = 0
        self.clear_sample_count = 0
        self.last_lane_change_ns = None
        self.lane_change_counts = {(1, 2): 0, (2, 1): 0}
        # lane_offset 노드가 어느 순서로 시작해도 현재 모드를 받을 수 있도록
        # 주기적으로 lane_info를 발행한다.
        self.create_timer(0.2, self.publish_lane_info)
        self.create_timer(0.05, self.update_green_straight_mode)

        self.get_logger().info(
            f'Subscribing {LOW_IMAGE_TOPIC}, {ULTRASONIC_TOPICS}, '
            f'{LANE_OFFSET_TOPIC}; publishing {LANE_INFO_TOPIC}, '
            f'{MOTOR_CONTROL_TOPIC}; starting in {self.driving_mode}'
        )

    @property
    def driving_mode(self):
        return f'{self.lane_number}lane'

    def low_image_callback(self, msg):
        self.low_image = msg
        frame = self.to_bgr(msg)
        if frame is None:
            return

        red_mask, green_mask = self.make_traffic_light_masks(frame)
        self.red_pixel_count = int(cv2.countNonZero(red_mask))
        self.green_pixel_count = int(cv2.countNonZero(green_mask))
        red_detected = self.red_pixel_count >= self.red_stop_pixel_count
        green_detected = self.green_pixel_count >= self.green_go_pixel_count

        if red_detected:
            self.red_confirm_count += 1
        else:
            self.red_confirm_count = 0

        # 평소에는 초록을 상태 전환에 사용하지 않는다. 신호등 빨강이 연속으로
        # 확인되어 정지한 뒤에만 초록 출발 조건을 활성화한다.
        if (
            not self.red_stop_active
            and self.red_confirm_count >= self.traffic_light_confirm_frames
        ):
            self.red_stop_active = True
            self.cancel_green_straight_mode()
            self.green_confirm_count = 0
            self.get_logger().info(
                f'Red traffic light confirmed: STOP '
                f'({self.red_pixel_count} pixels)'
            )

        if self.red_stop_active:
            # 빨강과 초록이 동시에 검출되거나 빨강이 충분히 사라지지 않았으면
            # 안전을 위해 빨강을 우선하고 초록 연속 확인을 초기화한다.
            if (
                self.red_pixel_count < self.red_resume_pixel_count
                and green_detected
            ):
                self.green_confirm_count += 1
            else:
                self.green_confirm_count = 0

            if (
                self.green_confirm_count
                >= self.traffic_light_confirm_frames
            ):
                self.red_stop_active = False
                self.red_confirm_count = 0
                self.green_confirm_count = 0
                self.get_logger().info(
                    f'Green traffic light confirmed: GO '
                    f'({self.green_pixel_count} pixels)'
                )
                self.start_green_straight_mode()
            else:
                self.publish_motor_command(0, 0)
        else:
            self.green_confirm_count = 0

        if self.debug_view:
            self.show_debug_view(frame)
            self.show_traffic_light_debug(frame, red_mask, green_mask)

    def ultrasonic_callback(self, msg, sensor_index):
        distance = float(msg.range)
        self.ultrasonic_ranges[sensor_index] = distance
        self.update_lane_change(sensor_index, distance)

    def update_lane_change(self, sensor_index, distance):
        """연속 근접 후 연속 이탈한 물체를 추월한 것으로 보고 차선을 전환한다."""
        cooldown_remaining = self.lane_change_cooldown_remaining_sec()
        if cooldown_remaining > 0.0:
            # 변경 직후 반대편 센서가 같은 장애물을 다시 감지해 원래 차선으로
            # 즉시 돌아가는 것을 막는다. 쿨다운 중 표본은 다음 변경에 넘기지 않는다.
            self.lane_change_armed = False
            self.close_sample_count = 0
            self.clear_sample_count = 0
            return False

        transition = (
            self.lane_number,
            1 if self.lane_number == 2 else 2,
        )
        if not self.lane_change_direction_available(transition):
            self.lane_change_armed = False
            self.close_sample_count = 0
            self.clear_sample_count = 0
            return False

        # 기존 3/4번 추월 감지는 일단 비활성화한다.
        # active_sensor = 3 if self.lane_number == 2 else 4
        # 2차선에서는 1번, 1차선에서는 2번 초음파로 옆 장애물 통과를 본다.
        active_sensor = 1 if self.lane_number == 2 else 2
        if sensor_index != active_sensor:
            return False

        # timeout/no-echo는 ultrasonic_node에서 +inf로 정규화된다. 근접 물체가
        # 센서 범위 밖으로 완전히 사라진 것이므로, armed 이후에는 +inf도
        # LANE_CHANGE_CLEAR_DISTANCE보다 먼 CLEAR 측정으로 사용한다.
        # NaN과 음수(-inf 포함)만 잘못된 측정으로 제외한다.
        if math.isnan(distance) or distance < 0.0:
            self.close_sample_count = 0
            self.clear_sample_count = 0
            return False

        if not self.lane_change_armed:
            if distance <= self.lane_change_close_distance:
                self.close_sample_count += 1
            else:
                self.close_sample_count = 0

            if (
                self.close_sample_count
                >= self.lane_change_close_confirm_samples
            ):
                self.lane_change_armed = True
                self.close_sample_count = 0
                self.clear_sample_count = 0
                self.get_logger().info(
                    f'{self.driving_mode}: ultrasonic {active_sensor} '
                    f'close confirmed ({distance:.2f} m); lane change armed'
                )
            return False

        if distance > self.lane_change_clear_distance:
            self.clear_sample_count += 1
        else:
            self.clear_sample_count = 0

        if self.clear_sample_count >= self.lane_change_clear_confirm_samples:
            previous_mode = self.driving_mode
            previous_lane = self.lane_number
            self.lane_number = 1 if self.lane_number == 2 else 2
            self.lane_change_counts[(previous_lane, self.lane_number)] += 1
            self.last_lane_change_ns = self.get_clock().now().nanoseconds
            self.lane_change_armed = False
            self.close_sample_count = 0
            self.clear_sample_count = 0
            self.publish_lane_info()
            self.get_logger().info(
                f'Lane change trigger: ultrasonic {active_sensor} '
                f'clear confirmed ({distance:.2f} m); '
                f'{previous_mode} -> {self.driving_mode}'
            )
            return True
        return False

    def lane_change_direction_available(self, transition=None):
        """지정 방향의 차선 변경 횟수가 허용 범위 안인지 확인한다."""
        if transition is None:
            transition = (
                self.lane_number,
                1 if self.lane_number == 2 else 2,
            )
        return (
            self.lane_change_counts[transition]
            < self.lane_change_max_per_direction
        )

    def lane_change_cooldown_remaining_sec(self):
        """최근 차선 변경 이후 남은 잠금 시간을 초 단위로 반환한다."""
        if self.last_lane_change_ns is None:
            return 0.0
        now_ns = self.get_clock().now().nanoseconds
        elapsed_sec = max(0.0, (now_ns - self.last_lane_change_ns) / 1e9)
        return max(0.0, self.lane_change_cooldown_sec - elapsed_sec)

    def lane_offset_callback(self, msg):
        offset = int(msg.data)
        self.last_lane_offset = offset
        if self.red_stop_active:
            self.publish_motor_command(0, 0)
            return
        if self.green_straight_active:
            if self.green_straight_deadline_ns is None:
                self.green_straight_deadline_ns = (
                    self.get_clock().now().nanoseconds
                    + int(self.green_straight_duration_sec * 1_000_000_000)
                )
                self.get_logger().info(
                    f'Fresh lane offset received after green; holding steer=0 '
                    f'for {self.green_straight_duration_sec:.1f} seconds'
                )
            self.publish_motor_command(0, self.base_speed)
            return

        self.publish_lane_offset_command(offset)

    def start_green_straight_mode(self):
        """초록 출발 후 새 차선 입력을 기다리며 조향 0으로 직진한다."""
        self.green_straight_active = True
        self.green_straight_deadline_ns = None
        self.publish_motor_command(0, self.base_speed)

    def cancel_green_straight_mode(self):
        self.green_straight_active = False
        self.green_straight_deadline_ns = None

    def update_green_straight_mode(self):
        """새 lane_offset 수신 후 3초가 지나면 중앙선 추적을 재개한다."""
        if not self.green_straight_active:
            return
        if self.red_stop_active or self.green_straight_deadline_ns is None:
            return
        if self.get_clock().now().nanoseconds < self.green_straight_deadline_ns:
            return

        self.cancel_green_straight_mode()
        self.get_logger().info(
            'Green straight interval complete; resuming lane tracking'
        )
        if self.last_lane_offset is not None:
            self.publish_lane_offset_command(self.last_lane_offset)

    def publish_lane_offset_command(self, offset):
        steer = offset
        steer = max(-self.max_steer, min(self.max_steer, steer))
        self.publish_motor_command(steer, self.base_speed)

    def publish_motor_command(self, steer, speed):
        command = Int16MultiArray()
        command.data = [int(steer), int(speed)]
        self.motor_pub.publish(command)

    def make_traffic_light_masks(self, frame):
        """HSV+HLS+YCrCb 교집합으로 상단의 빨강/초록 신호만 분리한다."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        bright_core_mask = cv2.inRange(hsv, (0, 0, 220), (179, 100, 255))

        red_hsv_low = cv2.inRange(
            hsv,
            (0, self.red_saturation_min, self.red_value_min),
            (self.red_hue_low_max, 255, 255),
        )
        red_hsv_high = cv2.inRange(
            hsv,
            (self.red_hue_high_min, self.red_saturation_min, self.red_value_min),
            (179, 255, 255),
        )
        red_hls = cv2.inRange(
            hls,
            (0, 0, self.red_hls_saturation_min),
            (179, 255, 255),
        )
        red_ycrcb = cv2.inRange(
            ycrcb,
            (0, self.red_ycrcb_cr_min, 0),
            (255, 255, 255),
        )
        red_mask = cv2.bitwise_and(
            cv2.bitwise_or(red_hsv_low, red_hsv_high), red_hls
        )
        red_mask = cv2.bitwise_and(red_mask, red_ycrcb)

        green_hsv = cv2.inRange(
            hsv,
            (
                self.green_hue_min,
                self.green_saturation_min,
                self.green_value_min,
            ),
            (self.green_hue_max, 255, 255),
        )
        green_hls = cv2.inRange(
            hls,
            (self.green_hue_min, 0, self.green_hls_saturation_min),
            (self.green_hue_max, 255, 255),
        )
        green_ycrcb = cv2.inRange(
            ycrcb,
            (0, 0, 0),
            (255, self.green_ycrcb_cr_max, 255),
        )
        green_mask = cv2.bitwise_and(green_hsv, green_hls)
        green_mask = cv2.bitwise_and(green_mask, green_ycrcb)

        roi_bottom = int(round(
            frame.shape[0] * self.traffic_light_roi_height_ratio
        ))
        roi_x_min = int(round(
            frame.shape[1] * self.traffic_light_roi_x_min_ratio
        ))
        roi_x_max = int(round(
            frame.shape[1] * self.traffic_light_roi_x_max_ratio
        ))
        red_mask[roi_bottom:, :] = 0
        green_mask[roi_bottom:, :] = 0
        red_mask[:, :roi_x_min] = 0
        red_mask[:, roi_x_max:] = 0
        green_mask[:, :roi_x_min] = 0
        green_mask[:, roi_x_max:] = 0

        return (
            self.filter_traffic_light_components(red_mask, bright_core_mask),
            self.filter_traffic_light_components(green_mask, bright_core_mask),
        )

    def filter_traffic_light_components(self, mask, bright_core_mask):
        """형태와 밝은 LED 중심을 만족하는 가장 큰 신호등 후보만 남긴다."""
        kernel = np.ones((3, 3), dtype=np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            cleaned
        )
        best_label = None
        best_area = 0
        for label in range(1, component_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.traffic_light_min_component_pixels:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            short_side = max(1, min(width, height))
            aspect_ratio = max(width, height) / short_side
            fill_ratio = area / max(1, width * height)
            bright_core_pixels = cv2.countNonZero(
                bright_core_mask[y:y + height, x:x + width]
            )
            if aspect_ratio > self.traffic_light_max_aspect_ratio:
                continue
            if fill_ratio < self.traffic_light_min_fill_ratio:
                continue
            if (
                bright_core_pixels
                < self.traffic_light_bright_core_min_pixels
            ):
                continue
            if area > best_area:
                best_area = area
                best_label = label

        filtered = np.zeros_like(cleaned)
        if best_label is not None:
            filtered[labels == best_label] = 255
        return filtered

    def show_traffic_light_debug(self, frame, red_mask, green_mask):
        """검출된 빨강과 초록 픽셀만 검은 배경 위에 표시한다."""
        debug = np.zeros_like(frame)
        debug[red_mask > 0] = (0, 0, 255)
        debug[green_mask > 0] = (0, 255, 0)
        cv2.imshow(self.traffic_light_debug_window_name, debug)
        cv2.waitKey(1)

    def publish_lane_info(self):
        lane_info = Int16()
        lane_info.data = self.lane_number
        self.lane_info_pub.publish(lane_info)

    def show_debug_view(self, frame):
        """저해상도 카메라 위에 미션 차선 및 1/2번 초음파 상태를 표시한다."""
        debug = frame.copy()
        # 기존 3/4번 표시도 추월 감지와 함께 일단 비활성화한다.
        # active_sensor = 3 if self.lane_number == 2 else 4
        active_sensor = 1 if self.lane_number == 2 else 2
        panel_height = min(debug.shape[0], 225)
        panel_width = min(debug.shape[1], 640)
        overlay = debug.copy()
        cv2.rectangle(
            overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), -1
        )
        cv2.addWeighted(overlay, 0.65, debug, 0.35, 0.0, debug)

        cooldown_remaining = self.lane_change_cooldown_remaining_sec()
        if cooldown_remaining > 0.0:
            state = f'COOLDOWN: {cooldown_remaining:.1f} SEC'
        elif not self.lane_change_direction_available():
            next_lane = 1 if self.lane_number == 2 else 2
            state = f'LIMIT: {self.lane_number}->{next_lane} USED'
        else:
            state = 'ARMED: WAIT CLEAR' if self.lane_change_armed else 'WATCH CLOSE'
        if self.red_stop_active:
            drive_state = 'STOP'
        elif self.green_straight_active:
            drive_state = (
                'STRAIGHT: WAIT LANE'
                if self.green_straight_deadline_ns is None
                else 'STRAIGHT: 3 SEC'
            )
        else:
            drive_state = 'LANE TRACK'
        lines = [
            ('MISSION LANE MAIN', (255, 255, 255)),
            (
                f'mode: {self.driving_mode}  active ultrasonic: {active_sensor}',
                (0, 255, 255),
            ),
            self.ultrasonic_debug_line(1, active_sensor),
            self.ultrasonic_debug_line(2, active_sensor),
            # self.ultrasonic_debug_line(3, active_sensor),
            # self.ultrasonic_debug_line(4, active_sensor),
            (
                f'state: {state}  close={self.close_sample_count}/'
                f'{self.lane_change_close_confirm_samples}  '
                f'clear={self.clear_sample_count}/'
                f'{self.lane_change_clear_confirm_samples}',
                (0, 255, 0) if self.lane_change_armed else (255, 255, 255),
            ),
            (
                f'thresholds: close<={self.lane_change_close_distance:.2f}m  '
                f'clear>{self.lane_change_clear_distance:.2f}m',
                (200, 200, 200),
            ),
            (
                f'changes: 1->2 {self.lane_change_counts[(1, 2)]}/'
                f'{self.lane_change_max_per_direction}  '
                f'2->1 {self.lane_change_counts[(2, 1)]}/'
                f'{self.lane_change_max_per_direction}',
                (200, 200, 200),
            ),
            (
                f'light: red={self.red_pixel_count}  green={self.green_pixel_count}  '
                f'{drive_state}  '
                f'(R {self.red_confirm_count}/'
                f'{self.traffic_light_confirm_frames}, G '
                f'{self.green_confirm_count}/'
                f'{self.traffic_light_confirm_frames})',
                (0, 0, 255) if self.red_stop_active else (0, 255, 0),
            ),
        ]
        for index, (text, color) in enumerate(lines):
            cv2.putText(
                debug,
                text,
                (10, 24 + index * 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(self.debug_window_name, debug)
        cv2.waitKey(1)

    def ultrasonic_debug_line(self, sensor_index, active_sensor):
        distance = self.ultrasonic_ranges.get(sensor_index)
        active_text = 'ACTIVE' if sensor_index == active_sensor else 'standby'
        if distance is None:
            value_text = '--'
            state_text = 'NO DATA'
            color = (128, 128, 128)
        elif math.isnan(distance) or distance < 0.0:
            value_text = 'NAN' if math.isnan(distance) else 'INVALID'
            state_text = 'INVALID'
            color = (0, 0, 255)
        elif math.isinf(distance):
            value_text = 'INF'
            state_text = 'CLEAR'
            color = (0, 255, 0)
        elif distance <= self.lane_change_close_distance:
            value_text = f'{distance:.2f} m'
            state_text = 'CLOSE'
            color = (0, 0, 255)
        elif distance > self.lane_change_clear_distance:
            value_text = f'{distance:.2f} m'
            state_text = 'CLEAR'
            color = (0, 255, 0)
        else:
            value_text = f'{distance:.2f} m'
            state_text = 'MID'
            color = (0, 255, 255)
        return (
            f'ultrasonic {sensor_index}: {value_text}  [{state_text}, {active_text}]',
            color,
        )

    def to_bgr(self, msg):
        data = np.frombuffer(msg.data, dtype=np.uint8)
        try:
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
        except ValueError as error:
            self.get_logger().warn(
                f'Invalid low camera image buffer: {error}',
                throttle_duration_sec=5.0,
            )
            return None

        self.get_logger().warn(
            f'Unsupported low camera encoding: {msg.encoding}',
            throttle_duration_sec=5.0,
        )
        return None

    def destroy_node(self):
        if self.debug_view:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MissionLaneMainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
