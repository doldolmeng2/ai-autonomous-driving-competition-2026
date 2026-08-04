import math

import cv2
import numpy as np
import rclpy
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Int16, Int16MultiArray


LOW_IMAGE_TOPIC = '/camera/low/image_raw'
SCAN_TOPIC = '/scan'
LANE_OFFSET_TOPIC = '/lane_offset'
LANE_INFO_TOPIC = '/lane_info'
LANE_CHANGE_COMPLETE_TOPIC = '/lane_change_complete'
MOTOR_CONTROL_TOPIC = '/motor_control'

######## 최대 조향각(angle임. +-45도) 및 속도(PWM. 0~255) ########
BASE_SPEED = 120
MAX_STEER = 45
##############################################################



##############  몇차선에서 출발할지 / 차선 변경 기준 ################
# 미션 시작 차선의 단일 설정값. lane_offset 노드는 /lane_info를 받아 이 값을
# 따라가므로 시작 차선을 바꿀 때는 이 상수만 수정한다.
DRIVING_MODE = '2lane'
# LiDAR 각도: 후진 0도, 오른쪽 +90도, 왼쪽 -90도, 전진 +/-180도.
# 2차선은 왼쪽(-100~-90도), 1차선은 오른쪽(+90~+100도)에서
# 0.7m 이내 물체가 보였다가 사라지면 추월이 완료된 것으로 본다.
LIDAR_RIGHT_ANGLE_MIN_DEG = 90.0
LIDAR_RIGHT_ANGLE_MAX_DEG = 100.0
LIDAR_LEFT_ANGLE_MIN_DEG = -100.0
LIDAR_LEFT_ANGLE_MAX_DEG = -90.0
LIDAR_OVERTAKE_MAX_DISTANCE_M = 1.0
LIDAR_OVERTAKE_CLEAR_DISTANCE_M = 1.2
LIDAR_DETECT_CONFIRM_SCANS = 3
LIDAR_CLEAR_CONFIRM_SCANS = 3
# 한 번의 노드 실행에서 각 방향(1->2, 2->1)으로 허용할 최대 변경 횟수.
LANE_CHANGE_MAX_PER_DIRECTION = 1
##############################################################



##################### 신호등 관련 파라미터 ######################
DEBUG_VIEW = True
DEBUG_WINDOW_NAME = 'mission_lane_main_debug'
TRAFFIC_LIGHT_DEBUG_WINDOW_NAME = 'mission_lane_traffic_light_mask'
LIDAR_DEBUG_WINDOW_NAME = 'mission_lane_lidar_debug'
LIDAR_DEBUG_IMAGE_SIZE = 640
LIDAR_DEBUG_MAX_RANGE_M = 2.0
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
###############################################################



class MissionLaneMainNode(Node):
    """Mission lane driving node.

    PDF flow:
        /camera/low/image_raw
        /scan
        /lane_offset
            -> mission_lane_main_node
            -> /lane_info, /motor_control
    """

    def __init__(self):
        super().__init__('mission_lane_main_node')

        self.declare_parameter('base_speed', BASE_SPEED)
        self.declare_parameter('max_steer', MAX_STEER)
        self.declare_parameter('driving_mode', DRIVING_MODE)
        self.declare_parameter('scan_topic', SCAN_TOPIC)
        self.declare_parameter(
            'lidar_right_angle_min_deg', LIDAR_RIGHT_ANGLE_MIN_DEG
        )
        self.declare_parameter(
            'lidar_right_angle_max_deg', LIDAR_RIGHT_ANGLE_MAX_DEG
        )
        self.declare_parameter(
            'lidar_left_angle_min_deg', LIDAR_LEFT_ANGLE_MIN_DEG
        )
        self.declare_parameter(
            'lidar_left_angle_max_deg', LIDAR_LEFT_ANGLE_MAX_DEG
        )
        self.declare_parameter(
            'lidar_overtake_max_distance_m', LIDAR_OVERTAKE_MAX_DISTANCE_M
        )
        self.declare_parameter(
            'lidar_overtake_clear_distance_m',
            LIDAR_OVERTAKE_CLEAR_DISTANCE_M,
        )
        self.declare_parameter(
            'lidar_detect_confirm_scans', LIDAR_DETECT_CONFIRM_SCANS
        )
        self.declare_parameter(
            'lidar_clear_confirm_scans', LIDAR_CLEAR_CONFIRM_SCANS
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
        self.declare_parameter('lidar_debug_window_name', LIDAR_DEBUG_WINDOW_NAME)
        self.declare_parameter('lidar_debug_image_size', LIDAR_DEBUG_IMAGE_SIZE)
        self.declare_parameter(
            'lidar_debug_max_range_m', LIDAR_DEBUG_MAX_RANGE_M
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
        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.lidar_right_angle_min_deg = float(
            self.get_parameter('lidar_right_angle_min_deg').value
        )
        self.lidar_right_angle_max_deg = float(
            self.get_parameter('lidar_right_angle_max_deg').value
        )
        self.lidar_left_angle_min_deg = float(
            self.get_parameter('lidar_left_angle_min_deg').value
        )
        self.lidar_left_angle_max_deg = float(
            self.get_parameter('lidar_left_angle_max_deg').value
        )
        self.lidar_overtake_max_distance_m = max(
            0.01,
            float(self.get_parameter('lidar_overtake_max_distance_m').value),
        )
        self.lidar_overtake_clear_distance_m = max(
            self.lidar_overtake_max_distance_m,
            float(
                self.get_parameter('lidar_overtake_clear_distance_m').value
            ),
        )
        self.lidar_detect_confirm_scans = max(
            1, int(self.get_parameter('lidar_detect_confirm_scans').value)
        )
        self.lidar_clear_confirm_scans = max(
            1, int(self.get_parameter('lidar_clear_confirm_scans').value)
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
        self.lidar_debug_window_name = str(
            self.get_parameter('lidar_debug_window_name').value
        )
        self.lidar_debug_image_size = max(
            320, int(self.get_parameter('lidar_debug_image_size').value)
        )
        self.lidar_debug_max_range_m = max(
            self.lidar_overtake_max_distance_m,
            float(self.get_parameter('lidar_debug_max_range_m').value),
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
        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, qos)
        self.create_subscription(
            Int16, LANE_OFFSET_TOPIC, self.lane_offset_callback, 10
        )
        self.create_subscription(
            Int16,
            LANE_CHANGE_COMPLETE_TOPIC,
            self.lane_change_complete_callback,
            10,
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
        self.lane_change_armed = False
        self.detect_sample_count = 0
        self.clear_sample_count = 0
        self.lidar_object_detected = False
        self.lidar_sector_state = 'NO DATA'
        self.lidar_sector_min_distance = None
        self.lidar_sector_point_count = 0
        self.latest_scan = None
        self.pending_lane_change_target = None
        self.lane_change_counts = {(1, 2): 0, (2, 1): 0}
        # lane_offset 노드가 어느 순서로 시작해도 현재 모드를 받을 수 있도록
        # 주기적으로 lane_info를 발행한다.
        self.create_timer(0.2, self.publish_lane_info)
        self.create_timer(0.05, self.update_green_straight_mode)

        self.get_logger().info(
            f'Subscribing {LOW_IMAGE_TOPIC}, {self.scan_topic}, '
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

    def scan_callback(self, msg):
        """Use the current lane's side sector to detect an overtaken object."""
        self.latest_scan = msg
        close, clear, min_distance, point_count = self.observe_overtake_sector(msg)
        self.lidar_object_detected = close
        self.lidar_sector_state = 'CLOSE' if close else 'CLEAR' if clear else 'MID'
        self.lidar_sector_min_distance = min_distance
        self.lidar_sector_point_count = point_count
        self.update_lane_change(close, clear)
        if self.debug_view:
            self.show_lidar_debug(msg)

    def observe_overtake_sector(self, msg):
        angle_min_deg, angle_max_deg = self.active_lidar_sector_deg()
        distances = []
        usable_max = float(msg.range_max)
        for index, raw_distance in enumerate(msg.ranges):
            distance = float(raw_distance)
            if not math.isfinite(distance):
                continue
            if distance < float(msg.range_min) or distance > usable_max:
                continue
            angle = float(msg.angle_min) + index * float(msg.angle_increment)
            angle_deg = math.degrees(math.atan2(math.sin(angle), math.cos(angle)))
            if angle_min_deg <= angle_deg <= angle_max_deg:
                distances.append(distance)

        if not distances:
            return False, True, None, 0
        min_distance = min(distances)
        close = min_distance <= self.lidar_overtake_max_distance_m
        clear = min_distance > self.lidar_overtake_clear_distance_m
        return close, clear, min_distance, len(distances)

    def active_lidar_sector_deg(self):
        if self.lane_number == 2:
            return self.lidar_left_angle_min_deg, self.lidar_left_angle_max_deg
        return self.lidar_right_angle_min_deg, self.lidar_right_angle_max_deg

    def update_lane_change(self, close, clear):
        """Change lanes after a side LiDAR object is seen and then disappears."""
        if self.pending_lane_change_target is not None:
            # 초 단위 쿨다운 대신, offset 노드가 목표 차선의 색상 경계
            # 흰 실선을 확인할 때까지 다음 차선 변경 판정을 잠근다.
            self.lane_change_armed = False
            self.detect_sample_count = 0
            self.clear_sample_count = 0
            return False

        transition = (
            self.lane_number,
            1 if self.lane_number == 2 else 2,
        )
        if not self.lane_change_direction_available(transition):
            self.lane_change_armed = False
            self.detect_sample_count = 0
            self.clear_sample_count = 0
            return False

        if not self.lane_change_armed:
            if close:
                self.detect_sample_count += 1
            else:
                self.detect_sample_count = 0

            if (
                self.detect_sample_count
                >= self.lidar_detect_confirm_scans
            ):
                self.lane_change_armed = True
                self.detect_sample_count = 0
                self.clear_sample_count = 0
                sector_min, sector_max = self.active_lidar_sector_deg()
                self.get_logger().info(
                    f'{self.driving_mode}: LiDAR object confirmed in '
                    f'{sector_min:+.0f}..{sector_max:+.0f} deg within '
                    f'{self.lidar_overtake_max_distance_m:.2f} m; '
                    'lane change armed'
                )
            return False

        if clear:
            self.clear_sample_count += 1
        else:
            self.clear_sample_count = 0

        if self.clear_sample_count >= self.lidar_clear_confirm_scans:
            previous_mode = self.driving_mode
            previous_lane = self.lane_number
            self.lane_number = 1 if self.lane_number == 2 else 2
            self.lane_change_counts[(previous_lane, self.lane_number)] += 1
            self.pending_lane_change_target = self.lane_number
            self.lane_change_armed = False
            self.detect_sample_count = 0
            self.clear_sample_count = 0
            self.publish_lane_info()
            self.get_logger().info(
                'Lane change trigger: LiDAR side object disappeared; '
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

    def lane_change_complete_callback(self, msg):
        """Unlock changes after offset acquires the destination boundary line."""
        completed_lane = int(msg.data)
        if completed_lane != self.pending_lane_change_target:
            return
        self.pending_lane_change_target = None
        self.lane_change_armed = False
        self.detect_sample_count = 0
        self.clear_sample_count = 0
        self.get_logger().info(
            f'Lane {completed_lane} boundary line acquired; '
            'next lane-change detection unlocked'
        )

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
        """저해상도 카메라 위에 미션 차선과 LiDAR 추월 상태를 표시한다."""
        debug = frame.copy()
        sector_min, sector_max = self.active_lidar_sector_deg()
        panel_height = min(debug.shape[0], 225)
        panel_width = min(debug.shape[1], 640)
        overlay = debug.copy()
        cv2.rectangle(
            overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), -1
        )
        cv2.addWeighted(overlay, 0.65, debug, 0.35, 0.0, debug)

        if self.pending_lane_change_target is not None:
            state = f'WAIT LANE {self.pending_lane_change_target} BOUNDARY LINE'
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
                f'mode: {self.driving_mode}  lidar: '
                f'{sector_min:+.0f}..{sector_max:+.0f} deg',
                (0, 255, 255),
            ),
            (
                f'lidar sector: {self.lidar_sector_state}  '
                f'min={self.format_lidar_distance()}  '
                f'points={self.lidar_sector_point_count}',
                (
                    (0, 0, 255) if self.lidar_sector_state == 'CLOSE'
                    else (0, 255, 255) if self.lidar_sector_state == 'MID'
                    else (0, 255, 0)
                ),
            ),
            (
                f'state: {state}  detect={self.detect_sample_count}/'
                f'{self.lidar_detect_confirm_scans}  '
                f'clear={self.clear_sample_count}/'
                f'{self.lidar_clear_confirm_scans}',
                (0, 255, 0) if self.lane_change_armed else (255, 255, 255),
            ),
            (
                f'lidar: close<={self.lidar_overtake_max_distance_m:.2f}m  '
                f'clear>{self.lidar_overtake_clear_distance_m:.2f}m',
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

    def format_lidar_distance(self):
        if self.lidar_sector_min_distance is None:
            return '--'
        return f'{self.lidar_sector_min_distance:.2f}m'

    def show_lidar_debug(self, msg):
        """Draw all scan points and overlay only the active overtake sector."""
        size = self.lidar_debug_image_size
        center = size // 2
        scale = size * 0.44 / self.lidar_debug_max_range_m
        image = np.zeros((size, size, 3), dtype=np.uint8)

        # 현재 차선에서 쓰는 10도 구간만 표시한다. 1.0m 영역은 CLEAR
        # 판정 경계, 그 안의 0.7m 영역은 CLOSE 판정 범위다.
        angle_min_deg, angle_max_deg = self.active_lidar_sector_deg()
        overlay = image.copy()
        self.fill_lidar_sector(
            overlay,
            center,
            scale,
            angle_min_deg,
            angle_max_deg,
            self.lidar_overtake_clear_distance_m,
            (0, 120, 160),
        )
        self.fill_lidar_sector(
            overlay,
            center,
            scale,
            angle_min_deg,
            angle_max_deg,
            self.lidar_overtake_max_distance_m,
            (0, 0, 220),
        )
        cv2.addWeighted(overlay, 0.28, image, 0.72, 0.0, image)

        for distance_m, color in (
            (self.lidar_overtake_max_distance_m, (0, 80, 255)),
            (self.lidar_overtake_clear_distance_m, (0, 200, 255)),
        ):
            arc = self.lidar_arc_points(
                center, scale, angle_min_deg, angle_max_deg, distance_m
            )
            cv2.polylines(image, [arc], False, color, 2, cv2.LINE_AA)

        nearest_point = None
        nearest_distance = math.inf
        for index, raw_distance in enumerate(msg.ranges):
            distance = float(raw_distance)
            if not math.isfinite(distance):
                continue
            if distance < float(msg.range_min) or distance > min(
                float(msg.range_max), self.lidar_debug_max_range_m
            ):
                continue
            angle = float(msg.angle_min) + index * float(msg.angle_increment)
            angle = math.atan2(math.sin(angle), math.cos(angle))
            angle_deg = math.degrees(angle)
            in_sector = angle_min_deg <= angle_deg <= angle_max_deg
            point = self.lidar_pixel(center, scale, angle, distance)
            color = (100, 100, 100)
            radius = 1
            if in_sector:
                if distance <= self.lidar_overtake_max_distance_m:
                    color = (0, 0, 255)
                elif distance <= self.lidar_overtake_clear_distance_m:
                    color = (0, 255, 255)
                else:
                    color = (255, 180, 0)
                radius = 2
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_point = point
            cv2.circle(image, point, radius, color, -1)

        if nearest_point is not None:
            cv2.circle(image, nearest_point, 7, (255, 255, 255), 2)
        cv2.line(image, (center, 16), (center, size - 16), (50, 50, 50), 1)
        cv2.line(image, (16, center), (size - 16, center), (50, 50, 50), 1)
        cv2.circle(image, (center, center), 7, (0, 255, 0), -1)
        cv2.arrowedLine(
            image, (center, center), (center, 28), (0, 255, 0), 2, tipLength=0.1
        )

        lines = [
            f'{self.driving_mode}: {angle_min_deg:+.0f}..{angle_max_deg:+.0f} deg',
            f'min={self.format_lidar_distance()}  state={self.lidar_sector_state}',
            (
                f'CLOSE <= {self.lidar_overtake_max_distance_m:.2f}m  '
                f'CLEAR > {self.lidar_overtake_clear_distance_m:.2f}m'
            ),
            'REAR 0 | RIGHT +90 | FRONT +/-180 | LEFT -90',
        ]
        for index, text in enumerate(lines):
            cv2.putText(
                image, text, (14, 24 + index * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1,
                cv2.LINE_AA,
            )
        cv2.imshow(self.lidar_debug_window_name, image)
        cv2.waitKey(1)

    @staticmethod
    def lidar_pixel(center, scale, angle, distance):
        return (
            int(round(center + math.sin(angle) * distance * scale)),
            int(round(center + math.cos(angle) * distance * scale)),
        )

    def lidar_arc_points(
        self, center, scale, angle_min_deg, angle_max_deg, distance
    ):
        return np.array([
            self.lidar_pixel(center, scale, math.radians(angle_deg), distance)
            for angle_deg in np.linspace(angle_min_deg, angle_max_deg, 32)
        ], dtype=np.int32)

    def fill_lidar_sector(
        self, image, center, scale, angle_min_deg, angle_max_deg, distance, color
    ):
        arc = self.lidar_arc_points(
            center, scale, angle_min_deg, angle_max_deg, distance
        )
        polygon = np.vstack((np.array([[center, center]], dtype=np.int32), arc))
        cv2.fillPoly(image, [polygon], color)

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
