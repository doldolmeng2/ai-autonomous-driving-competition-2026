import math

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
DRIVING_MODE = '1lane'
LANE_CHANGE_CLOSE_DISTANCE = 0.6
LANE_CHANGE_CLEAR_DISTANCE = 1.0
LANE_CHANGE_CLOSE_CONFIRM_SAMPLES = 5
LANE_CHANGE_CLEAR_CONFIRM_SAMPLES = 3


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
        self.ultrasonic_ranges = {}
        self.lane_change_armed = False
        self.close_sample_count = 0
        self.clear_sample_count = 0
        # lane_offset 노드가 어느 순서로 시작해도 현재 모드를 받을 수 있도록
        # 주기적으로 lane_info를 발행한다.
        self.create_timer(0.2, self.publish_lane_info)

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

    def ultrasonic_callback(self, msg, sensor_index):
        distance = float(msg.range)
        self.ultrasonic_ranges[sensor_index] = distance
        self.update_lane_change(sensor_index, distance)

    def update_lane_change(self, sensor_index, distance):
        """연속 근접 후 연속 이탈한 물체를 추월한 것으로 보고 차선을 전환한다."""
        active_sensor = 3 if self.lane_number == 2 else 4
        if sensor_index != active_sensor:
            return False

        # timeout/no-echo는 ultrasonic_node에서 inf로 정규화된다. 센서 오류를
        # "1m 밖으로 이탈"한 것으로 오인하지 않도록 유한한 측정만 사용한다.
        if not math.isfinite(distance) or distance < 0.0:
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
            self.lane_number = 1 if self.lane_number == 2 else 2
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

    def lane_offset_callback(self, msg):
        offset = int(msg.data)
        steer = offset
        steer = max(-self.max_steer, min(self.max_steer, steer))

        command = Int16MultiArray()
        command.data = [steer, self.base_speed]
        self.motor_pub.publish(command)

    def publish_lane_info(self):
        lane_info = Int16()
        lane_info.data = self.lane_number
        self.lane_info_pub.publish(lane_info)


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
