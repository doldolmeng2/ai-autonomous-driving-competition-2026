"""timed_lane_main_ngg.launch.py — 오른쪽 실선 기준(ngg) 시간주행.

timed_lane_main.launch.py 와 동일한 주행 체인이지만, 차선 인식 노드만
timed_lane_offset_node -> timed_lane_offset_node_ngg 로 바꾼 것이다.

체인:
    camera_node -> /camera/high/image_raw
      -> timed_lane_offset_node_ngg  : 오른쪽 실선을 target_right_x 에 맞춰 /lane_offset
      -> timed_lane_main_node        : /lane_offset -> /motor_control (steer, speed)
      -> drive_control_node          : /motor_control -> Arduino

주의:
    /lane_offset 은 공용 토픽이라 기존 timed_lane_offset_node 와 **동시에 띄우면
    안 된다**. 둘 다 발행하면 조향이 뒤섞인다.

실행:
    ros2 launch lane_main timed_lane_main_ngg.launch.py
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    sensor_topic_share = Path(get_package_share_directory('sensor_topic'))

    return LaunchDescription([
        Node(
            package='sensor_topic',
            executable='camera_node',
            output='screen',
            parameters=[str(sensor_topic_share / 'config' / 'camera.yaml')],
        ),
        # 카메라(high) -> 오른쪽 실선 기준 offset 계산 (새 top-down 카메라용)
        Node(
            package='lane_offset',
            executable='timed_lane_offset_node_ngg',
            output='screen',
            # 기준 x / 근접 밴드 / 실선 마스킹을 눈으로 확인할 수 있게 디버그 창을 띄운다.
            parameters=[{
                'debug_view': True,
            }],
        ),
        # offset -> steer/speed 계산
        Node(
            package='lane_main',
            executable='timed_lane_main_node',
            output='screen',
            parameters=[{
                'base_speed': 130,
                'max_steer': 45,
            }],
        ),
        Node(
            package='drive_control',
            executable='drive_control_node',
            output='screen',
            parameters=[{
                'max_drive_pwm': 110,
                'steer_pwm': 150,
                'steer_max_angle_deg': 45.0,
                'steer_center_time': 0.45,
                'steer_angle_tolerance_deg': 1.0,
            }],
        ),
    ])
