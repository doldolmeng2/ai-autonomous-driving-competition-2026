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
        # 카메라(high) -> 오른쪽 차선 기준 offset 계산
        Node(
            package='lane_offset',
            executable='timed_lane_offset_node_osy',
            output='screen',
            # OpenCV 차선/마스크 디버그 창을 항상 띄운다.
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
