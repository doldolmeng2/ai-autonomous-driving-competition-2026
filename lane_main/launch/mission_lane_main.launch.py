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
        Node(
            package='sensor_topic',
            executable='ultrasonic_node',
            output='screen',
            parameters=[str(sensor_topic_share / 'config' / 'ultrasonic.yaml')],
        ),
        Node(
            package='lane_offset',
            executable='mission_lane_offset_node',
            output='screen',
        ),
        Node(package='lane_main', executable='mission_lane_main_node', output='screen'),
        Node(
            package='drive_control',
            executable='drive_control_node',
            output='screen',
            parameters=[{
                'max_drive_pwm': 130,
                'steer_pwm': 150,
                'steer_max_angle_deg': 45.0,
                'steer_center_time': 0.45,
                'steer_angle_tolerance_deg': 1.0,
            }],
        ),
    ])
