from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    sensor_topic_share = Path(get_package_share_directory('sensor_topic'))
    sllidar_share = Path(get_package_share_directory('sllidar_ros2'))

    return LaunchDescription([
        Node(
            package='sensor_topic',
            executable='camera_node',
            output='screen',
            parameters=[str(sensor_topic_share / 'config' / 'camera.yaml')],
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(sllidar_share / 'launch' / 'sllidar_a1_launch.py')
            ),
            launch_arguments={
                'serial_port': '/dev/ttyUSB0',
                'serial_baudrate': '115200',
                'frame_id': 'laser',
            }.items(),
        ),
        Node(
            package='sensor_topic',
            executable='ultrasonic_node',
            output='screen',
            parameters=[str(sensor_topic_share / 'config' / 'ultrasonic.yaml')],
        ),
        Node(package='parking', executable='parking_node', output='screen'),
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
