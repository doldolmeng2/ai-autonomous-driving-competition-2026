from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    share = Path(get_package_share_directory('hardware'))
    sllidar_share = Path(get_package_share_directory('sllidar_ros2'))
    return LaunchDescription([
        Node(
            package='hardware',
            executable='camera_node',
            output='screen',
            parameters=[str(share / 'config' / 'camera.yaml')],
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
            package='hardware',
            executable='manual_controller_node',
            output='screen',
            parameters=[str(share / 'config' / 'manual_controller.yaml')],
        ),
        Node(package='hardware', executable='camera_viewer_node', output='screen'),
        Node(package='hardware', executable='lidar_viewer_node', output='screen'),
        Node(
            package='hardware',
            executable='ultrasonic_viewer_node',
            output='screen',
            parameters=[str(share / 'config' / 'ultrasonic.yaml')],
        ),
        Node(package='hardware', executable='controller_viewer_node', output='screen'),
    ])
