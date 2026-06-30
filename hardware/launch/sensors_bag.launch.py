from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    share = Path(get_package_share_directory('hardware'))
    return LaunchDescription([
        Node(
            package='hardware',
            executable='camera_node',
            output='screen',
            parameters=[str(share / 'config' / 'camera.yaml')],
        ),
        Node(
            package='hardware',
            executable='lidar_node',
            output='screen',
            parameters=[str(share / 'config' / 'lidar.yaml')],
        ),
        Node(
            package='hardware',
            executable='ultrasonic_node',
            output='screen',
            parameters=[str(share / 'config' / 'ultrasonic.yaml')],
        ),
        ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-a'],
            output='screen',
        ),
    ])
