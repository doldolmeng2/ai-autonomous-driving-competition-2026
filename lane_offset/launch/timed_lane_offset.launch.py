from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    hardware_share = Path(get_package_share_directory('hardware'))

    return LaunchDescription([
        Node(
            package='hardware',
            executable='camera_node',
            output='screen',
            parameters=[str(hardware_share / 'config' / 'camera.yaml')],
        ),
        Node(
            package='lane_offset',
            executable='timed_lane_offset_node',
            output='screen',
        ),
    ])
