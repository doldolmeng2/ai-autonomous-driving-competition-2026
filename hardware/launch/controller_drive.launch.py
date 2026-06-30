from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from pathlib import Path


def generate_launch_description():
    share = Path(get_package_share_directory('hardware'))
    return LaunchDescription([
        Node(
            package='hardware',
            executable='manual_controller_node',
            output='screen',
            parameters=[str(share / 'config' / 'manual_controller.yaml')],
        ),
    ])
