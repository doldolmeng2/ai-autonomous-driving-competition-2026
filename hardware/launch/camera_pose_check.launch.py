from launch import LaunchDescription
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
            executable='camera_pose_check_node',
            output='screen',
        ),
    ])
