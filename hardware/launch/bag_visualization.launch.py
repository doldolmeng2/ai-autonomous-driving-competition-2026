from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, LogInfo
from launch_ros.actions import Node


BAG_PATH = '/home/hailab/osy/260630/ai-autonomous-driving-competition-2026/rosbag2_2026_07_01-13_36_34'


def generate_launch_description():
    share = Path(get_package_share_directory('hardware'))
    bag_path = BAG_PATH.strip()

    if not bag_path:
        return LaunchDescription([
            LogInfo(
                msg='Set BAG_PATH in hardware/launch/bag_visualization.launch.py first.'
            ),
        ])

    return LaunchDescription([
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', bag_path],
            output='screen',
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
