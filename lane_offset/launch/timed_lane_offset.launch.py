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
            package='lane_offset',
            executable='timed_lane_offset_node',
            output='screen',
        ),
    ])
