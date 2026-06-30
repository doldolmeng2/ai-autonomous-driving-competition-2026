from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='lane_main', executable='timed_lane_main_node', output='screen'),
    ])
