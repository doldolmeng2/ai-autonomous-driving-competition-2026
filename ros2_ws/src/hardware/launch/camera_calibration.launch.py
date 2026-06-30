from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='hardware', executable='camera_calibration_node', output='screen'),
    ])
