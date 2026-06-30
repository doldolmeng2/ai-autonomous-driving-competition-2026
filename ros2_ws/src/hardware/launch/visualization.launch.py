from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='hardware', executable='camera_viewer_node', output='screen'),
        Node(package='hardware', executable='lidar_viewer_node', output='screen'),
        Node(package='hardware', executable='ultrasonic_viewer_node', output='screen'),
    ])
