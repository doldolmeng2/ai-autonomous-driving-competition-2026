from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='hardware', executable='camera_pose_check_node', output='screen'),
    ])
