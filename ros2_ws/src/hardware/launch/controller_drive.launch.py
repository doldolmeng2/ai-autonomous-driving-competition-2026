from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='hardware', executable='manual_controller_node', output='screen'),
        Node(package='hardware', executable='motor_serial_node', output='screen'),
    ])
