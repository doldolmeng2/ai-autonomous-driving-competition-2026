from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    hardware_share = Path(get_package_share_directory('hardware'))

    serial_port = LaunchConfiguration('serial_port')

    return LaunchDescription([
        DeclareLaunchArgument(
            'serial_port',
            default_value='auto',
            description='Arduino serial port. Use auto, /dev/ttyACM0, or /dev/ttyUSB0.',
        ),
        Node(
            package='hardware',
            executable='manual_controller_node',
            output='screen',
            parameters=[str(hardware_share / 'config' / 'manual_controller.yaml')],
        ),
        # 조이스틱(Joy) -> 공통 제어 토픽(/motor_control) 변환
        Node(
            package='hardware',
            executable='joy_to_motor_node',
            output='screen',
            parameters=[{
                'joy_topic': '/manual_controller/joy',
                'motor_control_topic': '/motor_control',
                'steer_axis': 3,
                'drive_axis': 1,
                'invert_steer_axis': False,
                'invert_drive_axis': True,
                'deadzone': 0.2,
                'max_speed': 130,
                'max_steer': 45,
            }],
        ),
        Node(
            package='drive_control',
            executable='drive_control_node',
            output='screen',
            parameters=[{
                'motor_control_topic': '/motor_control',
                'serial_port': serial_port,
                'baudrate': 115200,
                'command_rate': 20.0,
                'command_resend_interval': 0.1,
                'input_timeout': 0.5,
                'arduino_boot_delay': 5.0,
                'enable_arduino_debug_log': False,
                'enable_tx_debug_log': False,
                'max_drive_pwm': 130,
                'steer_pwm': 150,
                'steer_max_angle_deg': 45.0,
                'steer_center_time': 0.45,
                'steer_angle_tolerance_deg': 1.0,
            }],
        ),
    ])
