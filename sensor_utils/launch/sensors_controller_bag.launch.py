from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Sensor drivers and the controller are owned by sensors.launch.py.
        # /manual_controller/joy -> /motor_control [steer angle, drive PWM]
        Node(
            package='sensor_utils',
            executable='joy_to_motor_node',
            output='screen',
            parameters=[{
                'steer_axis': 3,
                'drive_axis': 1,
                'invert_steer_axis': False,
                'invert_drive_axis': True,
                'deadzone': 0.2,
                'max_speed': 130,
                'max_steer': 45,
            }],
        ),
        # /motor_control -> /arduino/motor_command [steer PWM, drive PWM]
        Node(
            package='drive_control',
            executable='drive_control_node',
            output='screen',
            parameters=[{'steer_pid_profile': 'timed'}],
        ),
        ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-a'],
            output='screen',
        ),
    ])
