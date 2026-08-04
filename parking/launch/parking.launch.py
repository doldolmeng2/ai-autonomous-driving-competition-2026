from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def start_after_readiness(actions):
    def handle_exit(event, _context):
        if event.returncode == 0:
            return [LogInfo(msg='Sensors ready; starting parking nodes'), *actions]
        return [
            LogInfo(msg='ERROR: Sensor readiness check failed; parking will not start'),
            EmitEvent(event=Shutdown(reason='Sensor readiness check failed')),
        ]

    return handle_exit


def generate_launch_description() -> LaunchDescription:
    sensor_topic_share = Path(get_package_share_directory('sensor_topic'))
    readiness = Node(
        package='sensor_topic',
        executable='sensor_readiness_node',
        output='screen',
    )
    application_actions = [
        Node(
            package='parking',
            executable='parking_node_yym',
            name='parking_node_yym',
            output='screen',
            parameters=[{
                'debug_view': ParameterValue(
                    LaunchConfiguration('debug_view'),
                    value_type=bool,
                ),
                'start_mode': LaunchConfiguration('start_mode'),
            }],
        ),
        Node(
            package='drive_control',
            executable='drive_control_node',
            name='parking_drive_control',
            output='screen',
            parameters=[{
                'steer_pid_profile': 'parking',
            }],
        ),
    ]

    return LaunchDescription([
        DeclareLaunchArgument('debug_view', default_value='true'),
        DeclareLaunchArgument('start_mode', default_value='recognition'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(sensor_topic_share / 'launch' / 'sensors.launch.py')
            )
        ),
        RegisterEventHandler(OnProcessExit(
            target_action=readiness,
            on_exit=start_after_readiness(application_actions),
        )),
        readiness,
    ])
