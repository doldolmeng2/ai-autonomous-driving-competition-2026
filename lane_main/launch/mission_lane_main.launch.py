from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def application_actions(use_hardware):
    actions = [
        Node(
            package='lane_offset',
            executable='mission_lane_offset_node',
            output='screen',
        ),
        Node(
            package='lane_main',
            executable='mission_lane_main_node',
            output='screen',
        ),
    ]
    if use_hardware:
        actions.append(Node(
            package='drive_control',
            executable='drive_control_node',
            output='screen',
            parameters=[{
                'steer_pid_profile': 'mission',
            }],
        ))
    return actions


def start_after_readiness(actions):
    def handle_exit(event, _context):
        if event.returncode == 0:
            return [LogInfo(msg='Sensors ready; starting mission nodes'), *actions]
        return [
            LogInfo(msg='ERROR: Sensor readiness check failed; mission will not start'),
            EmitEvent(event=Shutdown(reason='Sensor readiness check failed')),
        ]

    return handle_exit


def launch_setup(context):
    use_hardware = IfCondition(LaunchConfiguration('use_hardware')).evaluate(context)
    actions = application_actions(use_hardware)
    if not use_hardware:
        return [LogInfo(msg='Hardware disabled; starting mission nodes'), *actions]

    sensor_topic_share = Path(get_package_share_directory('sensor_topic'))
    readiness = Node(
        package='sensor_topic',
        executable='sensor_readiness_node',
        output='screen',
    )
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(sensor_topic_share / 'launch' / 'sensors.launch.py')
            )
        ),
        RegisterEventHandler(OnProcessExit(
            target_action=readiness,
            on_exit=start_after_readiness(actions),
        )),
        readiness,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_hardware', default_value='true'),
        OpaqueFunction(function=launch_setup),
    ])
