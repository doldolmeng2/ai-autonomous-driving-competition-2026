from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    EmitEvent,
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def start_after_readiness(actions):
    def handle_exit(event, _context):
        if event.returncode == 0:
            return [LogInfo(msg='Sensors ready; starting timed lane nodes'), *actions]
        return [
            LogInfo(msg='ERROR: Sensor readiness check failed; timed lane will not start'),
            EmitEvent(event=Shutdown(reason='Sensor readiness check failed')),
        ]

    return handle_exit


def generate_launch_description():
    sensor_topic_share = Path(get_package_share_directory('sensor_topic'))
    readiness = Node(
        package='sensor_topic',
        executable='sensor_readiness_node',
        output='screen',
        parameters=[{
            'required_topics': [
                '/scan',
                '/camera/high/image_raw',
            ],
        }],
    )
    application_actions = [
        Node(
            package='lane_offset',
            executable='timed_lane_offset_node',
            output='screen',
        ),
        Node(
            package='lane_main',
            executable='timed_lane_main_node',
            output='screen',
        ),
        Node(
            package='drive_control',
            executable='drive_control_node',
            output='screen',
            parameters=[{
                'steer_pid_profile': 'timed',
            }],
        ),
    ]

    return LaunchDescription([
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
