from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def find_lidar_port():
    """Prefer stable by-id LiDAR path; never select the ultrasonic Arduino."""
    by_id = Path('/dev/serial/by-id')
    if by_id.is_dir():
        for device in sorted(by_id.iterdir()):
            if 'arduino' not in device.name.lower():
                return str(device)
    ports = sorted(Path('/dev').glob('ttyUSB*'))
    return str(ports[0]) if ports else None


def start_lidar(context, *, sllidar_share):
    requested = LaunchConfiguration('lidar_serial_port').perform(context)
    port = find_lidar_port() if requested == 'auto' else requested
    if not port:
        return [LogInfo(msg='[parking] LiDAR device not found; sllidar_node skipped.')]
    return [
        LogInfo(msg=f'[parking] Starting SLLidar on {port}'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(sllidar_share / 'launch' / 'sllidar_a1_launch.py')
            ),
            launch_arguments={
                'serial_port': port,
                'serial_baudrate': LaunchConfiguration('lidar_baudrate').perform(context),
                'frame_id': 'laser',
            }.items(),
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    sensor_topic_share = Path(get_package_share_directory('sensor_topic'))
    sllidar_share = Path(get_package_share_directory('sllidar_ros2'))
    return LaunchDescription([
        DeclareLaunchArgument('lidar_serial_port', default_value='auto'),
        DeclareLaunchArgument('lidar_baudrate', default_value='115200'),

        # Parking owns the sensor drivers it needs: high camera, LiDAR, and
        # ultrasonic. Low camera/controller are deliberately not started.
        Node(
            package='sensor_topic',
            executable='camera_node',
            name='parking_high_camera',
            output='screen',
            parameters=[
                str(sensor_topic_share / 'config' / 'camera.yaml'),
                {'enable_high': True, 'enable_low': False},
            ],
        ),
        OpaqueFunction(function=start_lidar, kwargs={'sllidar_share': sllidar_share}),
        Node(
            package='sensor_topic',
            executable='ultrasonic_node',
            name='parking_ultrasonic',
            output='screen',
            parameters=[str(sensor_topic_share / 'config' / 'ultrasonic.yaml')],
        ),
        Node(
            package='parking',
            executable='parking_node_osy',
            name='parking_node_osy',
            output='screen',
            parameters=[{'debug_view': True}],
        ),
        # /motor_control [steer, speed] -> Arduino steering/drive commands.
        Node(
            package='drive_control',
            executable='drive_control_node',
            name='parking_drive_control',
            output='screen',
            parameters=[{
                'max_drive_pwm': 130,
                'steer_pwm': 150,
                'steer_max_angle_deg': 45.0,
                'steer_center_time': 0.45,
                'steer_angle_tolerance_deg': 1.0,
            }],
        ),
        Node(
            package='sensor_utils',
            executable='lidar_viewer_node',
            name='parking_lidar_viewer',
            output='screen',
            parameters=[{'max_range_m': 2.0, 'rear_quadrants_only': True}],
        ),
    ])
