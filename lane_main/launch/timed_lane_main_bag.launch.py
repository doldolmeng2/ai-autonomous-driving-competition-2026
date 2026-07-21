"""rosbag 재생으로 차선주행 파이프라인을 시각화(디버그 뷰) 확인하는 런치.

실제 카메라/구동 하드웨어 없이 bag만으로 lane_offset -> lane_main 이 잘 도는지
OpenCV 디버그 창으로 눈으로 확인하기 위한 용도.

  - camera_node / drive_control 은 실행하지 않는다(하드웨어 불필요).
  - bag의 /camera/left/image_raw 를 노드가 구독하는 /camera/high/image_raw 로 remap.
    (이 팀 셋업에서 left 카메라 = high 카메라)
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


REPO_DIR = '/home/hailab/osy/260711/ai-autonomous-driving-competition-2026'
DEFAULT_BAG = str(Path(REPO_DIR) / 'rosbag2_2026_07_01-15_30_56')


def generate_launch_description():
    bag_arg = DeclareLaunchArgument('bag', default_value=DEFAULT_BAG)
    rate_arg = DeclareLaunchArgument('rate', default_value='1.0')
    loop_arg = DeclareLaunchArgument('loop', default_value='true')

    bag = LaunchConfiguration('bag')
    rate = LaunchConfiguration('rate')
    loop_flag = PythonExpression([
        "'--loop' if '", LaunchConfiguration('loop'), "'.lower() in ('true','1') else ''",
    ])

    return LaunchDescription([
        bag_arg,
        rate_arg,
        loop_arg,
        LogInfo(msg=['Playing bag: ', bag]),
        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'play', bag,
                '--rate', rate,
                loop_flag,
                '--remap', '/camera/left/image_raw:=/camera/high/image_raw',
            ],
            output='screen',
        ),
        # 카메라(left=high) -> 중앙 점선 기준 offset 계산 + OpenCV 디버그 창
        Node(
            package='lane_offset',
            executable='timed_lane_offset_node_osy',
            output='screen',
            parameters=[{'debug_view': True}],
        ),
        # offset -> steer/speed 계산 (/motor_control 발행, 하드웨어로는 안 감)
        Node(
            package='lane_main',
            executable='timed_lane_main_node',
            output='screen',
            parameters=[{'base_speed': 130, 'max_steer': 45}],
        ),
    ])
