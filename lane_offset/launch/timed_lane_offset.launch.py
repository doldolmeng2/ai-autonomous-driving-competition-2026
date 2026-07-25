"""timed_lane_offset.launch.py — 인식/시각화 전용(모터 미구동).

현재 카메라로 ROI/사다리꼴/중앙 점선 기준선/슬라이딩 윈도우/검출 x좌표와
흰색·초록·회색 마스크 디버그 창만 띄운다. lane_main / drive_control 은 실행하지
않으므로 /motor_control 이 발행되지 않아 차는 움직이지 않는다.

실행:
    ros2 launch lane_offset timed_lane_offset.launch.py
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    sensor_topic_share = Path(get_package_share_directory('sensor_topic'))

    return LaunchDescription([
        # 카메라(high/low) 발행
        Node(
            package='sensor_topic',
            executable='camera_node',
            output='screen',
            parameters=[str(sensor_topic_share / 'config' / 'camera.yaml')],
        ),
        # 차선 인식 + OpenCV 디버그 창(ROI/윈도우/마스크). 모터 명령 없음.
        Node(
            package='lane_offset',
            executable='timed_lane_offset_node',
            output='screen',
            parameters=[{'debug_view': True}],
        ),
    ])
