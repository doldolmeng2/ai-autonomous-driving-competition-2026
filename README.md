2026 전국 대학생 AI 자율주행 경진대회

## Build

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

패키지별 빌드:

```bash
colcon build --symlink-install --packages-select sensor_topic sensor_utils
colcon build --symlink-install --packages-select drive_control
colcon build --symlink-install --packages-select lane_offset lane_main parking
```

## PDF Topic Flow

```text
sensor_topic/camera_node -> /camera/high/image_raw
sensor_topic/camera_node -> /camera/low/image_raw
sensor_topic/ultrasonic_node -> /ultrasonic/range_1 ... /ultrasonic/range_6
sllidar_ros2 -> /scan
sensor_topic/controller_node -> /manual_controller/joy
```

Timed lane driving:

```text
/camera/high/image_raw
-> lane_offset/timed_lane_offset_node
-> /lane_offset
-> lane_main/timed_lane_main_node
-> /motor_control
-> drive_control/drive_control_node
```

Mission lane driving:

```text
/camera/low/image_raw + /ultrasonic/range_1 ... /range_6 + /lane_offset
-> lane_main/mission_lane_main_node
-> /lane_info + /motor_control

/lane_info + /camera/high/image_raw
-> lane_offset/mission_lane_offset_node
-> /lane_offset
```

Parking:

```text
/camera/high/image_raw + /ultrasonic/range_1 ... /range_6 + /scan
-> parking/parking_node
-> /motor_control
-> drive_control/drive_control_node
```

Controller drive:

```text
/manual_controller/joy
-> sensor_utils/joy_to_motor_node
-> /motor_control
-> drive_control/drive_control_node
```

## Launch

센서 토픽 발행:

```bash
ros2 launch sensor_topic sensors.launch.py
```

수동 주행:

```bash
ros2 launch drive_control controller_drive.launch.py
```

시간주행:

```bash
ros2 launch lane_main timed_lane_main.launch.py
```

차선 미션:

```bash
ros2 launch lane_main mission_lane_main.launch.py
```

주차:

```bash
ros2 launch parking mission_parking.launch.py
```

센서 시각화/캘리브레이션/bag:

```bash
ros2 launch sensor_utils sensor_visualization.launch.py
ros2 launch sensor_utils camera_calibration.launch.py
ros2 launch sensor_utils camera_pose_check.launch.py
ros2 launch sensor_utils sensors_bag.launch.py
ros2 launch sensor_utils sensors_controller_bag.launch.py
ros2 launch sensor_utils bag_visualization.launch.py
```

## Topics

| Topic | Type |
| --- | --- |
| `/camera/high/image_raw` | `sensor_msgs/Image` |
| `/camera/high/camera_info` | `sensor_msgs/CameraInfo` |
| `/camera/low/image_raw` | `sensor_msgs/Image` |
| `/camera/low/camera_info` | `sensor_msgs/CameraInfo` |
| `/scan` | `sensor_msgs/LaserScan` |
| `/ultrasonic/range_1` ... `/ultrasonic/range_6` | `sensor_msgs/Range` |
| `/manual_controller/joy` | `sensor_msgs/Joy` |
| `/lane_info` | `std_msgs/Int16` |
| `/lane_offset` | `std_msgs/Int16` |
| `/motor_control` | `std_msgs/Int16MultiArray` |

## Device Notes

RPLidar A1이 `/dev/ttyUSB0`에서 `Permission denied`가 나면:

```bash
sudo usermod -aG dialout $USER
```

그 다음 로그아웃/로그인 후 다시 실행한다.
