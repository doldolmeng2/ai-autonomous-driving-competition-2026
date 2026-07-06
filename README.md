2026 전국 대학생 AI 자율주행 경진대회

## Build

ROS2 Humble 환경을 먼저 불러온 뒤 `colcon`으로 워크스페이스를 빌드한다.

```bash
cd /home/hailab/osy/260630/ai-autonomous-driving-competition-2026
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

특정 패키지만 빌드하려면:

```bash
colcon build --symlink-install --packages-select sensor_topic sensor_utils
source install/setup.bash
```

컨트롤러/아두이노 드라이브 패키지만 빌드하려면:

```bash
colcon build --symlink-install --packages-select drive_control
source install/setup.bash
```

## ROS2 Run / Launch

ROS1의 `rosrun`, `roslaunch` 대신 ROS2에서는 아래 명령을 사용한다.

```bash
ros2 run <패키지명> <실행파일명>
ros2 launch <패키지명> <launch파일명>
```

예시:

```bash
ros2 run sensor_topic controller_node
ros2 run drive_control drive_control_node
ros2 launch drive_control controller_drive.launch.py
```

launch 파일을 수정했거나 새로 옮겼으면 다시 빌드 후 `source install/setup.bash`를 실행해야 한다.

## Hardware Launch

센서 토픽 발행:

```bash
ros2 launch sensor_topic sensors.launch.py
```

라이다만 단독 실행:

```bash
ros2 launch sllidar_ros2 sllidar_a1_launch.py
```

`sensor_topic`의 센서 관련 launch는 내부에서 위 Slamtec A1 launch를 불러와 `/scan`을 발행한다.

발행 토픽:

```text
/camera/left/image_raw
/camera/left/camera_info
/camera/right/image_raw
/camera/right/camera_info
/scan
/ultrasonic/range_1
/ultrasonic/range_2
/ultrasonic/range_3
/ultrasonic/range_4
/ultrasonic/range_5
/ultrasonic/range_6
/ultrasonic/ranges
```

센서 시각화:

```bash
ros2 launch sensor_utils sensor_visualization.launch.py
```

컨트롤러 수신기 토픽 발행 + 아두이노 모터 제어:

```bash
ros2 launch drive_control controller_drive.launch.py
```

실행되는 노드:

```text
sensor_topic/controller_node -> /controller/joy 발행
drive_control/drive_control_node -> /controller/joy 구독 후 Arduino serial 송신
```

아두이노 포트가 자동 탐색되지 않으면 직접 지정한다.

```bash
ros2 launch drive_control controller_drive.launch.py serial_port:=/dev/ttyACM0
```

아두이노에는 `steer drive` 형식의 두 정수를 115200 baud로 보낸다. 현재 안전값은 다음과 같다.

```text
속도: -30 ~ 30 PWM
조향: -40 또는 40 PWM을 1초만 송신, 이후 0
```

센서 발행과 rosbag 기록:

```bash
ros2 launch sensor_utils sensors_bag.launch.py
```

센서, 컨트롤러, rosbag 기록:

```bash
ros2 launch sensor_utils sensors_controller_bag.launch.py
```

카메라 캘리브레이션 이미지 저장:

```bash
ros2 launch sensor_utils camera_calibration.launch.py
```

카메라 위치/각도 확인:

```bash
ros2 launch sensor_utils camera_pose_check.launch.py
```

## Device Notes

RPLidar A1이 `/dev/ttyUSB0`에서 `Permission denied`가 나면:

```bash
sudo usermod -aG dialout $USER
```

그 다음 로그아웃/로그인 후 다시 실행한다. 로그아웃 전 현재 터미널에서 임시로 테스트하려면:

```bash
sg dialout -c "ros2 launch sensor_topic sensors.launch.py"
```

컨트롤러 수신기는 기본값으로 `/dev/input/js*`를 자동 탐색한다. 현재 테스트된 장치는:

```text
Xbox 360 Wireless Receiver -> /dev/input/js0
```
