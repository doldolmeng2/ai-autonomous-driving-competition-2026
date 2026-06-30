2026 전국 대학생 AI 자율주행 경진대회

## Build

```bash
cd /home/hailab/osy/260630/ai-autonomous-driving-competition-2026
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

특정 패키지만 빌드하려면:

```bash
colcon build --symlink-install --packages-select hardware
source install/setup.bash
```

## Hardware Launch

센서 토픽 발행:

```bash
ros2 launch hardware sensors.launch.py
```

발행 토픽:

```text
/camera/left/image_raw
/camera/left/camera_info
/camera/right/image_raw
/camera/right/camera_info
/scan
/ultrasonic/front/range
/ultrasonic/left/range
/ultrasonic/right/range
/ultrasonic/ranges
```

센서 시각화:

```bash
ros2 launch hardware visualization.launch.py
```

컨트롤러 수신기 토픽 발행:

```bash
ros2 launch hardware controller_drive.launch.py
```

발행 토픽:

```text
/manual_controller/joy
```

센서 발행과 rosbag 기록:

```bash
ros2 launch hardware sensors_bag.launch.py
```

센서, 컨트롤러, rosbag 기록:

```bash
ros2 launch hardware sensors_controller_bag.launch.py
```

카메라 캘리브레이션 이미지 저장:

```bash
ros2 launch hardware camera_calibration.launch.py
```

카메라 위치/각도 확인:

```bash
ros2 launch hardware camera_pose_check.launch.py
```

## Device Notes

RPLidar A1이 `/dev/ttyUSB0`에서 `Permission denied`가 나면:

```bash
sudo usermod -aG dialout $USER
```

그 다음 로그아웃/로그인 후 다시 실행한다. 로그아웃 전 현재 터미널에서 임시로 테스트하려면:

```bash
sg dialout -c "ros2 launch hardware sensors.launch.py"
```

컨트롤러 수신기는 기본값으로 `/dev/input/js*`를 자동 탐색한다. 현재 테스트된 장치는:

```text
Xbox 360 Wireless Receiver -> /dev/input/js0
```
