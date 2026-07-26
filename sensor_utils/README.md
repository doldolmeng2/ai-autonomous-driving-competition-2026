# sensor_utils

시각화, 캘리브레이션, rosbag, 컨트롤러 변환 유틸 패키지다. 센서 원본 토픽은
`sensor_topic`에서 발행하고, 이 패키지는 그 토픽을 구독해 확인/변환만 한다.

| Node | Subscribe | Publish |
| --- | --- | --- |
| `camera_viewer_node` | `/camera/high/image_raw`, `/camera/low/image_raw` | OpenCV windows |
| `camera_calibration_node` | `/camera/high/image_raw`, `/camera/low/image_raw` | calibration image files |
| `camera_pose_check_node` | `/camera/high/image_raw`, `/camera/low/image_raw` | OpenCV window |
| `hsv_tuner_node` | camera image topic (또는 bag 파일 직접 열기) | OpenCV HSV tuner |
| `lidar_viewer_node` | `/scan` | OpenCV radar window |
| `ultrasonic_viewer_node` | `/ultrasonic/range_1` ... `/ultrasonic/range_6` | OpenCV range window |
| `controller_viewer_node` | `/manual_controller/joy` | OpenCV controller window |
| `joy_to_motor_node` | `/manual_controller/joy` | `/motor_control` |

Useful launches:

```bash
ros2 launch sensor_utils sensor_visualization.launch.py
ros2 launch sensor_utils camera_calibration.launch.py
ros2 launch sensor_utils camera_pose_check.launch.py
ros2 launch sensor_utils sensors_bag.launch.py
ros2 launch sensor_utils sensors_controller_bag.launch.py
ros2 launch sensor_utils bag_visualization.launch.py
```

## HSV 튜닝 (bag 모드)

`bag_path`를 주면 `ros2 bag play` 없이 bag을 직접 열어 동영상 플레이어처럼 다룬다.
재생바를 드래그하면 그 프레임에 멈춘 채로 HSV 트랙바를 조절할 수 있다.

```bash
ros2 run sensor_utils hsv_tuner_node --ros-args \
  -p bag_path:=/home/gill/bags/rosbag2_2026_07_25-22_08_25 \
  -p image_topic:=/camera/high/image_raw \
  -p preset:=white
```

| 키 | 동작 |
| --- | --- |
| `space` | 재생 / 일시정지 |
| `a`, `d` | 이전 / 다음 프레임 (자동으로 일시정지) |
| `j`, `l` | 1초 뒤로 / 1초 앞으로 |
| `r` | 처음으로 |
| `s` | 현재 HSV 값을 터미널에 출력 (`PRESETS`에 붙여넣는 형태) |
| `q`, `ESC` | 종료 |

`preset`은 `white` / `green` / `full`, 재생 속도는 컨트롤 창의 `speed %` 트랙바로 조절한다.
