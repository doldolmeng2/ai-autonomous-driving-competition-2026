# sensor_utils

시각화, 캘리브레이션, rosbag, 컨트롤러 변환 유틸 패키지다. 센서 원본 토픽은
`sensor_topic`에서 발행하고, 이 패키지는 그 토픽을 구독해 확인/변환만 한다.

| Node | Subscribe | Publish |
| --- | --- | --- |
| `camera_viewer_node` | `/camera/high/image_raw`, `/camera/low/image_raw` | OpenCV windows |
| `camera_calibration_node` | `/camera/high/image_raw`, `/camera/low/image_raw` | calibration image files |
| `camera_pose_check_node` | `/camera/high/image_raw`, `/camera/low/image_raw` | OpenCV window |
| `hsv_tuner_node` | camera image topic | OpenCV HSV tuner |
| `ycrcb_tuner_node` | camera image topic | OpenCV YCrCb tuner |
| `hsv_ycrcb_tuner_node` | camera image topic | OpenCV HSV/YCrCb AND-mask tuner |
| `lidar_viewer_node` | `/scan` | OpenCV radar window |
| `ultrasonic_viewer_node` | `/ultrasonic/range_1` ... `/ultrasonic/range_6` | OpenCV range window |
| `controller_viewer_node` | `/manual_controller/joy` | OpenCV controller window |
| `joy_to_motor_node` | `/manual_controller/joy` | `/motor_control` |

`lidar_viewer_node` uses rear 0°, right +90°, front ±180°, and left -90°.

HSV와 YCrCb 임계값을 한번에 조절하며 각 마스크와 AND 마스크를
확인하려면 다음과 같이 실행한다. `preset`은 `white`, `green`,
`light_gray`, `dark_gray`, `full` 중 하나다.

```bash
ros2 run sensor_utils hsv_ycrcb_tuner_node --ros-args \
  -p image_topic:=/camera/high/image_raw -p preset:=white
```

Useful launches:

```bash
ros2 launch sensor_utils sensor_visualization.launch.py
ros2 launch sensor_utils camera_calibration.launch.py
ros2 launch sensor_utils camera_pose_check.launch.py
ros2 launch sensor_utils sensors_bag.launch.py
ros2 launch sensor_utils sensors_controller_bag.launch.py
ros2 launch sensor_utils bag_visualization.launch.py
```

`sensors_controller_bag.launch.py`는 전체 센서와 컨트롤러를 기록하면서
조이스틱 수동 주행 연결도 함께 실행한다.

```text
/manual_controller/joy
  -> joy_to_motor_node
  -> /motor_control
  -> drive_control_node
  -> /arduino/motor_command
  -> arduino_communication_node
  -> Arduino serial
```

기본 축은 오른쪽 스틱 좌우 `axes[3]`(조향), 왼쪽 스틱 상하
`axes[1]`(주행)이며, 조향각은 ±45도, 주행 PWM은 ±130으로 제한된다.
컨트롤러 축 배열은 기종에 따라 다를 수 있으므로 바퀴를 띄운 상태에서
`ros2 topic echo /manual_controller/joy`로 먼저 확인한다.
