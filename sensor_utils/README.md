# sensor_utils

시각화, 캘리브레이션, rosbag, 컨트롤러 변환 유틸 패키지다. 센서 원본 토픽은
`sensor_topic`에서 발행하고, 이 패키지는 그 토픽을 구독해 확인/변환만 한다.

| Node | Subscribe | Publish |
| --- | --- | --- |
| `camera_viewer_node` | `/camera/high/image_raw`, `/camera/low/image_raw` | OpenCV windows |
| `camera_calibration_node` | `/camera/high/image_raw`, `/camera/low/image_raw` | calibration image files |
| `camera_pose_check_node` | `/camera/high/image_raw`, `/camera/low/image_raw` | OpenCV window |
| `hsv_tuner_node` | camera image topic | OpenCV HSV tuner |
| `lidar_viewer_node` | `/scan` | OpenCV radar window |
| `ultrasonic_viewer_node` | `/ultrasonic/range_1` ... `/ultrasonic/range_6` | OpenCV range window |
| `controller_viewer_node` | `/controller/joy` | OpenCV controller window |
| `joy_to_motor_node` | `/controller/joy` | `/motor_control` |

Useful launches:

```bash
ros2 launch sensor_utils sensor_visualization.launch.py
ros2 launch sensor_utils camera_calibration.launch.py
ros2 launch sensor_utils camera_pose_check.launch.py
ros2 launch sensor_utils sensors_bag.launch.py
ros2 launch sensor_utils sensors_controller_bag.launch.py
ros2 launch sensor_utils bag_visualization.launch.py
```
