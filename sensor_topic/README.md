# sensor_topic

PDF 기준 센서/컨트롤러 원본 토픽 발행 패키지다.

| Node | Subscribe | Publish |
| --- | --- | --- |
| `camera_node` | - | `/camera/high/image_raw`, `/camera/high/camera_info`, `/camera/low/image_raw`, `/camera/low/camera_info` |
| `arduino_communication_node` | `/arduino/motor_command` | `/arduino/ultrasonic_raw` |
| `ultrasonic_node` | `/arduino/ultrasonic_raw` | `/ultrasonic/range_1` ... `/ultrasonic/range_6`, `/ultrasonic/ranges` |
| `controller_node` | USB controller device | `/manual_controller/joy` |
| `sllidar_ros2` launch | RPLidar serial | `/scan` |

Launch:

```bash
ros2 launch sensor_topic sensors.launch.py
```

`sensors.launch.py` starts `arduino_communication_node`, `camera_node`,
`ultrasonic_node`, `controller_node`, and includes the Slamtec A1 launch.

LiDAR bearing convention:

```text
rear 0 deg, right +90 deg, front +/-180 deg, left -90 deg
(angle increases counter-clockwise)
```

Device defaults:

```text
/dev/video4 -> high camera
/dev/video6 or auto -> low camera
/dev/ttyUSB0 -> RPLidar A1
/dev/input/js* -> controller
```
