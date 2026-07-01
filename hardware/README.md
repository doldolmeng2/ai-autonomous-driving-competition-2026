# hardware package

## Topic flow

| Node | Subscribe | Publish |
| --- | --- | --- |
| `camera_node` | - | `/camera/left/image_raw` `sensor_msgs/Image`, `/camera/left/camera_info` `sensor_msgs/CameraInfo`, `/camera/right/image_raw` `sensor_msgs/Image`, `/camera/right/camera_info` `sensor_msgs/CameraInfo` |
| `lidar_node` | - | `/scan` `sensor_msgs/LaserScan` |
| `ultrasonic_node` | - | `/ultrasonic/front/range` `sensor_msgs/Range`, `/ultrasonic/left/range` `sensor_msgs/Range`, `/ultrasonic/right/range` `sensor_msgs/Range`, `/ultrasonic/ranges` `std_msgs/Float32MultiArray` |
| `manual_controller_node` | - | `/manual_controller/joy` `sensor_msgs/Joy` |
| `camera_viewer_node` | `/camera/left/image_raw` `sensor_msgs/Image`, `/camera/right/image_raw` `sensor_msgs/Image` | OpenCV windows |
| `lidar_viewer_node` | `/scan` `sensor_msgs/LaserScan` | OpenCV radar window |
| `ultrasonic_viewer_node` | `/ultrasonic/*/range` `sensor_msgs/Range` | OpenCV range window |

Manual driving should normally flow as:

```text
USB controller -> manual_controller_node -> /manual_controller/joy
-> drive_control node -> Arduino -> motor driver
```

Manual driving launch is owned by the `drive_control` package:

```bash
ros2 launch drive_control controller_drive.launch.py
```

## Device notes

Current tested camera mapping:

```text
/dev/video4 -> C920 left camera
/dev/video0 -> fallback right camera when /dev/video6 is not connected
```

Current tested controller mapping:

```text
/dev/input/js0 -> Xbox 360 Wireless Receiver
```

RPLidar A1 is expected on `/dev/ttyUSB0`. If it shows `Permission denied`,
grant serial access once and log out/in:

```bash
sudo usermod -aG dialout $USER
```

Before logging out/in, this session can test lidar with:

```bash
sg dialout -c "ros2 run hardware lidar_node --ros-args --params-file install/hardware/share/hardware/config/lidar.yaml"
```

For a temporary test without logging out, run:

```bash
sudo chmod a+rw /dev/ttyUSB0
```
