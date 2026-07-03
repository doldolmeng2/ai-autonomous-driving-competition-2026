# hardware package

## Topic flow

| Node | Subscribe | Publish |
| --- | --- | --- |
| `camera_node` | - | `/camera/left/image_raw` `sensor_msgs/Image`, `/camera/left/camera_info` `sensor_msgs/CameraInfo`, `/camera/right/image_raw` `sensor_msgs/Image`, `/camera/right/camera_info` `sensor_msgs/CameraInfo` |
| `sllidar_node` | - | `/scan` `sensor_msgs/LaserScan` |
| `ultrasonic_node` | - | `/ultrasonic/range_1` ... `/ultrasonic/range_6` `sensor_msgs/Range`, `/ultrasonic/ranges` `std_msgs/Float32MultiArray` |
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
/dev/video6 -> C920 right camera
```

`camera_node` requires the video device name to contain `C920`; it does not fall
back to non-C920 webcams.

Current tested controller mapping:

```text
/dev/input/js0 -> Xbox 360 Wireless Receiver
left stick vertical axis 1 -> drive forward/reverse
right stick horizontal axis 3 -> steering
```

RPLidar A1 is expected on `/dev/ttyUSB0`. If it shows `Permission denied`,
grant serial access once and log out/in:

```bash
sudo usermod -aG dialout $USER
```

Run the Slamtec A1 lidar directly with:

```bash
ros2 launch sllidar_ros2 sllidar_a1_launch.py
```

The `hardware` sensor launch files include that same Slamtec launch.

For a temporary test without logging out, run:

```bash
sudo chmod a+rw /dev/ttyUSB0
```
