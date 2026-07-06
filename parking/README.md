# parking

주차 제어 패키지다.

| Node | Subscribe | Publish |
| --- | --- | --- |
| `parking_node` | `/camera/high/image_raw`, `/ultrasonic/range_1` ... `/ultrasonic/range_6`, `/scan` | `/motor_control` |

PDF flow:

```text
/camera/high/image_raw + ultrasonic range topics + /scan
-> parking_node
-> /motor_control
-> drive_control_node
```

Launch:

```bash
ros2 launch parking mission_parking.launch.py
```
