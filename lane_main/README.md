# lane_main

차선 주행 제어 패키지다. 차선 인식 결과를 받아 공통 제어 토픽으로 변환한다.

| Node | Subscribe | Publish |
| --- | --- | --- |
| `timed_lane_main_node` | `/lane_offset` | `/motor_control` |
| `mission_lane_main_node` | `/lane_info`, `/lane_offset` | `/motor_control` |

PDF flow:

```text
timed:   /lane_offset -> lane_main -> /motor_control -> drive_control
mission: /lane_info + /lane_offset -> lane_main -> /motor_control -> drive_control
```

Launch:

```bash
ros2 launch lane_main timed_lane_main.launch.py
ros2 launch lane_main mission_lane_main.launch.py
```
