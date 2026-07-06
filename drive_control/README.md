# drive_control

공통 제어 토픽을 Arduino serial 명령으로 변환하는 패키지다.

| Node | Subscribe | Publish / Output |
| --- | --- | --- |
| `drive_control_node` | `/motor_control` `std_msgs/Int16MultiArray` | Arduino serial `steer speed\n` |

`/motor_control` 형식:

```text
data = [steer, speed]
steer: 목표 조향각
speed: 구동 PWM
```

수동 주행 launch:

```bash
ros2 launch drive_control controller_drive.launch.py
```

이 launch는 `sensor_topic/controller_node`, `sensor_utils/joy_to_motor_node`,
`drive_control_node`를 함께 실행한다.
