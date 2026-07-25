# drive_control

`drive_control_node` converts `/motor_control` steering targets and speed into
PWM commands on `/arduino/motor_command`. It never opens a serial device;
`sensor_topic/arduino_communication_node` exclusively owns communication.

Flash `arduino_code/integrated_ardunio_code/integrated_ardunio_code.ino`
to the Arduino Mega 2560.

공통 제어 토픽을 Arduino용 PWM 토픽으로 변환하는 패키지다.

| Node | Subscribe | Publish / Output |
| --- | --- | --- |
| `drive_control_node` | `/motor_control` `std_msgs/Int16MultiArray` | `/arduino/motor_command` `Int16MultiArray` |

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

이 launch는 `arduino_communication_node`, `sensor_topic/controller_node`,
`sensor_utils/joy_to_motor_node`, `drive_control_node`를 함께 실행한다.
