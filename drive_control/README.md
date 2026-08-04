# drive_control

`drive_control_node` converts `/motor_control` steering targets and speed into
PWM commands on `/arduino/motor_command`. It never opens a serial device;
`sensor_topic/arduino_communication_node` exclusively owns communication.

## Steering PID profiles

`steer_pid_profile` selects independent `timed`, `mission`, or `parking`
steering gains. Their defaults are defined in `drive_control_node.py` as
`TIMED_STEER_PID_*`, `MISSION_STEER_PID_*`, and `PARKING_STEER_PID_*`. The timed,
mission, and parking launch files select the matching profile explicitly.
`steer_pid_kp`, `steer_pid_ki`, and `steer_pid_kd` can still override the
selected profile for a single launch.

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

## 실차 조향 시간 측정

조향 모터의 최대 조향 및 원점 복귀 시간을 육안으로 측정할 때는
`drive_control_node`를 종료하고 Arduino 통신 노드만 실행한다.
차량 바퀴를 지면에서 띄우거나 차량이 움직이지 않도록 고정한 후 시험한다.

```bash
ros2 run sensor_topic arduino_communication_node
```

다른 터미널에서 다음 테스트를 실행한다. 아래 예시는 조향 PWM 150으로
오른쪽 0.2초, 정지 1초, 왼쪽 0.2초 순서로 동작한다. 주행 PWM은 항상 0이다.

```bash
ros2 run drive_control steering_time_test \
  --direction right \
  --outbound-sec 0.2 \
  --return-sec 0.2
```

`0.2`초부터 조금씩 시간을 늘려 최대 조향에 도달하는 시간을 찾는다.
기구 끝에 닿은 상태에서 계속 모터를 구동하지 않는다. 왼쪽도 별도로
`--direction left`로 시험한다.

## drive_control 주행 테스트

`drive_control_node`의 코드를 변경하지 않고 `/motor_control` 목표 명령으로
직진, 좌회전, 우회전을 시험한다. Arduino 통신 노드와
`drive_control_node`만 실행하고 조이스틱이나 자율주행처럼
`/motor_control`을 발행하는 다른 노드는 종료한다.

```bash
ros2 run drive_control drive_control_motion_test straight
ros2 run drive_control drive_control_motion_test left
ros2 run drive_control drive_control_motion_test right
```

기본값은 주행 PWM 80, 주행 시간 1초, 좌우 목표 조향각 30도다.

```bash
ros2 run drive_control drive_control_motion_test left \
  --drive-pwm 100 \
  --drive-sec 1.5 \
  --steer-angle 35
```

`drive_control_node`의 내부 조향각 기억과 중앙 복귀를 확인하려면 다음
시퀀스를 실행한다. 단계 사이에 정지 명령을 넣지 않고 연속 주행한다.

```bash
ros2 run drive_control drive_control_motion_test left_sequence
```

기본 동작은 `직진 2초 → 좌회전 1초 → 다시 직진 2초 → 정지`다.
