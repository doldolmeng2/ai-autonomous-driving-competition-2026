# ROS 2 후방 수직 주차 패키지 초안

## 피드백 반영

팀 피드백을 반영해 기존의 “카메라가 주차공간을 반드시 찾아야 한다”는 구조를 수정했습니다.

- 전방 카메라 화각이 좁을 수 있으므로 **LiDAR가 주차 공간 검출의 주 센서**
- 카메라는 화각 검증 전까지 **보조 힌트와 OUT 라인 확인**
- 초음파는 기본적으로 **감속·비상 정지**
- 정렬 단계의 초음파 조향 bias는 실차 검증 전까지 비활성화
- 제어는 **정지 → 조향 → 주행** FSM

## LiDAR 주차공간 검출

1. LaserScan을 실제 각도 정보로 XY 변환
2. TF로 `base_link` 좌표계로 변환
3. 좌·우 측면 ROI 분리
4. X축 binning으로 차량이 존재하는 구간 계산
5. `occupied → gap → occupied` 패턴 검색
6. gap 양쪽 점군에 PCA 직선 fitting
7. 슬롯 입구 중심, 폭, 축 방향, confidence 계산
8. 연속 스캔에서 위치·방향 분산이 작을 때만 슬롯 확정

## FSM

```text
SEARCH
→ APPROACH
→ STAGING
→ STEER_IN
→ REVERSE_ARC
→ COUNTER_STEER
→ ALIGN
→ FINAL_REVERSE
→ VERIFY
→ HOLD(4 s)
→ EXIT_STRAIGHT
→ OUT_CONFIRM
→ DONE
```

## 조향 보정

LiDAR에서 계산하는 값:

- `axis_error`: 차량 방향과 슬롯 축의 각도 차이
- `lateral_error`: 목표 rear-axle 위치와의 횡오차
- `depth_error`: 목표 깊이까지 남은 거리

정렬 단계에서 제한적으로:

```text
steer =
  reverse_steer_sign
  × (K_yaw × axis_error + K_lat × lateral_error)
  + optional_ultrasonic_bias
```

## 초음파 피드백

- `distance < hard_stop_distance` → 즉시 `[0, 0]`
- `hard_stop_distance ≤ distance < slow_distance` → 속도 감소
- 후진 원호 시 전면 바깥쪽 센서도 감시
- 측면 거리 기반 steering bias는 기본 OFF

센서 번호:

```text
1 front-left
2 front-right
3 side-left
4 side-right
5 rear-left
6 rear-right
```

## 입력/출력

입력:

```text
/scan
/camera/high/image_raw
/camera/low/image_raw
/ultrasonic/range_1 ... /ultrasonic/range_6
```

출력:

```text
/motor_control       Int16MultiArray [steer, speed]
/parking/state       String
/parking/reason      String
/parking/target_pose PoseStamped
/parking/clearances  Float32MultiArray
```

## 빌드

```bash
cd ~/your_ws/src
# 이 폴더를 parking 패키지로 복사
cd ~/your_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
ros2 launch parking parking.launch.py
```

시작:

```bash
ros2 topic pub --once /parking/start std_msgs/msg/Bool "{data: true}"
```

리셋:

```bash
ros2 topic pub --once /parking/reset std_msgs/msg/Bool "{data: true}"
```

## 실차에서 반드시 확인할 값

- 좌/우 슬롯 진입 조향 부호
- 최소 구동 PWM과 실제 속도
- staging_pass_distance
- counter_trigger_deg
- target_depth
- 후면 초음파 정지 거리
- LiDAR ROI와 차체 self-mask
- 두 카메라 화각과 OUT 라인 ROI
- 모든 센서 TF
- 조향 시간 추적의 좌우 비대칭·반복 오차
- 출차 방향과 exit_turn_steer

## 안전

- `/motor_control` Publisher는 주차 중 하나만 유지
- drive_control 또는 Arduino에 300~500 ms watchdog 추가
- 초음파 hard-stop 검증 전 자동 후진 금지
- Arduino 코드는 기존 motor firmware 대체본이 아니라 참고용
