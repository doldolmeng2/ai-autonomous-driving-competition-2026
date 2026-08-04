# `parking_node_yym.py` 인수인계 문서

작성 기준일: 2026-08-05  
대상 소스: `/home/hailab/osy/260801/ai-autonomous-driving-competition-2026/parking/parking/parking_node_yym.py`

## 1. 이 문서의 목적

이 문서는 다른 Codex 세션이나 개발자가 YYM 주차 노드를 바로 이어서 수정할 수 있도록 현재 코드의 상태 머신, LiDAR 좌표계, 조향 계산, 주요 파라미터, 실행 방법과 알려진 문제를 정리한 인수인계 문서다.

중요: 아래의 **현재 코드 동작**은 위 `260801` 경로의 파일을 직접 읽어 정리한 것이다. 문서 마지막의 **후속 작업 메모**는 이전 실차 테스트 세션에서 합의했지만 이 `260801` 파일에는 아직 반영되지 않은 내용이다. 두 내용을 혼동하지 말 것.

## 2. 수정 범위 원칙

- YYM 주차 작업에서는 사용자가 별도로 허용하지 않는 한 `parking/parking/parking_node_yym.py`만 수정한다.
- 이 노드는 조향 센서나 Arduino 토픽을 직접 제어하지 않는다.
- 모든 주행 요청은 `/motor_control`에 `std_msgs/msg/Int16MultiArray`로 발행한다.
- 메시지 형식은 `[steer_deg, speed]`이다.
- 조향 명령 범위는 `-45~+45`, 속도 명령 범위는 코드에서 `-130~+130`으로 제한한다.
- 현재 부호 규칙은 `음수=좌조향`, `양수=우조향`, `양수 속도=전진`, `음수 속도=후진`이다.

## 3. ROS 입출력

| 구분 | 기본 토픽 | 타입 | 설명 |
|---|---|---|---|
| 구독 | `/scan` | `sensor_msgs/msg/LaserScan` | 주차 차량 감지와 정지 조건에 사용 |
| 발행 | `/motor_control` | `std_msgs/msg/Int16MultiArray` | `[조향각, 속도]` 명령 |

노드 이름은 `parking_node_yym`이고 디버그 창 기본 이름은 `parking_yym_debug`다.

## 4. LiDAR 좌표계와 디버그 화면

차량에 장착된 LiDAR의 각도 표기는 다음과 같다.

```text
REAR 0° | RIGHT +90° | FRONT ±180° | LEFT -90°
```

코드 내부 차량 좌표는 다음과 같다.

- `+x`: 차량 전방
- `-x`: 차량 후방
- `+y`: 차량 좌측
- `-y`: 차량 우측
- 초록 가로선: `x=0`, 즉 후미 LiDAR 위치를 지나는 좌우선
- 초록 세로선: 차량 중심의 전후 기준선 `y=0`
- 빨간선: LiDAR 원점과 두 장애물 차량의 기준점 중간을 연결한 선

디버그 화면의 `ParkingPair.lower`는 우측 차량, `ParkingPair.upper`는 좌측 차량이다.

## 5. 현재 `260801` 코드의 전체 주차 시퀀스

### 5.1 시작과 첫 차량 인식

1. `WAIT_FOR_SCAN`: 유효한 `/scan`을 기다린다. 스캔이 오기 전에는 조향 명령도 보내지 않는다.
2. `START_DELAY`: 기본 5초 동안 `steer/speed=0/0`으로 정지한다.
3. `APPROACH_FIRST_CAR`: `steer=0`, `speed=110`으로 직진한다.
4. 우측 차량 클러스터 중 최근접 거리가 2m 이하인 물체가 3프레임 연속 검출되면 첫 차량으로 인정한다.
5. `SET_LEFT_STEER`: 정지한 상태로 최대 좌조향 `-45°`를 0.6초 동안 맞춘다.
6. `TURN_LEFT_TIMED`: `steer=-45`, `speed=110`으로 7초 주행한다.
7. `recognition_only=True`면 정지 후 프로그램을 종료하고, 기본값 `False`면 주차 모드로 넘어간다.

### 5.2 두 차량과 주차 공간 획득

1. `SETTLE_AND_ACQUIRE_GAP`에서 정지한다.
2. 엄격한 주차 간격 조건을 만족하는 두 클러스터를 우선 선택한다.
3. 엄격한 조건이 실패해도 두 클러스터가 보이면 fallback midpoint pair를 사용할 수 있다.
4. pair가 3프레임 연속 검출되고 최소 정렬 시간이 지나면 `REVERSE_CENTER`로 전환한다.
5. 4초 안에 pair를 얻지 못하면 `PARKING_FAILED`가 된다.

### 5.3 1m 원에 들어오기 전 후진

각 구간은 다음 순서로 반복된다.

1. 정지 상태에서 빨간선과 초록 세로선 사이 각도를 계산한다.
2. 계산각에 `reverse_steer_multiplier=10`을 곱하고 `±45°`로 제한한다.
3. 정지 상태로 해당 조향각을 0.6초 동안 맞춘다.
4. 같은 조향각과 `speed=-110`으로 1초 후진한다.
5. 기존 조향각을 유지한 채 0.4초 정지한다.
6. LiDAR pair를 다시 계산한 후 다음 1초 구간을 수행한다.

두 장애물 차량 중 하나라도 최근접 LiDAR 점이 1m 원 안에 들어오면 이 반복을 중단한다.

### 5.4 1m 진입 후 5초 정지와 최종 보정

1. `FINAL_STOP`: 즉시 `steer/speed=0/0`으로 5초 정지한다.
2. 그 뒤 최종 보정을 기본 2회 수행한다.
3. 매 보정 전에 0.4초 정지하며 pair와 조향각을 다시 계산한다.
4. 빨간선 오차가 `±5°`를 벗어나면 빨간선 각도에 `×10`을 적용한다.
5. 빨간선 오차가 `±5°` 이내이면 두 차량 PCA 축 각도의 평균에 `×5`를 적용한다.
6. 계산한 조향각을 정지 상태에서 0.6초 맞춘 뒤 0.5초 후진한다.
7. 두 번째 0.5초 후진이 끝나면 반드시 정지하고 `steer=0`을 0.6초 동안 맞춘다.
8. 이후 `FINAL_STRAIGHT_DRIVE`에서 `steer=0`, `speed=-110`으로 연속 후진한다.

### 5.5 주차 완료 조건

- 5초 정지 이후부터 처음 선택했던 좌우 차량을 별도로 추적한다.
- 각 기준 차량에 대해 초록 가로선 아래, 즉 `x<0`인 점이 존재하는지 확인한다.
- 좌측 또는 우측 기준 차량 중 **하나라도** 초록 가로선 아래에서 3프레임 연속 사라지면 `PARKED`로 전환한다.
- 이후 나타나는 기둥이나 다른 유닛이 원래 차량을 대체하지 않도록 원래 두 차량의 track center를 사용한다.

### 5.6 현재 코드의 출차 시퀀스

현재 `260801` 파일에는 출차가 활성화되어 있다.

1. `PARKED`에서 2초 정지
2. 조향 0으로 3초 전진
3. 정지 상태로 최대 우조향 `+45°` 설정
4. 최대 우조향으로 8초 전진
5. 정지 상태로 조향 0 정렬
6. 조향 0으로 10초 전진
7. `EXIT_COMPLETE`에서 `0/0`으로 대기

## 6. LiDAR 인식 방식

### 기본 필터

- 유효 각도 영역: 후방 0도를 중심으로 `±125°`
- 최소 거리: 0.15m
- 최대 클러스터 거리: 4.0m
- 점 간 연결 거리: 0.20m
- 최소 클러스터 점 개수: 7
- 최소 장애물 크기: 0.22m

### 차량 클러스터

1. 인접 점 거리가 0.20m 이하인 점들을 connected component로 묶는다.
2. 점이 7개 미만이면 제거한다.
3. x 또는 y 방향 크기가 0.22m 미만이면 차량 후보에서 제거한다.
4. PCA 주축으로 `axis_angle`을 계산한다.
5. 중심 y가 `-0.12m`보다 작으면 우측 차량 후보로 본다.

### 주차 pair

- y가 작은 클러스터가 `lower/right`, y가 큰 클러스터가 `upper/left`다.
- 두 차량 사이 간격 기본 허용 범위는 0.48~1.40m다.
- 기준점 `reference_point`는 두 클러스터 중앙점의 평균이다.
- strict pair가 없을 때는 중심선에 가깝고 두 클러스터가 중심을 감싸는 fallback pair를 선호한다.
- 1m 진입 전에는 `x<=0`인 점만 차량 클러스터에 사용한다.

## 7. 안전과 실패 조건

- `/scan`이 0.5초 이상 끊기면 `EMERGENCY_STOP`.
- 유효 점이 부족한 스캔이 5프레임 누적되면 `EMERGENCY_STOP`.
- 첫 차량을 30초 안에 찾지 못하면 `PARKING_FAILED`.
- 두 차량 pair를 4초 안에 얻지 못하면 `PARKING_FAILED`.
- 1m 진입 전 후방 `±11°` 안에서 0.18m 이하 물체가 감지되면 `PARKING_FAILED`.
- 실패 및 비상 정지 상태에서는 계속 `steer/speed=0/0`을 발행한다.

## 8. 주요 기본 파라미터

| 파라미터 | 기본값 | 의미 |
|---|---:|---|
| `startup_delay_sec` | 5.0 | 유효 스캔 이후 출발 대기 |
| `approach_speed` | 110 | 첫 차량까지 직진 속도 |
| `turn_speed` | 110 | 좌회전 속도 |
| `reverse_speed` | -110 | 후진 속도 |
| `left_max_steer_deg` | -45 | 최대 좌조향 |
| `left_turn_duration_sec` | 7.0 | 첫 차량 인식 후 좌회전 시간 |
| `first_car_max_distance_m` | 2.0 | 첫 차량 인식 최대 거리 |
| `first_car_confirm_frames` | 3 | 첫 차량 연속 확인 프레임 |
| `reverse_segment_duration_sec` | 1.0 | 1m 진입 전 후진 구간 |
| `reverse_measure_stop_sec` | 0.4 | 구간 사이 정지 측정 시간 |
| `steer_settle_sec` | 0.6 | 정지 상태 조향 정렬 시간 |
| `reverse_steer_multiplier` | 10.0 | 빨간선 각도 보정 배율 |
| `straight_reverse_radius_m` | 1.0 | 최종 단계 진입 원 반경 |
| `straight_reverse_stop_sec` | 5.0 | 1m 진입 후 정지 시간 |
| `final_line_alignment_tolerance_deg` | 5.0 | 빨간선 정렬 허용 범위 |
| `final_reverse_steer_multiplier` | 5.0 | 최종 차량 기울기 보정 배율 |
| `final_correction_duration_sec` | 0.5 | 최종 보정 1회 후진 시간 |
| `final_correction_segment_count` | 2 | 최종 보정 횟수 |
| `rear_half_empty_confirm_frames` | 3 | 주차 완료 확인 프레임 |
| `vehicle_width_m` | 0.38 | 자차 폭 |
| `minimum_side_clearance_m` | 0.05 | 요구 최소 측면 간격 |

## 9. 디버그 화면 읽는 법

- 노란색 상태: 진행 중
- 초록색 상태: `PARKED` 또는 `EXIT_COMPLETE`
- 빨간색 `FAILED`: 주차 실패 또는 비상 정지
- 회색 점: 필터 전 유효 LiDAR 점
- 색깔별 점 묶음: 차량 클러스터
- `V1`, `V2` 옆 각도: 각 클러스터의 PCA 축 각도
- 초록 세로선: 차량 중심 기준선
- 빨간선: 현재 pair 기준점 방향
- 상단 `cmd steer/speed`: `/motor_control`로 내보낸 마지막 명령
- `phase`: 세부 후진 단계
- `pair=STRICT/FALLBACK`: 현재 pair 획득 방식

## 10. 빌드와 실행

대상 저장소에서 코드 변경 후:

```bash
cd /home/hailab/osy/260801/ai-autonomous-driving-competition-2026
source /opt/ros/humble/setup.bash
colcon build --packages-select parking --symlink-install
source install/setup.bash
ros2 launch parking parking.launch.py
```

센서 런치가 별도라면 먼저 다른 터미널에서 실행한다.

```bash
cd /home/hailab/osy/260801/ai-autonomous-driving-competition-2026
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch sensor_topic sensors.launch.py
```

## 11. rosbag 기록과 재현

주차 실행 전에 별도 터미널에서:

```bash
cd /home/hailab/osy/260801/ai-autonomous-driving-competition-2026
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 bag record -o parking_test_01 \
  /scan \
  /motor_control \
  /arduino/steering_raw
```

분석할 때는 `/motor_control`의 구간별 명령과 `/arduino/steering_raw`의 실제 응답을 `/scan` 프레임과 시간 정렬한다. rosbag에 노드 내부 상태는 저장되지 않으므로 상태 전환은 명령 패턴으로 추정하거나, 향후 상태/계산값을 별도 디버그 토픽으로 발행하는 것이 좋다.

## 12. 알려진 문제와 실차 데이터에서 확인한 사항

### 박스의 L자 형상과 PCA 기울기 오류

장애물 차량을 박스로 대체했기 때문에 LiDAR에서 박스가 L자 형태로 보일 수 있다. 현재 `260801` 코드는 클러스터 전체 PCA 축 두 개를 단순 평균한다. 짧은 가로면이 `80~90°` 축으로 선택되면 실제 차량 기울기보다 훨씬 큰 조향을 만들 수 있다.

이전 실차 rosbag `parking_test_02~05` 분석에서는 다음이 확인됐다.

- 2번과 5번은 상대적으로 양호한 주차
- 3번과 4번은 최종 각도가 삐뚤어짐
- 구동계는 `/motor_control` 명령을 정상 추종함
- 연속 직선 후진 구간은 실제로 `steer=0, speed=-110`이었음
- 3·4번 문제는 직선 후진 중 새로 틀어진 것이 아니라 최종 보정 후 남은 각도가 유지된 것으로 판단됨
- 일부 프레임에서 실제 세로 경계는 약 `3~13°`인데 전체 PCA는 `80~90°`로 계산됨

참고 rosbag은 이전 작업 트리의 다음 경로에 있었다.

```text
/home/hailab/osy/260711/ai-autonomous-driving-competition-2026/parking_test_02
/home/hailab/osy/260711/ai-autonomous-driving-competition-2026/parking_test_03
/home/hailab/osy/260711/ai-autonomous-driving-competition-2026/parking_test_04
/home/hailab/osy/260711/ai-autonomous-driving-competition-2026/parking_test_05
```

## 13. 후속 작업 메모: `260801` 파일에 아직 없는 최근 합의 사항

다른 세션은 작업 시작 전에 반드시 실제 파일과 아래 요구를 비교해야 한다. 아래 내용은 이전 `260711` 작업 트리에서 구현 또는 합의됐지만, 이 문서 작성 시점의 `260801` 대상 파일에는 반영되지 않았다.

1. **출차 비활성화**
   - 주차 완료 후 `PARKED_HOLD`에서 `steer/speed=0/0`으로 계속 대기하는 것이 최근 요구였다.
   - 현재 `260801` 파일은 2초 후 자동 출차한다.

2. **초기 후방 노이즈 필터 강화**
   - 인식 모드에서 후방 및 초록 가로선 경계 부근 점을 첫 차량 후보에서 제외했다.
   - 이전 구현값은 전방 점 `x>=0.15m`, 클러스터 중심 `x>=0.25m`, 최소 12점, 크기 0.30m, 5프레임 연속이었다.
   - 현재 `260801` 파일은 거리 2m와 3프레임 조건 위주라 후방 노이즈 오인식 가능성이 더 크다.

3. **최종 빨간선 허용 오차**
   - 최근 합의값은 `±3°`였다.
   - 현재 `260801` 파일 기본값은 `±5°`다.

4. **최종 보정 횟수**
   - 여러 번 논의 후 최종 합의는 다시 **2회**, 회당 0.5초였다.
   - 현재 `260801` 파일도 이 부분은 2회로 일치한다.

5. **박스 세로 경계 기반 기울기**
   - 전체 PCA 평균 대신 주차 공간을 향한 박스의 세로 경계를 x 구간별로 추출하고 직선 적합하도록 개선했다.
   - x 방향 길이가 0.20m 미만인 짧은 가로면은 기울기 계산에서 제외했다.
   - 현재 `260801` 파일에는 이 개선이 없고 `final_average_alignment_angle()`이 두 PCA 각도를 단순 평균한다.

6. **최종 단계 우선순위**
   - 5초 정지 후 각 0.5초 보정마다 정지 상태에서 재측정한다.
   - 빨간선과 초록선 오차가 ±3°를 넘으면 빨간선 보정을 우선한다.
   - ±3° 이내일 때만 신뢰 가능한 차량 세로 경계 기울기를 사용한다.
   - 두 번째 보정 후 반드시 정지하고 조향 0을 맞춘 뒤 정지 조건까지 연속 후진한다.

## 14. 다음 세션 권장 시작 순서

1. 이 문서가 가리키는 `260801` 소스를 다시 읽는다.
2. `git status --short`로 사용자 변경사항을 확인하고 보존한다.
3. 사용자가 원하는 기준이 현재 `260801` 동작인지, 위 후속 합의 상태인지 확인한다.
4. 별도 허가가 없으면 `parking_node_yym.py` 외 파일은 수정하지 않는다.
5. 변경 후 `python3 -m py_compile`과 `git diff --check`를 실행한다.
6. `colcon build --packages-select parking --symlink-install`로 검증한다.
7. 실차 테스트는 `/scan`, `/motor_control`, `/arduino/steering_raw`를 함께 rosbag으로 남긴다.

