# lane_offset

차선 인식 결과를 토픽으로 발행하는 패키지다.

| Node | Subscribe | Publish |
| --- | --- | --- |
| `timed_lane_offset_node` | `/camera/high/image_raw` | `/lane_offset` |
| `mission_lane_offset_node` | `/lane_info`, `/camera/high/image_raw` | `/lane_offset` |
| `lane_offset_debug_viewer_node` | `/lane_offset/debug_image` | OpenCV window |

`mission_lane_offset_node`에는 현재 `timed_lane_offset_node`의 BEV, 색상 경계
실선 검출, 중앙 점선 fallback, 근접 x 측정, offset 매핑 파라미터와 구현을
독립적인 복사본으로 넣었다. 이후 한쪽 파일을 변경해도 다른 노드에는 자동으로
반영되지 않는다. 미션 복사본에는 `/lane_info`에 따른 1/2차로 선택과 목적 차로
경계선을 찾을 때까지 유지하는 차선 변경 조향이 추가되어 있다. 2차로는 오른쪽
초록 경계선을 우선하고, 보이지 않으면 중앙 점선을 사용한다. 1차로도 같은
알고리즘으로 왼쪽 밝은 회색 경계선과 중앙 점선을 사용한다.

PDF flow:

```text
timed:   /camera/high/image_raw -> lane_offset -> /lane_offset
mission: /lane_info + /camera/high/image_raw -> lane_offset -> /lane_offset
```

## mission_lane_offset_node의 바깥 차선 배제

차로를 옮길 때 바깥 실선을 중앙 점선으로 오인하던 문제를 색으로 막는다.
오른쪽 바깥 실선 밖에는 초록 매트, 왼쪽 바깥 실선 밖에는 밝은 회색 영역이
있고 중앙 점선 양옆은 아스팔트뿐이라는 점을 이용한다.

- 흰 덩어리 bbox의 바깥쪽 밴드에서 해당 색이 `outer_min_pixels` 이상 보이면
  바깥 실선으로 보고 중앙 점선 후보에서 제외한다.
- 한 번 바깥 실선으로 본 x는 기억해 두고, 초록/회색이 화면에서 사라진 뒤에도
  그 주변 덩어리를 계속 제외한다. 색 없이 움직일 때는 도로 바깥 방향으로만
  따라가며, 오래되거나(`outer_memory_max_age_frames`) 근처에 아무것도 없으면
  (`outer_memory_max_misses`) 기억을 버린다.
- 흰 선 가장자리 번짐은 밝은 회색으로 분류되기 쉬우므로 흰색에서
  `outer_halo_margin_px` 안의 색은 근거로 쓰지 않는다.

주요 파라미터(전부 `--ros-args -p` 로 조정 가능):

| 파라미터 | 기본값 | 뜻 |
| --- | --- | --- |
| `outer_near_distance_px` | 20 | 덩어리에서 이 거리 안의 색만 근거로 본다 |
| `outer_min_pixels` | 40 | 바깥 실선으로 인정할 최소 색 픽셀 수 |
| `outer_halo_margin_px` | 5 | 흰 선 주변 이 거리의 색은 무시(번짐 제거) |
| `outer_memory_tolerance_px` | 40 | 기억 위치에서 이 거리 안이면 같은 바깥 실선 |
| `outer_memory_max_age_frames` | 90 | 색 재확인 없이 유지할 최대 프레임 |
| `outer_memory_max_misses` | 10 | 근처에 덩어리가 없어도 버틸 프레임 |

디버그 화면에서 하늘색 박스가 배제된 바깥 실선(`color`=색으로 확인,
`memory`=기억으로 유지), 자홍 박스가 중앙 점선으로 선택된 덩어리다.
초록/회색 임계값은 `sensor_utils`의 `hsv_tuner_node`로 맞춘다.

## 장소별 색상 프로필

색상 임계값은 `config/color_profiles.yaml`에 장소별로 저장한다.
프로필은 노드 종류와 무관하게 `장소 -> colors -> 색상`으로 구성되며,
모든 차선 노드가 활성 장소의 같은 팔레트를 사용한다. 새 장소는 가장 비슷한
기존 프로필 전체를 복사하고 장소 이름과 HSV/YCrCb 값을 수정한다.
범위는 항상 `[최솟값, 최댓값]` 순서로 적는다.

빌드한 뒤 아래 명령으로 사용할 장소를 선택한다.

```bash
ros2 run lane_offset select_color_profile
```

장소 이름이나 화면에 표시된 번호를 입력하면 선택값이
`~/.config/lane_offset/active_color_profile`에 저장된다. 실행 중인 차선 노드에는
즉시 반영되지 않으며, `timed_lane_offset_node` 또는
`mission_lane_offset_node`를 재시작할 때 적용된다. 시작 로그의
`Color profile=...` 메시지로 실제 적용된 프로필을 확인할 수 있다.

YAML 형식이나 색상 범위가 잘못되면 선택 명령은 그 프로필을 저장하지 않고,
차선 노드도 임계값 없이 실행되지 않도록 시작 단계에서 오류를 발생시킨다.
