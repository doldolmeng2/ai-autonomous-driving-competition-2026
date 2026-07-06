# lane_offset

차선 인식 결과를 토픽으로 발행하는 패키지다.

| Node | Subscribe | Publish |
| --- | --- | --- |
| `timed_lane_offset_node` | `/camera/high/image_raw` | `/lane_offset` |
| `mission_lane_offset_node` | `/camera/high/image_raw`, `/camera/low/image_raw`, `/ultrasonic/range_1` ... `/ultrasonic/range_6` | `/lane_info`, `/lane_offset` |
| `lane_offset_debug_viewer_node` | `/lane_offset/debug_image` | OpenCV window |

PDF flow:

```text
timed:   /camera/high/image_raw -> lane_offset -> /lane_offset
mission: camera high/low + ultrasonic -> lane_offset -> /lane_info + /lane_offset
```
