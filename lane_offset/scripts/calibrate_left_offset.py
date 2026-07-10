#!/usr/bin/env python3
"""calibrate_left_offset.py.

왼쪽 차선(점선) 기준 offset(LEFT_TARGET_OFFSET_PX)을 실측하기 위한 오프라인
분석 스크립트. timed_lane_offset_node.py와 완전히 같은 검출 로직을 그대로
재사용해서, "오른쪽 기준으로 이미 잘 맞아떨어지는(=raw right offset이 거의
0인) 프레임"만 골라 그 프레임에서 왼쪽 차선의 x좌표를 같이 측정한다.
오른쪽 캘리브레이션(TARGET_OFFSET_PX=195)과 같은 기준을 공유하게 하기 위함
이며, 트랙에서 "직선 구간"을 따로 찾아 헤맬 필요가 없다.

사용법:
    source /opt/ros/humble/setup.bash
    source /home/gill/comp_ws/install/setup.bash
    python3 calibrate_left_offset.py <bag_dir> <topic> [--align-tol PX]

예:
    python3 calibrate_left_offset.py \
        /home/gill/bags/rosbag2_2026_07_01-15_30_56/rosbag2_2026_07_01-15_30_56 \
        /camera/left/image_raw
"""
import argparse
import statistics
import sys

import cv2
import rclpy
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from lane_offset.timed_lane_offset_node import TimedLaneOffsetNode


def read_messages(bag_dir, topic):
    storage_options = rosbag2_py.StorageOptions(uri=bag_dir, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in type_map:
        raise SystemExit(f'topic {topic} not in bag (available: {list(type_map)})')
    msg_type = get_message(type_map[topic])

    while reader.has_next():
        name, data, _t = reader.read_next()
        if name != topic:
            continue
        yield deserialize_message(data, msg_type)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bag_dir')
    parser.add_argument('topic')
    parser.add_argument(
        '--align-tol', type=float, default=15.0,
        help='오른쪽 raw offset이 이 값(px) 이내면 "정렬된 프레임"으로 인정',
    )
    args = parser.parse_args()

    rclpy.init(args=[])
    node = TimedLaneOffsetNode()

    center_x = None
    aligned_left_samples = []
    right_found = 0
    left_found = 0
    both_found = 0
    total = 0

    for msg in read_messages(args.bag_dir, args.topic):
        total += 1
        frame = node.to_bgr(msg)
        if frame is None:
            continue
        if center_x is None:
            center_x = frame.shape[1] / 2.0

        roi = frame[node.roi_top:node.roi_bottom, :]
        if roi.size == 0:
            continue

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white_mask = node.make_white_mask(hsv)
        green_mask = node.make_green_mask(hsv)
        road_mask = node.make_road_mask(hsv)

        near = white_mask[-node.near_field_rows:, :]
        near_white_ratio = float((near > 0).mean()) if near.size else 0.0
        if near_white_ratio > node.white_overload_ratio:
            continue

        right_x, left_x = node.find_lane_bases(white_mask, green_mask, road_mask)
        if right_x is not None:
            right_found += 1
        if left_x is not None:
            left_found += 1
        if right_x is None or left_x is None:
            continue
        both_found += 1

        width = frame.shape[1]
        line_x_r, _windows, _px, _py = node.track_line_with_sliding_window(white_mask, right_x)
        if not (0 <= line_x_r <= width):
            # 폴리핏이 화면 밖으로 심하게 extrapolate된 이상치는 버림
            continue
        raw_offset_r = (line_x_r - center_x) - node.target_offset_px
        if abs(raw_offset_r) > args.align_tol:
            continue

        line_x_l, _windows, _px, _py = node.track_line_with_sliding_window(white_mask, left_x)
        if not (0 <= line_x_l <= width):
            continue
        aligned_left_samples.append(line_x_l - center_x)

    print(f'total frames read: {total}')
    print(f'right found: {right_found}, left found: {left_found}, both found: {both_found}')
    print(
        f'aligned samples (right offset within {args.align_tol}px): '
        f'{len(aligned_left_samples)}'
    )
    if aligned_left_samples:
        mean = statistics.mean(aligned_left_samples)
        stdev = (
            statistics.pstdev(aligned_left_samples)
            if len(aligned_left_samples) > 1 else 0.0
        )
        median = statistics.median(aligned_left_samples)
        print(f'left x - center mean: {mean:.1f}px, stdev: {stdev:.1f}px')
        print(f'left x - center median: {median:.1f}px')
        print(f'min/max: {min(aligned_left_samples):.1f} / {max(aligned_left_samples):.1f}')
        print(f'=> LEFT_TARGET_OFFSET_PX candidate (median, robust to outliers): {round(median)}')
    else:
        print(
            'No aligned frames with both lanes visible found. Try a different '
            'bag/topic, or relax --align-tol.'
        )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
