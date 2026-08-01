#!/usr/bin/env python3
"""Standalone check that the high camera still aims where it should.

The camera looks down at the road with a checkerboard sheet on the floor just
inside the bottom of the frame, one row of squares deep.  Run this with the aim
known-good once to record a reference, then run it any time to see whether the
tripod has moved.

    python3 sensor_utils/scripts/check_camera_aim.py --save   # once
    python3 sensor_utils/scripts/check_camera_aim.py          # every time after

No ROS and no camera calibration: the check is done entirely in image pixels.
Stop camera_node first, since this opens the camera device directly.

Exit codes (with --headless): 0 aligned, 1 needs adjusting, 2 could not measure.
"""

import argparse
import sys
import time
from datetime import datetime
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'sensor_utils'))

from sensor_utils.board_strip import (  # noqa: E402
    DEFAULT_TOLERANCES,
    compare,
    measure,
)

REFERENCE_DIR = '~/.camera_aim'
MEASURED_KEYS = ('centre_offset_px', 'top_edge_y', 'roll_deg', 'square_px')
GREEN = (0, 200, 0)
RED = (0, 0, 235)
YELLOW = (0, 220, 255)
CYAN = (255, 220, 0)
WHITE = (255, 255, 255)
GREY = (150, 150, 150)


# ----------------------------------------------------------------- the camera

def device_name(path):
    try:
        video_name = Path(path).resolve().name
        return (Path('/sys/class/video4linux') / video_name / 'name').read_text().strip()
    except OSError:
        return ''


def capture_nodes(name_filter='C920'):
    """Capture-capable video nodes for the named camera, in stable USB order.

    Each C920 exposes a metadata node next to its capture node, and /dev/videoN
    numbering shuffles between boots.  The by-path symlinks ending in
    ``video-index0`` are exactly the capture nodes and stay in USB port order,
    which is what makes "high" mean the same camera every time.
    """
    return [
        link for link in sorted(glob('/dev/v4l/by-path/*video-index0'))
        if not name_filter or name_filter.lower() in device_name(link).lower()
    ]


def find_camera(requested, name_filter='C920'):
    """Resolve the camera to open, returning (path, error_message)."""
    if requested not in ('high', 'low', 'auto'):
        return requested, None

    nodes = capture_nodes(name_filter)
    if not nodes:
        return None, f'No {name_filter} capture device found under /dev/v4l/by-path/.'

    index = 1 if requested == 'low' else 0
    if index >= len(nodes):
        return None, (f'Asked for the {requested} camera but only {len(nodes)} '
                      f'{name_filter} found.')

    path = nodes[index]
    # Deliberately do not fall through to the other camera when this one is
    # busy: silently checking the wrong camera is worse than not checking.
    probe = cv2.VideoCapture(path, cv2.CAP_V4L2)
    busy = not probe.isOpened()
    probe.release()
    if busy:
        return None, (f'{Path(path).resolve()} ({requested}) is busy. Stop camera_node '
                      'or any ros2 launch using the camera, then retry.')
    return path, None


def open_camera(path, width, height):
    capture = cv2.VideoCapture(path, cv2.CAP_V4L2)
    if not capture.isOpened():
        return None
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


# ------------------------------------------------------------- the measurement

def median_measurement(samples):
    """Median of each measured number across frames.

    One frame's edge fit wanders by a few tenths of a pixel, which is the same
    size as the drift being looked for, so a raw readout flickers across the
    pass threshold.
    """
    usable = [s for s in samples if s is not None]
    if not usable:
        return None

    merged = dict(usable[-1])
    for key in MEASURED_KEYS:
        values = [s[key] for s in usable if s.get(key) is not None]
        merged[key] = float(np.median(values)) if values else None
    return merged


def collect(capture, frames, roi_fraction, warmup=5):
    for _ in range(warmup):
        capture.read()

    samples = []
    frame = None
    for _ in range(frames):
        ok, grabbed = capture.read()
        if not ok or grabbed is None:
            continue
        frame = grabbed
        samples.append(measure(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), roi_fraction))
    return median_measurement(samples), frame


# ---------------------------------------------------------------- the reference

def save_reference(path, measurement):
    record = {
        'saved_at': datetime.now().isoformat(timespec='seconds'),
        **{key: (None if measurement.get(key) is None else float(measurement[key]))
           for key in MEASURED_KEYS},
    }
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(record, sort_keys=False))
    return record


def load_reference(path):
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    record = yaml.safe_load(path.read_text()) or {}
    missing = [key for key in ('centre_offset_px', 'top_edge_y', 'roll_deg')
               if record.get(key) is None]
    if missing:
        raise ValueError(f'{path} is missing: {", ".join(missing)}')
    return record


# ------------------------------------------------------------------ presenting

def summarise(result, measurement):
    axes = result['axes']
    rows = axes['pitch'].get('rows')
    line = (
        f'{result["verdict"]}  '
        f'yaw {axes["yaw"]["delta"]:+.1f}px  '
        f'pitch {axes["pitch"]["delta"]:+.1f}px'
        + (f' ({rows:+.2f} rows)' if rows is not None else '')
        + f'  roll {axes["roll"]["delta"]:+.2f}deg'
    )
    split = measurement.get('split')
    if split:
        line += f'  [{split["left"]}:{split["right"]} squares]'
    if result['hints']:
        line += '  ->  turn ' + ' + '.join(result['hints'])
    return line


def dim(panel, x, y, width, height, strength=0.6):
    x2, y2 = min(panel.shape[1], x + width), min(panel.shape[0], y + height)
    if x < x2 and y < y2:
        panel[y:y2, x:x2] = (panel[y:y2, x:x2] * (1.0 - strength)).astype(panel.dtype)


def draw(frame, measurement, result, message):
    panel = frame.copy()
    height, width = panel.shape[:2]
    centre_x = width // 2

    if measurement is not None:
        slope = measurement['top_edge_slope']
        intercept = measurement['top_edge_intercept']
        x0, x1 = int(measurement['strip_x0']), int(measurement['strip_x1'])
        cv2.line(panel, (x0, int(slope * x0 + intercept)),
                 (x1, int(slope * x1 + intercept)), CYAN, 1, cv2.LINE_AA)
        split = measurement.get('split')
        if split:
            for boundary in split['boundaries']:
                cv2.line(panel, (int(round(boundary)), int(measurement['top_edge_y'])),
                         (int(round(boundary)), height - 1), YELLOW, 1)

    cv2.line(panel, (centre_x, 0), (centre_x, height - 1), RED, 1)

    verdict = result['verdict'] if result else (message or 'NO BOARD')
    colour = {'PASS': GREEN, 'ADJUST': RED}.get(verdict, YELLOW)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), colour, 6)
    dim(panel, 6, 6, 330, 132)
    cv2.putText(panel, verdict, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2, cv2.LINE_AA)

    if result:
        axes = result['axes']
        rows = axes['pitch'].get('rows')
        lines = [
            ('yaw  ', f'{axes["yaw"]["delta"]:+6.1f} px', axes['yaw']),
            ('pitch', f'{axes["pitch"]["delta"]:+6.1f} px'
                     + (f'  {rows:+.2f} rows' if rows is not None else ''), axes['pitch']),
            ('roll ', f'{axes["roll"]["delta"]:+6.2f} deg', axes['roll']),
        ]
        for index, (name, text, axis) in enumerate(lines):
            hint = '' if axis['ok'] else f'  -> {axis["direction"]}'
            cv2.putText(panel, f'{name} {text}{hint}', (16, 72 + index * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        GREEN if axis['ok'] else RED, 2 if not axis['ok'] else 1, cv2.LINE_AA)
    elif message:
        cv2.putText(panel, message, (16, 76), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, YELLOW, 1, cv2.LINE_AA)

    if measurement is not None and measurement.get('split'):
        split = measurement['split']
        cv2.putText(panel, f'{split["left"]} : {split["right"]} squares',
                    (16, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1, cv2.LINE_AA)

    cv2.putText(panel, 's=save reference   q=quit', (16, height - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    return panel


# ------------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--device', default='high',
                        help='high, low, auto, or an explicit /dev/videoN path (default: high)')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=360,
                        help='capture size; the default matches the driving pipeline')
    parser.add_argument('--reference', default=None,
                        help=f'reference file (default: {REFERENCE_DIR}/<device>_strip.yaml)')
    parser.add_argument('--roi-fraction', type=float, default=0.28,
                        help='how much of the frame, up from the bottom, to search (default: 0.28)')
    parser.add_argument('--frames', type=int, default=9,
                        help='frames to median-filter each measurement over (default: 9)')
    parser.add_argument('--yaw-tolerance', type=float, default=DEFAULT_TOLERANCES['centre_offset_px'])
    parser.add_argument('--pitch-tolerance', type=float, default=DEFAULT_TOLERANCES['top_edge_y'])
    parser.add_argument('--roll-tolerance', type=float, default=DEFAULT_TOLERANCES['roll_deg'])
    parser.add_argument('--save', action='store_true',
                        help='save the current view as the reference and exit')
    parser.add_argument('--headless', action='store_true',
                        help='print one verdict and exit instead of opening a window')
    args = parser.parse_args()

    if args.reference is None:
        label = args.device if args.device in ('high', 'low', 'auto') else Path(args.device).name
        args.reference = f'{REFERENCE_DIR}/{label}_strip.yaml'
    tolerances = {
        'centre_offset_px': args.yaw_tolerance,
        'top_edge_y': args.pitch_tolerance,
        'roll_deg': args.roll_tolerance,
    }

    device, error = find_camera(args.device)
    if device is None:
        print(error, file=sys.stderr)
        return 2

    capture = open_camera(device, args.width, args.height)
    if capture is None:
        print(f'Could not open {device}.', file=sys.stderr)
        return 2

    print(f'Camera: {device} ({device_name(device)}) at '
          f'{int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x'
          f'{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}')

    reference = None
    message = ''
    try:
        reference = load_reference(args.reference)
        print(f'Reference: {args.reference} (saved {reference.get("saved_at", "?")})')
    except FileNotFoundError:
        message = 'no reference yet - press s'
        print(f'Reference: none at {args.reference}')
    except ValueError as exc:
        message = str(exc)
        print(f'Reference unusable: {exc}', file=sys.stderr)

    try:
        if args.save:
            return do_save(capture, args)
        if args.headless:
            return do_headless(capture, args, reference, tolerances)
        return do_window(capture, args, reference, tolerances, message)
    finally:
        capture.release()
        cv2.destroyAllWindows()


def do_save(capture, args):
    measurement, _ = collect(capture, args.frames, args.roi_fraction)
    if measurement is None:
        print('Checkerboard strip not found; nothing saved. Is the board in view '
              'along the bottom of the frame?', file=sys.stderr)
        return 2
    record = save_reference(args.reference, measurement)
    print(f'Saved reference to {Path(args.reference).expanduser()}')
    for key in MEASURED_KEYS:
        print(f'  {key}: {record[key]}')
    return 0


def do_headless(capture, args, reference, tolerances):
    if reference is None:
        print('No usable reference. Run with --save first.', file=sys.stderr)
        return 2
    measurement, _ = collect(capture, args.frames, args.roi_fraction)
    if measurement is None:
        print('Could not measure: checkerboard strip not found.', file=sys.stderr)
        return 2

    result = compare(reference, measurement, tolerances)
    print(summarise(result, measurement))
    return 0 if result['verdict'] == 'PASS' else 1


def do_window(capture, args, reference, tolerances, message):
    window = 'camera aim check'
    samples = []
    last_report = 0.0

    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue

        samples.append(measure(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), args.roi_fraction))
        del samples[:-args.frames]
        measurement = median_measurement(samples) if len(samples) >= args.frames else None

        result = None
        if measurement is not None and reference is not None:
            result = compare(reference, measurement, tolerances)

        cv2.imshow(window, draw(frame, measurement, result, message))

        if result is not None and time.monotonic() - last_report > 1.0:
            print(summarise(result, measurement))
            last_report = time.monotonic()

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            return 0
        if key == ord('s'):
            if measurement is None:
                message = 'no strip detected - cannot save'
                continue
            save_reference(args.reference, measurement)
            reference = load_reference(args.reference)
            message = ''
            print(f'Saved reference to {Path(args.reference).expanduser()}')


if __name__ == '__main__':
    sys.exit(main())
