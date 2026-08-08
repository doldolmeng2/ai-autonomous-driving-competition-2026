#!/usr/bin/env python3
"""
주차 기준선(GT) 기반 수평/상하각도 정렬 도구  (심플 오버레이 + SUCCESS)

표시(요청):
  - 검출된 양쪽 '세로 측면선' = 파란 선 (실제 기울기대로 길게)
  - 검출된 '가로 끝선'        = 분홍 선
  - 수평·상하 각도가 모두 GT에 맞으면 화면에 크게 'SUCCESS'
  (ROI 박스/GT 기준선 등 잡다한 표시는 기본 숨김. SHOW_GT_REF로 켤 수 있음.)

정렬 로직:
  - 수평(좌우)   : 양쪽 세로선의 '바깥쪽 edge' x (원근 기울기 피팅, 두 선 평균)
  - 상하(각도)   : 가로 끝선의 '바깥쪽(위) edge' y
  - 2단계: 먼저 수평을 맞추고, 그다음 상하 각도를 맞춤 -> 둘 다 맞으면 SUCCESS

의존성: pip install opencv-python numpy
실행:  python3 camera_check_1.py --source high   (기본값, /dev/video4 C920)
       python3 camera_check_1.py --source low    (/dev/video6 C920)
       python3 camera_check_1.py --source 0      (노트북 내장캠)
       python3 camera_check_1.py --source clip.mp4  /  --source gt_src.png
키:    g=GT저장, s=스냅샷, q=종료
"""

import argparse
import json
import os

import cv2
import numpy as np


# ===================== 설정 =====================
SAT_MAX = 60
VAL_MIN = 140

ROI_LEFT = (0.00, 0.62, 0.17, 0.27)
ROI_RIGHT = (0.66, 0.62, 0.26, 0.38)
ROI_H = (0.12, 0.55, 0.56, 0.17)

MORPH_KERNEL = 3
MIN_ROWS = 10
MIN_COLS = 20
EDGE_MARGIN = 6

X_TOLERANCE_PX = 8.0
Y_TOLERANCE_PX = 6.0
X_SIGN = +1
Y_SIGN = +1

SHOW_GT_REF = False      # True로 하면 GT 기준선(얇은 회색)도 표시
SHOW_HINT = True         # 정렬 전 좌상단에 작은 안내(dx/dy) 표시

# 색 (BGR)
COL_SIDE = (255, 0, 0)     # 파랑 - 세로 측면선
COL_END = (255, 0, 255)    # 분홍 - 가로 끝선
COL_OK = (0, 220, 0)       # 초록 - SUCCESS
COL_HINT = (255, 255, 255)

GT_PATH = 'gt.json'

# 카메라 장치/포맷 — sensor_topic/config/camera.yaml 과 동일하게 맞춘다.
# (ROI가 비율 기반이라 화면비가 다르면 GT가 안 맞음)
CAM_DEVICES = {'high': '/dev/video4', 'low': '/dev/video6'}
CAP_WIDTH = 640
CAP_HEIGHT = 360
CAP_FPS = 30.0
CAP_FOURCC = 'YUYV'
# ===============================================


def is_camera_source(source):
    return (source in CAM_DEVICES or source.isdigit()
            or source.startswith('/dev/video'))


def open_camera(source):
    """'high'/'low'/장치경로/인덱스를 열고 ROS 카메라 노드와 같은 포맷으로 설정."""
    dev = CAM_DEVICES.get(source, source)
    if isinstance(dev, str) and dev.isdigit():
        dev = int(dev)
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAP_FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print('camera opened: %s -> %dx%d @%.0f' % (
        dev, cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS)))
    return cap


def roi_to_px(roi_frac, w, h):
    x, y, rw, rh = roi_frac
    return int(x * w), int(y * h), int(rw * w), int(rh * h)


def white_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, VAL_MIN], np.uint8),
                       np.array([179, SAT_MAX, 255], np.uint8))
    if MORPH_KERNEL > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (MORPH_KERNEL, MORPH_KERNEL))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


def measure_vertical_outer(bgr, roi_frac, side):
    """세로 측면선 바깥 edge를 x=a*y+b로 피팅. 그리기용 a,b,rx,ry 포함."""
    h, w = bgr.shape[:2]
    rx, ry, rw, rh = roi_to_px(roi_frac, w, h)
    sub = white_mask(bgr[ry:ry + rh, rx:rx + rw])
    pts = []
    for r in range(sub.shape[0]):
        xs = np.where(sub[r] > 0)[0]
        if xs.size < 3:
            continue
        ox = xs.min() if side == 'left' else xs.max()
        pts.append((float(ox), float(r)))
    if len(pts) < MIN_ROWS:
        return None
    pts = np.array(pts, np.float32)
    a, b = np.polyfit(pts[:, 1], pts[:, 0], 1)   # x_local = a*y_local + b
    ref_y = rh // 2
    x_ref = float(a * ref_y + b) + rx
    edge_warn = (x_ref <= EDGE_MARGIN or x_ref >= w - EDGE_MARGIN
                 or x_ref <= rx + EDGE_MARGIN or x_ref >= rx + rw - EDGE_MARGIN)
    return {'x': x_ref, 'a': float(a), 'b': float(b), 'rx': rx, 'ry': ry,
            'n': len(pts), 'edge_warn': bool(edge_warn)}


def measure_horizontal_outer(bgr, roi_frac):
    """가로 끝선 바깥(위) edge를 y=c*x+d로 피팅. 대표 y와 그리기용 c,d,rx,ry."""
    h, w = bgr.shape[:2]
    rx, ry, rw, rh = roi_to_px(roi_frac, w, h)
    sub = white_mask(bgr[ry:ry + rh, rx:rx + rw])
    cols, tops = [], []
    for c in range(sub.shape[1]):
        ys = np.where(sub[:, c] > 0)[0]
        if ys.size < 3:
            continue
        cols.append(float(c))
        tops.append(float(ys.min()))
    if len(tops) < MIN_COLS:
        return None
    cc, dd = np.polyfit(np.array(cols), np.array(tops), 1)  # y_local = cc*x + dd
    return {'y': float(np.median(tops)) + ry, 'c': float(cc), 'd': float(dd),
            'rx': rx, 'ry': ry, 'n': len(tops)}


class ParkingAligner:
    def __init__(self, gt=None):
        self.gt = gt
        self.step = 1

    def measure(self, frame):
        return {
            'L': measure_vertical_outer(frame, ROI_LEFT, 'left'),
            'R': measure_vertical_outer(frame, ROI_RIGHT, 'right'),
            'H': measure_horizontal_outer(frame, ROI_H),
        }

    def set_gt_from_frame(self, frame):
        m = self.measure(frame)
        if m['L'] is None or m['R'] is None or m['H'] is None:
            return False
        self.gt = {'left_x': m['L']['x'], 'right_x': m['R']['x'],
                   'y_top': m['H']['y']}
        return True

    def save_gt(self, path=GT_PATH):
        if self.gt is None:
            return False
        with open(path, 'w') as f:
            json.dump(self.gt, f, indent=2)
        return True

    @staticmethod
    def load_gt(path=GT_PATH):
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def process(self, frame):
        m = self.measure(frame)
        out = {'L': m['L'], 'R': m['R'], 'H': m['H'],
               'has_gt': self.gt is not None}
        if self.gt is None:
            return out

        deltas = []
        if m['L'] and not m['L']['edge_warn']:
            deltas.append(m['L']['x'] - self.gt['left_x'])
        if m['R'] and not m['R']['edge_warn']:
            deltas.append(m['R']['x'] - self.gt['right_x'])
        dx = X_SIGN * float(np.mean(deltas)) if deltas else None
        dy = Y_SIGN * (m['H']['y'] - self.gt['y_top']) if m['H'] else None
        out['dx'], out['dy'] = dx, dy

        x_ok = dx is not None and abs(dx) <= X_TOLERANCE_PX
        y_ok = dy is not None and abs(dy) <= Y_TOLERANCE_PX
        out['x_aligned'] = x_ok
        out['y_aligned'] = y_ok
        out['success'] = x_ok and y_ok
        out['step'] = 2 if x_ok else 1
        return out


# =============================== 시각화 ===============================
def _draw_vline(vis, m, color, w, h):
    if not m:
        return
    a, b, rx, ry = m['a'], m['b'], m['rx'], m['ry']
    y0, y1 = 0, h
    x0 = int(a * (y0 - ry) + b + rx)
    x1 = int(a * (y1 - ry) + b + rx)
    cv2.line(vis, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)


def _draw_hline(vis, m, color, w, h):
    if not m:
        return
    c, d, rx, ry = m['c'], m['d'], m['rx'], m['ry']
    x0, x1 = 0, w
    y0 = int(c * (x0 - rx) + d + ry)
    y1 = int(c * (x1 - rx) + d + ry)
    cv2.line(vis, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)


def draw_overlay(frame, aligner, out):
    h, w = frame.shape[:2]
    vis = frame.copy()

    if SHOW_GT_REF and aligner.gt is not None:
        cv2.line(vis, (int(aligner.gt['left_x']), 0),
                 (int(aligner.gt['left_x']), h), (150, 150, 150), 1)
        cv2.line(vis, (int(aligner.gt['right_x']), 0),
                 (int(aligner.gt['right_x']), h), (150, 150, 150), 1)
        cv2.line(vis, (0, int(aligner.gt['y_top'])),
                 (w, int(aligner.gt['y_top'])), (150, 150, 150), 1)

    # 검출된 선만 깔끔하게
    _draw_vline(vis, out.get('L'), COL_SIDE, w, h)
    _draw_vline(vis, out.get('R'), COL_SIDE, w, h)
    _draw_hline(vis, out.get('H'), COL_END, w, h)

    if not out.get('has_gt'):
        cv2.putText(vis, "press 'g' to set GT", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_HINT, 2, cv2.LINE_AA)
        return vis

    if out.get('success'):
        text = 'SUCCESS'
        scale, thick = 2.2, 5
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        org = ((w - tw) // 2, (h + th) // 2)
        cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), thick + 4, cv2.LINE_AA)          # 검은 외곽
        cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    COL_OK, thick, cv2.LINE_AA)
    elif SHOW_HINT:
        dx, dy = out.get('dx'), out.get('dy')
        if out.get('step') == 1:
            hint = ('ALIGN X: %+.0f px' % dx) if dx is not None else 'X: side line lost'
        else:
            hint = ('ADJUST ANGLE: %+.0f px' % dy) if dy is not None else 'Y: end line lost'
        cv2.putText(vis, hint, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, hint, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    COL_HINT, 2, cv2.LINE_AA)
    return vis


def _handle_key(key, aligner, frame, gt_path, vis):
    if key == ord('g'):
        if aligner.set_gt_from_frame(frame) and aligner.save_gt(gt_path):
            print('GT saved ->', gt_path, aligner.gt)
        else:
            print('GT save failed: a line was not detected')
    elif key == ord('s'):
        cv2.imwrite('overlay_snapshot.png', vis)
        print('snapshot saved: overlay_snapshot.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='high',
                    help="high(/dev/video4), low(/dev/video6), 카메라 인덱스, "
                         "동영상/이미지 경로")
    ap.add_argument('--gt', default=GT_PATH)
    args = ap.parse_args()
    aligner = ParkingAligner(gt=ParkingAligner.load_gt(args.gt))
    src = args.source

    if src.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        frame = cv2.imread(src)
        if frame is None:
            print('image load failed:', src)
            return
        while True:
            out = aligner.process(frame)
            vis = draw_overlay(frame, aligner, out)
            cv2.imshow('parking_gt_align', vis)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            _handle_key(key, aligner, frame, args.gt, vis)
        cv2.destroyAllWindows()
        return

    if is_camera_source(src):
        cap = open_camera(src)
        if cap is None:
            print('camera open failed: %s (%s)'
                  % (src, CAM_DEVICES.get(src, src)))
            print('  -> ROS camera_node 가 장치를 잡고 있으면 먼저 종료하세요.')
            return
    else:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print('source open failed:', src)
            return
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = aligner.process(frame)
        vis = draw_overlay(frame, aligner, out)
        cv2.imshow('parking_gt_align', vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        _handle_key(key, aligner, frame, args.gt, vis)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()