#!/usr/bin/env python3
"""
주차 기준선(GT) 기반 수평/상하각도 정렬 도구
  - GT 저장 전 : 화면 전체에서 세 선(양쪽 세로 측면선 + 가로 끝선)을 자동 검출해 표시.
  - 'g' 저장   : 검출된 세 선 위치를 GT로 저장하고, 그 선 주변으로 ROI를 '자동 생성'해서
                 함께 gt_roi.json 에 저장한다. (ROI가 GT를 기준으로 그어짐)
  - GT 저장 후 : gt_roi.json 의 ROI 안에서만 선을 안정적으로 추적하며 정렬 판정.

주의: GT 포맷이 camera_check_1.py 와 다르다. 저 파일은 gt.json(left_x/right_x/y_top),
      이 파일은 gt_roi.json(left/right/h + roi)을 쓴다. 서로 바꿔 쓸 수 없다.

표시:
  - 세로 측면선 = 파란 선,  가로 끝선 = 분홍 선  (검출된 선만 깔끔하게)
  - 수평·상하 각도가 모두 GT에 맞으면 화면에 크게 'SUCCESS'

정렬 로직(2단계):
  - 수평(좌우) : 양쪽 세로선 '바깥쪽 edge' x (원근 기울기 피팅, 두 선 평균)
  - 상하(각도) : 가로 끝선 '바깥쪽(위) edge' y
  - 수평 먼저 맞추고 -> 상하 각도 -> 둘 다 맞으면 SUCCESS

의존성: pip install opencv-python numpy
실행:  python3 camera_check_2.py --source high   (기본값, /dev/video4 C920)
       python3 camera_check_2.py --source low    (/dev/video6 C920)
       python3 camera_check_2.py --source 0      (노트북 내장캠)
       python3 camera_check_2.py --source clip.mp4  /  --source gt_src.png
키:    g=GT+ROI 저장,  s=스냅샷,  q=종료
"""

import argparse
import json
import os

import cv2
import numpy as np


# ===================== 설정 =====================
SAT_MAX = 60          # HSV 흰색 채도 상한
VAL_MIN = 140         # HSV 흰색 명도 하한
MORPH_KERNEL = 3

# 자동 검출(HoughLinesP) 파라미터
HOUGH_THRESH = 50
HOUGH_MIN_LEN = 90
HOUGH_MAX_GAP = 40
HORIZ_MAX_ANG = 20    # |각도|<=이 값 이면 가로선 후보
STEEP_MIN_ANG = 40    # |각도|>=이 값 이면 세로(측면)선 후보

# ROI 자동 생성 시 검출선 둘레에 줄 여유(px) — 드리프트 허용폭
ROI_MARGIN = 45

MIN_ROWS = 10
MIN_COLS = 20
EDGE_MARGIN = 6

X_TOLERANCE_PX = 8.0
Y_TOLERANCE_PX = 6.0
X_SIGN = +1
Y_SIGN = +1

SHOW_HINT = True
SHOW_ROI = False      # True면 저장된 ROI 박스도 얇게 표시(디버그용)

COL_SIDE = (255, 0, 0)     # 파랑 - 세로 측면선
COL_END = (255, 0, 255)    # 분홍 - 가로 끝선
COL_OK = (0, 220, 0)       # 초록 - SUCCESS
COL_HINT = (255, 255, 255)

# camera_check_1.py 의 gt.json 과 스키마가 달라서 파일명을 분리한다.
GT_PATH = 'gt_roi.json'

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


def white_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, VAL_MIN], np.uint8),
                       np.array([179, SAT_MAX, 255], np.uint8))
    if MORPH_KERNEL > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (MORPH_KERNEL, MORPH_KERNEL))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


# ----------------- 전체 화면 자동 검출 -----------------
def autodetect(frame):
    """화면 전체에서 가로 끝선(H)과 양쪽 세로 측면선(L,R) 세그먼트를 찾는다."""
    h, w = frame.shape[:2]
    mask = white_mask(frame)
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, HOUGH_THRESH,
                            minLineLength=HOUGH_MIN_LEN, maxLineGap=HOUGH_MAX_GAP)
    segs = []
    for l in (lines if lines is not None else []):
        x1, y1, x2, y2 = [int(v) for v in l[0]]
        length = float(np.hypot(x2 - x1, y2 - y1))
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        ang = (ang + 90) % 180 - 90
        segs.append(dict(x1=x1, y1=y1, x2=x2, y2=y2, L=length, ang=ang,
                         cx=(x1 + x2) / 2, cy=(y1 + y2) / 2))

    def longest(lst):
        return max(lst, key=lambda s: s['L']) if lst else None

    H = longest([s for s in segs if abs(s['ang']) <= HORIZ_MAX_ANG])
    hy = H['cy'] if H else h / 2
    steep = [s for s in segs if abs(s['ang']) >= STEEP_MIN_ANG]
    L = longest([s for s in steep if s['cx'] < w * 0.45 and s['cy'] > hy - 40])
    R = longest([s for s in steep if s['cx'] > w * 0.55 and s['cy'] > hy - 40])
    return {'H': H, 'L': L, 'R': R}


def seg_bbox_roi(seg, w, h, margin=ROI_MARGIN):
    """세그먼트 둘레 bounding box를 여유(margin) 포함해 비율 ROI로."""
    xs = [seg['x1'], seg['x2']]
    ys = [seg['y1'], seg['y2']]
    x0 = max(0, min(xs) - margin)
    y0 = max(0, min(ys) - margin)
    x1 = min(w, max(xs) + margin)
    y1 = min(h, max(ys) + margin)
    return (x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h)


def seg_vfit(seg):
    x1, y1, x2, y2 = seg['x1'], seg['y1'], seg['x2'], seg['y2']
    if y2 == y1:
        y2 += 1
    a = (x2 - x1) / (y2 - y1)
    b = x1 - a * y1
    return {'a': a, 'b': b, 'rx': 0, 'ry': 0,
            'x': a * ((y1 + y2) / 2) + b, 'edge_warn': False}


def seg_hfit(seg):
    x1, y1, x2, y2 = seg['x1'], seg['y1'], seg['x2'], seg['y2']
    if x2 == x1:
        x2 += 1
    c = (y2 - y1) / (x2 - x1)
    d = y1 - c * x1
    return {'c': c, 'd': d, 'rx': 0, 'ry': 0, 'y': c * ((x1 + x2) / 2) + d}


# ----------------- ROI 안에서 정밀 측정 -----------------
def roi_to_px(roi_frac, w, h):
    x, y, rw, rh = roi_frac
    return int(x * w), int(y * h), int(rw * w), int(rh * h)


def measure_vertical_outer(bgr, roi_frac, side):
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
    a, b = np.polyfit(pts[:, 1], pts[:, 0], 1)
    ref_y = rh // 2
    x_ref = float(a * ref_y + b) + rx
    edge_warn = (x_ref <= EDGE_MARGIN or x_ref >= w - EDGE_MARGIN
                 or x_ref <= rx + EDGE_MARGIN or x_ref >= rx + rw - EDGE_MARGIN)
    return {'x': x_ref, 'a': float(a), 'b': float(b), 'rx': rx, 'ry': ry,
            'edge_warn': bool(edge_warn)}


def measure_horizontal_outer(bgr, roi_frac):
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
    cc, dd = np.polyfit(np.array(cols), np.array(tops), 1)
    return {'y': float(np.median(tops)) + ry, 'c': float(cc), 'd': float(dd),
            'rx': rx, 'ry': ry}


class ParkingAligner:
    def __init__(self, gt=None):
        self.gt = gt   # {'left':{'roi','x'}, 'right':{'roi','x'}, 'h':{'roi','y'}}

    # ---- GT + ROI 자동 저장 ----
    def capture_gt(self, frame):
        h, w = frame.shape[:2]
        det = autodetect(frame)
        if not (det['H'] and det['L'] and det['R']):
            return False
        roi_l = seg_bbox_roi(det['L'], w, h)
        roi_r = seg_bbox_roi(det['R'], w, h)
        roi_h = seg_bbox_roi(det['H'], w, h)
        mL = measure_vertical_outer(frame, roi_l, 'left')
        mR = measure_vertical_outer(frame, roi_r, 'right')
        mH = measure_horizontal_outer(frame, roi_h)
        if not (mL and mR and mH):
            return False
        self.gt = {'left': {'roi': list(roi_l), 'x': mL['x']},
                   'right': {'roi': list(roi_r), 'x': mR['x']},
                   'h': {'roi': list(roi_h), 'y': mH['y']}}
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
            gt = json.load(f)
        # camera_check_1.py 가 만든 gt.json(left_x/right_x/y_top)은 ROI가 없어
        # 여기서 쓸 수 없다. 크래시 대신 무시하고 GT 재촬영을 유도한다.
        if not all(k in gt and 'roi' in gt[k] for k in ('left', 'right', 'h')):
            print('WARNING: %s 는 이 스크립트 포맷이 아닙니다 (camera_check_1 형식?).'
                  % path)
            print("  -> 무시하고 시작합니다. 'g' 를 눌러 새로 GT를 저장하세요.")
            return None
        return gt

    def process(self, frame):
        # GT 없으면 전체화면 자동검출 결과만(그리기용) 반환
        if self.gt is None:
            det = autodetect(frame)
            out = {'has_gt': False}
            out['L'] = seg_vfit(det['L']) if det['L'] else None
            out['R'] = seg_vfit(det['R']) if det['R'] else None
            out['H'] = seg_hfit(det['H']) if det['H'] else None
            return out

        mL = measure_vertical_outer(frame, self.gt['left']['roi'], 'left')
        mR = measure_vertical_outer(frame, self.gt['right']['roi'], 'right')
        mH = measure_horizontal_outer(frame, self.gt['h']['roi'])
        out = {'has_gt': True, 'L': mL, 'R': mR, 'H': mH}

        deltas = []
        if mL and not mL['edge_warn']:
            deltas.append(mL['x'] - self.gt['left']['x'])
        if mR and not mR['edge_warn']:
            deltas.append(mR['x'] - self.gt['right']['x'])
        dx = X_SIGN * float(np.mean(deltas)) if deltas else None
        dy = Y_SIGN * (mH['y'] - self.gt['h']['y']) if mH else None
        out['dx'], out['dy'] = dx, dy

        x_ok = dx is not None and abs(dx) <= X_TOLERANCE_PX
        y_ok = dy is not None and abs(dy) <= Y_TOLERANCE_PX
        out['x_aligned'], out['y_aligned'] = x_ok, y_ok
        out['success'] = x_ok and y_ok
        out['step'] = 2 if x_ok else 1
        return out


# =============================== 시각화 ===============================
def _draw_vline(vis, m, color, w, h):
    if not m:
        return
    a, b, rx, ry = m['a'], m['b'], m['rx'], m['ry']
    x0 = int(a * (0 - ry) + b + rx)
    x1 = int(a * (h - ry) + b + rx)
    cv2.line(vis, (x0, 0), (x1, h), color, 2, cv2.LINE_AA)


def _draw_hline(vis, m, color, w, h):
    if not m:
        return
    c, d, rx, ry = m['c'], m['d'], m['rx'], m['ry']
    y0 = int(c * (0 - rx) + d + ry)
    y1 = int(c * (w - rx) + d + ry)
    cv2.line(vis, (0, y0), (w, y1), color, 2, cv2.LINE_AA)


def _put(vis, text, org, scale, color, thick):
    cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)


def draw_overlay(frame, aligner, out):
    h, w = frame.shape[:2]
    vis = frame.copy()

    if SHOW_ROI and aligner.gt is not None:
        for key in ('left', 'right', 'h'):
            rx, ry, rw, rh = roi_to_px(aligner.gt[key]['roi'], w, h)
            cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (90, 90, 90), 1)

    _draw_vline(vis, out.get('L'), COL_SIDE, w, h)
    _draw_vline(vis, out.get('R'), COL_SIDE, w, h)
    _draw_hline(vis, out.get('H'), COL_END, w, h)

    if not out.get('has_gt'):
        _put(vis, "press 'g' to set GT", (12, 30), 0.7, COL_HINT, 2)
        return vis

    if out.get('success'):
        text = 'SUCCESS'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 5)
        _put(vis, text, ((w - tw) // 2, (h + th) // 2), 2.2, COL_OK, 5)
    elif SHOW_HINT:
        dx, dy = out.get('dx'), out.get('dy')
        if out.get('step') == 1:
            hint = ('ALIGN X: %+.0f px' % dx) if dx is not None else 'X: side line lost'
        else:
            hint = ('ADJUST ANGLE: %+.0f px' % dy) if dy is not None else 'Y: end line lost'
        _put(vis, hint, (12, 30), 0.7, COL_HINT, 2)
    return vis


def _handle_key(key, aligner, frame, gt_path, vis):
    if key == ord('g'):
        if aligner.capture_gt(frame) and aligner.save_gt(gt_path):
            print('GT + ROI saved ->', gt_path)
        else:
            print('GT save failed: 3 lines not all detected (조명/각도 확인)')
    elif key == ord('s'):
        cv2.imwrite('overlay_snapshot.png', vis)
        print('snapshot saved: overlay_snapshot.png')


def _loop_frame(aligner, frame, gt_path):
    out = aligner.process(frame)
    vis = draw_overlay(frame, aligner, out)
    cv2.imshow('parking_gt_align', vis)
    return vis


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
            vis = _loop_frame(aligner, frame, args.gt)
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
        vis = _loop_frame(aligner, frame, args.gt)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        _handle_key(key, aligner, frame, args.gt, vis)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()