"""hsv_tuner_node.

트랙바로 HSV 최소/최대값을 조절하며 마스크 결과를 실시간으로 보여준다.
lane_offset의 흰색/초록색 임계값을 눈으로 보면서 정확히 잡을 때 사용.

두 가지 모드가 있다.
  1) 토픽 모드 : 카메라 토픽을 구독 (실제 카메라든 ros2 bag play 든 상관없음)
  2) bag 모드  : bag 파일을 직접 열어서 동영상 플레이어처럼 재생/일시정지/구간이동
                 (bag_path 파라미터를 주면 자동으로 이 모드)

사용 예:
    ros2 run sensor_utils hsv_tuner_node --ros-args -p image_topic:=/camera/high/image_raw
    ros2 run sensor_utils hsv_tuner_node --ros-args -p preset:=green
    ros2 run sensor_utils hsv_tuner_node --ros-args \\
        -p bag_path:=/home/gill/bags/rosbag2_2026_07_25-22_08_25 \\
        -p image_topic:=/camera/high/image_raw -p preset:=white

bag 모드 단축키 (OpenCV 창을 클릭해서 포커스를 준 뒤 누른다):
    space   재생 / 일시정지
    a , d   이전 / 다음 프레임 (누르면 일시정지 상태가 된다)
    j , l   1초 뒤로 / 1초 앞으로
    r       처음으로
    s       현재 HSV 값을 터미널에 출력 (PRESETS 에 붙여넣기 좋은 형태)
    q, ESC  종료
"""

import glob
import os
import sqlite3
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image

# ============================================================================
# 파라미터 기본값 - 튜닝은 대부분 여기서만 하면 된다.
# (전부 ROS 파라미터로도 선언되므로 --ros-args -p 로 실행 중 덮어쓰기도 가능)
# ============================================================================
IMAGE_TOPIC = '/camera/high/image_raw'
PRESET = 'white'
BAG_PATH = ''          # 비워두면 토픽 모드, 경로를 주면 bag 모드
START_PAUSED = True    # bag 모드에서 첫 프레임을 멈춘 채로 띄울지
CACHE_SIZE = 16        # 되감기할 때 바로 나오도록 최근 디코딩 프레임을 몇 장 들고 있을지

WIN_CONTROLS = 'hsv_tuner_controls'
WIN_ORIGINAL = 'hsv_tuner_original'
WIN_MASK = 'hsv_tuner_mask'
WIN_RESULT = 'hsv_tuner_result'

TB_FRAME = 'frame'
TB_SPEED = 'speed %'

# lane_offset 에서 이미 쓰고 있는 값들을 시작점으로 제공 (거기서부터 미세 조정)
PRESETS = {
    # H는 흰색 판별에 안 쓰므로 전체 범위로 둠
    'white': {'h': (70, 109), 's': (0, 13), 'v': (168, 255)},
    'green': {'h': (30, 90), 's': (40, 255), 'v': (70, 255)},
    'full': {'h': (0, 179), 's': (0, 255), 'v': (0, 255)},
}


class BagImageSource:
    """rosbag2(sqlite3) 안의 이미지 토픽을 프레임 단위로 랜덤 액세스한다.

    ros2 bag play 는 앞으로만 흘러가지만, 여기서는 db3 를 직접 열어
    (프레임 번호 -> 메시지) 인덱스를 만들어 두기 때문에 아무 위치나 바로 띄울 수 있다.
    """

    def __init__(self, bag_path, topic):
        self.topic = topic
        self.db_files = self.find_db_files(bag_path)
        if not self.db_files:
            raise FileNotFoundError(
                f'No .db3 file under "{bag_path}". '
                '(sqlite3 저장 형식의 rosbag2 폴더나 .db3 파일 경로를 주어야 한다)'
            )

        self.connections = {}
        self.index = []  # [(db_file, message_id, timestamp_ns), ...] 시간순
        for db_file in self.db_files:
            conn = self.connect(db_file)
            row = conn.execute(
                'SELECT id FROM topics WHERE name = ?', (topic,)
            ).fetchone()
            if row is None:
                continue
            rows = conn.execute(
                'SELECT id, timestamp FROM messages WHERE topic_id = ? ORDER BY timestamp',
                (row[0],),
            ).fetchall()
            self.index.extend((db_file, msg_id, stamp) for msg_id, stamp in rows)

        self.index.sort(key=lambda item: item[2])
        if not self.index:
            available = self.list_topics()
            raise LookupError(
                f'Topic "{topic}" not found in bag. 사용 가능한 이미지 토픽: {available}'
            )

        self.t0 = self.index[0][2]
        self.duration = (self.index[-1][2] - self.t0) / 1e9
        self.cache = {}
        self.cache_order = []

    @staticmethod
    def find_db_files(bag_path):
        bag_path = os.path.expanduser(bag_path)
        if os.path.isfile(bag_path):
            return [bag_path]
        # 폴더 안에 또 폴더가 있는 경우(bags/xxx/xxx/)도 잡히도록 재귀 탐색
        return sorted(glob.glob(os.path.join(bag_path, '**', '*.db3'), recursive=True))

    def connect(self, db_file):
        if db_file not in self.connections:
            self.connections[db_file] = sqlite3.connect(f'file:{db_file}?mode=ro', uri=True)
        return self.connections[db_file]

    def list_topics(self):
        names = []
        for db_file in self.db_files:
            rows = self.connect(db_file).execute(
                "SELECT name FROM topics WHERE type = 'sensor_msgs/msg/Image'"
            ).fetchall()
            names.extend(name for (name,) in rows)
        return sorted(set(names))

    def __len__(self):
        return len(self.index)

    def stamp(self, idx):
        """해당 프레임의 bag 시작 기준 시간(초)."""
        return (self.index[idx][2] - self.t0) / 1e9

    def index_at_time(self, seconds):
        """시간(초)에 가장 가까운(그 이하의) 프레임 번호."""
        target = self.t0 + int(seconds * 1e9)
        lo, hi = 0, len(self.index) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.index[mid][2] <= target:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def raw_message(self, idx):
        db_file, msg_id, _ = self.index[idx]
        row = self.connect(db_file).execute(
            'SELECT data FROM messages WHERE id = ?', (msg_id,)
        ).fetchone()
        return deserialize_message(bytes(row[0]), Image)

    def frame(self, idx, to_bgr):
        """idx 번째 프레임을 BGR 로 돌려준다 (최근 것들은 캐시에서)."""
        if idx in self.cache:
            return self.cache[idx]

        bgr = to_bgr(self.raw_message(idx))
        self.cache[idx] = bgr
        self.cache_order.append(idx)
        if len(self.cache_order) > CACHE_SIZE:
            self.cache.pop(self.cache_order.pop(0), None)
        return bgr

    def close(self):
        for conn in self.connections.values():
            conn.close()
        self.connections.clear()


class HsvTunerNode(Node):
    """트랙바로 HSV 범위를 조절하며 원본/마스크/결과 화면을 동시에 보여준다."""

    def __init__(self):
        super().__init__('hsv_tuner_node')

        self.declare_parameter('image_topic', IMAGE_TOPIC)
        self.declare_parameter('preset', PRESET)
        self.declare_parameter('bag_path', BAG_PATH)
        self.declare_parameter('start_paused', START_PAUSED)

        self.image_topic = self.get_parameter('image_topic').value
        self.bag_path = self.get_parameter('bag_path').value
        preset = PRESETS.get(self.get_parameter('preset').value, PRESETS['white'])

        self.frame = None
        self.bag = None
        self.frame_idx = 0
        self.playing = not self.get_parameter('start_paused').value
        self.last_advance = time.monotonic()
        self.seeking = False  # 트랙바를 코드가 움직일 때 콜백이 되먹임되는 것 방지

        if self.bag_path:
            self.bag = BagImageSource(self.bag_path, self.image_topic)
            self.setup_windows(preset, bag_frames=len(self.bag))
            self.show_frame(0)
            self.get_logger().info(
                f'Bag mode: {len(self.bag)} frames of {self.image_topic} '
                f'({self.bag.duration:.1f} s). '
                'space=play/pause, a/d=frame step, j/l=+-1s, r=restart, s=print HSV, q=quit'
            )
        else:
            self.setup_windows(preset)
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
            self.create_subscription(Image, self.image_topic, self.image_callback, qos)
            self.get_logger().info(
                f'Subscribing {self.image_topic}. Adjust trackbars in "{WIN_CONTROLS}" '
                'and watch the mask window.'
            )

        self.timer = self.create_timer(1.0 / 30.0, self.tick)

    def setup_windows(self, preset, bag_frames=0):
        cv2.namedWindow(WIN_CONTROLS, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_CONTROLS, 480, 320)
        cv2.namedWindow(WIN_ORIGINAL, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_MASK, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_RESULT, cv2.WINDOW_NORMAL)

        def nothing(_):
            pass

        h_lo, h_hi = preset['h']
        s_lo, s_hi = preset['s']
        v_lo, v_hi = preset['v']
        cv2.createTrackbar('H min', WIN_CONTROLS, h_lo, 179, nothing)
        cv2.createTrackbar('H max', WIN_CONTROLS, h_hi, 179, nothing)
        cv2.createTrackbar('S min', WIN_CONTROLS, s_lo, 255, nothing)
        cv2.createTrackbar('S max', WIN_CONTROLS, s_hi, 255, nothing)
        cv2.createTrackbar('V min', WIN_CONTROLS, v_lo, 255, nothing)
        cv2.createTrackbar('V max', WIN_CONTROLS, v_hi, 255, nothing)

        if bag_frames:
            # 동영상 플레이어의 재생바 역할. 드래그하면 그 프레임에서 멈춘다.
            cv2.createTrackbar(TB_FRAME, WIN_CONTROLS, 0, bag_frames - 1, self.on_seek)
            cv2.createTrackbar(TB_SPEED, WIN_CONTROLS, 100, 400, nothing)

    # ======================================================================
    # bag 모드 재생 제어
    # ======================================================================
    def on_seek(self, pos):
        if self.seeking or self.bag is None:
            return
        self.playing = False  # 사용자가 재생바를 잡으면 그 화면에 멈춘다
        self.show_frame(pos)

    def show_frame(self, idx):
        idx = max(0, min(idx, len(self.bag) - 1))
        self.frame_idx = idx
        bgr = self.bag.frame(idx, self.to_bgr)
        if bgr is not None:
            self.frame = bgr
        self.seeking = True
        cv2.setTrackbarPos(TB_FRAME, WIN_CONTROLS, idx)
        self.seeking = False

    def seek_seconds(self, delta):
        target = max(0.0, min(self.bag.stamp(self.frame_idx) + delta, self.bag.duration))
        self.show_frame(self.bag.index_at_time(target))

    def advance_playback(self):
        """bag 의 원래 타임스탬프 간격 x 배속 만큼 시간이 흐르면 다음 프레임으로."""
        now = time.monotonic()
        if not self.playing:
            self.last_advance = now
            return

        speed = max(10, cv2.getTrackbarPos(TB_SPEED, WIN_CONTROLS)) / 100.0
        while self.playing:
            nxt = self.frame_idx + 1
            if nxt >= len(self.bag):
                self.playing = False  # 끝에서 멈춤 (r 로 처음으로)
                break
            gap = (self.bag.stamp(nxt) - self.bag.stamp(self.frame_idx)) / speed
            if now - self.last_advance < gap:
                break
            self.last_advance += gap
            self.show_frame(nxt)

    def handle_key(self, key):
        if key in (ord('q'), 27):
            raise KeyboardInterrupt
        if key == ord('s'):
            self.print_values()
            return
        if self.bag is None or key == 255 or key == -1:
            return

        if key == ord(' '):
            self.playing = not self.playing
            self.last_advance = time.monotonic()
        elif key in (ord('a'), 81):        # 81 = 좌측 방향키
            self.playing = False
            self.show_frame(self.frame_idx - 1)
        elif key in (ord('d'), 83):        # 83 = 우측 방향키
            self.playing = False
            self.show_frame(self.frame_idx + 1)
        elif key == ord('j'):
            self.playing = False
            self.seek_seconds(-1.0)
        elif key == ord('l'):
            self.playing = False
            self.seek_seconds(1.0)
        elif key == ord('r'):
            self.playing = False
            self.show_frame(0)

    def print_values(self):
        h_min, h_max, s_min, s_max, v_min, v_max = self.read_trackbars()
        self.get_logger().info(
            f"PRESETS entry -> {{'h': ({h_min}, {h_max}), "
            f"'s': ({s_min}, {s_max}), 'v': ({v_min}, {v_max})}}"
        )

    # ======================================================================
    # 공통 처리
    # ======================================================================
    def image_callback(self, msg):
        self.frame = self.to_bgr(msg)

    def read_trackbars(self):
        return (
            cv2.getTrackbarPos('H min', WIN_CONTROLS),
            cv2.getTrackbarPos('H max', WIN_CONTROLS),
            cv2.getTrackbarPos('S min', WIN_CONTROLS),
            cv2.getTrackbarPos('S max', WIN_CONTROLS),
            cv2.getTrackbarPos('V min', WIN_CONTROLS),
            cv2.getTrackbarPos('V max', WIN_CONTROLS),
        )

    def tick(self):
        if self.bag is not None:
            self.advance_playback()

        if self.frame is None:
            cv2.waitKey(1)
            return

        h_min, h_max, s_min, s_max, v_min, v_max = self.read_trackbars()

        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))
        result = cv2.bitwise_and(self.frame, self.frame, mask=mask)
        pixel_count = int(np.count_nonzero(mask))
        ratio = pixel_count / mask.size

        overlay = self.frame.copy()
        cv2.putText(
            overlay,
            f'H[{h_min},{h_max}] S[{s_min},{s_max}] V[{v_min},{v_max}] '
            f'px={pixel_count} ratio={ratio:.3f}',
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        if self.bag is not None:
            state = 'PLAY' if self.playing else 'PAUSE'
            cv2.putText(
                overlay,
                f'[{state}] frame {self.frame_idx + 1}/{len(self.bag)} '
                f't={self.bag.stamp(self.frame_idx):.2f}/{self.bag.duration:.2f}s '
                'space a/d j/l r s q',
                (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(WIN_ORIGINAL, overlay)
        cv2.imshow(WIN_MASK, mask)
        cv2.imshow(WIN_RESULT, result)
        self.handle_key(cv2.waitKey(1) & 0xFF)

    # ======================================================================
    # YUYV -> BGR 변환 (sensor_utils/camera_viewer_node.py 와 동일한 방식)
    # ======================================================================
    def to_bgr(self, msg):
        data = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding in ('yuv422_yuy2', 'yuyv', 'yuyv422'):
            yuyv = data.reshape((msg.height, msg.width, 2))
            return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
        if msg.encoding in ('bgr8', '8UC3'):
            return data.reshape((msg.height, msg.width, 3))
        if msg.encoding == 'rgb8':
            rgb = data.reshape((msg.height, msg.width, 3))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if msg.encoding in ('mono8', '8UC1'):
            mono = data.reshape((msg.height, msg.width))
            return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

        self.get_logger().warn(
            f'Unsupported camera encoding: {msg.encoding}', throttle_duration_sec=5.0
        )
        return None

    def destroy_node(self):
        if self.bag is not None:
            self.bag.close()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HsvTunerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
