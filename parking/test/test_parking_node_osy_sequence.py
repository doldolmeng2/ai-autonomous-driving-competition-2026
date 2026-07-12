import math
from unittest.mock import patch

import rclpy
from sensor_msgs.msg import LaserScan

from parking.parking_node_osy import ParkingNodeOsy, ParkingState


def make_scan(segments=(), *, invalid=False):
    sample_count = 721
    scan = LaserScan()
    scan.angle_min = -math.pi
    scan.angle_max = math.pi
    scan.angle_increment = 2.0 * math.pi / (sample_count - 1)
    scan.range_min = 0.05
    scan.range_max = 12.0
    scan.ranges = [math.inf if invalid else 5.0] * sample_count
    for start_deg, end_deg, distance in segments:
        start = round((math.radians(start_deg) - scan.angle_min) / scan.angle_increment)
        end = round((math.radians(end_deg) - scan.angle_min) / scan.angle_increment)
        for index in range(max(0, start), min(sample_count, end + 1)):
            scan.ranges[index] = float(distance)
    return scan


def test_full_perpendicular_t_parking_sequence():
    rclpy.init(args=['--ros-args', '-p', 'debug_view:=false'])
    node = ParkingNodeOsy()
    commands = []
    node.publish = lambda steer, speed: commands.append((int(steer), int(speed)))
    clock = [0.0]

    empty = make_scan()
    right_one = make_scan([(-165, -140, 1.15)])
    # The second vehicle is in the extended RIGHT-side field near its boundary.
    right_two = make_scan([(-175, -158, 1.50), (-110, -82, 0.85)])
    # During reverse, a second RIGHT bundle can appear near the boundary and
    # must be added as B2 while the original RIGHT bundle remains B1.
    reverse_pair = make_scan([(-175, -155, 1.35), (-110, -82, 0.80)])
    reverse_b2_left = make_scan([(-175, -155, 1.35), (82, 110, 0.80)])
    reverse_b2_approaching = make_scan([(-175, -155, 1.35), (82, 110, 0.75)])
    reverse_b2_close = make_scan([(-175, -155, 1.35), (82, 110, 0.45)])
    reverse_b2_receding = make_scan([(-175, -155, 1.35), (82, 110, 0.65)])
    reverse_single = make_scan([(-175, -150, 1.25)])

    def step(scan=empty, dt=0.1):
        clock[0] += dt
        node.scan_callback(scan)
        node.control_tick()

    def step_until(target, scan=empty, limit=200):
        for _ in range(limit):
            step(scan)
            if node.state == target:
                return
        raise AssertionError(f'did not reach {target}; current={node.state}')

    try:
        with patch('parking.parking_node_osy.time.monotonic', side_effect=lambda: clock[0]):
            node.state_started_at = clock[0]

            # Initial clutter is gone, then RIGHT follows the requested
            # 0→1→2→1 pattern.
            step_until(ParkingState.WAIT_CAR1_ENTRY, empty)
            for _ in range(3):
                step(right_one)
            assert node.state == ParkingState.WAIT_CAR2_ENTRY
            for _ in range(3):
                step(right_two)
            assert node.right_two_bundles_seen
            for _ in range(3):
                step(right_one)
            assert node.state == ParkingState.PASS_SECOND_CAR

            step_until(ParkingState.SET_REVERSE_STEER, empty)
            step_until(ParkingState.REVERSE_HARD_RIGHT, empty)

            # Reverse starts by reacquiring the seeded RIGHT B1 alone.  The
            # new RIGHT B2 must then persist for three real scans.
            step(reverse_single)
            assert node.observation.bundle1_visible
            assert not node.observation.bundle2_visible
            for _ in range(2):
                step(reverse_pair)
            assert node.state == ParkingState.REVERSE_HARD_RIGHT
            step(reverse_pair)
            assert node.reverse_pair_confirmed
            assert node.state == ParkingState.REVERSE_HARD_RIGHT
            for _ in range(2):
                step(reverse_b2_left)
            assert node.state == ParkingState.REVERSE_HARD_RIGHT
            step(reverse_b2_approaching)
            step(reverse_b2_close)
            assert node.state == ParkingState.REVERSE_HARD_RIGHT
            step(reverse_b2_receding)
            assert node.state == ParkingState.PARK_STOP
            assert commands[-1] == (0, 0)

            step_until(ParkingState.EXIT_SET_RIGHT_STEER, empty)
            step_until(ParkingState.EXIT_RIGHT_TURN, empty)
            step_until(ParkingState.EXIT_FORWARD, empty)
            step_until(ParkingState.DONE, empty)
            step(empty)
            assert commands[-1] == (0, node.forward_speed)

            # Failure cannot fall through to PARK_STOP/successful exit.
            node.transition(ParkingState.REVERSE_HARD_RIGHT, clock[0])
            step_until(ParkingState.PARKING_FAILED, empty)
            assert commands[-1] == (0, 0)

            # All-inf scans are a sensor fault, not an empty parking slot.
            node.transition(ParkingState.REVERSE_HARD_RIGHT, clock[0])
            node.invalid_scan_count = 0
            invalid = make_scan(invalid=True)
            for _ in range(node.invalid_scan_confirm_frames):
                step(invalid)
            assert node.state == ParkingState.EMERGENCY_STOP
            assert commands[-1] == (0, 0)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
