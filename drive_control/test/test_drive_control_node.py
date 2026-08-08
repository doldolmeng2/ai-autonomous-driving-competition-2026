import pytest

from drive_control.drive_control_node import (
    DriveControlNode,
    MISSION_STEER_PID_KD,
    MISSION_STEER_PID_KI,
    MISSION_STEER_PID_KP,
    PARKING_STEER_PID_KD,
    PARKING_STEER_PID_KI,
    PARKING_STEER_PID_KP,
    TIMED_STEER_PID_KD,
    TIMED_STEER_PID_KI,
    TIMED_STEER_PID_KP,
    steer_pid_gains,
)


class FakePublisher:
    def __init__(self):
        self.data = None

    def publish(self, message):
        self.data = list(message.data)


@pytest.mark.parametrize(
    ('profile', 'expected'),
    [
        (
            'timed',
            (TIMED_STEER_PID_KP, TIMED_STEER_PID_KI, TIMED_STEER_PID_KD),
        ),
        (
            'mission',
            (
                MISSION_STEER_PID_KP,
                MISSION_STEER_PID_KI,
                MISSION_STEER_PID_KD,
            ),
        ),
        (
            'parking',
            (
                PARKING_STEER_PID_KP,
                PARKING_STEER_PID_KI,
                PARKING_STEER_PID_KD,
            ),
        ),
    ],
)
def test_named_steering_pid_profiles(profile, expected):
    assert steer_pid_gains(profile) == (profile, expected)


def test_unknown_steering_pid_profile_is_rejected():
    with pytest.raises(ValueError, match='Unknown steer_pid_profile'):
        steer_pid_gains('unknown')


def test_pwm_command_is_published_without_serial_access():
    node = object.__new__(DriveControlNode)
    node.max_drive_pwm = 130
    node.command_publisher = FakePublisher()

    node.publish_command(180, 200)

    assert node.command_publisher.data == [180, 130]


def make_drive_ramp_node():
    node = object.__new__(DriveControlNode)
    node.max_drive_pwm = 140
    node.command_rate_hz = 20.0
    node.drive_accel_duration_sec = 0.75
    node.drive_decel_duration_sec = 0.75
    node.current_drive_pwm = 0.0
    return node


def test_drive_acceleration_reaches_maximum_in_075_seconds():
    node = make_drive_ramp_node()

    outputs = [node.ramp_drive_output(140) for _ in range(15)]

    assert outputs[-2] < 140
    assert outputs[-1] == 140


def test_drive_deceleration_reaches_zero_in_075_seconds():
    node = make_drive_ramp_node()
    node.current_drive_pwm = 140.0

    outputs = [node.ramp_drive_output(0) for _ in range(15)]

    assert outputs[-2] > 0
    assert outputs[-1] == 0


def make_closed_loop_node():
    node = object.__new__(DriveControlNode)
    node.steer_raw_left = 560
    node.steer_raw_center = 490
    node.steer_raw_right = 420
    node.steer_max_angle_deg = 45.0
    node.steer_angle_tolerance_deg = 1.0
    node.steer_pwm = 150
    node.steer_min_pwm = 40
    node.steer_pid_kp = 2.0
    node.steer_pid_ki = 0.0
    node.steer_pid_kd = 0.8
    node.steer_pid_integral_limit_pwm = 30.0
    node.pid_integral_error = 0.0
    node.steer_angle_velocity_deg_per_sec = 0.0
    node.command_rate_hz = 20.0
    node.target_steer_angle_deg = 0.0
    node.steer_angle_deg = 0.0
    return node


def test_raw_feedback_maps_to_calibrated_angles():
    node = make_closed_loop_node()

    assert node.raw_to_steer_angle(560) == -45.0
    assert node.raw_to_steer_angle(490) == 0.0
    assert node.raw_to_steer_angle(420) == 45.0
    assert node.raw_to_steer_angle(525) == -22.5
    assert node.raw_to_steer_angle(455) == 22.5


def test_positive_error_commands_right_pwm():
    node = make_closed_loop_node()
    node.target_steer_angle_deg = 45.0
    node.steer_angle_deg = -45.0

    assert node.calculate_steer_pwm(0.05) == 150


def test_negative_error_commands_left_pwm():
    node = make_closed_loop_node()
    node.target_steer_angle_deg = -45.0
    node.steer_angle_deg = 45.0

    assert node.calculate_steer_pwm(0.05) == -150


def test_pwm_stops_inside_angle_tolerance():
    node = make_closed_loop_node()
    node.target_steer_angle_deg = 10.0
    node.steer_angle_deg = 9.5

    assert node.calculate_steer_pwm(0.05) == 0


def test_pid_reduces_pwm_near_target():
    node = make_closed_loop_node()
    node.target_steer_angle_deg = 10.0
    node.steer_angle_deg = 0.0

    assert node.calculate_steer_pwm(0.05) == 40


def test_pid_integral_is_bounded_during_saturation():
    node = make_closed_loop_node()
    node.target_steer_angle_deg = 45.0
    node.steer_angle_deg = -45.0

    for _ in range(100):
        assert node.calculate_steer_pwm(0.05) == 150

    assert node.pid_integral_error == 0.0
