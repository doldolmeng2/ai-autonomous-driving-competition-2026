from drive_control.drive_control_node import DriveControlNode


class FakePublisher:
    def __init__(self):
        self.data = None

    def publish(self, message):
        self.data = list(message.data)


def test_pwm_command_is_published_without_serial_access():
    node = object.__new__(DriveControlNode)
    node.max_drive_pwm = 130
    node.command_publisher = FakePublisher()

    node.publish_command(180, 200)

    assert node.command_publisher.data == [180, 130]


def make_steer_filter_node():
    node = object.__new__(DriveControlNode)
    node.steer_accumulation_threshold_deg = 15.0
    node.steer_accumulator_deg = 0.0
    return node


def test_small_steer_is_zero_until_accumulation_exceeds_threshold():
    node = make_steer_filter_node()

    assert node.filter_steer_target(5) == 0.0
    assert node.filter_steer_target(5) == 0.0
    assert node.filter_steer_target(5) == 0.0
    assert node.steer_accumulator_deg == 15.0

    assert node.filter_steer_target(1) == 16.0
    assert node.steer_accumulator_deg == 0.0


def test_negative_small_steer_accumulates_and_resets():
    node = make_steer_filter_node()

    assert node.filter_steer_target(-8) == 0.0
    assert node.filter_steer_target(-8) == -16.0
    assert node.steer_accumulator_deg == 0.0


def test_large_steer_is_added_to_pending_accumulation_then_resets():
    node = make_steer_filter_node()

    assert node.filter_steer_target(10) == 0.0
    assert node.filter_steer_target(20) == 30.0
    assert node.steer_accumulator_deg == 0.0
