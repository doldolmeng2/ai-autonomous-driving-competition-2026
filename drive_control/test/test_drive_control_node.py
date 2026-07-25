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
