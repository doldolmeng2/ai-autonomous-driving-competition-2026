import math

import pytest

from sensor_topic.arduino_communication_node import (
    ArduinoCommunicationNode,
    parse_ultrasonic_line,
)


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(list(message.data))


class FakeSerial:
    def __init__(self):
        self.data = bytearray()
        self.events = []

    @property
    def in_waiting(self):
        return len(self.data)

    def feed(self, data):
        self.data.extend(data)

    def read(self, size):
        self.events.append('read')
        result = bytes(self.data[:size])
        del self.data[:size]
        return result

    def write(self, data):
        self.events.append(('write', data))
        return len(data)

    def flush(self):
        self.events.append('flush')


def test_parse_complete_ultrasonic_frame():
    values = parse_ultrasonic_line('U,0.898,2.411,nan,0.850,1.200,0.300')
    assert values[:2] == [0.898, 2.411]
    assert math.isinf(values[2])
    assert values[3:] == [0.850, 1.200, 0.300]


def test_reject_partial_or_malformed_frame():
    assert parse_ultrasonic_line('U,0.991') is None
    assert parse_ultrasonic_line('U') is None
    assert parse_ultrasonic_line('U,1,2,bad,4,5,6') is None
    assert parse_ultrasonic_line('debug') is None


def test_fragmented_and_bad_frames_do_not_close_serial():
    node = object.__new__(ArduinoCommunicationNode)
    node.serial = FakeSerial()
    node.rx_buffer = bytearray()
    node.sensor_count = 6
    node.debug_serial_lines = False
    node.ultrasonic_publisher = FakePublisher()

    node.serial.feed(b'U,0.1,0.2,0.')
    node.read_available()
    assert node.ultrasonic_publisher.messages == []

    node.serial.feed(b'3,0.4,0.5,0.6\nU,broken\n')
    node.read_available()
    assert len(node.ultrasonic_publisher.messages) == 1
    assert node.ultrasonic_publisher.messages[0] == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    )
    assert node.serial is not None


def test_poll_writes_motor_command_before_reading_ultrasonics():
    node = object.__new__(ArduinoCommunicationNode)
    node.serial = FakeSerial()
    node.serial.feed(b'U,1,2,3,4,5,6\n')
    node.rx_buffer = bytearray()
    node.sensor_count = 6
    node.debug_serial_lines = False
    node.ultrasonic_publisher = FakePublisher()
    node.command = (15, -20)
    node.last_command_received = __import__('time').monotonic()
    node.command_timeout_sec = 0.5

    node.poll()

    assert node.serial.events[0] == ('write', b'15 -20\n')
    assert node.serial.events[2] == 'read'
    assert node.ultrasonic_publisher.messages == [[1, 2, 3, 4, 5, 6]]


def test_bridge_watchdog_sends_stop_for_stale_command():
    node = object.__new__(ArduinoCommunicationNode)
    node.serial = FakeSerial()
    node.rx_buffer = bytearray()
    node.command = (150, 130)
    node.last_command_received = 0.0
    node.command_timeout_sec = 0.5

    node.poll()

    assert node.serial.events[0] == ('write', b'0 0\n')
