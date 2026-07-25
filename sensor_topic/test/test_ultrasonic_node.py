import math

from sensor_topic.ultrasonic_node import UltrasonicNode


class Message:
    def __init__(self, data):
        self.data = data


def test_incomplete_bridge_array_is_ignored():
    node = object.__new__(UltrasonicNode)
    node.sensor_names = ['1', '2', '3', '4', '5', '6']
    published = []
    node.publish = lambda values: published.append(values)

    node.raw_callback(Message([1.0, 2.0]))

    assert published == []


def test_invalid_distance_becomes_no_echo():
    assert math.isinf(UltrasonicNode.normalize_distance(float('nan')))
    assert math.isinf(UltrasonicNode.normalize_distance(-1.0))
