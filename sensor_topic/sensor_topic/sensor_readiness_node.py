"""Exit successfully after every required sensor topic has produced data."""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Float32MultiArray


DEFAULT_REQUIRED_TOPICS = [
    '/camera/high/image_raw',
    '/camera/low/image_raw',
    '/scan',
    '/ultrasonic/ranges',
]

TOPIC_TYPES = {
    '/camera/high/image_raw': Image,
    '/camera/low/image_raw': Image,
    '/scan': LaserScan,
    '/ultrasonic/ranges': Float32MultiArray,
}


class SensorReadinessNode(Node):
    """Observe one message per required topic, then let the launch proceed."""

    def __init__(self):
        super().__init__('sensor_readiness')
        self.declare_parameter('required_topics', DEFAULT_REQUIRED_TOPICS)
        self.required_topics = list(
            self.get_parameter('required_topics').get_parameter_value().string_array_value
        )
        unknown_topics = set(self.required_topics) - set(TOPIC_TYPES)
        if unknown_topics:
            raise ValueError(
                f'Unsupported sensor readiness topics: {sorted(unknown_topics)}'
            )

        self.received_topics = set()
        self.ready = not self.required_topics
        self._subscriptions = [
            self.create_subscription(
                TOPIC_TYPES[topic],
                topic,
                lambda _message, topic=topic: self._mark_ready(topic),
                qos_profile_sensor_data,
            )
            for topic in self.required_topics
        ]
        self.status_timer = self.create_timer(5.0, self._log_pending_topics)
        self.get_logger().info(
            f'Waiting for sensor data: {", ".join(self.required_topics)}'
        )

    def _mark_ready(self, topic):
        if topic in self.received_topics:
            return
        self.received_topics.add(topic)
        self.get_logger().info(f'Sensor ready: {topic}')
        if self.received_topics == set(self.required_topics):
            self.ready = True
            self.get_logger().info('All required sensors are ready')

    def _log_pending_topics(self):
        pending = set(self.required_topics) - self.received_topics
        if pending:
            self.get_logger().info(
                f'Still waiting for sensors: {", ".join(sorted(pending))}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = SensorReadinessNode()
    try:
        while rclpy.ok() and not node.ready:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
