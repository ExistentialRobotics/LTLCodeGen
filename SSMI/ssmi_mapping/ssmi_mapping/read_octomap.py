#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from octomap_msgs.msg import Octomap

from rclpy.qos import QoSProfile, QoSReliabilityPolicy


class ReadOctoMap(Node):
    def __init__(self):
        super().__init__("semantic_colors_node")

        # Declare and read the topic parameter
        self.declare_parameter("topic_name", "/octomap_full")
        topic_name = self.get_parameter("topic_name").value

        # QoS configuration
        qos_profile = QoSProfile(depth=100)
        qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT

        # Create the subscriber
        self.subscription = self.create_subscription(
            Octomap,
            topic_name,
            self.callback,
            qos_profile
        )

        self.get_logger().info(f"Subscribed to topic: {topic_name}")

    @staticmethod
    def parse_semantic_colors(octomap_msg):
        """
        Parse the semantic colors from the octomap message.
        This function should be adjusted based on how semantic colors
        are stored in the octomap message.

        Args:
            octomap_msg: The incoming Octomap message.

        Returns:
            A set of unique semantic colors.
        """
        # For example purposes, let's assume colors are stored as part of the data field.
        # Replace this logic with actual parsing based on your message format.
        semantic_colors = set()

        # Assuming `octomap_msg.data` contains color information (replace as needed)
        for i in range(0, len(octomap_msg.data), 4):  # Example: each color is 4 bytes (RGBA)
            r = octomap_msg.data[i]
            g = octomap_msg.data[i + 1]
            b = octomap_msg.data[i + 2]
            a = octomap_msg.data[i + 3]
            semantic_colors.add((r, g, b, a))

        return semantic_colors

    def callback(self, octomap_msg):
        """Callback function for the Octomap subscriber."""
        try:
            unique_colors = self.parse_semantic_colors(octomap_msg)
            self.get_logger().info(f"Unique semantic colors: {unique_colors}")
        except Exception as e:
            self.get_logger().error(f"Error parsing semantic colors: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ReadOctoMap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
