#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
from nav_msgs.msg import OccupancyGrid
import os
import signal

from rclpy.qos import QoSProfile, QoSReliabilityPolicy


class OccupancyMapSaver(Node):
    def __init__(self):
        super().__init__("map_saver")

        # Initialize map variables
        self.occ_map = None
        self.semantic_map = None

        # Map save path
        default_save_path = os.path.expanduser("~/ssmi_mapping/saved_maps")
        os.makedirs(default_save_path, exist_ok=True)

        # Declare parameters
        self.declare_parameter("occ_map_topic", "/occupancy_map_2D")
        self.declare_parameter("semantic_map_topic", "/semantic_map_2D")
        self.declare_parameter("save_dir", default_save_path)
        self.declare_parameter("occ_file_name", "occ_map.npy")
        self.declare_parameter("semantic_file_name", "semantic_map.npy")
        self.declare_parameter("occ_metadata_file_name", "occ_map_metadata.npy")
        self.declare_parameter("semantic_metadata_file_name", "semantic_map_metadata.npy")

        # Topic to subscribe to
        self.occ_map_topic = self.get_parameter("occ_map_topic").value
        self.semantic_map_topic = self.get_parameter("semantic_map_topic").value

        # Directory to save the numpy array
        self.save_dir = self.get_parameter("save_dir").value

        self.occ_file_name = self.get_parameter("occ_file_name").value
        self.semantic_file_name = self.get_parameter("semantic_file_name").value

        self.occ_metadata_file_name = self.get_parameter("occ_metadata_file_name").value
        self.semantic_metadata_file_name = self.get_parameter("semantic_metadata_file_name").value

        self.occ_metadata_file_name = self.get_parameter("occ_metadata_file_name").value
        self.semantic_metadata_file_name = self.get_parameter("semantic_metadata_file_name").value

        # Full path for the numpy file
        self.occ_file_path = os.path.join(self.save_dir, self.occ_file_name)
        self.semantic_file_path = os.path.join(self.save_dir, self.semantic_file_name)

        self.occ_metadata_file_path = os.path.join(self.save_dir, self.occ_metadata_file_name)
        self.semantic_metadata_file_path = os.path.join(self.save_dir, self.semantic_metadata_file_name)

        # QoS configuration
        qos_profile = QoSProfile(depth=100)
        qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT

        # Subscribe to the occupancy grid
        self.create_subscription(OccupancyGrid, self.occ_map_topic, self.occ_callback, qos_profile)
        self.get_logger().info(f"Subscribed to occ topic: {self.occ_map_topic}")

        # Subscribe to the semantic grid
        self.create_subscription(OccupancyGrid, self.semantic_map_topic, self.semantic_callback, qos_profile)
        self.get_logger().info(f"Subscribed to semantic topic: {self.semantic_map_topic}")

        # Set up signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown_handler)

    def occ_callback(self, msg):
        self.occ_map = msg

    def semantic_callback(self, msg):
        self.semantic_map = msg

    def save_map(self, msg, file_path, metadata_file_path):
        if msg is None:
            self.get_logger().warning("Received None message, skipping save.")
            return

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_file_path), exist_ok=True)

        # Extract metadata
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin = msg.info.origin
        self.get_logger().info(f"Map received: width={width}, height={height}, resolution={resolution}")

        # Convert the data to a numpy array and reshape to 2D
        map_data = np.array(msg.data, dtype=np.int8).reshape((height, width))

        # Optionally, save additional metadata
        metadata = {
            "width": width,
            "height": height,
            "resolution": resolution,
            "origin": {
                "position": {
                    "x": origin.position.x,
                    "y": origin.position.y,
                    "z": origin.position.z,
                },
                "orientation": {
                    "x": origin.orientation.x,
                    "y": origin.orientation.y,
                    "z": origin.orientation.z,
                    "w": origin.orientation.w,
                },
            },
        }

        np.save(file_path, map_data)
        self.get_logger().info(f"Map saved as numpy array at: {file_path}")

        np.save(metadata_file_path, metadata)
        self.get_logger().info(f"Metadata saved as numpy array at: {metadata_file_path}")

    def shutdown_handler(self, signum, frame):
        self.get_logger().info("Shutting down... Saving maps.")

        self.save_map(self.occ_map, self.occ_file_path, self.occ_metadata_file_path)
        self.save_map(self.semantic_map, self.semantic_file_path, self.semantic_metadata_file_path)

        self.get_logger().info("Maps saved successfully. Exiting node.")
        rclpy.shutdown()


def main():
    rclpy.init()
    node = OccupancyMapSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
