#!/usr/bin/env python

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
import os
import signal


class OccupancyMapSaver:
    def __init__(self):
        rospy.init_node("map_saver", anonymous=True)

        # Initialize map variables
        self.occ_map = None
        self.semantic_map = None

        # Topic to subscribe to
        self.occ_map_topic = rospy.get_param("~occ_map_topic", "/occupancy_map_2D")
        self.semantic_map_topic = rospy.get_param(
            "~semantic_map_topic", "/semantic_map_2D"
        )

        # Directory to save the numpy array
        self.save_dir = rospy.get_param("~save_dir", "/home/brabiei/SOLAR_WS/src/SOLAR/label_map/maps")

        self.occ_file_name = rospy.get_param("~file_name", "occ_map.npy")
        self.semantic_file_name = rospy.get_param(
            "~semantic_file_name", "semantic_map.npy"
        )

        self.occ_metadata_file_name = rospy.get_param(
            "~occ_metadata_file_name", "occ_map_metadata.npy"
        )
        self.semantic_metadata_file_name = rospy.get_param(
            "~semantic_metadata_file_name", "semantic_map_metadata.npy"
        )

        # Full path for the numpy file
        self.occ_file_path = os.path.join(self.save_dir, self.occ_file_name)
        self.semantic_file_path = os.path.join(self.save_dir, self.semantic_file_name)

        self.occ_metadata_file_path = os.path.join(
            self.save_dir, self.occ_metadata_file_name
        )
        self.semantic_metadata_file_path = os.path.join(
            self.save_dir, self.semantic_metadata_file_name
        )

        # Subscribe to the occupancy grid
        rospy.Subscriber(self.occ_map_topic, OccupancyGrid, self.occ_callback)
        rospy.loginfo(f"Subscribed to occ topic: {self.occ_map_topic}")

        # Subscribe to the semantic grid
        rospy.Subscriber(self.semantic_map_topic, OccupancyGrid, self.semantic_callback)
        rospy.loginfo(f"Subscribed to semantic topic: {self.semantic_map_topic}")

        # Set up signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self.shutdown_handler)

        rospy.spin()

    def occ_callback(self, msg):
        self.occ_map = msg

    def semantic_callback(self, msg):
        self.semantic_map = msg

    def save_map(self, msg, file_path, metadata_file_path):
        # Extract metadata
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin = msg.info.origin
        rospy.loginfo(
            f"Map received: width={width}, height={height}, resolution={resolution}"
        )

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
        rospy.loginfo(f"Map saved as numpy array at: {file_path}")

        np.save(metadata_file_path, metadata)
        rospy.loginfo(f"Metadata saved as numpy array at: {metadata_file_path}")

    def shutdown_handler(self, signum, frame):
        rospy.loginfo("Shutting down... Saving maps.")

        self.save_map(self.occ_map, self.occ_file_path, self.occ_metadata_file_path)
        self.save_map(
            self.semantic_map, self.semantic_file_path, self.semantic_metadata_file_path
        )

        rospy.loginfo("Maps saved successfully. Exiting node.")
        rospy.signal_shutdown("Maps saved and node terminated.")


if __name__ == "__main__":
    try:
        OccupancyMapSaver()
    except rospy.ROSInterruptException:
        rospy.loginfo("Occupancy map saver node terminated.")
