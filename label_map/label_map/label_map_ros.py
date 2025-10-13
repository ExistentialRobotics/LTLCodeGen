#!/usr/bin/env python3

import numpy as np
import ast

from label_map.labelmap_radius import generate_label_map

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String


class LabelMap(Node):
    def __init__(self):
        super().__init__('label_map_node')

        # Read parameters
        self.declare_parameter("radius", 1)
        self.declare_parameter("ap_dict", "ap_dict")
        self.declare_parameter("ap_id", "ap_id")
        self.declare_parameter("semantic_map", "semantic_map")
        self.declare_parameter("label_map_topic", "label_map")
        self.declare_parameter("label_map_viz_topic", "label_map_viz")

        self.radius = self.get_parameter("radius").value
        self.ap_dict_topic = self.get_parameter("ap_dict").value
        self.ap_id_topic = self.get_parameter("ap_id").value
        self.semantic_map_topic = self.get_parameter("semantic_map").value
        self.label_map_topic = self.get_parameter("label_map_topic").value
        self.label_map_viz_topic = self.get_parameter("label_map_viz_topic").value

        # ROS communication
        qos_profile = QoSProfile(depth=100)
        qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT

        self.sub_ap_dict = self.create_subscription(String, self.ap_dict_topic, self.ap_dict_cb, qos_profile)
        self.sub_ap_id = self.create_subscription(String, self.ap_id_topic, self.ap_id_cb, qos_profile)
        self.sub_semantic_map = self.create_subscription(OccupancyGrid, self.semantic_map_topic, self.semantic_cb, qos_profile)
        self.pub_label_map = self.create_publisher(OccupancyGrid, self.label_map_topic, 1)
        self.pub_label_map_viz = self.create_publisher(OccupancyGrid, self.label_map_viz_topic, 1)

        self.ap_id = None
        self.ap_dict = None

    def semantic_cb(self, map):
        # Get semantic class data
        if self.ap_dict and self.ap_id:
            height, width = map.info.height, map.info.width
            semantic_map = np.array(map.data, dtype=np.int8).reshape((height, width)) # convert into array
            try:
                label_map, label_map_viz = generate_label_map(semantic_map, self.ap_dict, self.ap_id, self.radius)

                # Publish label map
                label_msg = OccupancyGrid()
                label_msg.header = map.header
                label_msg.header.stamp = Clock().now().to_msg()
                label_msg.info = map.info
                label_msg.data = label_map.flatten().tolist()
                self.pub_label_map.publish(label_msg)

                # Publish label map for visualization
                viz_msg = OccupancyGrid()
                viz_msg.header = map.header
                viz_msg.header.stamp = Clock().now().to_msg()
                viz_msg.info = map.info
                viz_msg.data = label_map_viz.flatten().tolist()
                self.pub_label_map_viz.publish(viz_msg)

            except Exception as e:
                self.get_logger().info(f"{e}")

    def ap_dict_cb(self, msg):
        self.ap_dict = ast.literal_eval(msg.data)

    def ap_id_cb(self, msg):
        self.ap_id = ast.literal_eval(msg.data)


def main():
    rclpy.init()
    node = LabelMap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
