#!/usr/bin/env python3

import rospy
import numpy as np
import ast
from nav_msgs.msg import OccupancyGrid
from labelmap_radius import generate_label_map
from std_msgs.msg import String

class Labelmap:
    def __init__(self):
        # Read parameters
        self.radius = rospy.get_param("~radius", 1) # radius is matches grid resolution
        self.ap_dict_topic = rospy.get_param("~ap_dict", "ap_dict") # dictionary matching ap to task
        self.ap_id_topic = rospy.get_param("~ap_id", "ap_id")
        self.semantic_map_topic = rospy.get_param("~semantic_map", "semantic_map") # bird's eye view

        self.label_map_topic = rospy.get_param("~label_map_topic", "label_map") # this is what we will publish
        self.label_map_viz_topic = rospy.get_param("~label_map_viz_topic", "label_map_viz")

        # ROS communication
        self.sub_ap_dict = rospy.Subscriber(self.ap_dict_topic, String, self.ap_dict_cb, queue_size= 1)
        self.sub_ap_id = rospy.Subscriber(self.ap_id_topic, String, self.ap_id_cb, queue_size= 1)
        self.sub_semantic_map = rospy.Subscriber(self.semantic_map_topic, OccupancyGrid, self.semantic_cb, queue_size= 1) # this comes from ssmi
        self.pub_label_map = rospy.Publisher(self.label_map_topic, OccupancyGrid, queue_size = 1)
        self.pub_label_map_viz = rospy.Publisher(self.label_map_viz_topic, OccupancyGrid, queue_size = 1)

        self.ap_id = None
        self.ap_dict = None

    def semantic_cb(self, map):
        # Get semantic class data
        if self.ap_dict and self.ap_id:
            height, width = map.info.height, map.info.width
            semantic_map = np.array(map.data, dtype=np.int8).reshape((height, width)) # this will turn into array
            try:
                label_map, label_map_viz = generate_label_map(semantic_map, self.ap_dict, self.ap_id, self.radius)

                # Publish label map
                label_msg = OccupancyGrid()
                label_msg.header = map.header
                label_msg.header.stamp = rospy.Time.now() # Set current time
                label_msg.info = map.info
                label_msg.data = label_map.flatten().tolist()
                self.pub_label_map.publish(label_msg)

                # Publish label map for visualization
                viz_msg = OccupancyGrid()
                viz_msg.header = map.header
                viz_msg.header.stamp = rospy.Time.now() # Set current time
                viz_msg.info = map.info
                viz_msg.data = label_map_viz.flatten().tolist()
                self.pub_label_map_viz.publish(viz_msg)

            except Exception as e:
                rospy.logerr(e)

    def ap_dict_cb(self, msg):
        self.ap_dict = ast.literal_eval(msg.data)

    def ap_id_cb(self, msg):
        rospy.loginfo(f"Received message data: {msg.data}")
        self.ap_id = ast.literal_eval(msg.data)



if __name__ == "__main__":
    # Initialize ROS node
    rospy.init_node("label_map_node")
    label_map = Labelmap()
    rospy.spin()
