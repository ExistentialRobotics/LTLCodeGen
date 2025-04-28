#!/usr/bin/env python

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid

class OccupancyMapUnique:
    def __init__(self):
        # Initialize the node
        rospy.init_node('occupancy_map_unique', anonymous=True)
        
        # Parameters for topic subscription
        self.map_topic = rospy.get_param("~map_topic", "/label_map")
        rospy.loginfo(f"Subscribing to topic: {self.map_topic}")
        
        # Subscribe to the occupancy grid
        self.map_subscriber = rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_callback)
        
        rospy.loginfo("Occupancy map unique IDs node started.")
        rospy.spin()

    def map_callback(self, msg):
        """
        Callback function to process the occupancy grid data.
        """
        try:
            # Extract map dimensions and resolution
            width = msg.info.width
            height = msg.info.height
            resolution = msg.info.resolution
            rospy.loginfo(f"Map received: width={width}, height={height}, resolution={resolution:.2f}m")

            # Convert the flattened map data to a 2D NumPy array
            map_data = np.array(msg.data, dtype=np.int8).reshape((height, width))
            
            # Identify unique IDs in the map
            unique_ids = np.unique(map_data)
            rospy.loginfo(f"Unique IDs in the map: {unique_ids}")
            print(f"Unique IDs: {unique_ids}")

        except Exception as e:
            rospy.logerr(f"Error processing the map data: {e}")

if __name__ == '__main__':
    try:
        OccupancyMapUnique()
    except rospy.ROSInterruptException:
        rospy.loginfo("Occupancy map unique IDs node terminated.")
