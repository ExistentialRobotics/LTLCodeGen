#!/usr/bin/env python3

import os
import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

def load_map_from_npy(meta_file, data_file):
    # Load metadata and map data from .npy files
    map_metadata = np.load(meta_file, allow_pickle=True).item()  # Metadata stored as a dictionary
    occupancy_array = np.load(data_file)

    # Return the modified array and metadata
    return occupancy_array, map_metadata

def publish_occupancy_map(occupancy_meta_file_path,
                          occupancy_data_file_path,
                          semantic_meta_file_path,
                          semantic_data_file_path):

    # Initialize the ROS node
    rospy.init_node('map_publisher', anonymous=True)

    # Create a publisher for the OccupancyGrid message
    pub_occ = rospy.Publisher('/occupancy_map_2D', OccupancyGrid, queue_size=1)
    pub_sem = rospy.Publisher('/semantic_map', OccupancyGrid, queue_size=1)

    # Load occupancy map data and metadata
    occ_map_array, occ_map_metadata = load_map_from_npy(occupancy_meta_file_path, occupancy_data_file_path)
    occ_height, occ_width = occ_map_array.shape

    # Load label map data and metadata
    semantic_map_array, semantic_map_metadata = load_map_from_npy(semantic_meta_file_path, semantic_data_file_path)
    semantic_height, semantic_width = semantic_map_array.shape

    print("Unique class IDs in semantic map:")
    print(np.unique(semantic_map_array))  # Check unique values for debugging

    # Create the OccupancyGrid message
    occupancy_grid_msg = OccupancyGrid()
    occupancy_grid_msg.header = Header()
    # occupancy_grid_msg.header.frame_id = "odom"  # Frame ID for the map
    occupancy_grid_msg.header.frame_id = "world"  # Frame ID for the map

    occupancy_grid_msg.info.resolution = occ_map_metadata['resolution']  # Resolution from the metadata
    occupancy_grid_msg.info.width = occ_width
    occupancy_grid_msg.info.height = occ_height
    occupancy_grid_msg.info.origin.position.x = occ_map_metadata['origin']['position']['x']
    occupancy_grid_msg.info.origin.position.y = occ_map_metadata['origin']['position']['y']
    occupancy_grid_msg.info.origin.position.z = 0.0
    occupancy_grid_msg.info.origin.orientation.w = 1.0

    # Flatten the occupancy array and convert it to a list
    occupancy_grid_msg.data = occ_map_array.flatten().tolist()

    # Create the OccupancyGrid message
    semantic_grid_msg = OccupancyGrid()
    semantic_grid_msg.header = Header()
    # semantic_grid_msg.header.frame_id = "odom"  # Frame ID for the map
    semantic_grid_msg.header.frame_id = "world"  # Frame ID for the map

    semantic_grid_msg.info.resolution = semantic_map_metadata['resolution']  # Resolution from the metadata
    semantic_grid_msg.info.width = semantic_width
    semantic_grid_msg.info.height = semantic_height
    semantic_grid_msg.info.origin.position.x = semantic_map_metadata['origin']['position']['x']
    semantic_grid_msg.info.origin.position.y = semantic_map_metadata['origin']['position']['y']
    semantic_grid_msg.info.origin.position.z = 0.0
    semantic_grid_msg.info.origin.orientation.w = 1.0

    # Flatten the occupancy array and convert it to a list
    semantic_grid_msg.data = semantic_map_array.flatten().tolist()

    # Publish the occupancy grid
    rate = rospy.Rate(1)  # 1 Hz
    while not rospy.is_shutdown():
        # Publish occupancy map
        occupancy_grid_msg.header.stamp = rospy.Time.now()  # Update the timestamp
        pub_occ.publish(occupancy_grid_msg)

        # Publish semantic map
        semantic_grid_msg.header.stamp = rospy.Time.now()  # Update the timestamp
        pub_sem.publish(semantic_grid_msg)
        rate.sleep()

PROJ_DIR = os.path.join(os.path.dirname(__file__), "../../")
PROJ_DIR = os.path.realpath(PROJ_DIR)

if __name__ == '__main__':
    try:
        # Proviude the paths to occupancy map files
        occupancy_meta_file_path = f"{PROJ_DIR}/label_map/maps/occ_map_metadata_real_env1.npy"
        occupancy_data_file_path = f"{PROJ_DIR}/label_map/maps/occ_map_real_env1.npy"

        # Provide the paths to semantic map files

        semantic_meta_file_path = f"{PROJ_DIR}/label_map/maps/semantic_map_metadata_real_env1.npy"
        semantic_data_file_path = f"{PROJ_DIR}/label_map/maps/semantic_map_real_env1.npy"

        publish_occupancy_map(occupancy_meta_file_path,
                              occupancy_data_file_path,
                              semantic_meta_file_path,
                              semantic_data_file_path)
    except rospy.ROSInterruptException:
        pass
