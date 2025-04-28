import rospy
from octomap_msgs.msg import Octomap
import numpy as np

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

def callback(octomap_msg):
    """Callback function for the Octomap subscriber."""
    try:
        unique_colors = parse_semantic_colors(octomap_msg)
        rospy.loginfo(f"Unique semantic colors: {unique_colors}")
    except Exception as e:
        rospy.logerr(f"Error parsing semantic colors: {e}")

def main():
    rospy.init_node('semantic_colors_node', anonymous=True)

    # Replace 'semantic_octomap_topic' with your topic name
    topic_name = rospy.get_param('~topic_name', '/octomap_full')

    rospy.Subscriber(topic_name, Octomap, callback)

    rospy.loginfo(f"Subscribed to topic: {topic_name}")

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
