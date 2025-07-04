#!/usr/bin/env python
import rospy
from std_msgs.msg import String

def publisher():
    
    rospy.init_node('ap_publisher', anonymous=True)
    # Get ROS params
    ap_dict_topic = rospy.get_param("~dict_topic", "/ap_dict")
    ap_id_topic = rospy.get_param("~id_topic", "/ap_id")
    # Set up AP Dict and Id publishers
    dict_pub = rospy.Publisher(ap_dict_topic, String, queue_size=50)
    id_pub = rospy.Publisher(ap_id_topic, String, queue_size=50)
    rate = rospy.Rate(rospy.get_param('~rate', 1))

    dict_message = rospy.get_param("~ap_dict_message")
    id_message = rospy.get_param("~ap_id_message")

    # Publish AP Dict and Id
    while not rospy.is_shutdown():
        dict_pub.publish(dict_message)
        id_pub.publish(id_message)
        rate.sleep()

if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
