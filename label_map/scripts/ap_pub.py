#!/usr/bin/env python
import rospy
from std_msgs.msg import String

def publisher():
    
    rospy.init_node('ap_dict_publisher', anonymous=True)
    ap_topic = rospy.get_param("~topic", "/ap_dict")
    pub = rospy.Publisher(ap_topic, String, queue_size=50)
    rate = rospy.Rate(rospy.get_param('~rate', 1))

    message = rospy.get_param("~ap_dict_message")

    while not rospy.is_shutdown():
        pub.publish(message)
        rate.sleep()

if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
