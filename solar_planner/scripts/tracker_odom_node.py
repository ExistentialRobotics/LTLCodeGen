#!/usr/bin/env python

import rospy
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

def publish_odom():
    rospy.init_node('tracker_odom_node', anonymous=True)

    # Publisher for the /tracker_odom topic
    odom_pub = rospy.Publisher('/tracker_odom', Odometry, queue_size=10)

    # TF listener to get transformations
    listener = tf.TransformListener()

    rate = rospy.Rate(10.0)  # 10 Hz
    while not rospy.is_shutdown():
        try:
            # Get the transformation from world to husky_1/base_link
            (trans, rot) = listener.lookupTransform('/world', '/husky_1/base_link', rospy.Time(0))

            # Create an Odometry message
            odom_msg = Odometry()

            # Set the header
            odom_msg.header.stamp = rospy.Time.now()
            odom_msg.header.frame_id = "world"
            odom_msg.child_frame_id = "husky_1/base_link"

            # Set the pose (position and orientation)
            odom_msg.pose.pose.position.x = trans[0]
            odom_msg.pose.pose.position.y = trans[1]
            odom_msg.pose.pose.position.z = trans[2]
            odom_msg.pose.pose.orientation.x = rot[0]
            odom_msg.pose.pose.orientation.y = rot[1]
            odom_msg.pose.pose.orientation.z = rot[2]
            odom_msg.pose.pose.orientation.w = rot[3]

            # Set pose covariance (optional, here it's set to zero)
            odom_msg.pose.covariance = [0] * 36

            # Set twist (velocity) to zero, as TF doesn't provide velocity directly
            odom_msg.twist.twist = Twist()
            odom_msg.twist.covariance = [0] * 36

            # Publish the odometry message
            odom_pub.publish(odom_msg)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("TF lookup failed")
            continue

        rate.sleep()

if __name__ == '__main__':
    try:
        publish_odom()
    except rospy.ROSInterruptException:
        pass
