#!/usr/bin/env python3


import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class ImageInverter:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('image_inverter', anonymous=True)
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Get parameters from parameter server
        input_topic = rospy.get_param('~input_topic', '/jackal1/camera/color/image_raw')
        
        # Create subscriber and publisher
        self.image_sub = rospy.Subscriber(input_topic, Image, self.image_callback)
        self.image_pub = rospy.Publisher("/output_img", Image, queue_size=1)

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            
            flipped_image = cv2.flip(cv_image, -1)
            
            # Convert back to ROS Image message
            flipped_msg = self.bridge.cv2_to_imgmsg(flipped_image, encoding=msg.encoding)
            
            # Copy header information from original message
            flipped_msg.header = msg.header
            flipped_msg.height = msg.height
            flipped_msg.width = msg.width
            flipped_msg.encoding = msg.encoding
            flipped_msg.is_bigendian = msg.is_bigendian
            flipped_msg.step = msg.step
            
            # Publish the flipped image
            self.image_pub.publish(flipped_msg)
            
        except CvBridgeError as e:
            rospy.logerr("CV Bridge Error: %s", e)
        except Exception as e:
            rospy.logerr("Error processing image: %s", e)


def main():
    try:
        # Create and run the image inverter
        inverter = ImageInverter()
        
        # Keep the node running
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Image Inverter Node Shutdown")
    except Exception as e:
        rospy.logerr("Error in main: %s", e)


if __name__ == '__main__':
    main()
