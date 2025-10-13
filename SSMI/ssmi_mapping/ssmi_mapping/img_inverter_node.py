#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class ImageInverter(Node):
    def __init__(self):
        super().__init__('image_inverter')

        # Declare parameters
        self.declare_parameter('input_topic', '/jackal1/camera/color/image_raw')
        self.declare_parameter('output_topic', '/output_img')

        # Get parameters from parameter server
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        # Initialize CV Bridge
        self.bridge = CvBridge()

        # QoS configuration
        qos_profile = QoSProfile(depth=100)
        qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT

        # Create subscriber and publisher
        self.image_pub = self.create_publisher(Image, output_topic, 1)
        self.image_sub = self.create_subscription(Image, input_topic, self.image_callback, qos_profile)

        self.get_logger().info(
            f"ImageInverterNode started. Subscribing: {input_topic}  Publishing: {output_topic}"
        )

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            flipped_image = cv2.flip(cv_image, -1)

            # Convert back to ROS Image message
            flipped_msg = self.bridge.cv2_to_imgmsg(flipped_image, encoding=msg.encoding)

            # Copy header information from original message
            flipped_msg.header = msg.header

            # Publish
            self.image_pub.publish(flipped_msg)

        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")


def main():
    rclpy.init()
    node = ImageInverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Image Inverter Node Shutdown')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
