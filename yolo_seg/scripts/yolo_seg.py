#!/usr/bin/python

import rospy
import numpy as np
import cv2
import cv_bridge
from sensor_msgs.msg import Image
import message_filters

from ultralytics import YOLO
import logging

# Suppress Ultralytics log output
# logging.getLogger('ultralytics').setLevel(logging.CRITICAL)

class YoloSeg:
    def __init__(self, weights):
        self.model = YOLO(weights)
        # self.model.export(format='engine')
        # self.tensor_rt_model = YOLO('yolo11x-seg.engine')
    
    def detect(self, img, img_size = (480,640,3)):
        results = self.model(img, conf=0.8)
        # results = self.tensor_rt_model(img, conf=0.8)
        # print(f"Current Device: {self.model.device}")
        if results[0].masks:  # Check if masks exist
            try:
                masks = results[0].masks.data.cpu().numpy()
                combined_mask = np.ones((*masks[0].shape, 3), dtype=np.uint8)
                label_mask = 100 * np.ones((*masks[0].shape, 3), dtype=np.uint8) # The number is so that we can exceed all possible class ids
                mask_ids = results[0].boxes.cls.cpu().numpy().astype(dtype=np.uint).tolist()
                
                # Combine masks and assign unique colors to each label
                for i in range(len(mask_ids)):
                    color = np.array(rospy.get_param("~" + str(mask_ids[i]))[:3]) #* 255  # Get the RGB values for the i-th label
                    mask = masks[i].astype(bool)
                    combined_mask[mask] = color
                    label_mask[mask][:] = mask_ids[i]

                return combined_mask, label_mask
            
            except Exception as e:
                rospy.logerr(f"Error processing masks: {e}")
                return  np.zeros(img_size, dtype=np.uint8),  np.zeros(img_size, dtype=np.uint8)

        return np.zeros(img_size, dtype=np.uint8),  np.zeros(img_size, dtype=np.uint8) # black image for capturing depth information

class Segmenter:
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()

        # Read parameters
        self._weights = rospy.get_param('~weights', 'yolov8x-seg.pt')
        self._input_rgb_topic = rospy.get_param('~input_rgb_topic', 'camera/color/img_raw')
        self._input_depth_topic = rospy.get_param('~input_depth_topic', 'camera/depth/img_raw')
        self._output_seg_topic = rospy.get_param('~output_segmentation_topic', 'segmented_image')
        self._output_rgb_topic = rospy.get_param('~output_rgb_topic', 'rgb_image')
        self._output_depth_topic = rospy.get_param('~output_depth_topic', 'depth_image')
        # self._class_topic = rospy.get_param('~class_topic', 'class_image') # this is for simulation

        # Load YOLO model
        self.segmenter = YoloSeg(self._weights)

        # ROS communication
        rgb_sub = message_filters.Subscriber(self._input_rgb_topic, Image)
        depth_sub = message_filters.Subscriber(self._input_depth_topic, Image)

        # Synchronize RGB and depth topics
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=1, slop=0.1)
        self.ts.registerCallback(self.callback)

        # Publishers
        self.pub_seg_out = rospy.Publisher(self._output_seg_topic, Image, queue_size=1)
        self.pub_rgb_out = rospy.Publisher(self._output_rgb_topic, Image, queue_size=1)
        self.pub_depth_out = rospy.Publisher(self._output_depth_topic, Image, queue_size=1)
        # self.pub_img_class = rospy.Publisher(self._class_topic, Image, queue_size=1)

    def callback(self, rgb_msg, depth_msg):
        try:
            # Convert RGB image to OpenCV format
            color_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Flip color image vertically because camera on jackal is upside down
            color_image = cv2.flip(color_image, 0)

            # Perform segmentation
            detections, _ = self.segmenter.detect(color_image, color_image.shape)
            print(f"Image Size: {color_image.shape}")

        
            # Flip detections back to original orientation
            detections = cv2.flip(detections, 0)
            
            # Convert segmentation result to ROS Image message
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            seg_msg = self.bridge.cv2_to_imgmsg(detections, encoding='bgr8')
            seg_msg.header = rgb_msg.header

            # Convert class labels to ROS Image message
            # class_msg = self.bridge.cv2_to_imgmsg(labels, encoding='bgr8')
            # class_msg.header = rgb_msg.header

            # Publish segmentation image
            self.pub_seg_out.publish(seg_msg)

            # Publish original RGB image
            self.pub_rgb_out.publish(rgb_msg)

            # Publish synchronized depth image
            self.pub_depth_out.publish(depth_msg)

            # Publish class labels image
            # self.pub_img_class.publish(class_msg)

        except Exception as e:
            rospy.logerr(f"Error in synchronized callback: {e}")
            
            
if __name__ == "__main__":
    
    rospy.init_node("yolo_seg_node")
    segmenter = Segmenter()
    rospy.spin()
    
