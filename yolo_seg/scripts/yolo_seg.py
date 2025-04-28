#!/usr/bin/python

import rospy
import numpy as np
import yaml
import cv2
import cv_bridge
import sensor_msgs.msg as sensor_msgs
import vision_msgs.msg as vision_msgs

from ultralytics import YOLO

class YoloSeg:
    def __init__(self, weights):
        self.model = YOLO(weights)
    
    def detect(self, img, img_size = (360,480,3)):
        results = self.model(img, conf=0.25)
        if results[0].masks:  # Check if masks exist
            try:
                masks = results[0].masks.data.cpu().numpy()
                combined_mask = np.ones((*masks[0].shape, 3), dtype=np.uint8)
                label_mask = 80 * np.ones((*masks[0].shape, 3), dtype=np.uint8)
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
                return None
        return np.zeros(img_size, dtype=np.uint8) # black image for capturing depth information

class Segmenter:
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        # Read parameters
        self._weights = rospy.get_param('~weights', 'yolov8x-seg.pt')
        self._input_topic = rospy.get_param('~input_topic', '/oakd/rgb/preview/image_raw')
        self._output_topic = rospy.get_param('~output_topic', 'segmented_image')
        self._class_topic = rospy.get_param('~class_topic', 'class_image')
        # Load Yolo model
        self.segmenter = YoloSeg(self._weights)

        # ROS communication
        self.sub_img_in = rospy.Subscriber(self._input_topic, sensor_msgs.Image, self.img_cb, queue_size=1)
        self.pub_img_out = rospy.Publisher(self._output_topic, sensor_msgs.Image, queue_size=1)
        self.pub_img_class = rospy.Publisher(self._class_topic, sensor_msgs.Image, queue_size=1)

    def img_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections, labels = self.segmenter.detect(frame)
            out = self.bridge.cv2_to_imgmsg(detections, encoding='bgr8')
            out.header = msg.header
            class_msg = self.bridge.cv2_to_imgmsg(labels, encoding='bgr8')
            class_msg.header = msg.header
            self.pub_img_out.publish(out)
            self.pub_img_class.publish(class_msg)
        except Exception as e:
            rospy.logerr(f"Error in image callback: {e}")

if __name__ == "__main__":
    
    rospy.init_node("yolo_seg_node")
    segmenter = Segmenter()
    rospy.spin()
    
