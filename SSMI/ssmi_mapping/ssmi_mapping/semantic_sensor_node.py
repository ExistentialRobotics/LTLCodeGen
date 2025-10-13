#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor

import sys
import numpy as np
import message_filters

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from skimage.transform import resize
from sensor_msgs.msg import PointCloud2
from ssmi_mapping.semantic_sensor import PointType, SemanticPclGenerator


class SemanticCloud(Node):
    """
    Class for ros node to take in a color image (bgr) and a semantic segmentation image (bgr)
    Then produce point cloud based on depth information
    """

    def __init__(self):
        super().__init__('semantic_sensor_node')

        # Get point type
        self.declare_parameter('semantic_pcl.point_type', 0)
        point_type = self.get_parameter('semantic_pcl.point_type').value

        # Point type warning
        if point_type == 0:
            self.point_type = PointType.SEMANTIC
            self.get_logger().info("Generate semantic point cloud.")
        else:
            self.get_logger().warning("Invalid point type.")
            return

        # Get maximum synchronization delay parameter
        self.declare_parameter('semantic_pcl.max_delay', 1.0)
        max_delay = self.get_parameter('semantic_pcl.max_delay').value

        # Get Unit conversion factor
        self.declare_parameter('semantic_pcl.unit_conversion', 1000.0)
        unit_conversion = self.get_parameter('semantic_pcl.unit_conversion').value

        # Get image size
        self.declare_parameter('camera.width', 640)
        self.img_width = self.get_parameter('camera.width').value
        self.declare_parameter('camera.height', 480)
        self.img_height = self.get_parameter('camera.height').value

        # Set up ROS
        self.bridge = CvBridge()  # CvBridge to transform ROS Image message to OpenCV image

        # Set up ros image subscriber
        # Set buff_size to average msg size to avoid accumulating delay
        # Point cloud frame id
        self.declare_parameter('semantic_pcl.frame_id', 'camera_optic')
        frame_id = self.get_parameter('semantic_pcl.frame_id').value

        # Camera intrinsic matrix
        self.declare_parameter('camera.fx', 320)
        self.declare_parameter('camera.fy', 320)
        self.declare_parameter('camera.cx', 320)
        self.declare_parameter('camera.cy', 240)
        fx = self.get_parameter('camera.fx').value
        fy = self.get_parameter('camera.fy').value
        cx = self.get_parameter('camera.cx').value
        cy = self.get_parameter('camera.cy').value
        intrinsic = np.matrix([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        # Noise configuration
        self.noisy_obs = False
        self.declare_parameter('semantic_pcl.depth_noise_std', 0.0)
        self.depth_noise_std = self.get_parameter('semantic_pcl.depth_noise_std').value
        self.declare_parameter('semantic_pcl.true_class_prob', 1.0)
        self.true_class_prob = self.get_parameter('semantic_pcl.true_class_prob').value
        self.declare_parameter('semantic_pcl.is_semantic_img_classes', False)
        self.is_semantic_img_classes = self.get_parameter('semantic_pcl.is_semantic_img_classes').value
        self.class_id_color_dict = {}
        if self.depth_noise_std > 0 or self.true_class_prob < 1:
            self.noisy_obs = True
            self.declare_parameter('class_labels.num_classes')
            num_classes = self.get_parameter('class_labels.num_classes').value
            self.class_colors = []
            for i in range(num_classes):
                self.declare_parameter('class_labels.color_' + str(i + 1))
                color = self.get_parameter('class_labels.color_' + str(i + 1)).value
                self.class_colors.append(255 * np.array([color["b"], color["g"], color["r"]]))
            self.class_colors = np.array(self.class_colors).astype(np.uint8)

        # Semantic point cloud topic
        self.declare_parameter('octomap.pointcloud_topic', '/semantic_pcl/semantic_pcl')
        _pointcloud_topic = self.get_parameter('octomap.pointcloud_topic').value

        # QoS configuration
        qos_profile = QoSProfile(depth=100)
        qos_profile.reliability = QoSReliabilityPolicy.BEST_EFFORT

        # Create publisher
        self.pcl_pub = self.create_publisher(PointCloud2, _pointcloud_topic, 1)

        # increase buffer size to avoid delay (despite queue_size = 1)
        self.declare_parameter('semantic_pcl.color_image_topic', '/jackal2/camera/color/image_raw')
        self.declare_parameter('semantic_pcl.semantic_image_topic', '/jackal2/semantic_img')
        self.declare_parameter('semantic_pcl.depth_image_topic', '/jackal2/camera/depth/image_rect_raw')

        _color_sub = message_filters.Subscriber(self, Image,
                                                self.get_parameter('semantic_pcl.color_image_topic').value,
                                                qos_profile=qos_profile)
        _semantic_sub = message_filters.Subscriber(self, Image,
                                                   self.get_parameter('semantic_pcl.semantic_image_topic').value,
                                                   qos_profile=qos_profile)
        _depth_sub = message_filters.Subscriber(self, Image,
                                                self.get_parameter('semantic_pcl.depth_image_topic').value,
                                                qos_profile=qos_profile)

        # time synchronization
        _time_synchronizer = message_filters.ApproximateTimeSynchronizer([_color_sub, _semantic_sub, _depth_sub],
                                                                         queue_size=1, slop=max_delay)
        _time_synchronizer.registerCallback(self.color_semantic_depth_callback)
        self.cloud_generator = SemanticPclGenerator(
            intrinsic, self.img_width, self.img_height, unit_conversion, frame_id, self.point_type
        )
        self.get_logger().info("Semantic point cloud ready!")

    def color_semantic_depth_callback(self, color_img_ros, semantic_img_ros, depth_img_ros):
        """
        Callback function to produce point cloud registered with semantic class color based
        on input color image and depth image
        """
        # Convert ros Image message to numpy array
        try:
            color_img = self.bridge.imgmsg_to_cv2(color_img_ros, "bgr8")
            semantic_img = self.bridge.imgmsg_to_cv2(semantic_img_ros, "bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_img_ros, "32FC1")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error: {e}')

        assert (
                color_img.shape[0] == self.img_height and color_img.shape[1] == self.img_width
        ), "Color image size does not match the expected size!"

        # Resize depth
        if depth_img.shape[0] != self.img_height or depth_img.shape[1] != self.img_width:
            self.get_logger().info(f"Resizing depth image from {depth_img.shape} to ({self.img_height}, {self.img_width})")
            depth_img = resize(
                depth_img,
                (self.img_height, self.img_width),
                order=0,
                mode="reflect",
                anti_aliasing=False,
                preserve_range=True,
            )  # order = 0, nearest neighbour
            depth_img = depth_img.astype(np.float32)

        # Resize semantic
        if semantic_img.shape[0] != self.img_height or semantic_img.shape[1] != self.img_width:
            self.get_logger().info(f"Resizing semantic image from {semantic_img.shape} to ({self.img_height}, {self.img_width})")
            semantic_img = resize(
                semantic_img,
                (self.img_height, self.img_width),
                order=0,
                mode="reflect",
                anti_aliasing=False,
                preserve_range=True,
            )  # order = 0, nearest neighbour
            semantic_img = semantic_img.astype(np.uint8)

        # Add noise
        if self.noisy_obs:
            depth_img, semantic_img = self.add_noise(depth_img, semantic_img)

        cloud_ros = None
        if self.point_type == PointType.SEMANTIC:
            if self.is_semantic_img_classes:
                if np.any(semantic_img[..., -1] == 0):
                    self.get_logger().warning("Replace Zeros in Class Image with 7 (class 'rest')", once=True)
                    semantic_img[..., :-1][semantic_img[..., :-1] == 0] = 7
                class_id_pool = np.unique(semantic_img[:, :, 0].flatten())
                segmentation_img = np.zeros_like(semantic_img)
                for class_id in class_id_pool:
                    try:
                        if class_id in self.class_id_color_dict:
                            color = self.class_id_color_dict[class_id]
                        else:
                            descriptor = ParameterDescriptor(dynamic_typing=True)
                            self.declare_parameter(str(class_id), descriptor=descriptor)
                            color = np.array(self.get_parameter(str(class_id)).value[:3])
                            self.class_id_color_dict[class_id] = color
                        class_mask = semantic_img[:, :, 0] == class_id
                        segmentation_img[class_mask] = color
                    except KeyError:
                        self.get_logger().warning("Error While Converting Class ID to Semantic Color!")

                    cloud_ros = self.cloud_generator.generate_cloud_semantic(
                        color_img, segmentation_img, depth_img, color_img_ros.header.stamp
                    )
            else:
                cloud_ros = self.cloud_generator.generate_cloud_semantic(
                    color_img, semantic_img, depth_img, color_img_ros.header.stamp
                )
        else:
            self.get_logger().warning("Point type not supported!")

        # Publish point cloud
        self.get_logger().info("Published Cloud!!!", throttle_duration_sec=10.0)
        self.pcl_pub.publish(cloud_ros)

    def add_noise(self, depth_img, semantic_img):
        noisy_depth_img = depth_img + np.random.normal(0, self.depth_noise_std, depth_img.shape).astype(np.float32)
        np.place(noisy_depth_img, depth_img == 0, 0)
        error_mask = np.random.sample(size=semantic_img.shape[:2]) > self.true_class_prob
        random_classes = np.random.choice(self.class_colors.shape[0], size=np.count_nonzero(error_mask))
        error_mask = np.repeat(error_mask[:, :, None], repeats=3, axis=2)
        np.place(semantic_img, error_mask, self.class_colors[random_classes, :])

        return noisy_depth_img, semantic_img


def main(args=None):
    rclpy.init(args=args)
    node = SemanticCloud()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Semantic cloud shutting down!')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
