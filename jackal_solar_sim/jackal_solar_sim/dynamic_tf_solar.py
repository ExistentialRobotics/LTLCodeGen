#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped

class DynamicTfSolarBroadcaster(Node):
    def __init__(self):
        super().__init__('dynamic_tf_solar_broadcaster')
        
        self.subscriber = self.create_subscription(
            PoseStamped,
            '/model/husky_seg_cam/pose',
            self.pose_callback,
            10
        )
        self.broadcaster = tf2_ros.TransformBroadcaster(self)
        self.get_logger().info('Dynamic TF solar broadcaster started')

    def pose_callback(self, pose_stamped):
        """
        Callback function that receives pose updates and broadcasts them.
        """
        transform = TransformStamped()

        # Set the header information on the transform
        transform.header.stamp = pose_stamped.header.stamp
        transform.header.frame_id = 'world'
        transform.child_frame_id = 'semantic_segmentation_world'

        # Set the pose information on the transform
        transform.transform.translation.x = pose_stamped.pose.position.x
        transform.transform.translation.y = pose_stamped.pose.position.y
        transform.transform.translation.z = pose_stamped.pose.position.z
        transform.transform.rotation.x = pose_stamped.pose.orientation.x
        transform.transform.rotation.y = pose_stamped.pose.orientation.y
        transform.transform.rotation.z = pose_stamped.pose.orientation.z
        transform.transform.rotation.w = pose_stamped.pose.orientation.w

        # Broadcast the transform
        self.broadcaster.sendTransform(transform)

def main(args=None):
    rclpy.init(args=args)
    broadcaster = DynamicTfSolarBroadcaster()
    
    try:
        rclpy.spin(broadcaster)
    except KeyboardInterrupt:
        broadcaster.get_logger().info('TF solar broadcaster shutting down!')
    finally:
        broadcaster.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
