#!/usr/bin/env python3
"""
ROS2 Launch file for semantic initialization
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare arguments
    world_arg = DeclareLaunchArgument('world', default_value='room')
    robot_name_arg = DeclareLaunchArgument('robot_name', default_value='husky')
    x_arg = DeclareLaunchArgument('x', default_value='0')
    y_arg = DeclareLaunchArgument('y', default_value='0')
    z_arg = DeclareLaunchArgument('z', default_value='0')
    roll_arg = DeclareLaunchArgument('roll', default_value='0')
    pitch_arg = DeclareLaunchArgument('pitch', default_value='0')
    yaw_arg = DeclareLaunchArgument('yaw', default_value='0')
    
    body_frame_arg = DeclareLaunchArgument('body_frame', default_value='base_link')
    optic_frame_arg = DeclareLaunchArgument('optic_frame', default_value='camera_optic')
    camera_frame_arg = DeclareLaunchArgument('camera_frame', default_value='camera_regular')
    camera_height_arg = DeclareLaunchArgument('camera_height', default_value='0.5')
    
    # Get package directory
    pkg_share = FindPackageShare('jackal_solar_sim')
    
    # Dynamic TF broadcaster node
    dynamic_tf_node = Node(
        package='jackal_solar_sim',
        executable='dynamics_tf.py',
        name='dynamic_tf_broadcaster',
        output='screen',
        parameters=[{
            'agent_name': LaunchConfiguration('robot_name'),
        }],
        respawn=True
    )
    
    # Static transform: camera to body
    camera_to_body_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera2body',
        arguments=[
            '0', '0', LaunchConfiguration('camera_height'),
            '0', '0', '0',
            [LaunchConfiguration('robot_name'), '/', LaunchConfiguration('body_frame')],
            [LaunchConfiguration('robot_name'), '/', LaunchConfiguration('camera_frame')],
        ],
        respawn=True
    )
    
    # Static transform: optic to camera
    optic_to_camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='optic2camera',
        arguments=[
            '0', '0', '0',
            '-1.5708', '0', '-1.5708',
            [LaunchConfiguration('robot_name'), '/', LaunchConfiguration('camera_frame')],
            [LaunchConfiguration('robot_name'), '/', LaunchConfiguration('optic_frame')],
        ],
        respawn=True
    )
    
    # Spawn robot
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        name='robot_respawn',
        output='screen',
        arguments=[
            '-world', LaunchConfiguration('world'),
            '-file', PathJoinSubstitution([pkg_share, 'models', 'husky.sdf']),
            '-name', LaunchConfiguration('robot_name'),
            '-x', LaunchConfiguration('x'),
            '-y', LaunchConfiguration('y'),
            '-z', LaunchConfiguration('z'),
            '-R', LaunchConfiguration('roll'),
            '-P', LaunchConfiguration('pitch'),
            '-Y', LaunchConfiguration('yaw'),
        ]
    )
    
    # Bridge between ROS2 and Gazebo - all sensor topics
    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        output='screen',
        arguments=[
            # RGB-D Camera image
            ['/world/', LaunchConfiguration('world'), '/model/', LaunchConfiguration('robot_name'), 
             '/link/sensors_link/sensor/rgbd_camera/image@sensor_msgs/msg/Image[gz.msgs.Image'],
            # RGB-D Camera depth
            ['/world/', LaunchConfiguration('world'), '/model/', LaunchConfiguration('robot_name'),
             '/link/sensors_link/sensor/rgbd_camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image'],
            # Semantic segmentation colored map
            ['/world/', LaunchConfiguration('world'), '/model/', LaunchConfiguration('robot_name'),
             '/link/sensors_link/sensor/semantic_segmentation_camera/segmentation/colored_map@sensor_msgs/msg/Image[gz.msgs.Image'],
            # Semantic segmentation labels map
            ['/world/', LaunchConfiguration('world'), '/model/', LaunchConfiguration('robot_name'),
             '/link/sensors_link/sensor/semantic_segmentation_camera/segmentation/labels_map@sensor_msgs/msg/Image[gz.msgs.Image'],
            # Robot pose
            ['/model/', LaunchConfiguration('robot_name'), '/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose'],
            # Robot cmd_vel
            ['/model/', LaunchConfiguration('robot_name'), '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist'],
        ],
        remappings=[
            (['/model/', LaunchConfiguration('robot_name'), '/pose'], 'pose'),
            (['/model/', LaunchConfiguration('robot_name'), '/cmd_vel'], 'cmd_vel'),
            (['/world/', LaunchConfiguration('world'), '/model/', LaunchConfiguration('robot_name'), 
             '/link/sensors_link/sensor/rgbd_camera/image'], 'camera/color/image_raw'),
            (['/world/', LaunchConfiguration('world'), '/model/', LaunchConfiguration('robot_name'), 
             '/link/sensors_link/sensor/rgbd_camera/depth_image'], 'camera/depth/image_raw'),
            (['/world/', LaunchConfiguration('world'), '/model/', LaunchConfiguration('robot_name'), 
             '/link/sensors_link/sensor/semantic_segmentation_camera/segmentation/colored_map'], 'camera/semantic/colored_map'),
            (['/world/', LaunchConfiguration('world'), '/model/', LaunchConfiguration('robot_name'), 
             '/link/sensors_link/sensor/semantic_segmentation_camera/segmentation/labels_map'], 'camera/semantic/class_map'),
        ],
        respawn=True
    )
    
    # Joy node
    joy_config_arg = DeclareLaunchArgument('joy_config', default_value='ps3')
    joy_dev_arg = DeclareLaunchArgument('joy_dev', default_value='/dev/input/js0')
    
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[{
            'dev': LaunchConfiguration('joy_dev'),
            'deadzone': 0.3,
            'autorepeat_rate': 20.0,
        }],
        remappings=[('joy', 'joy')]
    )
    
    # Teleop twist joy node - using PS4 controller config
    teleop_node = Node(
        package='teleop_twist_joy',
        executable='teleop_node',
        name='teleop_twist_joy',
        parameters=[
            PathJoinSubstitution([pkg_share, 'config', 'ps4_teleop.yaml'])
        ],
        remappings=[('joy', 'joy')]
    )
    
    # Group all nodes under robot namespace
    robot_group = GroupAction([
        PushRosNamespace(LaunchConfiguration('robot_name')),
        dynamic_tf_node,
        camera_to_body_tf,
        optic_to_camera_tf,
        spawn_robot,
        bridge_node,
        joy_node,
        teleop_node,
    ])
    
    return LaunchDescription([
        world_arg,
        robot_name_arg,
        x_arg, y_arg, z_arg,
        roll_arg, pitch_arg, yaw_arg,
        body_frame_arg,
        optic_frame_arg,
        camera_frame_arg,
        camera_height_arg,
        joy_config_arg,
        joy_dev_arg,
        robot_group,
    ])
