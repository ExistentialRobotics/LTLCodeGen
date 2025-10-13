#!/usr/bin/env python3
"""
ROS2 Launch file for Jackal solar simulation
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os

PROJ_DIR = os.path.join(os.path.dirname(__file__), "../../")
PROJ_DIR = os.path.realpath(PROJ_DIR)


def generate_launch_description():
    # Declare arguments
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='office',
        description='World name to load'
    )
    
    want_gui_arg = DeclareLaunchArgument(
        'want_gui',
        default_value='false',
        description='Whether to launch Gazebo GUI'
    )
    
    sim_arg = DeclareLaunchArgument(
        'sim',
        default_value='true',
        description='Whether to launch simulation'
    )
    
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='husky_1',
        description='Robot name'
    )
    
    active_mapping_arg = DeclareLaunchArgument(
        'active_mapping',
        default_value='false',
        description='Whether to enable active mapping'
    )
    
    llm_instructions_arg = DeclareLaunchArgument(
        'llm_instructions',
        default_value='Go to car then the fire hydrant.',
        description='Natural language instructions for LTL translation'
    )
    
    # Get package directory
    pkg_share = FindPackageShare('jackal_solar_sim')
    
    # Include the Jackal office launch file
    jackal_office_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([pkg_share, 'launch', 'launch_jackal_office.launch.py'])
        ]),
        launch_arguments={
            'want_gui': LaunchConfiguration('want_gui'),
            'robot_name': LaunchConfiguration('robot_name'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('sim'))
    )
    
    # Include robot SSMI mapping launch file
    ssmi_pkg_share = get_package_share_directory('ssmi_mapping')
    octomap_config = PathJoinSubstitution([ssmi_pkg_share, 'params', 'octomap_generator.yaml'])
    semantic_sensor_config = PathJoinSubstitution([ssmi_pkg_share, 'params', 'semantic_cloud.yaml'])
    color_id = PathJoinSubstitution([ssmi_pkg_share, 'params', 'officesim_color_id.yaml'])
    reverse_color_id = PathJoinSubstitution([ssmi_pkg_share, 'params', 'reverse_color_id_officesim.yaml'])

    mapping_node = Node(
        package='ssmi_mapping', executable='octomap_generator_ros', name='octomap_generator', output='screen',
        parameters=[
            octomap_config,
            reverse_color_id,
        ],
    )

    sensor_node = Node(
        package='ssmi_mapping', executable='semantic_sensor_node.py', name='semantic_sensor_node', output='screen',
        parameters=[
            semantic_sensor_config,
            color_id,
        ],
    )
    
    # Group mapping nodes under robot namespace
    mapping_group = GroupAction([
        PushRosNamespace(LaunchConfiguration('robot_name')),
        mapping_node,
        sensor_node,
    ])

    # Speech to LTL node
    label_map_pkg_share = get_package_share_directory('label_map')
    speech_to_ltl_node = Node(
        package='speech_to_ltl',
        executable='ltl_translate_node',
        name='ltl_translate_node',
        output='screen',
        parameters=[{
            "semantic_file_path": f"{label_map_pkg_share}/maps/semantic_map.npy",
            "all_classes_file_path": f"{ssmi_pkg_share}/params/officesim_label.yaml",
            "ltl_translator_alg": "code",
            "enable_semantic_check": False,
            "enable_syntactic_check": False,
            "load_all_possible_ids": True,
            "llm_instructions": LaunchConfiguration('llm_instructions'),
        }],
    )

    label_map_node = Node(
            package='label_map',
            executable='label_map_ros',
            name='label_map_node',
            output='screen',
            parameters=[{
                "radius": 1,
                "ap_dict": 'ap_dict',
                "ap_id": 'ap_id',
                "semantic_map": 'semantic_map_2D',
                "label_map_topic": "label_map",
                "label_map_viz_topic": "label_map_viz",
            }],
        )
    
    # Solar planner node
    planner_node = Node(
        package="solar_planner",
        executable="ltl_planner_node",
        name="ltl_planner_node",
        output="screen",
        parameters=[{
            "world_frame_id": "world",
            "robot_frame_id": [LaunchConfiguration('robot_name'), '/base_link'],
            "debug_mode": False,
            "debug_pose_x": 0.0,
            "debug_pose_y": 0.0,
        }],
    )
    
    # Group planner nodes under robot namespace
    planner_group = GroupAction([
        PushRosNamespace(LaunchConfiguration('robot_name')),
        speech_to_ltl_node,
        planner_node,
        label_map_node,
    ])
    
    # RViz node
    # rviz_config_file = PathJoinSubstitution([
    #     pkg_share, 'rviz', 'mapping.rviz'
    # ])
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        # arguments=['-d', rviz_config_file],
        output='screen'
    )
    
    return LaunchDescription([
        world_arg,
        want_gui_arg,
        sim_arg,
        robot_name_arg,
        active_mapping_arg,
        llm_instructions_arg,
        jackal_office_launch,
        mapping_group,
        planner_group,
        rviz_node,
    ])
