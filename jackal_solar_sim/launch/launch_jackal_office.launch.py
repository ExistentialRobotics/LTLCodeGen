#!/usr/bin/env python3
"""
ROS2 Launch file for Jackal office environment
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command, TextSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare arguments
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='office',
        description='World name'
    )
    
    want_gui_arg = DeclareLaunchArgument(
        'want_gui',
        default_value='false',
        description='Launch with GUI'
    )
    
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='husky_1',
        description='Robot name'
    )
    
    # Get package directories
    pkg_jackal = FindPackageShare('jackal_solar_sim')
    pkg_ros_gz_sim = FindPackageShare('ros_gz_sim')
    
    # Set model path environment variable
    models_path = PathJoinSubstitution([pkg_jackal, 'models'])
    set_gazebo_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=[models_path]
    )
    
    # Gazebo simulation with GUI
    world_file = PathJoinSubstitution([
        pkg_jackal, 'models', 
        [LaunchConfiguration('world'), TextSubstitution(text='.sdf')]
    ])
    
    gz_sim_with_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py'])
        ]),
        launch_arguments={
            'gz_args': ['-r -v 0 ', world_file],
        }.items(),
        condition=IfCondition(LaunchConfiguration('want_gui'))
    )
    
    # Gazebo simulation without GUI  
    gz_sim_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py'])
        ]),
        launch_arguments={
            'gz_args': ['-r -s -v 0 ', world_file],
        }.items(),
        condition=UnlessCondition(LaunchConfiguration('want_gui'))
    )
    
    # Include semantic init launch
    semantic_init_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([pkg_jackal, 'launch', 'semantic_init.launch.py'])
        ]),
        launch_arguments={
            'world': LaunchConfiguration('world'),
            'robot_name': LaunchConfiguration('robot_name'),
            'x': '32',
            'y': '12',
            'z': '0.2',
            'yaw': '1.5',
        }.items()
    )
    
    return LaunchDescription([
        world_arg,
        want_gui_arg,
        robot_name_arg,
        set_gazebo_resource_path,
        gz_sim_with_gui,
        gz_sim_headless,
        semantic_init_launch,
    ])
