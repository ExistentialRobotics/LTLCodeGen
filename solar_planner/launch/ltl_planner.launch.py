# launch/ltl_planner.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    planner_node = Node(
        package="solar_planner",
        executable="ltl_planner_node",
        name="ltl_planner_node",
        output="screen",
        parameters=[{
            "world_frame_id": "odom",
            "robot_frame_id": "husky_1/base_link",
            "debug_mode": False,
            "debug_pose_x": 0.0,
            "debug_pose_y": 0.0,
        }],
    )

    return LaunchDescription([planner_node])
