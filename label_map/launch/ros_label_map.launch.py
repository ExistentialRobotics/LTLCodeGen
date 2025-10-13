from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='label_map',
            executable='label_map_ros',
            name='label_map_node',
            output='screen',
            parameters=[{
                "radius": 1,
                "ap_dict": "ap_dict",
                "ap_id": "ap_id",
                "semantic_map": "semantic_map",
                "label_map_topic": "label_map",
                "label_map_viz_topic": "label_map_viz",
            }],
        ),
    ])
