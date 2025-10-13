from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('ssmi_mapping')
    octomap_config = PathJoinSubstitution([pkg_share, 'params/robot', 'octomap_generator.yaml'])
    semantic_sensor_config = PathJoinSubstitution([pkg_share, 'params/robot', 'semantic_cloud.yaml'])

    mapping_node = Node(
        package='ssmi_mapping', executable='octomap_generator_ros', name='octomap_generator', output='screen',
        parameters=[
            octomap_config
        ],
    )

    sensor_node = Node(
        package='ssmi_mapping', executable='semantic_sensor_node.py', name='semantic_sensor_node', output='screen',
        parameters=[
            semantic_sensor_config
        ],
    )

    return LaunchDescription([
        sensor_node,
        mapping_node,
    ])
