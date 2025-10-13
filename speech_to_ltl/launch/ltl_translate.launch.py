from launch import LaunchDescription
from launch_ros.actions import Node
import os


PROJ_DIR = os.path.join(os.path.dirname(__file__), "../../")
PROJ_DIR = os.path.realpath(PROJ_DIR)


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='speech_to_ltl',
            executable='ltl_translate_node',
            name='ltl_translate_node',
            output='screen',
            parameters=[{
                "semantic_file_path": f"{PROJ_DIR}/label_map/maps/semantic_map.npy",
                "all_classes_file_path": f"{PROJ_DIR}/SSMI/ssmi_mapping/params/officesim_color_id.yaml",
                "ltl_translator_alg": "code",
                "enable_semantic_check": False,
                "enable_syntactic_check": False,
                "load_all_possible_ids": True,
                "llm_instructions": "Go to car then the fire hydrant.",
            }],
        ),
    ])
