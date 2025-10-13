#!/usr/bin/env python3
import sys

sys.path.append("/usr/local/lib/python3.8/site-packages")  # ADD THE PATH WHERE SPOT WAS BUILT

import spot
import numpy as np
import yaml
import time

from langchain.chains import LLMChain
from speech_to_ltl.model import llm_init
from speech_to_ltl.scenegraph_prompt import (
    get_translate_prompt,
    get_yolo_id_convert_prompt,
)
from speech_to_ltl.yolo_parser import (
    gpt_to_spot,
    find_env_elements,
    list_yolo_ids,
)

import os
from speech_to_ltl.ltl_translator import get_repeated_outputs, do_syntactic_check, do_semantic_check
from speech_to_ltl.code_ltl_translator import code_ltl_translator

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from rclpy.qos import QoSProfile, QoSReliabilityPolicy


n_chain_repeat = 1
verbose = False
for_debug = True

PROJ_DIR = os.path.join(os.path.dirname(__file__), "../../")
PROJ_DIR = os.path.realpath(PROJ_DIR)


class Translate(Node):
    def __init__(self):
        super().__init__('ltl_translate_node')

        self.declare_parameter("semantic_file_path", f"{PROJ_DIR}/label_map/maps/semantic_map.npy")
        self.declare_parameter("all_classes_file_path", f"{PROJ_DIR}/SSMI/ssmi_mapping/params/officesim_label.yaml")
        self.declare_parameter("ltl_translator_alg", "code") # or scenegraph
        self.declare_parameter("enable_semantic_check", False)
        self.declare_parameter("enable_syntactic_check", False)
        self.declare_parameter("load_all_possible_ids", True)
        self.declare_parameter("llm_instructions", "Go to car then the fire hydrant.")

        self.semantic_file_path = self.get_parameter("semantic_file_path").value
        self.all_classes_file_path = self.get_parameter("all_classes_file_path").value
        self.ltl_translator_alg = self.get_parameter("ltl_translator_alg").value
        self.enable_semantic_check = self.get_parameter("enable_semantic_check").value
        self.enable_syntactic_check = self.get_parameter("enable_syntactic_check").value
        self.load_all_possible_ids = self.get_parameter("load_all_possible_ids").value
        instruction = self.get_parameter("llm_instructions").value

        # Publishers for the automaton and atomic propositions
        self.automation_pub = self.create_publisher(String, 'aut_str', 1)
        self.ap_pub = self.create_publisher(String, 'ap_dict', 1)

        # Load the semantic map
        if self.load_all_possible_ids:
            with open(self.all_classes_file_path, "r") as file:
                all_classes = yaml.load(file, Loader=yaml.FullLoader)
                ids = [int(k) for k in all_classes.keys()]
                self.unique_ids = np.unique(ids)
        else:
            self.semantic_map = np.load(self.semantic_file_path)
            self.unique_ids = np.unique(self.semantic_map)

        if for_debug:
            self.get_logger().info(f"Unique ids: {self.unique_ids}")

        # Get the intersection of the unique ids and the yolo ids
        self.present_ids = self.get_present_ids()

        # Load OpenAI API Key
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key is None:
            raise ValueError("Environment variable `OPENAI_API_KEY` is required but not set.")

        print("Initializing LLM model...")
        self.llm = llm_init(gpt_api=openai_key)

        # Get the yolo ids as strings
        yolo_ids_str = list_yolo_ids(self.present_ids)

        # Get the yolo id conversion chain
        yolo_id_conversion_prompt = get_yolo_id_convert_prompt()
        yolo_id_conversion_chain = LLMChain(llm=self.llm, prompt=yolo_id_conversion_prompt, verbose=verbose)

        self.get_logger().info(f"instruction: {instruction} \n")

        tick = time.time()

        yolo_id_instruction = get_repeated_outputs(
            input_dict={"yolo_ids": yolo_ids_str, "NL_input": instruction},
            chain=yolo_id_conversion_chain,
            n_repeat=n_chain_repeat,
        )
        if for_debug:
            self.get_logger().info(yolo_id_instruction)

        _, env_elements_str = find_env_elements(yolo_id_instruction)

        if for_debug:
            self.get_logger().info(env_elements_str)

        scenegraph_translate_chain = LLMChain(llm=self.llm, prompt=get_translate_prompt(), verbose=verbose)

        if self.ltl_translator_alg == "scenegraph":
            ltl_formula = get_repeated_outputs(
                input_dict={
                    "env_elements": env_elements_str,
                    "instruction": yolo_id_instruction,
                },
                chain=scenegraph_translate_chain,
                n_repeat=n_chain_repeat,
            )
        elif self.ltl_translator_alg == "code":
            ltl_formula = code_ltl_translator(self.llm, yolo_id_instruction, verbose=verbose)[0]
        else:
            raise ValueError(f"Invalid LTL translator algorithm specified: {self.ltl_translator_alg}")

        if for_debug:
            self.get_logger().info(ltl_formula)

        spot_desc, ap_dict = gpt_to_spot(ltl_formula)
        if for_debug:
            self.get_logger().info(f"{spot_desc} \n {ap_dict}")

        if self.enable_syntactic_check:
            formula, ap_dict = do_syntactic_check(self.llm, yolo_id_instruction, spot_desc, ap_dict, verbose=verbose)
        else:
            formula = spot.formula(spot_desc)
        if self.enable_semantic_check:
            formula, ap_dict = do_semantic_check(self.llm, ltl_formula, formula, yolo_id_instruction, ap_dict,
                                                 verbose=verbose)

        self.get_logger().info("AP Dictionary:{final_ap_dict}".format(final_ap_dict=ap_dict))
        self.get_logger().info("LTL FORMULA:{ltl_formula}".format(ltl_formula=formula))

        automaton = spot.translate(formula, "sbacc", "complete")

        self.automatonAsString = automaton.to_str("hoa")
        self.ap_dict_str= str(ap_dict)

        tock = time.time()
        self.get_logger().info(f"Execution Time: {tock - tick} !!!!!!")
        if for_debug:
            self.get_logger().info(self.automatonAsString)

        self.create_timer(1.0, self.publish_outputs)

    def get_present_ids(self):
        with open(self.all_classes_file_path, "r") as file:
            all_classes = yaml.load(file, Loader=yaml.FullLoader)

        present_dict = {all_classes.get(str(obj_id)): f"object_{obj_id}" for obj_id in self.unique_ids if all_classes.get(str(obj_id))}

        if for_debug:
            self.get_logger().info(f"Present dict: {present_dict}")

        return present_dict

    def publish_outputs(self):
        self.automation_pub.publish(String(data=self.automatonAsString))
        self.ap_pub.publish(String(data=self.ap_dict_str))


def main():
    rclpy.init()
    node = Translate()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
