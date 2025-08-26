#!/usr/bin/env python3
import sys

sys.path.append("/usr/local/lib/python3.8/site-packages")  # ADD THE PATH WHERE SPOT WAS BUILT

import spot

import numpy as np
import rospy
from std_msgs.msg import String
import yaml
import time
from langchain.chains import LLMChain
from model import llm_init
from scenegraph_prompt import (
    get_translate_prompt,
    get_yolo_id_convert_prompt,
)
from yolo_parser import (
    gpt_to_spot,
    find_env_elements,
    list_yolo_ids,
)

# from speech2text import*
import os
from ltl_translator import get_repeated_outputs, do_syntactic_check, do_semantic_check
from code_ltl_translator import code_ltl_translator

n_chain_repeat = 1
verbose = False
for_debug = False

PROJ_DIR = os.path.join(os.path.dirname(__file__), "../../")
PROJ_DIR = os.path.realpath(PROJ_DIR)


class Translate:
    def __init__(self):
        # Get file paths as parameters with default values
        self.semantic_file_path = rospy.get_param(
            "~semantic_file_path",
            f"{PROJ_DIR}/label_map/maps/semantic_map.npy",
        )
        self.all_classes_file_path = rospy.get_param(
            "~all_classes_file_path",
            f"{PROJ_DIR}/SSMI/SSMI-Mapping/params/officesim_color_id.yaml",
            
        )
        self.ltl_translator_alg = rospy.get_param(
            "~ltl_translator_alg",
            "code",  # or scenegraph
        )
        self.enable_semantic_check = rospy.get_param(
            "~enable_semantic_check",
            False,
        )
        self.enable_syntactic_check = rospy.get_param(
            "~enable_syntactic_check",
            False,
        )

        # Publishers for the automaton and atomic propositions
        self.automaton_pub = rospy.Publisher("aut_str", String, queue_size=1)  # Might need to write to file
        self.ap_pub = rospy.Publisher("ap_dict", String, queue_size=1)

        # Load the semantic map
        self.semantic_map = np.load(self.semantic_file_path)
        self.unique_ids = np.unique(self.semantic_map)

        if for_debug:
            print("Unique ids: ", self.unique_ids)

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

        instruction = "Go to car then the monitor."
        print("instruction:", instruction, "\n")

        tick = time.time()

        yolo_id_instruction = get_repeated_outputs(
            input_dict={"yolo_ids": yolo_ids_str, "NL_input": instruction},
            chain=yolo_id_conversion_chain,
            n_repeat=n_chain_repeat,
        )
        if for_debug:
            print(yolo_id_instruction)

        _, env_elements_str = find_env_elements(yolo_id_instruction)

        if for_debug:
            print(env_elements_str)

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
            print(ltl_formula)

        spot_desc, ap_dict = gpt_to_spot(ltl_formula)
        if for_debug:
            print(spot_desc, "\n", ap_dict)

        if self.enable_syntactic_check:
            formula, ap_dict = do_syntactic_check(self.llm, yolo_id_instruction, spot_desc, ap_dict, verbose=verbose)
        else:
            formula = spot.formula(spot_desc)
        if self.enable_semantic_check:
            formula, ap_dict = do_semantic_check(self.llm, ltl_formula, formula, yolo_id_instruction, ap_dict, verbose=verbose)

        print("AP Dictionary:{final_ap_dict}".format(final_ap_dict=ap_dict))
        print("LTL FORMULA:{ltl_formula}".format(ltl_formula=formula))

        automaton = spot.translate(formula, "sbacc", "complete")

        automatonAsString = automaton.to_str("hoa")

        tock = time.time()
        print(f"Execution Time: {tock -tick} !!!!!!")
        if for_debug:
            print(automatonAsString)

        while not rospy.is_shutdown():
            # Publish the automaton and atomic propositions
            self.automaton_pub.publish(automatonAsString)
            self.ap_pub.publish(str(ap_dict))
            rospy.sleep(1)

    def get_present_ids(self):
        with open(self.all_classes_file_path, "r") as file:
            all_classes = yaml.load(file, Loader=yaml.FullLoader)

        present_dict = {all_classes.get(str(id))[3]: f"object_{id}" for id in self.unique_ids if all_classes.get(str(id))}

        if for_debug:
            print("Present dict: ", present_dict)

        return present_dict


if __name__ == "__main__":
    rospy.init_node("ltl_translate_node", anonymous=True)
    Translate()
    rospy.spin()
