#!/usr/bin/env python
import sys
sys.path.append('/usr/local/lib/python3.8/site-packages') # ADD THE PATH WHERE SPOT WAS BUILT

import spot

import numpy as np
import rospy
from std_msgs.msg import String
import yaml
from langchain.chains import LLMChain
from model import llm_init
from yolo_prompt import (
    get_yolo_translate_prompt,
    get_yolo_id_convert_prompt,
)
from yolo_parser import (
    gpt_to_spot,
    find_env_elements,
    list_yolo_ids,
)

from ltl_translator import get_repeated_outputs, do_syntactic_check, do_semantic_check

n_chain_repeat = 1
verbose = False
for_debug = False


class Translate:
    def __init__(self):
        # Get file paths as parameters with default values
        self.semantic_file_path = rospy.get_param(
            "~semantic_file_path",
            "/home/brabiei/SOLAR_WS/src/SOLAR/label_map/maps/semantic_map.npy",
        )
        self.all_classes_file_path = rospy.get_param(
            "~all_classes_file_path",
            "/home/brabiei/SOLAR_WS/src/SOLAR/SSMI/SSMI-Mapping/params/officesim_color_id.yaml",
        )
        self.api_path = rospy.get_param(
            "~api_path",
            "/home/brabiei/SOLAR_WS/src/SOLAR/speech_to_ltl/config/config_openai.yml",
        )

        # Publishers for the automaton and atomic propositions
        self.automaton_pub = rospy.Publisher(
            "aut_str", String, queue_size=1
        )  # Might need to write to file
        self.ap_pub = rospy.Publisher("ap_dict", String, queue_size=1)

        # Load the semantic map
        self.semantic_map = np.load(self.semantic_file_path)
        self.unique_ids = np.unique(self.semantic_map)
        
        if for_debug:
            print("Unique ids: ", self.unique_ids)

        # Get the intersection of the unique ids and the yolo ids
        self.present_ids = self.get_present_ids()

        # Initialize the model
        with open(self.api_path, "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        openai_key = config["openai_api_key"]

        print("Initializing LLM model...")
        self.llm = llm_init(gpt_api=openai_key)

        # Get the yolo ids as strings
        yolo_ids_str = list_yolo_ids(self.present_ids)

        # Get the yolo id conversion chain
        yolo_id_conversion_prompt = get_yolo_id_convert_prompt()
        yolo_id_conversion_chain = LLMChain(
            llm=self.llm, prompt=yolo_id_conversion_prompt, verbose=verbose
        )

        instruction = "Eventually get to the vehicle and eventually get to a table while avoiding wall."  # Instruction for testing
        print(instruction, "\n")

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

        yolo_translate_chain = LLMChain(
            llm=self.llm, prompt=get_yolo_translate_prompt(), verbose=verbose
        )

        yolo_translate_output = get_repeated_outputs(
            input_dict={
                "env_elements": env_elements_str,
                "instruction": yolo_id_instruction,
            },
            chain=yolo_translate_chain,
            n_repeat=n_chain_repeat,
        )
        if for_debug:
            print(yolo_translate_output)

        spot_desc, ap_dict = gpt_to_spot(yolo_translate_output)
        if for_debug:
            print(spot_desc, "\n", ap_dict)

        formula, ap_dict = do_syntactic_check(
            self.llm, yolo_translate_output, spot_desc, ap_dict, verbose=verbose
        )
        formula, ap_dict = do_semantic_check(
            self.llm, yolo_translate_output, formula, yolo_id_instruction, ap_dict, verbose=verbose
        )
        
        
        print("AP Dictionary:{final_ap_dict}".format(final_ap_dict=ap_dict))
        print("LTL FORMULA:{ltl_formula}".format(ltl_formula=formula))


        automaton = spot.translate(formula, "sbacc", "complete")

        automatonAsString = automaton.to_str("hoa")
        
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

        present_dict = {
            all_classes.get(str(id))[3]: f"object_{id}"
            for id in self.unique_ids
            if all_classes.get(str(id))
        }

        if for_debug:
            print("Present dict: ", present_dict)

        return present_dict


if __name__ == "__main__":
    rospy.init_node("ltl_translate_node", anonymous=True)
    Translate()
    rospy.spin()
