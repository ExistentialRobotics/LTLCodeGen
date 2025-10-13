# code inspired from https://github.com/RoboCoachTechnologies/GPT-Synthesizer/blob/master/gpt_synthesizer/main.py
import sys

sys.path.append("/usr/local/lib/python3.8/site-packages")  # ADD THE PATH WHERE SPOT WAS BUILT

import spot
from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate

from speech_to_ltl.scenegraph_prompt import (
    get_translate_prompt,
    get_syntactic_check_prompt,
    get_yolo_id_convert_prompt,
    get_check_semantics_prompt,
    get_reasoning_prompt,
    get_semantic_correction_prompt,
)
from speech_to_ltl.yolo_parser import (
    gpt_to_spot,
    parse_syntax_error,
)


import os

PROJ_DIR = os.path.join(os.path.dirname(__file__), "../../")
PROJ_DIR = os.path.realpath(PROJ_DIR)

openai_key = os.getenv("OPENAI_API_KEY")
if openai_key is None:
    raise ValueError("Environment variable `OPENAI_API_KEY` is required but not set.")

spot.setup()


def get_repeated_outputs(input_dict, chain, n_repeat) -> str:
    output_dict = dict()
    most_repeated = ("", 0)

    for n in range(n_repeat):
        output = chain(input_dict)["text"]
        if output in output_dict.keys():
            output_dict[output] += 1
            if output_dict[output] > most_repeated[1]:
                most_repeated = (output, output_dict[output])
        else:
            output_dict[output] = 1

    if most_repeated[1] == 0:
        most_repeated = (list(output_dict.keys())[0], 1)

    return most_repeated[0]


def do_syntactic_check(llm, instruction, spot_desc, ap_dict, n_checks=4, verbose=True):  # get the chain instead of llm
    success = False
    formula = None
    check_prompt = get_syntactic_check_prompt()

    # Initialize memory
    all_preds = []

    check_chain = LLMChain(llm=llm, prompt=check_prompt, verbose=verbose)
    for n in range(n_checks):
        try:
            formula = spot.formula(spot_desc)
            success = True
        except SyntaxError as e:
            print("Attempt {synt_n}: Unsuccessful\nSyntax Error:".format(synt_n=n))
            print(e)

            wrong_gpt_formula, error_str = parse_syntax_error(e.msg, ap_dict)

            translate_output = check_chain.predict(
                instruction=instruction + "Here is a history\n" + "\n".join(all_preds),
                syntax_error=error_str,
                wrong_LTL=wrong_gpt_formula,
            )

            all_preds.append(translate_output)

            spot_desc, ap_dict = gpt_to_spot(translate_output)

        if success:
            return formula, ap_dict

    if not success:
        try:
            formula = spot.formula(spot_desc)
            return formula, ap_dict
        except SyntaxError as e:
            print(e)
            print("All of the syntactic checker reties were unsuccessful! Exiting the program.")
            return


def do_semantic_check(llm, prefix_output, formula, instruction, ap_dict, n_checks=3, verbose=True):  # get chain instead of llm object

    check_prompt = get_check_semantics_prompt()  # update the chain prompt
    check_chain = LLMChain(llm=llm, prompt=check_prompt, verbose=verbose)
    semantic_check_output = check_chain.predict(instruction=instruction, ap_dict=ap_dict, formula=formula)

    if int(semantic_check_output):
        return formula, ap_dict
    else:
        reasoning_prompt = get_reasoning_prompt()  # update prompt instead of creating a new chain
        reasoning_chain = LLMChain(llm=llm, prompt=reasoning_prompt, verbose=verbose)
        reasoning_output = reasoning_chain.predict(instruction=instruction, ap_dict=ap_dict, formula=formula, prefix=prefix_output)
        print("Attempt 0: Unsuccessful\nReason:")
        print(reasoning_output)
        correction_prompt = get_semantic_correction_prompt()  # update prompt instead of creating new chain
        correction_chain = LLMChain(llm=llm, prompt=correction_prompt, verbose=verbose)

        for n in range(n_checks):
            translate_output = correction_chain.predict(instruction=instruction, reasoning=reasoning_output, wrong_LTL=formula)
            print(translate_output)
            spot_desc, ap_dict = gpt_to_spot(translate_output)
            formula, ap_dict = do_syntactic_check(llm, instruction, spot_desc, ap_dict)  # wont this be an error if syntax check fails how do you handle?
            if formula:
                check_output = check_chain.predict(instruction=instruction, ap_dict=ap_dict, formula=formula)
                if int(check_output):
                    return formula, ap_dict
                else:
                    reasoning_output = reasoning_chain.predict(instruction=instruction, ap_dict=ap_dict, formula=formula, prefix=prefix_output)
                    print("Attempt {sem_n}: Unsuccessful\nReason:".format(sem_n=n + 1))
                    print(reasoning_output)
            else:
                break

        print("All of the semantic checker retries were unsuccessful! Exiting the program.")
        return formula, ap_dict

