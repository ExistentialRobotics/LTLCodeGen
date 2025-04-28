# code inspired from https://github.com/RoboCoachTechnologies/GPT-Synthesizer/blob/master/gpt_synthesizer/main.py
import sys
sys.path.append('/usr/local/lib/python3.8/site-packages') # ADD THE PATH WHERE SPOT WAS BUILT

import spot
from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate

from model import llm_init
from yolo_prompt import get_yolo_translate_prompt, get_yolo_syntactic_check_prompt, get_yolo_id_convert_prompt, \
                        get_check_semantics_prompt, get_reasoning_prompt, get_semantic_correction_prompt 
from yolo_parser import gpt_to_spot, parse_syntax_error, find_env_elements,\
                   load_yolo_ids, get_yolo_id_instruction, list_yolo_ids 
        
# from speech2text import*
import yaml

with open("/home/brabiei/SOLAR_WS/src/SOLAR/speech_to_ltl/config/config_openai.yml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

openai_key = config['openai_api_key']
spot.setup()


def get_repeated_outputs(input_dict, chain, n_repeat):
    output_dict = dict()
    most_repeated = ('', 0)

    for n in range(n_repeat):
        output = chain(input_dict)['text']
        if output in output_dict.keys():
            output_dict[output] += 1
            if output_dict[output] > most_repeated[1]:
                most_repeated = (output, output_dict[output])
        else:
            output_dict[output] = 1

    if most_repeated[1] == 0:
        most_repeated = (list(output_dict.keys())[0], 1)
        print('Warning: LLM produced unique outputs across all attempts!\nInput Keys: {keys}'
              .format(keys=input_dict.keys()))

    return most_repeated[0]

def do_syntactic_check(llm, instruction, spot_desc, ap_dict, n_checks=4, verbose=True): # get the chain instead of llm 
    success = False
    formula = None
    check_prompt = get_yolo_syntactic_check_prompt()
    check_chain = LLMChain(
        llm=llm,
        prompt=check_prompt,
        verbose=verbose
    )
    for n in range(n_checks):
        try:
            # print(spot_desc)
            formula = spot.formula(spot_desc)
            success = True
        except SyntaxError as e:
            print('Attempt {synt_n}: Unsuccessful\nSyntax Error:'.format(synt_n=n))
            print(e)

            wrong_gpt_formula, error_str = parse_syntax_error(e.msg, ap_dict)

            translate_output = check_chain.predict(instruction=instruction,
                                                             syntax_error=error_str,
                                                             wrong_LTL=wrong_gpt_formula)
            spot_desc, ap_dict = gpt_to_spot(translate_output)

        if success:
            return formula, ap_dict

    if not success:
        try:
            formula = spot.formula(spot_desc)
            return formula, ap_dict
        except SyntaxError as e:
            print(e)
            print('All of the syntactic checker reties were unsuccessful! Exiting the program.')
            return 

def do_semantic_check(llm, prefix_output, formula, instruction, ap_dict, n_checks=3, verbose=True): # get chain instead of llm object
    
    check_prompt = get_check_semantics_prompt() # update the chain prompt
    check_chain = LLMChain( 
        llm=llm,
        prompt= check_prompt,
        verbose=verbose
    )
    semantic_check_output = check_chain.predict(instruction = instruction,
                                                         ap_dict = ap_dict,
                                                         formula = formula)
    
    if int(semantic_check_output):
        return formula, ap_dict
    else:
        reasoning_prompt = get_reasoning_prompt() # update prompt instead of creating a new chain
        reasoning_chain = LLMChain(
            llm=llm,
            prompt=reasoning_prompt,
            verbose=verbose
        )
        reasoning_output = reasoning_chain.predict(instruction = instruction,
                                                         ap_dict = ap_dict,
                                                         formula = formula,
                                                         prefix = prefix_output)
        print('Attempt 0: Unsuccessful\nReason:')
        print(reasoning_output)
        correction_prompt = get_semantic_correction_prompt() #update prompt instead of creating new chain
        correction_chain = LLMChain(
            llm=llm,
            prompt=correction_prompt,
            verbose=verbose
        )

        for n in range(n_checks):
            translate_output = correction_chain.predict(instruction = instruction,
                                                         reasoning = reasoning_output,
                                                         wrong_LTL = formula)
            print(translate_output)
            spot_desc, ap_dict = gpt_to_spot(translate_output)
            formula, ap_dict = do_syntactic_check(llm, instruction, spot_desc, ap_dict) # wont this be an error if syntax check fails how do you handle?
            if formula:
                check_output = check_chain.predict(instruction = instruction,
                                                            ap_dict = ap_dict,
                                                            formula = formula)
                if int(check_output):
                    return formula, ap_dict
                else:
                    reasoning_output = reasoning_chain.predict(instruction = instruction,
                                                            ap_dict = ap_dict,
                                                            formula = formula,
                                                            prefix = prefix_output)
                    print('Attempt {sem_n}: Unsuccessful\nReason:'.format(sem_n=n+1))
                    print(reasoning_output)
            else:
                break
            
        print('All of the semantic checker retries were unsuccessful! Exiting the program.')        
        return
   

def main(n_chain_repeat=1, print_results=True, verbose=False, speech=False):
    
    yolo_ids = load_yolo_ids("/home/brabiei/SOLAR_WS/src/SOLAR/SSMI/SSMI-Mapping/params/officesim_color_id.yaml")
    llm = llm_init(gpt_api=openai_key)
    
    print ("Yolo IDs: ", yolo_ids)
    
    if speech:
        yolo_ids_str = list_yolo_ids(yolo_ids)
        yolo_id_conversion_prompt = get_yolo_id_convert_prompt()
        yolo_id_conversion_chain = LLMChain(
            llm=llm,
            prompt=yolo_id_conversion_prompt,
            verbose=verbose)
        
        instruction = speech2instruction(openai_key)
        print(instruction, "\n")
    
        yolo_id_instruction = get_repeated_outputs(input_dict={'yolo_ids': yolo_ids_str,
                                                                  'NL_input': instruction},
                                                      chain=yolo_id_conversion_chain,
                                                      n_repeat=n_chain_repeat)
        print(yolo_id_instruction)
        
        _, env_elements_str = find_env_elements(yolo_id_instruction)

    else:
        instruction = "Eventually get to the sink but first go to the toothbrush and avoid dining table" # Instruction for testing  
        env_elements_str, yolo_id_instruction = get_yolo_id_instruction(instruction, yolo_ids)
        print(yolo_id_instruction)
    
    yolo_translate_chain = LLMChain(
        llm=llm,
        prompt=get_yolo_translate_prompt(),
        verbose=verbose)
    
    yolo_translate_output = get_repeated_outputs(input_dict={'env_elements': env_elements_str,
                                                        'instruction': yolo_id_instruction},
                                            chain=yolo_translate_chain,
                                            n_repeat=n_chain_repeat)
    print(yolo_translate_output)
    
    
    spot_desc, ap_dict = gpt_to_spot(yolo_translate_output)
    print(spot_desc, "\n", ap_dict)
    

    
    formula,ap_dict = do_syntactic_check(llm, yolo_translate_output, spot_desc, ap_dict)
    formula,ap_dict = do_semantic_check(llm, yolo_translate_output, formula, yolo_id_instruction, ap_dict)
    print("AP Dictionary:{final_ap_dict}".format(final_ap_dict=ap_dict))
    print("LTL FORMULA:{ltl_formula}".format(ltl_formula=formula))

    
    print("Works!!!!")
    # automaton = spot.translate(formula)
    # automatonAsString = automaton.to_str()
    # return automatonAsString

if __name__ == '__main__':
    main()

