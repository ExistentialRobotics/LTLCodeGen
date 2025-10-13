# code inspired from https://github.com/RoboCoachTechnologies/GPT-Synthesizer/blob/master/gpt_synthesizer/parser.py
import ast
import re
import numpy as np
import yaml

GPT_SPOT_SYNTAX = {'NEGATION': '!',
                   'IMPLY': 'i',
                   'AND': '&',
                   'EQUAL': 'e',
                   'UNTIL': 'U',
                   'ALWAYS': 'G',
                   'EVENTUALLY': 'F',
                   'OR': '|'}


GPT_SPOT_SYNTAX_INV = {value: key for (key, value) in GPT_SPOT_SYNTAX.items()}


def gpt_to_spot(gpt_str):
    if isinstance(gpt_str, str):
        gpt_str.strip()
    tl_list = ast.literal_eval(gpt_str)
    ap_dict = dict()
    spot_desc = ""
    for tl in tl_list:
        if 'enter' in tl or 'reach' in tl:
            if tl not in ap_dict.keys():
                ap_dict[tl] = 'p' + str(len(ap_dict))
            spot_desc += ' {prop}'.format(prop=ap_dict[tl])
        else:
            spot_desc += ' {oper}'.format(oper=GPT_SPOT_SYNTAX[tl])

    return spot_desc, ap_dict


def save_ap_desc(ap_dict): # Needs some modification, as I progress to automaton saving or maybe not required for us at all. 
    ap_desc = {}
    for k in ap_dict.keys():
        if k.startswith('enter'):
            ap_desc[ap_dict[k]] = {
                'type': 'kEnterRoom',
                'uuid': int(k.split('_')[1].strip()[:-1]),
            }
        elif k.startswith('reach'):
            ap_desc[ap_dict[k]] = {
                'type': 'kReachObject',
                'uuid': int(k.split('_')[1].strip()[:-1]),
                'reach_distance': -1.  # negative means not specified
            }
        else:
            raise ValueError('Unknown AP type: {}'.format(k))
    print("Save AP descriptions to ap_desc.yaml")
    print(ap_desc)
    with open('ap_desc.yaml', 'w') as f:
        yaml.dump(ap_desc, f)


def save_ap_desc_npz(ap_dict):
    ap_desc = {}
    for k in ap_dict.keys():
        ap_desc[ap_dict[k]] = k
    np.savez("ap_desc.npz", **ap_desc)


def load_yolo_ids(yaml_file):
    with open(yaml_file, "r") as file:
        yolo_dict = yaml.load(file, Loader=yaml.FullLoader)

    yolo_ids = { yolo_dict[id][-1] : "object_" + id  for id in yolo_dict.keys()}
    
    return yolo_ids

def list_yolo_ids(yolo_ids):
    list_str = """"""
    for key in yolo_ids:
        list_str += "'{id}' : '{object}' \n".format(object = key, 
                                                            id = yolo_ids[key])
        
    return list_str

def find_env_elements(input_str):
    env_elements = set()
    room_ind_tuple_list = [(m.start(), m.end()) for m in re.finditer('room_\d+', input_str)]
    object_ind_tuple_list = [(m.start(), m.end()) for m in re.finditer('object_\d+', input_str)]

    if len(room_ind_tuple_list) > 0:
        for room_ind_tuple in room_ind_tuple_list:
            env_elements.add(input_str[room_ind_tuple[0]:room_ind_tuple[1]])

    if len(object_ind_tuple_list) > 0:
        for object_ind_tuple in object_ind_tuple_list:
            env_elements.add(input_str[object_ind_tuple[0]:object_ind_tuple[1]])

    env_elements = list(env_elements)
    env_elements_str = str(env_elements).replace("'", "")

    return env_elements, env_elements_str

def replace_and_track(match, yolo_dict, env_elements):
    object_name = match.group(0)
    object_id = yolo_dict[object_name]
    env_elements.add(object_id)
    return object_id
    

def get_yolo_id_instruction(instruction, yolo_dict):
    env_elements = set()
    pattern = "|".join(re.escape(object_name) for object_name in yolo_dict.keys())
    
    yolo_instruction = re.sub(pattern, 
                              lambda match: replace_and_track(match, yolo_dict, env_elements),
                              instruction)    
    
    env_elements = list(env_elements)
    env_elements_str = "[" + ", ".join(element for element in env_elements) + "]"
    
    return env_elements_str, yolo_instruction


def parse_syntax_error(syntax_error, ap_dict):
    ap_dict_inv = {value: key for (key, value) in ap_dict.items()}
    spot_formula_str = syntax_error.splitlines()[1][5:]
    gpt_formula = []
    for element in spot_formula_str.split(' '):
        if element in ap_dict_inv.keys():
            gpt_formula.append(ap_dict_inv[element])
        elif element in GPT_SPOT_SYNTAX_INV.keys():
            gpt_formula.append(GPT_SPOT_SYNTAX_INV[element])
        else:
            print('Error: syntax error is invalid!')

    indicator_str = syntax_error.splitlines()[2][5:]
    error_element_ind = spot_formula_str[:len(indicator_str)].count(' ')
    gpt_formula[error_element_ind] = "{error_element} --> INCORRECT".format(error_element=gpt_formula[error_element_ind])

    error_str = syntax_error.splitlines()[3]

    return str(gpt_formula), error_str
