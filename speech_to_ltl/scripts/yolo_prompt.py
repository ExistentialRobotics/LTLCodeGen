
from langchain.prompts.prompt import PromptTemplate


def get_yolo_id_convert_prompt():
    template = """Your task is to convert the object names in a text to their unique id based on a given object id correspondences.

Use the object IDs provided in the object id correspondences for conversion.

The conversion should take the context of each sentence into account, so that objects with similar names but different locations can be correctly distinguished.

Here are a few examples:

Object ID correspondence:

    'object_27' : 'sink' 
    'object_41' : 'toilet' 
    'object_51' : 'potted plant' 
    'object_46' : 'chair'   
    'object_39' : 'bowl' 
    'object_55' : 'dining table' 
    'object_21' : 'microwave' 
    'object_28' : 'refrigerator'
    'object_31' : 'bottle' 
    'object_36' : 'teddy bear'  
    'object_48' : 'couch' 
    'object_60' : 'tv' 

Input text: Take the bear doll, then pick the water bottle. Always avoid the refrigerator.
Output text: Take object_36, then pick object_31. Always avoid the object_28.

Object ID correspondence:

    'object_24' : 'sink' 
    'object_30' : 'book'
    'object_34' : 'vase'
    'object_40' : 'toilet' 
    'object_51' : 'potted plant' 
    'object_47' : 'chair'   
    'object_39' : 'bowl' 
    'object_54' : 'bed' 
    'object_21' : 'microwave' 
    'object_28' : 'refrigerator'
    'object_33' : 'bottle' 
    'object_35' : 'teddy bear'  
    'object_48' : 'couch' 
    'object_59' : 'tv' 

Input text: Water the potted plant, move the vase. Always avoid going near the couch and sink.
Output text: Water the object_51, move the object_34. Always avoid going near the object_48 and object_24.


Using the provided examples, convert the objects in the following text into their unique IDs.

Object ID correspondence:
{yolo_ids} 
Input text: {NL_input}
Output text: """
    return PromptTemplate(template=template, input_variables=["yolo_ids", "NL_input"])


def get_yolo_translate_prompt():
    template = """Please help transform natural language statements into linear temporal logic (LTL) descriptions.

The LTL operators are: NEGATION, IMPLY, AND, EQUAL, UNTIL, ALWAYS, EVENTUALLY, OR.

The LTL description should follow pre-order expression.

The available action is: reach(object_x).

Some examples of natural language statements and their corresponding LTL descriptions are:

natural language: Reach object_2 and object_1.
LTL:  ['AND', 'reach(object_2)', 'reach(object_1)']

natural language: Get object_2, object_3, object_1.
LTL:  ['AND', 'AND', 'reach(object_2)', 'reach(object_3)', 'reach(object_1)']

natural language: Get every object_8, object_13, object_14, object_17, object_2, object_10, object_12, object_15, object_16, object_18, object_4, object_6, object_7, object_11 in the room.
LTL:  ['AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'reach(object_8)', 'reach(object_13)', 'reach(object_14)', 'reach(object_17)', 'reach(object_2)', 'reach(object_10)', 'reach(object_12)', 'reach(object_15)', 'reach(object_16)', 'reach(object_18)', 'reach(object_4)', 'reach(object_6)', 'reach(object_7)', 'reach(object_11)']

natural language: Picking object_1 always follows picking object_2.
LTL:  ['ALWAYS', 'IMPLY', 'reach(object_2)', 'EVENTUALLY', 'reach(object_1)']

natural language: Maintain object_1 until object_2 is satisfied.
LTL:  ['UNTIL', 'reach(object_1)', 'reach(object_2)']

natural language: Go to object_1 and always avoid both object_2 and object_3.
LTL:  ['AND', 'EVENTUALLY', 'reach(object_1)', 'ALWAYS', 'AND', 'NEGATION', 'reach(object_2)', 'NEGATION', 'reach(object_3)']

natural language: Reach object_2 after reaching to object_1.
LTL:  ['AND', 'EVENTUALLY', 'reach(object_2)', 'UNTIL', 'NEGATION', 'reach(object_2)', 'reach(object_1)']

natural language: Always if reaching object_1 happens before reaching object_3, then start object_2 and eventually cancel object_4.
LTL: ['ALWAYS', 'IMPLY' , 'AND', 'reach(object_3)', 'UNTIL', 'NEGATION', 'reach(object_3)', 'reach(object_1)', 'AND', 'reach(object_2)', 'EVENTUALLY', 'NEGATION', 'reach(object_4)']

natural language: For all time steps, until reach(object_1) and reach(object_2) is true, don't start reach(object_3).
LTL:  ['ALWAYS', 'UNTIL', 'NEGATION', 'reach(object_3)', 'AND', 'reach(object_1)', 'reach(object_2)']

natural language: If object_1 and object_2 and not object_3 or object_4, then object_5 happens eventually.
LTL:  ['IMPLY', 'AND', 'AND', 'reach(object_1)', 'reach(object_2)', 'NEGATION', 'OR', 'reach(object_3)', 'reach(object_4)', 'EVENTUALLY', 'reach(object_5)']

I will give you the list of objects in the environment, and the instruction involves reaching some objects, and avoiding some objects.

Your task is to transform the natural language instruction into a LTL description with pre-order format.

Here are some examples:
Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5, object_6, object_7, object_8]
    natural language instruction: Finally reach object_7, and you have to go to object_4 ahead to reach object_1.
Output:
    LTL: ['AND', 'EVENTUALLY', 'reach(object_7)', 'UNTIL', 'reach(object_4)', 'reach(object_1)']


Input:
    available environment elements: [object_1, object_2, object_3, object_6, object_7, object_8]
    natural language instruction: Finally reach object_2, and you have to reach an object, such as object_6 or object_8, ahead to get object_1. Remember do not reach object_3 at any time.
Output:
    LTL: ['AND', 'AND', 'EVENTUALLY', 'reach(object_2)', 'UNTIL', 'NEGATION', 'reach(object_1)', 'OR', 'reach(object_6)', 'reach(object_8)', 'ALWAYS', 'NEGATION', 'reach(object_3)']


Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5 object_6, object_7, object_8]
    natural language instruction: reach object_1, then object_2 and keep it until reaching object_8, remember always keep away from object_4 and object_6.
Output:
    LTL: ['AND', 'AND', 'AND', 'reach(object_2)', 'UNTIL', 'NEGATION', 'reach(object_2)', 'reach(object_1)', 'UNTIL', 'reach(object_2)', 'reach(object_8)', 'ALWAYS', 'NEGATION', 'OR', 'reach(object_4)', 'reach(object_6)']


Using the provided examples, transform the following natural language instruction into LTL specification:
Input:
    available environment elements: {env_elements}
    natural language instruction: {instruction}
    
The returned output should only contain the LTL formula in the above mentioned list format with no additional headers. Do not provide explaination.    
    
Output:
LTL: """
    return PromptTemplate(template=template, input_variables=["env_elements", "instruction"])


def get_yolo_syntactic_check_prompt():
    template = """Please help transform natural language instructions into linear temporal logic (LTL) formulas.

The LTL operators are: NEGATION, IMPLY, AND, EQUAL, UNTIL, ALWAYS, EVENTUALLY, OR.

The LTL description should follow pre-order expression.

The available action is: reach(object_x).

Some examples of natural language statements and their corresponding LTL descriptions are:

natural language: Pick object_2 if you place object_1.
LTL:  ['IMPLY', 'NEGATION', 'reach(object_1)', 'EVENTUALLY', 'reach(object_2)']

natural language: Reach object_2 only if you don't reach object_1.
LTL:  ['IMPLY', 'NEGATION', 'EVENTUALLY', 'reach(object_1)', 'ALWAYS', 'EVENTUALLY', 'reach(object_2)']

natural language: Reach object_2 and object_1.
LTL:  ['AND', 'reach(object_2)', 'reach(object_1)']

natural language: Get object_2, object_3, object_1.
LTL:  ['AND', 'AND', 'reach(object_2)', 'reach(object_3)', 'reach(object_1)']

natural language: Get every object_8, object_13, object_14, object_17, object_2, object_10, object_12, object_15, object_16, object_18, object_4, object_6, object_7, object_11 in the room.
LTL:  ['AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'reach(object_8)', 'reach(object_13)', 'reach(object_14)', 'reach(object_17)', 'reach(object_2)', 'reach(object_10)', 'reach(object_12)', 'reach(object_15)', 'reach(object_16)', 'reach(object_18)', 'reach(object_4)', 'reach(object_6)', 'reach(object_7)', 'reach(object_11)']

natural language: Picking object_1 always follows picking object_2.
LTL:  ['ALWAYS', 'IMPLY', 'reach(object_2)', 'EVENTUALLY', 'reach(object_1)']

natural language: Maintain object_1 until object_2 is satisfied.
LTL:  ['UNTIL', 'reach(object_1)', 'reach(object_2)']

natural language: Go to object_1 and always avoid both object_2 and object_3.
LTL:  ['AND', 'EVENTUALLY', 'reach(object_1)', 'ALWAYS', 'AND', 'NEGATION', 'reach(object_2)', 'NEGATION', 'reach(object_3)']

natural language: Reach object_2 after reaching to object_1.
LTL:  ['AND', 'EVENTUALLY', 'reach(object_2)', 'UNTIL', 'NEGATION', 'reach(object_2)', 'reach(object_1)']

natural language: Always if reaching object_1 happens before reaching object_3, then start object_2 and eventually cancel object_4.
LTL: ['ALWAYS', 'IMPLY' , 'AND', 'reach(object_3)', 'UNTIL', 'NEGATION', 'reach(object_3)', 'reach(object_1)', 'AND', 'reach(object_2)', 'EVENTUALLY', 'NEGATION', 'reach(object_4)']

natural language: For all time steps, until reach(object_1) and reach(object_2) is true, don't start reach(object_3).
LTL:  ['ALWAYS', 'UNTIL', 'NEGATION', 'reach(object_3)', 'AND', 'reach(object_1)', 'reach(object_2)']

natural language: If object_1 and object_2 and not object_3 or object_4, then object_5 happens eventually.
LTL:  ['IMPLY', 'AND', 'AND', 'reach(object_1)', 'reach(object_2)', 'NEGATION', 'OR', 'reach(object_3)', 'reach(object_4)', 'EVENTUALLY', 'reach(object_5)']

I will give you the list of objects in the environment, and the instruction involves reaching some objects, and avoiding some objects.

Your task is to transform the natural language instruction into a LTL description with pre-order format.

Here are some examples:
Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5, object_6, object_7, object_8]
    natural language instruction: Finally reach object_7, and you have to go to object_4 ahead to reach object_1.
Output:
    LTL: ['AND', 'EVENTUALLY', 'reach(object_7)', 'UNTIL', 'reach(object_4)', 'reach(object_1)']


Input:
    available environment elements: [object_1, object_2, object_3, object_6, object_7, object_8]
    natural language instruction: Finally reach object_2, and you have to reach an object, such as object_6 or object_8, ahead to get object_1. Remember do not reach object_3 at any time.
Output:
    LTL: ['AND', 'AND', 'EVENTUALLY', 'reach(object_2)', 'UNTIL', 'NEGATION', 'reach(object_1)', 'OR', 'reach(object_6)', 'reach(object_8)', 'ALWAYS', 'NEGATION', 'reach(object_3)']


Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5 object_6, object_7, object_8]
    natural language instruction: reach object_1, then object_2 and keep it until reaching object_8, remember always keep away from object_4 and object_6.
Output:
    LTL: ['AND', 'AND', 'AND', 'reach(object_2)', 'UNTIL', 'NEGATION', 'reach(object_2)', 'reach(object_1)', 'UNTIL', 'reach(object_2)', 'reach(object_8)', 'ALWAYS', 'NEGATION', 'OR', 'reach(object_4)', 'reach(object_6)']


Input:
    available environment elements: [object_1, object_2, object_3, object_4]
    natural language instruction: Every robot should reach object_3 every time they leave object_1. After object_2, the robot should visit object_4 at some point.
Output:
    LTL: ['AND', 'ALWAYS', 'IMPLY', 'NEGATION', 'reach(object_1)', 'reach(object_3)', 'AND', 'EVENTUALLY', 'reach(object_4)', 'UNTIL', 'NEGATION', 'reach(object_4)', 'reach(object_2)']


Input:
    available environment elements: [object_1, object_2, object_3, object_6, object_7, object_8]
    natural language instruction: 1) Every robot should visit object_3 every time they leave object_2. 2) After reaching object_7, the robot should visit object_3, to transmit the collected data to the remote control. 3) The robots should avoid object_1.
Output:
    LTL: ['AND', 'AND', 'ALWAYS', 'IMPLY', 'NEGATION', 'reach(object_2)', 'EVENTUALLY', 'reach(object_3)', 'AND', 'reach(object_3)', 'UNTIL', 'NEGATION', 'reach(object_3)', 'reach(object_7)', 'ALWAYS', 'NEGATION', 'reach(object_1)']


Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5, object_6]
    natural language instruction: Go to object_1, then reach object_2 and keep it until reaching object_4, and finally reach object_3. Remember always do not touch object_6.
Output:
    LTL: ['AND', 'AND', 'AND', 'AND', 'reach(object_2)', 'UNTIL', 'NEGATION', 'reach(object_2)', 'reach(object_1)', 'UNTIL', 'reach(object_2)', 'reach(object_4)', 'EVENTUALLY', 'reach(object_3)', 'ALWAYS', 'NEGATION', 'reach(object_6)']

Trained by the above examples, an AI has generated a syntactically INCORRECT LTL formula for the following natural language instruction:
natural language: {instruction}
Incorrect LTL: {wrong_LTL}

The incorrect part of the LTL formula is shown with '--> INCORRECT', meaning that this part is causing the error and needs to be modified.

More specifically, here is a description of the error caused by the incorrect part:
{syntax_error}

Generate a syntactically correct revision of the LTL formula similar to the examples provided above. 

Pay attention to the number of elements that each LTL operator requires also pay attention to temporal ordering and relationship between actions.

For example, 'AND', 'OR', 'EQUAL', 'IMPLY', 'UNTIL' operators take two inputs: 'AND', 'reach(object_1)', 'reach(object_2)'
On the other hand, 'NEGATION', 'ALWAYS', 'EVENTUALLY' only take a single input: 'EVENTUALLY', 'reach(object_2)'

natural language: {instruction}

The returned output should only contain the LTL formula in the above mentioned list format with no additional headers. Do not provide explaination.

Corrected LTL: """
    return PromptTemplate(template=template, input_variables=["instruction", "syntax_error", "wrong_LTL"])


def get_check_semantics_prompt():
    template = """Given the following information

The LTL operators are: NEGATION, IMPLY, AND, EQUAL, UNTIL, ALWAYS, EVENTUALLY, OR.

The operation to symbol association :           'NEGATION': '!',
                                                'IMPLY': 'i',
                                                'AND': '&',
                                                'EQUAL': 'e',
                                                'UNTIL': 'U',
                                                'ALWAYS': 'G',
                                                'EVENTUALLY': 'F',
                                                'OR': '|'
                                                
Natural Language specification of the task: {instruction}

Associated atomic proposition dictionary: {ap_dict}

Generated LTL formula: {formula}

Please help check if the generated LTL correctly represents the given task.

Return only 1 or 0. 

"""
    return PromptTemplate(template=template, input_variables=["instruction", "ap_dict", "formula"])

def get_reasoning_prompt():
    template = """Given the following information

The LTL operators are: NEGATION, IMPLY, AND, EQUAL, UNTIL, ALWAYS, EVENTUALLY, OR.

The operation to symbol association :           'NEGATION': '!',
                                                'IMPLY': 'i',
                                                'AND': '&',
                                                'EQUAL': 'e',
                                                'UNTIL': 'U',
                                                'ALWAYS': 'G',
                                                'EVENTUALLY': 'F',
                                                'OR': '|'
                                                
Natural Language specification of the task: {instruction}

Associated atomic proposition dictionary: {ap_dict}

Generated LTL formula: {formula}

Associated prefix format: {prefix}

Provide reasoning for why the generated LTL is wrong. Do not provide the correct formula, just give a reasoning for why it is wrong. 
"""
    return PromptTemplate(template=template, input_variables=["instruction", "ap_dict", "formula", "prefix"])

def get_semantic_correction_prompt():
    template = """Please help transform natural language instructions into linear temporal logic (LTL) formulas.

The LTL operators are: NEGATION, IMPLY, AND, EQUAL, UNTIL, ALWAYS, EVENTUALLY, OR.

The LTL description should follow pre-order expression.

The available action is: reach(object_x).

Some examples of natural language statements and their corresponding LTL descriptions are:

natural language: Reach object_2 and object_1.
LTL:  ['AND', 'reach(object_2)', 'reach(object_1)']

natural language: Get object_2, object_3, object_1.
LTL:  ['AND', 'AND', 'reach(object_2)', 'reach(object_3)', 'reach(object_1)']

natural language: Get every object_8, object_13, object_14, object_17, object_2, object_10, object_12, object_15, object_16, object_18, object_4, object_6, object_7, object_11 in the room.
LTL:  ['AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'reach(object_8)', 'reach(object_13)', 'reach(object_14)', 'reach(object_17)', 'reach(object_2)', 'reach(object_10)', 'reach(object_12)', 'reach(object_15)', 'reach(object_16)', 'reach(object_18)', 'reach(object_4)', 'reach(object_6)', 'reach(object_7)', 'reach(object_11)']

natural language: Picking object_1 always follows picking object_2.
LTL:  ['ALWAYS', 'IMPLY', 'reach(object_2)', 'EVENTUALLY', 'reach(object_1)']

natural language: Maintain object_1 until object_2 is satisfied.
LTL:  ['UNTIL', 'reach(object_1)', 'reach(object_2)']

natural language: Go to object_1 and always avoid both object_2 and object_3.
LTL:  ['AND', 'EVENTUALLY', 'reach(object_1)', 'ALWAYS', 'AND', 'NEGATION', 'reach(object_2)', 'NEGATION', 'reach(object_3)']

natural language: Reach object_2 after reaching to object_1.
LTL:  ['AND', 'EVENTUALLY', 'reach(object_2)', 'UNTIL', 'NEGATION', 'reach(object_2)', 'reach(object_1)']

natural language: Always if reaching object_1 happens before reaching object_3, then start object_2 and eventually cancel object_4.
LTL: ['ALWAYS', 'IMPLY' , 'AND', 'reach(object_3)', 'UNTIL', 'NEGATION', 'reach(object_3)', 'reach(object_1)', 'AND', 'reach(object_2)', 'EVENTUALLY', 'NEGATION', 'reach(object_4)']


I will give you the list of objects in the environment, and the instruction involves reaching some objects, and avoiding some objects.

Your task is to transform the natural language instruction into a LTL description with pre-order format.

Here are some examples:
Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5, object_6, object_7, object_8]
    natural language instruction: Finally reach object_7, and you have to go to object_4 ahead to reach object_1.
Output:
    LTL: ['AND', 'EVENTUALLY', 'reach(object_7)', 'UNTIL', 'reach(object_4)', 'reach(object_1)']


Input:
    available environment elements: [object_1, object_2, object_3, object_6, object_7, object_8]
    natural language instruction: Finally reach object_2, and you have to reach an object, such as object_6 or object_8, ahead to get object_1. Remember do not reach object_3 at any time.
Output:
    LTL: ['AND', 'AND', 'EVENTUALLY', 'reach(object_2)', 'UNTIL', 'NEGATION', 'reach(object_1)', 'OR', 'reach(object_6)', 'reach(object_8)', 'ALWAYS', 'NEGATION', 'reach(object_3)']


Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5 object_6, object_7, object_8]
    natural language instruction: reach object_1, then object_2 and keep it until reaching object_8, remember always keep away from object_4 and object_6.
Output:
    LTL: ['AND', 'AND', 'AND', 'reach(object_2)', 'UNTIL', 'NEGATION', 'reach(object_2)', 'reach(object_1)', 'UNTIL', 'reach(object_2)', 'reach(object_8)', 'ALWAYS', 'NEGATION', 'OR', 'reach(object_4)', 'reach(object_6)']


Trained by the above examples, an AI has generated a semantically INCORRECT LTL formula for the following natural language instruction:
natural language: {instruction}
Incorrect LTL: {wrong_LTL}


Here is the reasoning for why the generated LTL is wrong:
{reasoning}

Generate a semantically any syntactically correct revision of the LTL formula. 

natural language: {instruction}

The returned output should only contain the LTL formula in the above mentioned list format. Do not provide explanation.

Corrected LTL: """
    return PromptTemplate(template=template, input_variables=["instruction", "reasoning", "wrong_LTL"])


# Test with the examples and without the examples on higher end models to see if there is any difference.
def get_syntactic_check_prompt():
    template = """Please help transform natural language instructions into linear temporal logic (LTL) formulas.

The LTL operators are: NEGATION, IMPLY, AND, EQUAL, UNTIL, ALWAYS, EVENTUALLY, OR.

The LTL description should follow pre-order expression.

The available action is: reach(object_x).

Some examples of natural language statements and their corresponding LTL descriptions are:

natural language: Pick object_2 if you place object_1.
LTL:  ['IMPLY', 'NEGATION', 'reach(object_1)', 'EVENTUALLY', 'reach(object_2)']

natural language: Reach object_2 only if you don't reach object_1.
LTL:  ['IMPLY', 'NEGATION', 'EVENTUALLY', 'reach(object_1)', 'ALWAYS', 'EVENTUALLY', 'reach(object_2)']

natural language: Reach object_2 and object_1.
LTL:  ['AND', 'reach(object_2)', 'reach(object_1)']

natural language: Get object_2, object_3, object_1.
LTL:  ['AND', 'AND', 'reach(object_2)', 'reach(object_3)', 'reach(object_1)']

natural language: Get every object_8, object_13, object_14, object_17, object_2, object_10, object_12, object_15, object_16, object_18, object_4, object_6, object_7, object_11 in the room.
LTL:  ['AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'AND', 'reach(object_8)', 'reach(object_13)', 'reach(object_14)', 'reach(object_17)', 'reach(object_2)', 'reach(object_10)', 'reach(object_12)', 'reach(object_15)', 'reach(object_16)', 'reach(object_18)', 'reach(object_4)', 'reach(object_6)', 'reach(object_7)', 'reach(object_11)']

natural language: Picking object_1 always follows picking object_2.
LTL:  ['ALWAYS', 'IMPLY', 'reach(object_2)', 'EVENTUALLY', 'reach(object_1)']

natural language: Maintain object_1 until object_2 is satisfied.
LTL:  ['UNTIL', 'reach(object_1)', 'reach(object_2)']

natural language: Go to object_1 and always avoid both object_2 and object_3.
LTL:  ['AND', 'EVENTUALLY', 'reach(object_1)', 'ALWAYS', 'AND', 'NEGATION', 'reach(object_2)', 'NEGATION', 'reach(object_3)']

natural language: Reach object_2 after reaching to object_1.
LTL:  ['AND', 'EVENTUALLY', 'reach(object_2)', 'UNTIL', 'NEGATION', 'reach(object_2)', 'reach(object_1)']

natural language: Always if reaching object_1 happens before reaching object_3, then start object_2 and eventually cancel object_4.
LTL: ['ALWAYS', 'IMPLY' , 'AND', 'reach(object_3)', 'UNTIL', 'NEGATION', 'reach(object_3)', 'reach(object_1)', 'AND', 'reach(object_2)', 'EVENTUALLY', 'NEGATION', 'reach(object_4)']

natural language: For all time steps, until reach(object_1) and reach(object_2) is true, don't start reach(object_3).
LTL:  ['ALWAYS', 'UNTIL', 'NEGATION', 'reach(object_3)', 'AND', 'reach(object_1)', 'reach(object_2)']

natural language: If object_1 and object_2 and not object_3 or object_4, then object_5 happens eventually.
LTL:  ['IMPLY', 'AND', 'AND', 'reach(object_1)', 'reach(object_2)', 'NEGATION', 'OR', 'reach(object_3)', 'reach(object_4)', 'EVENTUALLY', 'reach(object_5)']

I will give you the list of objects in the environment, and the instruction involves reaching some objects, and avoiding some objects.

Your task is to transform the natural language instruction into a LTL description with pre-order format.

Here are some examples:
Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5, object_6, object_7, object_8]
    natural language instruction: Finally reach object_7, and you have to go to object_4 ahead to reach object_1.
Output:
    LTL: ['AND', 'EVENTUALLY', 'reach(object_7)', 'UNTIL', 'reach(object_4)', 'reach(object_1)']


Input:
    available environment elements: [object_1, object_2, object_3, object_6, object_7, object_8]
    natural language instruction: Finally reach object_2, and you have to reach an object, such as object_6 or object_8, ahead to get object_1. Remember do not reach object_3 at any time.
Output:
    LTL: ['AND', 'AND', 'EVENTUALLY', 'reach(object_2)', 'UNTIL', 'NEGATION', 'reach(object_1)', 'OR', 'reach(object_6)', 'reach(object_8)', 'ALWAYS', 'NEGATION', 'reach(object_3)']


Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5 object_6, object_7, object_8]
    natural language instruction: reach object_1, then object_2 and keep it until reaching object_8, remember always keep away from object_4 and object_6.
Output:
    LTL: ['AND', 'AND', 'AND', 'reach(object_2)', 'UNTIL', 'NEGATION', 'reach(object_2)', 'reach(object_1)', 'UNTIL', 'reach(object_2)', 'reach(object_8)', 'ALWAYS', 'NEGATION', 'OR', 'reach(object_4)', 'reach(object_6)']


Input:
    available environment elements: [object_1, object_2, object_3, object_4]
    natural language instruction: Every robot should reach object_3 every time they leave object_1. After object_2, the robot should visit object_4 at some point.
Output:
    LTL: ['AND', 'ALWAYS', 'IMPLY', 'NEGATION', 'reach(object_1)', 'reach(object_3)', 'AND', 'EVENTUALLY', 'reach(object_4)', 'UNTIL', 'NEGATION', 'reach(object_4)', 'reach(object_2)']


Input:
    available environment elements: [object_1, object_2, object_3, object_6, object_7, object_8]
    natural language instruction: 1) Every robot should visit object_3 every time they leave object_2. 2) After reaching object_7, the robot should visit object_3, to transmit the collected data to the remote control. 3) The robots should avoid object_1.
Output:
    LTL: ['AND', 'AND', 'ALWAYS', 'IMPLY', 'NEGATION', 'reach(object_2)', 'EVENTUALLY', 'reach(object_3)', 'AND', 'reach(object_3)', 'UNTIL', 'NEGATION', 'reach(object_3)', 'reach(object_7)', 'ALWAYS', 'NEGATION', 'reach(object_1)']


Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5, object_6]
    natural language instruction: Go to object_1, then reach object_2 and keep it until reaching object_4, and finally reach object_3. Remember always do not touch object_6.
Output:
    LTL: ['AND', 'AND', 'AND', 'AND', 'reach(object_2)', 'UNTIL', 'NEGATION', 'reach(object_2)', 'reach(object_1)', 'UNTIL', 'reach(object_2)', 'reach(object_4)', 'EVENTUALLY', 'reach(object_3)', 'ALWAYS', 'NEGATION', 'reach(object_6)']

Trained by the above examples, an AI has generated a syntactically INCORRECT LTL formula for the following natural language instruction:
natural language: {instruction}
Incorrect LTL: {wrong_LTL}

The incorrect part of the LTL formula is shown with '--> INCORRECT', meaning that this part is causing the error and needs to be modified.

Description of the error caused by the incorrect part:
{syntax_error}

Generate a syntactically correct revision of the LTL formula similar to the examples provided above. 

Additional context about the task:
{reason}

For example, 'AND', 'OR', 'EQUAL', 'IMPLY', 'UNTIL' operators take two inputs: 'AND', 'reach(object_1)', 'reach(object_2)'
On the other hand, 'NEGATION', 'ALWAYS', 'EVENTUALLY' only take a single input: 'EVENTUALLY', 'reach(object_2)'

Natural language: {instruction}

The returned output should only contain the LTL formula in the above mentioned list format with no additional headers. 
Do not provide explaination.

Corrected LTL: """
    return PromptTemplate(template=template, input_variables=["instruction", "syntax_error", "wrong_LTL", "reason"])
