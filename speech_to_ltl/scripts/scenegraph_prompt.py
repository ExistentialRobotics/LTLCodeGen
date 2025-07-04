from langchain.prompts.prompt import PromptTemplate


def get_yolo_id_convert_prompt():
    template = """Your task is to convert the object names in a text to their unique id based on given object id correspondences.

Use the object IDs provided in the object id correspondences for conversion.
The conversion should take the context of each sentence into account, so that objects with similar meaning but different locations can be correctly distinguished.

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
Input text: Take the teddy bear, then pick the bottle. Always avoid the refrigerator.
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

Object ID correspondence:
    'object_24' : 'sink' 
    'object_30' : 'book'
    'object_34' : 'vase'
    'object_40' : 'toilet'
    'object_49' : 'potted plant'
    'object_46' : 'chair'
    'object_39' : 'bowl'
    'object_54' : 'bed'
    'object_21' : 'microwave'
    'object_22' : 'oven'
    'object_28' : 'refrigerator'
    'object_31' : 'bottle'
    'object_36' : 'teddy bear'
    'object_48' : 'couch'
    'object_55' : 'dining table'
    'object_60' : 'tv'

Input text: Get the bottle out of the refrigerator, place it on the dining table. Avoid the going near the oven.
Output text: Get the object_31 out of the object_28, place it on the object_55. Avoid the going near the object_22.

Using the provided examples, convert the objects in the following text into their unique IDs.
Object ID correspondence:
{yolo_ids} 
Input text: {NL_input}
Output text: """
    return PromptTemplate(template=template, input_variables=["yolo_ids", "NL_input"])


def get_translate_prompt():
    template = """Please help transform natural language statements into linear temporal logic (LTL) descriptions.

The LTL operators are: NEGATION, IMPLY, AND, EQUAL, UNTIL, ALWAYS, EVENTUALLY, OR.

The LTL description should follow pre-order expression.

The available action is: reach(object_x).

Given below are examples of natural language statements their corresponding LTL descriptions and their explanation for your understanding:

NOTE: You can differentiate between Sequential and revisiting tasks by key phrases such as 'revisit', 'go back to', 'come back' 'return', 'again' etc.

natural language: Check object_2 and object_1.
LTL:  ['AND', 'reach(object_2)', 'reach(object_1)']
explanation: In this LTL 'AND' checks if both objects are visited at the current time. If not you reach sink state. 

natural language:  object_1 until object_2 is satisfied.
LTL:  ['UNTIL', 'reach(object_1)', 'reach(object_2)']
explanation: In this LTL 'UNTIL' checks if object_1 is true until object_2 is satisfied. If object_1 is false you reach sink state.

natural language: Reach object_2 and object_1.
LTL:  ['AND', 'EVENTUALLY', 'reach(object_2)', 'EVENTUALLY', 'reach(object_1)']
explanation: In this LTL 'AND' ensures both the objects are visited at current time step. 'EVENTUALLY' is added to relax the 
             condition of visiting them at the current time. Accomodating cases where objects are visited in future time. 

natural language: Reach object_2, object_3, object_1.
LTL:  ['AND', 'EVENTUALLY', 'reach(object_2)', 'AND', 'EVENTUALLY', 'reach(object_3)', 'EVENTUALLY', 'reach(object_1)']
explanation: In this LTL 'AND' ensures all objects are visited at the current time step. 'EVENTUALLY' relaxes the condition of 
             visiting them them at the current time. Accomodating cases where objects are visited in future time steps.

natural language: Go to object_1 and always avoid both object_2 and object_3.
LTL:  ['AND', 'EVENTUALLY', 'reach(object_1)', 'ALWAYS', 'AND', 'NEGATION', 'reach(object_2)', 'NEGATION', 'reach(object_3)']
explanation: In this LTL 'EVENTUALLY' allows object_1 to be reached at any time step. 'ALWAYS' ensures any violation of the 
             condition leads to a sink state. 'AND' is used to penalize visting either object_2 or object_3.

 natural language: Reach object_8, object_13, object_14, object_17, object_2, object_10, object_12, object_15, object_16, object_18, object_4, object_6, object_7, object_11 in the room.
LTL:  ['AND', 'EVENTUALLY', 'reach(object_8)', 'AND', 'EVENTUALLY', 'reach(object_13)', 'AND', 'EVENTUALLY', 'reach(object_14)', 'AND', 'EVENTUALLY', 'reach(object_17)', 'AND', 'EVENTUALLY', 'reach(object_2)', 'AND', 'EVENTUALLY', 'reach(object_10)', 'AND', 'EVENTUALLY', 'reach(object_12)', 'AND', 'EVENTUALLY', 'reach(object_15)', 'AND', 'EVENTUALLY', 'reach(object_16)', 'AND', 'EVENTUALLY', 'reach(object_18)', 'AND', 'EVENTUALLY', 'reach(object_4)', 'AND', 'EVENTUALLY', 'reach(object_6)', 'AND', 'EVENTUALLY', 'reach(object_7)', 'EVENTUALLY', 'reach(object_11)']
explanation: In this LTL 'AND' ensures all objects are visited at the current time step. 'EVENTUALLY' relaxes the condition of 
             visiting them them at the current time. Accomodating cases where objects are visited in future time steps.

natural language: Reach object_2 if you leave object_1.
LTL:  ['IMPLY', 'NEGATION', 'reach(object_1)', 'EVENTUALLY', 'reach(object_2)']
explanation: In this LTL 'IMPLY' checks the condition not object_1(leave object_1). If the condition is met, try to reach object_2. 
             'EVENTUALLY' indicates the robot can reach object_2 at any time step.

natural language: Reach object_2 only if you don't reach object_1.
LTL:  ['IMPLY', 'NEGATION', 'EVENTUALLY', 'reach(object_1)', 'EVENTUALLY', 'reach(object_2)']
explanation: In this LTL 'IMPLY' checks the condition not reach object_1. If the condition is met you can still reach object_2. 
             'EVENTUALLY' allows both objects to be reached at any time step.

natural language: If you have reached object_1 and object_2 and not reached object_3 or object_4, then you reach object_5 eventually.
LTL:  ['IMPLY', 'AND', 'AND', 'reach(object_1)', 'reach(object_2)', 'NEGATION', 'OR', 'reach(object_3)', 'reach(object_4)', 'EVENTUALLY', 'reach(object_5)']
explanation: In this LTL 'IMPLY' checks the condition if you have already reached object_1 and object_2 and not object_3 and object_4. 
             If the condition is met then execute reaching object_5 otherwise do nothing. 'EVENTUALLY' allows the object_5 to be reached any time step.

natural language: For all time steps, until reach(object_1) and reach(object_2) is true, don't start reach(object_3).
LTL:  ['UNTIL', 'NEGATION', 'reach(object_3)', 'AND', 'reach(object_1)', 'reach(object_2)']
explanation: In this LTL the condition is not object_3 'UNTIL' object_1 and object_2, which enforces visit order. The robot should first visit object_1 and object_2 
             before visiting object_3.

natural language: Reach object3 after object_2 and object_2 after reaching to object_1.
LTL:  ['AND', 'EVENTUALLY', 'reach(object_3)', 'AND', 'EVENTUALLY', 'reach(object_2)', 'AND', 'UNTIL', 'NEGATION', 'reach(object_2)', 'reach(object_1)', 'UNTIL', 'NEGATION', 'reach(object_3)', 'reach(object_2)']
explanation: In this LTL conditions not object_2 'UNTIL' object_1 and not object_3 'UNTIL' object_2 enforce the visiting order. 
             'EVENTUALLY' allows objects to be reached at any time.

natural language: Reach object_2 subsequently visit object_1 and come back to object_2
LTL: ['AND',  'ALWAYS', 'EVENTUALLY', 'reach(object_2)', 'AND', 'EVENTUALLY', 'reach(object_1)', 'UNTIL', 'NEGATION', 'reach(object_1)', 'reach(object_2)']
explanation: In this LTL not object_1 'UNTIL' object_2 ensures object_2 is first visited. 'EVENTUALLY' allows objects to 
             be reached at any time step. 'ALWAYS' eventually object_2 sets up a loop allowing revisiting.

natural language: Reach object_1 next go to object_2 and last to object_3.  Revisit object_1.
LTL:  ['AND', 'ALWAYS', 'EVENTUALLY', 'reach(object_1)', 'AND', 'EVENTUALLY', 'reach(object_2)', 'AND', 'EVENTUALLY', 'reach(object_3)', 'AND', 'UNTIL', 'NEGATION', 'reach(object_2)', 'reach(object_1)', 'UNTIL', 'NEGATION', 'reach(object_3)', 'reach(object_2)']
explanation:  In this LTL not object_2 'UNTIL' object_1 and  not object_3 'UNTIL' object_2 ensures the visiting order is 1->2->3. 
              'EVENTUALLY' allows objects to be reached at any time step. 'ALWAYS' eventually object_1 sets up a loop allowing revisiting.

I will give you the list of objects in the environment, and the instruction involves reaching some objects, and avoiding some objects.

Your task is to transform the natural language instruction into a LTL description with pre-order format.

Here are some examples:

Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5 object_6, object_7, object_8]
    natural language instruction: I think object_3 maybe on object_2 or object_5 check it out.
Output:
    LTL: ['EVENTUALLY', 'AND', 'reach(object_3)', 'OR', 'reach(object_2)', 'reach(object_5)' ]

Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5 object_6, object_7, object_8]
    natural language instruction: Try going to object_3 near object_5 otherwise go to object_2
Output:
    LTL: ['OR', 'EVENTUALLY', 'AND', 'reach(object_3)', 'reach(object_5)', 'EVENTUALLY', 'reach(object_2)']

Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5 object_6, object_7, object_8]
    natural language instruction: If you are near object_4 go to object_5 otherwise go to object_1
Output:
    LTL: ['AND', 'IMPLY', 'reach(object_4)', 'EVENTUALLY', 'reach(object_5)', 'IMPLY', 'NEGATION', 'reach(object_4)', 'EVENTUALLY', 'reach(object_1)']

Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5 object_6, object_7, object_8]
    natural language instruction: Visit object_1, object_2 and object_8 in mentioned order, always keep away from object_4 and object_6.
Output:
    LTL: ['AND', 'EVENTUALLY', 'reach(object_8)', 'AND', 'EVENTUALLY', 'reach(object_2)', 'AND', 'UNTIL', 'NEGATION', 'reach(object_2)', 'reach(object_1)', 'AND', 'UNTIL', 'NEGATION', 'reach(object_8)', 'reach(object_2)', 'ALWAYS', 'NEGATION', 'OR', 'reach(object_4)', 'reach(object_6)']

Input:
    available environment elements: [object_1, object_2, object_3, object_4]
    natural language instruction: Every robot should reach object_3 every time they leave object_1. After object_2, the robot should visit object_4 at some point.

Output:
    LTL: ['AND', 'IMPLY', 'NEGATION', 'reach(object_1)', 'reach(object_3)', 'AND', 'EVENTUALLY', 'reach(object_4)', 'UNTIL', 'NEGATION', 'reach(object_4)', 'reach(object_2)']

Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5, object_6]
    natural language instruction: Robots are given a sequence of objects. They are expected to visit each object in order of appearance. Sequence: object_4, object_2, object_3, object_1, object_5 

Output:
    LTL: ['AND', 'EVENTUALLY', 'reach(object_5)', 'AND', 'EVENTUALLY', 'reach(object_1)', 'AND', 'EVENTUALLY', 'reach(object_3)', 'AND', 'EVENTUALLY', 'reach(object_2)', 'AND', 'UNTIL', 'NEGATION', 'reach(object_2)', 'reach(object_4)', 'AND', 'UNTIL', 'NEGATION', 'reach(object_3)', 'reach(object_2)', 'AND', 'UNTIL', 'NEGATION', 'reach(object_1)', 'reach(object_3), 'UNTIL', 'NEGATION', 'reach(object_5)', 'reach(object_1)']

Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5, object_6] 
    natural language instruction: Get the chicken from object_6 then and heat it in object_3, go clean yourself in object_2 and check if chicken is cooked in object 3.

Output:
    LTL: ['AND', 'ALWAYS', 'EVENTUALLY', 'reach(object_3)', 'AND', 'EVENTUALLY', 'reach(object_2)', 'AND', 'UNTIL', 'NEGATION', 'reach(object_3)', 'reach(object_6)', 'UNTIL', 'NEGATION', 'reach(object_2)', 'reach(object_3)']

Using the provided examples, transform the following natural language instruction into LTL specification:

Input:
    available environment elements: {env_elements}
    natural language instruction: {instruction}
    
The returned output should only contain the LTL formula in the above mentioned list format with no additional headers like 'LTL:'. Do not provide explaination.        
Output:
 """
    return PromptTemplate(template=template, input_variables=["env_elements", "instruction"])


def get_syntactic_check_prompt():
    template = """Please help transform natural language statements into linear temporal logic (LTL) descriptions.

The LTL operators are: NEGATION, IMPLY, AND, EQUAL, UNTIL, ALWAYS, EVENTUALLY, OR.

The LTL description should follow pre-order expression.

The available action is: reach(object_x).

Given below are examples of natural language statements their corresponding LTL descriptions and their explanation for your understanding:

natural language: Check object_2 and object_1.
LTL:  ['AND', 'reach(object_2)', 'reach(object_1)']
explanation: In this LTL 'AND' checks if both objects are visited at the current time. If not you reach sink state. 

natural language:  object_1 until object_2 is satisfied.
LTL:  ['UNTIL', 'reach(object_1)', 'reach(object_2)']
explanation: In this LTL 'UNTIL' checks if object_1 is true until object_2 is satisfied. If object_1 is false you reach sink state.

natural language: Reach object_2 and object_1.
LTL:  ['AND', 'EVENTUALLY', 'reach(object_2)', 'EVENTUALLY', 'reach(object_1)']
explanation: In this LTL 'AND' ensures both the objects are visited at current time step. 'EVENTUALLY' is added to relax the 
             condition of visiting them at the current time. Accomodating cases where objects are visited in future time. 

natural language: Reach object_2, object_3, object_1.
LTL:  ['AND', 'EVENTUALLY', 'reach(object_2)', 'AND', 'EVENTUALLY', 'reach(object_3)', 'EVENTUALLY', 'reach(object_1)']
explanation: In this LTL 'AND' ensures all objects are visited at the current time step. 'EVENTUALLY' relaxes the condition of 
             visiting them them at the current time. Accomodating cases where objects are visited in future time steps.

natural language: Go to object_1 and always avoid both object_2 and object_3.
LTL:  ['AND', 'EVENTUALLY', 'reach(object_1)', 'ALWAYS', 'AND', 'NEGATION', 'reach(object_2)', 'NEGATION', 'reach(object_3)']
explanation: In this LTL 'EVENTUALLY' allows object_1 to be reached at any time step. 'ALWAYS' ensures any violation of the 
             condition leads to a sink state. 'AND' is used to penalize visting either object_2 or object_3.

 natural language: Reach object_8, object_13, object_14, object_17, object_2, object_10, object_12, object_15, object_16, object_18, object_4, object_6, object_7, object_11 in the room.
LTL:  ['AND', 'EVENTUALLY', 'reach(object_8)', 'AND', 'EVENTUALLY', 'reach(object_13)', 'AND', 'EVENTUALLY', 'reach(object_14)', 'AND', 'EVENTUALLY', 'reach(object_17)', 'AND', 'EVENTUALLY', 'reach(object_2)', 'AND', 'EVENTUALLY', 'reach(object_10)', 'AND', 'EVENTUALLY', 'reach(object_12)', 'AND', 'EVENTUALLY', 'reach(object_15)', 'AND', 'EVENTUALLY', 'reach(object_16)', 'AND', 'EVENTUALLY', 'reach(object_18)', 'AND', 'EVENTUALLY', 'reach(object_4)', 'AND', 'EVENTUALLY', 'reach(object_6)', 'AND', 'EVENTUALLY', 'reach(object_7)', 'EVENTUALLY', 'reach(object_11)']
explanation: In this LTL 'AND' ensures all objects are visited at the current time step. 'EVENTUALLY' relaxes the condition of 
             visiting them them at the current time. Accomodating cases where objects are visited in future time steps.

natural language: Reach object_2 if you leave object_1.
LTL:  ['IMPLY', 'NEGATION', 'reach(object_1)', 'EVENTUALLY', 'reach(object_2)']
explanation: In this LTL 'IMPLY' checks the condition not object_1(leave object_1). If the condition is met, try to reach object_2. 
             'EVENTUALLY' indicates the robot can reach object_2 at any time step.

natural language: Reach object_2 only if you don't reach object_1.
LTL:  ['IMPLY', 'NEGATION', 'EVENTUALLY', 'reach(object_1)', 'EVENTUALLY', 'reach(object_2)']
explanation: In this LTL 'IMPLY' checks the condition not reach object_1. If the condition is met you can still reach object_2. 
             'EVENTUALLY' allows both objects to be reached at any time step.

natural language: If you have reached object_1 and object_2 and not reached object_3 or object_4, then you reach object_5 eventually.
LTL:  ['IMPLY', 'AND', 'AND', 'reach(object_1)', 'reach(object_2)', 'NEGATION', 'OR', 'reach(object_3)', 'reach(object_4)', 'EVENTUALLY', 'reach(object_5)']
explanation: In this LTL 'IMPLY' checks the condition if you have already reached object_1 and object_2 and not object_3 and object_4. 
             If the condition is met then execute reaching object_5 otherwise do nothing. 'EVENTUALLY' allows the object_5 to be reached any time step.

natural language: For all time steps, until reach(object_1) and reach(object_2) is true, don't start reach(object_3).
LTL:  ['AND', 'EVENTUALLY', 'reach(object_3)', 'UNTIL', 'NEGATION', 'reach(object_3)', 'AND', 'reach(object_1)', 'reach(object_2)']
explanation: In this LTL the condition not object_3 'UNTIL' object_1 and object_2, enforces visit order. The robot would have to first visit object_1 and object_2 
             then object_3. 'EVENTUALLY' allows object_3 to be reached at any time step.

natural language: Reach object3 after object_2 and object_2 after reaching to object_1.
LTL:  ['AND', 'EVENTUALLY', 'reach(object_3)', 'AND', 'EVENTUALLY', 'reach(object_2)', 'AND', 'UNTIL', 'NEGATION', 'reach(object_2)', 'reach(object_1)', 'UNTIL', 'NEGATION', 'reach(object_3)', 'reach(object_2)']
explanation: In this LTL conditions not object_2 'UNTIL' object_1 and not object_3 'UNTIL' object_2 enforce the visiting order. 
             'EVENTUALLY' allows objects to be reached at any time.

natural language: Reach object_2 subsequently visit object_1 and come back to object_2
LTL: ['AND',  'ALWAYS', 'EVENTUALLY', 'reach(object_2)', 'AND', 'EVENTUALLY', 'reach(object_1)', 'UNTIL', 'NEGATION', 'reach(object_1)', 'reach(object_2)']
explanation: In this LTL not object_1 'UNTIL' object_2 ensures object_2 is first visited. 'EVENTUALLY' allows objects to 
             be reached at any time step. 'ALWAYS' eventually object_2 sets up a loop allowing revisiting.

natural language: Reach object_1 next go to object_2 and last to object_3.  Revisit object_1.
LTL:  ['AND', 'ALWAYS', 'EVENTUALLY', 'reach(object_1)', 'AND', 'EVENTUALLY', 'reach(object_2)', 'AND', 'EVENTUALLY', 'reach(object_3)', 'AND', 'UNTIL', 'NEGATION', 'reach(object_2)', 'reach(object_1)', 'UNTIL', 'NEGATION', 'reach(object_3)', 'reach(object_2)']
explanation:  In this LTL not object_2 'UNTIL' object_1 and  not object_3 'UNTIL' object_2 ensures the visiting order is 1->2->3. 
              'EVENTUALLY' allows objects to be reached at any time step. 'ALWAYS' eventually object_1 sets up a loop allowing revisiting.

I will give you the list of objects in the environment, and the instruction involves reaching some objects, and avoiding some objects.

Your task is to transform the natural language instruction into a LTL description with pre-order format.

Here are some examples:

Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5 object_6, object_7, object_8]
    natural language instruction: I think object_3 maybe on object_2 or object_5 check it out.
Output:
    LTL: ['EVENTUALLY', 'AND', 'reach(object_3)', 'OR', 'reach(object_2)', 'reach(object_5)' ]

Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5 object_6, object_7, object_8]
    natural language instruction: Try going to object_3 near object_5 otherwise go to object_2
Output:
    LTL: ['OR', 'EVENTUALLY', 'AND', 'reach(object_3)', 'reach(object_5)', 'EVENTUALLY', 'reach(object_2)']

Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5 object_6, object_7, object_8]
    natural language instruction: If you are near object_4 go to object_5 otherwise go to object_1
Output:
    LTL: ['AND', 'IMPLY', 'reach(object_4)', 'EVENTUALLY', 'reach(object_5)', 'IMPLY', 'NEGATION', 'reach(object_4)', 'EVENTUALLY', 'reach(object_1)']

Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5 object_6, object_7, object_8]
    natural language instruction: Visit object_1, object_2 and object_8 in mentioned order, always keep away from object_4 and object_6.
Output:
    LTL: ['AND', 'EVENTUALLY', 'reach(object_8)', 'AND', 'EVENTUALLY', 'reach(object_2)', 'AND', 'UNTIL', 'NEGATION', 'reach(object_2)', 'reach(object_1)', 'AND', 'UNTIL', 'NEGATION', 'reach(object_8)', 'reach(object_2)', 'ALWAYS', 'NEGATION', 'OR', 'reach(object_4)', 'reach(object_6)']

Input:
    available environment elements: [object_1, object_2, object_3, object_4]
    natural language instruction: Every robot should reach object_3 every time they leave object_1. After object_2, the robot should visit object_4 at some point.

Output:
    LTL: ['AND', 'IMPLY', 'NEGATION', 'reach(object_1)', 'reach(object_3)', 'AND', 'EVENTUALLY', 'reach(object_4)', 'UNTIL', 'NEGATION', 'reach(object_4)', 'reach(object_2)']

Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5, object_6]
    natural language instruction: Robots are given a sequence of objects. They are expected to visit each object in order of appearance. Sequence: object_4, object_2, object_3, object_1, object_5 

Output:
    LTL: ['AND', 'EVENTUALLY', 'reach(object_5)', 'AND', 'EVENTUALLY', 'reach(object_1)', 'AND', 'EVENTUALLY', 'reach(object_3)', 'AND', 'EVENTUALLY', 'reach(object_2)', 'AND', 'UNTIL', 'NEGATION', 'reach(object_2)', 'reach(object_4)', 'AND', 'UNTIL', 'NEGATION', 'reach(object_3)', 'reach(object_2)', 'AND', 'UNTIL', 'NEGATION', 'reach(object_1)', 'reach(object_3), 'UNTIL', 'NEGATION', 'reach(object_5)', 'reach(object_1)']

Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5, object_6] 
    natural language instruction: Get the chicken from object_6 then and heat it in object_3, go clean yourself in object_2 and check if chicken is cooked in object 3.

Output:
    LTL: ['AND', 'ALWAYS', 'EVENTUALLY', 'reach(object_3)', 'AND', 'EVENTUALLY', 'reach(object_2)', 'AND', 'UNTIL', 'NEGATION', 'reach(object_3)', 'reach(object_6)', 'UNTIL', 'NEGATION', 'reach(object_2)', 'reach(object_3)']

Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5, object_6] 
    natural language instruction: Get the chicken from object_6 then and heat it in object_3, go clean yourself in object_2 and check if chicken is cooked in object 3.

Output:
    LTL: ['AND', 'ALWAYS', 'EVENTUALLY', 'reach(object_3)', 'AND', 'EVENTUALLY', 'reach(object_2)', 'AND', 'UNTIL', 'NEGATION', 'reach(object_3)', 'reach(object_6)', 'UNTIL', 'NEGATION', 'reach(object_2)', 'reach(object_3)']

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

Provide reasoning for why the generated LTL is wrong. 
Your suggestions must be limited LTL operations specified above. 
Restrict your answer to why the given LTL is wrong and your suggestions.

"""
    return PromptTemplate(template=template, input_variables=["instruction", "ap_dict", "formula", "prefix"])


def get_semantic_correction_prompt():
    template = """Please help transform natural language statements into linear temporal logic (LTL) descriptions.

The LTL operators are: NEGATION, IMPLY, AND, EQUAL, UNTIL, ALWAYS, EVENTUALLY, OR.

The LTL description should follow pre-order expression.

The available action is: reach(object_x).

Given below are examples of natural language statements their corresponding LTL descriptions and their explanation for your understanding:

natural language: Check object_2 and object_1.
LTL:  ['AND', 'reach(object_2)', 'reach(object_1)']
explanation: In this LTL 'AND' checks if both objects are visited at the current time. If not you reach sink state. 

natural language:  object_1 until object_2 is satisfied.
LTL:  ['UNTIL', 'reach(object_1)', 'reach(object_2)']
explanation: In this LTL 'UNTIL' checks if object_1 is true until object_2 is satisfied. If object_1 is false you reach sink state.

natural language: Reach object_2 and object_1.
LTL:  ['AND', 'EVENTUALLY', 'reach(object_2)', 'EVENTUALLY', 'reach(object_1)']
explanation: In this LTL 'AND' ensures both the objects are visited at current time step. 'EVENTUALLY' is added to relax the 
             condition of visiting them at the current time. Accomodating cases where objects are visited in future time. 

natural language: Reach object_2, object_3, object_1.
LTL:  ['AND', 'EVENTUALLY', 'reach(object_2)', 'AND', 'EVENTUALLY', 'reach(object_3)', 'EVENTUALLY', 'reach(object_1)']
explanation: In this LTL 'AND' ensures all objects are visited at the current time step. 'EVENTUALLY' relaxes the condition of 
             visiting them them at the current time. Accomodating cases where objects are visited in future time steps.

natural language: Go to object_1 and always avoid both object_2 and object_3.
LTL:  ['AND', 'EVENTUALLY', 'reach(object_1)', 'ALWAYS', 'AND', 'NEGATION', 'reach(object_2)', 'NEGATION', 'reach(object_3)']
explanation: In this LTL 'EVENTUALLY' allows object_1 to be reached at any time step. 'ALWAYS' ensures any violation of the 
             condition leads to a sink state. 'AND' is used to penalize visting either object_2 or object_3.

 natural language: Reach object_8, object_13, object_14, object_17, object_2, object_10, object_12, object_15, object_16, object_18, object_4, object_6, object_7, object_11 in the room.
LTL:  ['AND', 'EVENTUALLY', 'reach(object_8)', 'AND', 'EVENTUALLY', 'reach(object_13)', 'AND', 'EVENTUALLY', 'reach(object_14)', 'AND', 'EVENTUALLY', 'reach(object_17)', 'AND', 'EVENTUALLY', 'reach(object_2)', 'AND', 'EVENTUALLY', 'reach(object_10)', 'AND', 'EVENTUALLY', 'reach(object_12)', 'AND', 'EVENTUALLY', 'reach(object_15)', 'AND', 'EVENTUALLY', 'reach(object_16)', 'AND', 'EVENTUALLY', 'reach(object_18)', 'AND', 'EVENTUALLY', 'reach(object_4)', 'AND', 'EVENTUALLY', 'reach(object_6)', 'AND', 'EVENTUALLY', 'reach(object_7)', 'EVENTUALLY', 'reach(object_11)']
explanation: In this LTL 'AND' ensures all objects are visited at the current time step. 'EVENTUALLY' relaxes the condition of 
             visiting them them at the current time. Accomodating cases where objects are visited in future time steps.

natural language: Reach object_2 if you leave object_1.
LTL:  ['IMPLY', 'NEGATION', 'reach(object_1)', 'EVENTUALLY', 'reach(object_2)']
explanation: In this LTL 'IMPLY' checks the condition not object_1(leave object_1). If the condition is met, try to reach object_2. 
             'EVENTUALLY' indicates the robot can reach object_2 at any time step.

natural language: Reach object_2 only if you don't reach object_1.
LTL:  ['IMPLY', 'NEGATION', 'EVENTUALLY', 'reach(object_1)', 'EVENTUALLY', 'reach(object_2)']
explanation: In this LTL 'IMPLY' checks the condition not reach object_1. If the condition is met you can still reach object_2. 
             'EVENTUALLY' allows both objects to be reached at any time step.

natural language: If you have reached object_1 and object_2 and not reached object_3 or object_4, then you reach object_5 eventually.
LTL:  ['IMPLY', 'AND', 'AND', 'reach(object_1)', 'reach(object_2)', 'NEGATION', 'OR', 'reach(object_3)', 'reach(object_4)', 'EVENTUALLY', 'reach(object_5)']
explanation: In this LTL 'IMPLY' checks the condition if you have already reached object_1 and object_2 and not object_3 and object_4. 
             If the condition is met then execute reaching object_5 otherwise do nothing. 'EVENTUALLY' allows the object_5 to be reached any time step.

natural language: For all time steps, until reach(object_1) and reach(object_2) is true, don't start reach(object_3).
LTL:  ['AND', 'EVENTUALLY', 'reach(object_3)', 'UNTIL', 'NEGATION', 'reach(object_3)', 'AND', 'reach(object_1)', 'reach(object_2)']
explanation: In this LTL the condition not object_3 'UNTIL' object_1 and object_2, enforces visit order. The robot would have to first visit object_1 and object_2 
             then object_3. 'EVENTUALLY' allows object_3 to be reached at any time step.

natural language: Reach object3 after object_2 and object_2 after reaching to object_1.
LTL:  ['AND', 'EVENTUALLY', 'reach(object_3)', 'AND', 'EVENTUALLY', 'reach(object_2)', 'AND', 'UNTIL', 'NEGATION', 'reach(object_2)', 'reach(object_1)', 'UNTIL', 'NEGATION', 'reach(object_3)', 'reach(object_2)']
explanation: In this LTL conditions not object_2 'UNTIL' object_1 and not object_3 'UNTIL' object_2 enforce the visiting order. 
             'EVENTUALLY' allows objects to be reached at any time.

natural language: Reach object_2 subsequently visit object_1 and come back to object_2
LTL: ['AND',  'ALWAYS', 'EVENTUALLY', 'reach(object_2)', 'AND', 'EVENTUALLY', 'reach(object_1)', 'UNTIL', 'NEGATION', 'reach(object_1)', 'reach(object_2)']
explanation: In this LTL not object_1 'UNTIL' object_2 ensures object_2 is first visited. 'EVENTUALLY' allows objects to 
             be reached at any time step. 'ALWAYS' eventually object_2 sets up a loop allowing revisiting.

natural language: Reach object_1 next go to object_2 and last to object_3.  Revisit object_1.
LTL:  ['AND', 'ALWAYS', 'EVENTUALLY', 'reach(object_1)', 'AND', 'EVENTUALLY', 'reach(object_2)', 'AND', 'EVENTUALLY', 'reach(object_3)', 'AND', 'UNTIL', 'NEGATION', 'reach(object_2)', 'reach(object_1)', 'UNTIL', 'NEGATION', 'reach(object_3)', 'reach(object_2)']
explanation:  In this LTL not object_2 'UNTIL' object_1 and  not object_3 'UNTIL' object_2 ensures the visiting order is 1->2->3. 
              'EVENTUALLY' allows objects to be reached at any time step. 'ALWAYS' eventually object_1 sets up a loop allowing revisiting.

I will give you the list of objects in the environment, and the instruction involves reaching some objects, and avoiding some objects.

Your task is to transform the natural language instruction into a LTL description with pre-order format.

Here are some examples:

Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5 object_6, object_7, object_8]
    natural language instruction: I think object_3 maybe on object_2 or object_5 check it out.
Output:
    LTL: ['EVENTUALLY', 'AND', 'reach(object_3)', 'OR', 'reach(object_2)', 'reach(object_5)' ]

Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5 object_6, object_7, object_8]
    natural language instruction: Try going to object_3 near object_5 otherwise go to object_2
Output:
    LTL: ['OR', 'EVENTUALLY', 'AND', 'reach(object_3)', 'reach(object_5)', 'EVENTUALLY', 'reach(object_2)']

Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5 object_6, object_7, object_8]
    natural language instruction: If you are near object_4 go to object_5 otherwise go to object_1
Output:
    LTL: ['AND', 'IMPLY', 'reach(object_4)', 'EVENTUALLY', 'reach(object_5)', 'IMPLY', 'NEGATION', 'reach(object_4)', 'EVENTUALLY', 'reach(object_1)']

Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5 object_6, object_7, object_8]
    natural language instruction: Visit object_1, object_2 and object_8 in mentioned order, always keep away from object_4 and object_6.
Output:
    LTL: ['AND', 'EVENTUALLY', 'reach(object_8)', 'AND', 'EVENTUALLY', 'reach(object_2)', 'AND', 'UNTIL', 'NEGATION', 'reach(object_2)', 'reach(object_1)', 'AND', 'UNTIL', 'NEGATION', 'reach(object_8)', 'reach(object_2)', 'ALWAYS', 'NEGATION', 'OR', 'reach(object_4)', 'reach(object_6)']

Input:
    available environment elements: [object_1, object_2, object_3, object_4]
    natural language instruction: Every robot should reach object_3 every time they leave object_1. After object_2, the robot should visit object_4 at some point.

Output:
    LTL: ['AND', 'IMPLY', 'NEGATION', 'reach(object_1)', 'reach(object_3)', 'IMPLY', 'reach(object_2)', 'reach(object_4)']

Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5, object_6]
    natural language instruction: Robots are given a sequence of objects. They are expected to visit each object in order of appearance. Sequence: object_4, object_2, object_3, object_1, object_5 

Output:
    LTL: ['AND', 'EVENTUALLY', 'reach(object_5)', 'AND', 'EVENTUALLY', 'reach(object_1)', 'AND', 'EVENTUALLY', 'reach(object_3)', 'AND', 'EVENTUALLY', 'reach(object_2)', 'AND', 'UNTIL', 'NEGATION', 'reach(object_2)', 'reach(object_4)', 'AND', 'UNTIL', 'NEGATION', 'reach(object_3)', 'reach(object_2)', 'AND', 'UNTIL', 'NEGATION', 'reach(object_1)', 'reach(object_3), 'UNTIL', 'NEGATION', 'reach(object_5)', 'reach(object_1)']

Input:
    available environment elements: [object_1, object_2, object_3, object_4, object_5, object_6] 
    natural language instruction: Get the chicken from object_6 then and heat it in object_3, go clean yourself in object_2 and check if chicken is cooked in object 3.

Output:
    LTL: ['AND', 'ALWAYS', 'EVENTUALLY', 'reach(object_3)', 'AND', 'EVENTUALLY', 'reach(object_2)', 'AND', 'UNTIL', 'NEGATION', 'reach(object_3)', 'reach(object_6)', 'UNTIL', 'NEGATION', 'reach(object_2)', 'reach(object_3)']

    
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
