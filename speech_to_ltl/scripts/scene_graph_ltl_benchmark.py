import os
import time

from langchain.chains import LLMChain
from tqdm import tqdm

from model import llm_init
from yolo_parser import load_yolo_ids, get_yolo_id_instruction
from yolo_prompt import get_yolo_translate_prompt

openai_key = "Enter API Key..."
instructions = dict(
    tier1=[
        "Oh today was such a long day. I’m looking for a object_2 to sit on.",
        "I think today is going to rain! Can you take me to my object_3?",
        "I need to do some homework, can you find my object_5 please?",
        "I’m looking for my object_4, where is he?",
        "Oh no! I forgot to water my object_1! Take me to it!",
    ],
    tier2=[
        "Can you check up on the object_1 and the object_3 please.",
        "I just watered the object_1 so stay away from it. Can you get my object_5 please and visit that object_4.",
        "Take me to a object_2 but avoid that object_4. I don’t want to say hi.",
        "Go to my object_5 but avoid any object_2, object_1, or people.",
        "Check up on the object_1, object_5, and object_3 all while avoiding object_2.",
    ],
    tier3=[
        "I need to water the object_1 but the water is in my object_5. Go to the object_5 and then go to the object_1.",
        "I want you to entertain my kids. It would be super cool if you did a lap around the house. Take them to that object_4 in the corner then to my object_5 then to my object_3 then to the object_1 and at this point they are tired so take them to a object_2.",
        "I wanna see how fast you can go back and forth. I want you to go to the object_3 then to my object_5 and then back to the object_3 again.",
        "Go to the object_1 then to the object_4 and return to the object_1.",
        "Hey, I’m in a hurry for school! Go get the object_3 then bring it to the object_5 and bring the object_5 to me (object_4). Try to avoid the object_2 along the way.",
    ],
)
verbose = False
PROJ_DIR = os.path.join(os.path.dirname(__file__), "../../")
PROJ_DIR = os.path.realpath(PROJ_DIR)

llm = llm_init(gpt_api=openai_key)
yolo_ids = load_yolo_ids(f"{PROJ_DIR}/SSMI/SSMI-Mapping/params/yolo_color_id.yaml")

output_dir = "output_cur_ltl"
os.makedirs(output_dir, exist_ok=True)
results = []
yolo_translate_chain = LLMChain(llm=llm, prompt=get_yolo_translate_prompt(), verbose=verbose)
for tier, tier_instructions in instructions.items():
    with open(os.path.join(output_dir, f"results_{tier}.txt"), "w") as file:
        for i, instruction in tqdm(enumerate(tier_instructions), ncols=80, total=len(tier_instructions)):
            env_elements_str, yolo_id_instruction = get_yolo_id_instruction(instruction, yolo_ids)
            ltl = yolo_translate_chain({"env_elements": env_elements_str, "instruction": yolo_id_instruction})["text"]
            results.append((instruction, ltl))

            file.write("====================\n")
            file.write("instruction: " + instruction + "\n")
            file.write("ltl: " + ltl + "\n")
            file.write("\n\n")

            time.sleep(1)
