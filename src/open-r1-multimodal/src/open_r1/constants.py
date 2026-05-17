from open_r1.rewards import *
# REWARD MAPING
reward_funcs_registry = {
    "iou":iou_reward,
    "format": format_reward,
}

QWEN2_SYS = (
    "You are a helpful assistant. "
)

system_prompt_registry = {
    "default": QWEN2_SYS,
    "qwen": QWEN2_SYS,
}

question_template_registry= {
    "default": "{question}",


     "cot":"""
Your task is to locate the object matching "{question}" in the Target Image.
Data provided:
1. Reference Image {ref_bboxes}.
2. Target Image: The image to locate.
Think through the reasoning process in your mind， induce the visual rule {check_prompt} in  the reference image, apply this rule to locate the corresponding object in the Target Image.
Finally,  provide the bounding box  and a 1-2 word  noun phrase for the object in the target image. 
Output strictly in the following format: <think>[Your step-by-step analysis and reasoning]</think>  {check_answer} <bbox>[x3, y3, x4, y4]</bbox> <answer>concise noun phrase for target object</answer>
     """,

    "reasonseg_cot": """
Your task is to locate the object matching "{question}" in the Target Image.
Data provided:
1. Reference Image: N/A.
2. Target Image: The image to locate.
Think through the reasoning process in your mind， induce the visual rule, apply this rule to locate the corresponding object in the Target Image.
Finally,  provide the bounding box  and a 1-2 word  noun phrase for the object in the target image. 
Output strictly in the following format: <think>[Your step-by-step analysis and reasoning]</think>  <rule>Visual rule </rule> <bbox>[x3, y3, x4, y4]</bbox> <answer>concise noun phrase for target object</answer>
 """,

}

answer_template_registry = {
    "default": "{answer}",
    "r1v": "<answer> {answer} </answer>"
}
