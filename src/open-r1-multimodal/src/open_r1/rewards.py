import re
import os
import numpy as np
import warnings
from datetime import datetime
def log(content, sol, other_info, reward, tag=None):
    log_dir = os.getenv("LOG_DIR", None)
    os.makedirs(log_dir, exist_ok=True)
    if log_dir is None:
        warnings.warn("LOG_DIR is not set, log will not be saved")
        return
    log_path = os.path.join(log_dir, f"{tag}.log")
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    with open(log_path, "a") as f:
        try:
            f.write(f"------------- {current_time} {tag} reward: {reward} -------------\n")
            f.write(f"Content: {content}\n")
            f.write(f"Solution: {sol}\n")
            if other_info is not None:
                for k, v in other_info.items():
                    f.write(f"{k}: {v}\n")
        except:
            f.write("writeing error")



def parse_custom_format(content: str,bbox_name="bbox"):
    """Parse custom formatted string to extract points, labels and bbox.

    Supported format:
    <points>[[x1,y1],[x2,y2],...]</points>
    <labels>[1,0,...]</labels>
    <bbox>[x1,y1,x2,y2]</bbox>

    Args:
        content: String containing the formatted data

    Returns:
        Tuple of (points, labels, bbox) as numpy arrays
    """
    bbox_pattern = rf"<{bbox_name}>\s*(\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\])\s*</{bbox_name}>"

    bbox_matches = re.findall(bbox_pattern, content)

    try:
        bboxes = None
        for bbox_str in bbox_matches:
            bbox = np.array(eval(bbox_str))
            if len(bbox.shape) == 1 and bbox.shape[0] == 4:
                bboxes = bbox

        return bboxes

    except Exception as e:
        print("Error parsing content:", e)
        return None

def cal_iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2] - 1, box2[2] - 1)
    inter_y2 = min(box1[3] - 1, box2[3] - 1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        inter = 0
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    return float(inter) / union
def iou_reward(completions, solution, **kwargs):
    """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""



    def resize_bbox(bbox, input_height, input_width, image_height, image_width):
        bbox[0] = bbox[0] / input_width * image_width
        bbox[1] = bbox[1] / input_height * image_height
        bbox[2] = bbox[2] / input_width * image_width
        bbox[3] = bbox[3] / input_height * image_height
        return bbox

    contents = [completion[0]["content"] for completion in completions]
    # if random.random() < 0.01:
    #     print(contents)
    rewards = []

    for i, (content, sol,gt_miss_ref_box) in enumerate(zip(contents, solution,kwargs['miss_ref_bbox'])):
        reward = 0.0
        # Try symbolic verification first
        try:
            bbox = parse_custom_format(content)
            verifybbox = parse_custom_format(content,"check")

            if len(gt_miss_ref_box) == 0:
                reward =cal_iou(bbox, sol)**2#**2
            else:
                reward =cal_iou(bbox, sol)*cal_iou(verifybbox, gt_miss_ref_box)
        except Exception as e:
            print("IOU Reward Cal Error",e,bbox)  # Continue to next verification method if this fails

        rewards.append(reward)
    return rewards

def format_reward(completions, **kwargs):
    # def format_reward(completions, **kwargs):
    """Calculate reward based on format compliance.

    Args:
        completions: List of model completion dictionaries

    Returns:
        List of reward scores (1.0 for valid format, 0.0 otherwise)
    """
    import re
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for i, (content,gt_miss_ref_box) in enumerate(zip(contents, kwargs['miss_ref_bbox'])):
        pattern = r"<think>.*?</think>\s*<rule>.*?</rule>\s*<bbox>.*?</bbox>\s*<answer>.*?</answer>"
        if len(gt_miss_ref_box) > 0:
            pattern = r"<think>.*?</think>\s*<rule>.*?</rule>\s*<check>\s*(\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\])\s*</check>\s*<bbox>.*?</bbox>\s*<answer>.*?</answer>"
        match = re.search(pattern, content, re.DOTALL)
        if match is not None:
            rewards.append(1.0)
        else:
            rewards.append(0)
    return rewards
