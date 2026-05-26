import copy
import os
import time


from transformers import AutoProcessor
from open_r1.mymodels.conceptr1 import  ConceptSegR1ForConditionalGeneration_qwen2p5
import re
import torch
import json
from tqdm import tqdm
import os
import argparse
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
from PIL import Image
from src.eval.utils import AverageMeter, Summary, intersectionAndUnionGPU
from open_r1.constants import system_prompt_registry, question_template_registry
import cv2
import glob
from qwen_vl_utils import smart_resize
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
from PIL import Image as PILImage
from open_r1.mydataset import ConceptSegDataset
from open_r1.rewards import parse_custom_format
from typing import List
import random
random.seed(42)
import torch.nn.functional as F
from metrics import cal_mae,cal_sm,cal_wfm, cal_dice, cal_iou_f,cal_iou_b,cal_ber

from open_r1.trainer import VLMGRPOTrainer
def get_mask_from_json(json_path, img):
    """
    Read polygon annotation JSON file and generate mask.

    Returns:
        mask (np.uint8): 0 for background, 1 for target, 255 for ignore regions.
        comments: annotation comments
        is_sentence: whether the annotation is sentence-level
    """
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]
    comments = anno["text"]
    is_sentence = anno["is_sentence"]

    height, width = img.shape[:2]

    # Sort polygons by area
    area_list = []
    valid_poly_list = []
    for i in inform:
        label_id = i["label"]
        points = i["points"]
        if "flag" == label_id.lower():  # Deprecated, meaningless annotations
            continue

        tmp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        tmp_area = tmp_mask.sum()

        area_list.append(tmp_area)
        valid_poly_list.append(i)

    sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    sort_inform = [valid_poly_list[s] for s in sort_index]

    # Generate GT mask
    mask = np.zeros((height, width), dtype=np.uint8)
    for i in sort_inform:
        label_id = i["label"]
        points = i["points"]

        if "ignore" in label_id.lower():
            label_value = 255  # Ignored in evaluation
        else:
            label_value = 1  # Foreground

        cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    return mask, comments, is_sentence


def save_mask_visualization(ref_pil_image,pil_image, pred_mask, gt_mask, save_path, box):
    """
    Save as two separate images: original and prediction overlay.
    No matplotlib dependency, only PIL and torch.
    """
    if isinstance(pil_image,List):
        pil_image = pil_image[-1]
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = max(0, x1), max(0, y1), max(0, x2), max(0, y2)
    img_w, img_h = pil_image.size
    x2, y2 = min(x2, img_w - 1), min(y2, img_h - 1)
    box = (x1, y1, x2, y2)

    image = TF.to_tensor(pil_image) * 255
    image = image.to(torch.uint8)

    size = pil_image.size[::-1]
    pred_mask = TF.resize(pred_mask.unsqueeze(0).float(), size, interpolation=Image.NEAREST)[0].int()
    mask = TF.resize(gt_mask.unsqueeze(0).float(), size, interpolation=Image.NEAREST)[0].int()
    ref_pil_image = ref_pil_image.resize(pil_image.size)
    def overlay_mask(image, mask, color):
        color_layer = torch.zeros_like(image)
        if color == "green":
            color_layer[1] = 255
        elif color == "red":
            color_layer[0] = 255
        mask = mask.bool().unsqueeze(0)
        overlay = image * 0.6 + torch.where(mask, color_layer * 0.4, torch.zeros_like(image))
        return overlay.clamp(0, 255).byte()

    pred_overlay = overlay_mask(image, pred_mask, "red")
    img_ori = TF.to_pil_image(image)
    img_pred = TF.to_pil_image(pred_overlay)
    mask = Image.fromarray(gt_mask.numpy()*255).resize(size)
    draw = ImageDraw.Draw(img_pred)
    if y2 >= y1 and x2 >= x1:
        draw.rectangle(box, outline="green", width=3)
    final_image = Image.new("RGB",(size[0]*4,size[1]))
    final_image.paste(ref_pil_image,(0*size[0],0))
    final_image.paste(img_ori,(1*size[0],0))
    final_image.paste(img_pred,(2*size[0],0))
    final_image.paste(mask,(3*size[0],0))
    final_image.save(save_path)
    return final_image
class RefCOCOEvaluator:
    """Evaluator class for ReasonSeg dataset"""
    def __init__(self, args):
        self.args = args
        self.dtype = torch.bfloat16
        self.model = ConceptSegR1ForConditionalGeneration_qwen2p5.from_pretrained(
            args.model_path, torch_dtype=self.dtype, attn_implementation="flash_attention_2"
        ).cuda()
        self.processor = AutoProcessor.from_pretrained(args.model_path)
        self.model.init_sam()

    @torch.no_grad()
    def evaluate_single(self, input_data):

        gts,preds = [],[]
        # new_input_data = []
        # for cur_data in input_data:
        #     vis_dir = f'{self.args.model_path}/{args.save_name}/{cur_data["task_name"]}/visualization'
        #     os.makedirs(vis_dir, exist_ok=True)
        #     name = cur_data['mask_path'].split("/")[-1]
        #     save_path = os.path.join(vis_dir, f"{name[:-4]}_*.png")
        #     if len(glob.glob(save_path)) == 0:
        #         new_input_data.append(cur_data)
        # print(len(new_input_data),len(input_data))
        # input_data = new_input_data

        input_data,sam_gts, sam_preds   = self.pre_process_by_sam3(input_data)
        if len(input_data)==0:
            return sam_gts, sam_preds
        gts.extend(sam_gts)
        preds.extend(sam_preds)
        messages = [data['prompt'] for data in input_data]
        image_inputs = [data["image"]  for data in input_data]# * len(texts)
        sam_img = [data["sam_image"]  for data in input_data]
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]

        inputs = self.processor(text=texts, images=image_inputs, padding=True, return_tensors="pt").to(
            device="cuda", dtype=self.dtype
        )

        llm_out = self.model.generate(max_length=2048+768, use_cache=True, do_sample=False, **inputs)
        prompt_length =inputs['input_ids'].shape[1]
        completion_text = self.processor.batch_decode(llm_out[:,prompt_length:], skip_special_tokens=True)
        out_text = [re.search(r'<answer>(.*?)</answer>', text) for text in completion_text]
        sam_text_prompt = [text.group(1).strip() if text is not None else "" for text in out_text]

        new_attention_mask = torch.ones_like(llm_out, dtype=torch.int64)
        pos = torch.where(llm_out == self.processor.tokenizer.pad_token_id)
        new_attention_mask[pos] = 0
        inputs.update({"input_ids": llm_out, "attention_mask": new_attention_mask})
        inputs.update({"sam_images": sam_img})

        inputs.update({"sam_text_prompt":sam_text_prompt})
        inputs.update({"prompt_length":prompt_length})


        _, low_res_masks = self.model(output_hidden_states=True, use_learnable_query=True, **inputs)


        completion_text = self.processor.batch_decode(llm_out[:,prompt_length:], skip_special_tokens=False)
        for index,(data,pred_mask,out_text) in enumerate(zip(input_data,low_res_masks,completion_text)):
            gt,pred = self.post_process( data, [pred_mask], [out_text])
            gts.append(gt.squeeze(0))
            preds.append(pred.squeeze(0))
        return  gts,preds


    def pre_process_by_sam3(self,input_data):

        processor = self.model.sam
        new_data = []
        sam_gts, sam_preds = [],[]
        for data in input_data:
            image = data["image"][-1]
            prompt = data["question"]
            prompt = re.sub(r'\([^)]*\)', '', prompt).strip()
            length = len(prompt.split())
            confidence_limit = 0.1 * (pow(2, length - 1))
            if confidence_limit>=1:
                new_data.append(data)
                continue
            inference_state = processor.set_image(image)
            output = processor.set_text_prompt(state=inference_state, prompt=prompt)
            confidence = output["presence_score"].item()
            if confidence < confidence_limit:
                new_data.append(data)
                continue
            # Get the masks, bounding boxes, and scores
            gt_masks = data["mask"] #
            gt_masks = torch.tensor(gt_masks>0).int()

            pred_masks = F.interpolate(output['semantic_seg'], gt_masks.shape, mode="bilinear", align_corners=False)
            pred_masks = (pred_masks>0).int()[0, 0]

            completion_text = f"sam3 mode. presence_score:{output['presence_score'].item()} instance_score:{output['scores'].tolist()}"
            sam_gts.append(gt_masks)
            sam_preds.append(pred_masks)

        return new_data,sam_gts, sam_preds
    def post_process(self,input_data,low_res_masks,completion_text):
        pred_masks = [self.model.postprocess_masks(mask, orig_hw=input_data["mask"].shape[-2:]) for mask in
                      low_res_masks]
        pred_masks = torch.cat(pred_masks, dim=0)
        pred_masks = (pred_masks[:, 0] > 0).int()
        masks_list = input_data["mask"].int().unsqueeze(0).repeat(pred_masks.size(0), 1, 1)
        intersection, union, acc_iou = 0.0, 0.0, 0.0

        for index,(cur_data, mask_i, output_i) in enumerate(zip([input_data], masks_list, pred_masks)):

            pred_box = parse_custom_format(completion_text[index])
            if pred_box is None:
                pred_box = [0,0,0,0]
            intersection_i,union_i,iou = self.save_image(input_data,output_i,mask_i,pred_box,completion_text[index])

            intersection += intersection_i
            union += union_i
            acc_iou += iou
            acc_iou[union_i == 0] += 1.0  # Perfect score if no object present

        return masks_list, pred_masks
    def save_image(self,cur_data,pred_mask,gt_mask,pred_box=[0,0,0,0],completion_text="sam3 mode w/o text output"):
        size = cur_data["size"]
        task_name = cur_data["task_name"]
        path = cur_data["image_path"][-1]

        intersection_i, union_i, _ = intersectionAndUnionGPU(
            pred_mask.contiguous().clone(), gt_mask.contiguous().cuda(), 2, ignore_index=255
        )
        iou = intersection_i / (union_i + 1e-5)
        mask_iou = iou[1].cpu().item()

        mask_dir = f'{self.args.model_path}/{args.save_name}/{cur_data["task_name"]}'
        name = cur_data['mask_path'].split("/")[-1]
        mask_file_path = os.path.join(mask_dir, f"{name[:-4]}.png")
        os.makedirs(mask_dir, exist_ok=True)
        save_mask = PILImage.fromarray(pred_mask.cpu().numpy().astype(np.uint8) * 255)  # .show()
        save_mask = save_mask.resize(size)
        save_mask.save(mask_file_path)

        vis_dir = f'{self.args.model_path}/{args.save_name}/{task_name}/visualization'
        os.makedirs(vis_dir, exist_ok=True)
        save_path = os.path.join(vis_dir, f"{name[:-4]}_{mask_iou:.4f}.png")
        save_mask_visualization(cur_data["image"][0], cur_data["image"][-1], pred_mask.cpu(), gt_mask.cpu(),
                                save_path, pred_box)
        cur_data["sam_image"] = ""
        cur_data["image"] = ""
        cur_data["mask"] = ""
        cur_data["image_grid_thw"] = ""
        cur_data["out_all_text"] = completion_text

        with open(os.path.join(vis_dir, f"{name[:-4]}_{mask_iou:.4f}.txt"), "w") as f:
            json.dump(cur_data, f, indent=2)
        return intersection_i, union_i,iou
def main(args):
    os.makedirs(f"{args.model_path}/{args.save_name}", exist_ok=True)

    evaluator = RefCOCOEvaluator(args)
    dataset = ConceptSegDataset(
        argparse.Namespace(train_sample_size=1000, min_pixels=4, max_pixels=360000, question_template=args.template,
        data_file_paths= args.data_files,
        image_folders = args.image_folders),
        mode="test", dataset_names=args.dataset_names,
    )


    assert args.batch_size == 1, f"batch_size must be 1, got {args.batch_size}"
    dataloader = DataLoader(dataset, args.batch_size, False, collate_fn=lambda batch: list(batch))

    bar = tqdm(dataloader)
    total,mllm_total = 0,0
    mae,sm, wfm, m_dice, iou_f, iou_b, ber=  cal_mae(), cal_sm(), cal_wfm(), cal_dice(), cal_iou_f(), cal_iou_b(), cal_ber()

    for batch_data in bar:
        # assert len(batch_data) == 1, "Only batch_size=1 is supported"
        gts,preds  = evaluator.evaluate_single(batch_data)
        for res,gt in zip(preds,gts):
            try:
                res,gt = res.cpu().numpy(),gt.cpu().numpy()
                mae.update(res, gt)
                sm.update(res, gt)
                wfm.update(res, gt)
                m_dice.update(res, gt)
                iou_f.update(res, gt)
                iou_b.update(res, gt)
                ber.update(res, gt)
            except Exception as e:
                print(e)
                print(res.shape, gt.shape)
                raise  Exception("error")
        log = (f" datasetname:{args.dataset_names} "
               f" MAE:{mae.show():.4f} ber:{ber.show():.4f}  "
               f" wfm:{wfm.show():.4f} sm:{sm.show():.4f}  "
               f" miou:{(iou_f.show() + iou_b.show()) / 2:.4f} m_dice:{m_dice.show():.4f}")
        bar.desc =log

    cur_time = time.strftime("%Y-%m-%d %H:%M:%S")
    log ="\n" +cur_time+log
    print("-"*10,args.dataset_names,"-"*10)
    print(log)
    with open(f"result.log", "a") as f:
        f.write(log)
    print(f"model path is: {args.model_path}")
    print("-"*25)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Localization Evaluation Script")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--data_files", type=str, required=True, default=None, help="Path to data files")
    parser.add_argument("--image_folders", type=str, required=True, default=None, help="Path to image folders")
    parser.add_argument("--save_name", type=str, required=False,default="evalution_del", help="Model path")
    parser.add_argument("--dataset_names", type=str, required=True, help="dataset_names")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--sample_num", type=int, default=-1, help="Number of samples (debugging)")
    parser.add_argument("--coord_norm_type", type=str, default="qwen2p5vl", choices=["qwen2vl", "qwen2p5vl"], help="Coordinate normalization type")
    parser.add_argument("--image_dir", type=str, default="./datasets", help="Path to reasonseg dir")
    parser.add_argument("--template", type=str, help="Enable chain-of-thought prompt",default="cot")
    parser.add_argument("--vis", action="store_true", help="Visualize segmentation results")
    args = parser.parse_args()
    print("args:", args)
    main(args)
