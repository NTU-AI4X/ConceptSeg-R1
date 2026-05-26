
from transformers import AutoProcessor
from open_r1.mymodels.conceptr1 import  ConceptSegR1ForConditionalGeneration_qwen2p5
import re
import torch
import json
from tqdm import tqdm
import os
import argparse
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from src.eval.utils import AverageMeter, Summary,intersectionAndUnionGPU
from open_r1.constants import system_prompt_registry, question_template_registry
import cv2
from datasets import load_from_disk, load_dataset
import random
random.seed(102)

class ReasonSegVal(Dataset):
    def __init__(self,  if_reasonseg_val=False):
        if if_reasonseg_val:
            self.datas  = load_dataset("Ricky06662/ReasonSeg_val", split='test')
            self.question_template = question_template_registry["reasonseg_cot"]
        else:
            self.datas  = load_dataset("Ricky06662/ReasonSeg_test", split='test')
            self.question_template = question_template_registry["reasonseg_cot"]

        self.system_template = system_prompt_registry["default"]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        image = data["image"]
        H,W  = 600,600
        llm_image =  image.resize(( H,W))
        sam_image = image.resize((1008,1008))
        problems = [data["text"]]
        message = [
            {"role": "system", "content": self.system_template},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": self.question_template.format(question=problems[0])}
            ]},
        ]

        masks = torch.tensor(data["mask"]).float()[None]

        return {
            "size": image.size,
            "task_name":"reasonseg",
            "image": llm_image,
            "messages": [message],
            "sam_image": sam_image,
            "mask": masks[0],
            "problems": problems,
            "box": [],
            "id":data['ann_id'],
        }
class ReasoningSegEvaluator:
    """Evaluator class for ReasonSeg dataset"""
    def __init__(self, args):
        self.args = args
        self.dtype = torch.bfloat16
        self.model = ConceptSegR1ForConditionalGeneration_qwen2p5.from_pretrained(
            args.model_path, torch_dtype=self.dtype, attn_implementation="flash_attention_2"
        ).cuda()
        self.model.init_sam()

        self.processor = AutoProcessor.from_pretrained(args.model_path)

    def postprocess_masks(self, masks, orig_hw):
        # 假设输入是 (C, H, W)

        masks_np = masks[0, 0].cpu().numpy()   # H, W, C
        resized = cv2.resize(masks_np, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_CUBIC)

        return torch.from_numpy(resized).to(masks.device)[None,None]
    @staticmethod
    def remove_last_think_part(text):
        """Remove the last <think>...</think> block from the string."""
        matches = list(re.finditer(r'<think>.*?</think>\n?', text, flags=re.DOTALL))
        if not matches:
            return text
        start, end = matches[-1].span()
        return text[:start] + text[end:]

    @torch.no_grad()
    def evaluate_single(self, input_data):
        """Evaluate segmentation for a single image."""
        messages = input_data['messages']
        # messages = [input_data['prompt']]
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]

        image_inputs = [input_data["image"]] * len(texts)
        inputs = self.processor(text=texts, images=image_inputs, padding=True, return_tensors="pt").to(
            device="cuda", dtype=self.dtype
        )

        llm_out = self.model.generate(max_length=2048, use_cache=True, do_sample=False, **inputs)
        prompt_length = inputs['input_ids'].shape[1]
        completion_text = self.processor.batch_decode(llm_out[:, inputs['input_ids'].shape[1]:],
                                                      skip_special_tokens=True)
        out_text = [re.search(r'<answer>(.*?)</answer>', text) for text in completion_text]
        sam_text_prompt = [text.group(1).strip() if text is not None else "" for text in out_text]
        inputs.update({"sam_text_prompt": sam_text_prompt})
        # inputs.update({"sam_text_prompt": input_data['problems']})
        inputs.update({"prompt_length":prompt_length})

        new_attention_mask = torch.ones_like(llm_out, dtype=torch.int64)
        pos = torch.where(llm_out == self.processor.tokenizer.pad_token_id)
        new_attention_mask[pos] = 0
        inputs.update({"input_ids": llm_out, "attention_mask": new_attention_mask})
        inputs.update({"sam_images": [input_data["sam_image"]]})

        output, low_res_masks = self.model(output_hidden_states=True, use_learnable_query=True, **inputs)

        inference_state = self.model.sam.set_image(input_data["sam_image"])
        output = self.model.sam.set_text_prompt(state=inference_state, prompt=[input_data['problems'][0], None])
        presence_score,sam_mask = output['presence_score']  ,   output['semantic_seg']
        thr = 0.1*pow(2,len(input_data['problems'][0].split())-1)
        if presence_score > thr:low_res_masks = [sam_mask]


        pred_masks = [self.postprocess_masks(mask, orig_hw=input_data["mask"].shape[-2:]) for mask in low_res_masks]
        pred_masks = torch.cat(pred_masks, dim=0)
        pred_masks = (pred_masks[:, 0] > 0).int()

        masks_list = input_data["mask"].int().unsqueeze(0).repeat(pred_masks.size(0), 1, 1)
        intersection, union, acc_iou = 0.0, 0.0, 0.0
        index = 0

        for mask_i, output_i in zip(masks_list, pred_masks):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous().cuda(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            iou = intersection_i / (union_i + 1e-5)
            # mask_iou = iou[1].cpu().item()
            #
            # pred_box = parse_float_sequence_within(completion_text[index])


            index += 1
            acc_iou += iou
            acc_iou[union_i == 0] += 1.0  # Perfect score if no object present

        intersection = intersection.cpu().numpy() / len(masks_list)
        union = union.cpu().numpy() / len(masks_list)
        acc_iou = acc_iou.cpu().numpy() / len(masks_list)
        return intersection, union, acc_iou, len(masks_list)


def main(args):
    os.makedirs(f"{args.model_path}/evaluations", exist_ok=True)
    dataset = ReasonSegVal( if_reasonseg_val=args.cot)

    evaluator = ReasoningSegEvaluator(args)
    # dataset = ReasonSegDataset(Namespace(max_pixels=360000,min_pixels=100),base_image_dir=args.image_dir,mode="val")
    dataloader = DataLoader(dataset, 1, False, collate_fn=lambda batch: list(batch))

    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    bar = tqdm(dataloader)
    result = []
    for batch_data in bar:
        assert len(batch_data) == 1, "Only batch_size=1 is supported"
        intersection, union, acc_iou, num_mask = evaluator.evaluate_single(batch_data[0])
        intersection_meter.update(intersection, n=num_mask)
        union_meter.update(union, n=num_mask)
        acc_iou_meter.update(acc_iou, n=num_mask)
        # result.append((batch_data[0]["id"],acc_iou_meter.avg[1]))
        # print(result[-1])
        bar.desc =f"evaluation on ReasonSeg {'Val' if args.cot else 'Test'}: giou={acc_iou_meter.avg[1]}, ciou={(intersection_meter.sum / (union_meter.sum + 1e-8))[1]}"
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-8)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    print(f"model path is: {args.model_path}")
    print(f"evaluation on reasonseg {'Val' if args.cot else 'Test'}: giou={giou}, ciou={ciou}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Localization Evaluation Script")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--sample_num", type=int, default=-1, help="Number of samples (debugging)")
    parser.add_argument("--cot", action="store_true", help="True: Reasonseg val False: Reasonseg Test")
    parser.add_argument("--vis", action="store_true", help="Visualize segmentation results")
    args = parser.parse_args()
    print("args:", args)
    main(args)
