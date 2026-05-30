#!/usr/bin/env python3
"""
ConceptSeg-R1 Single Example Inference Script
Supports single image or dual image (reference + inference) input modes
"""

import os
import argparse
import re
import torch
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
from transformers import AutoProcessor
from open_r1.mymodels.conceptr1 import ConceptSegR1ForConditionalGeneration_qwen2p5
from open_r1.mydataset import build_prompt, system_prompt_registry
import sam3.visualization_utils as utils
from open_r1.mydataset import draw_bboxes

# Visualization tool color configuration
COLORS = utils.pascal_color_map()[1:]

question_template = """
Your task is to locate the object matching "{question}" in the Target Image.
Data provided:
1. Reference Image {ref_bboxes}.
2. Target Image: The image to locate.
Think through the reasoning process in your mind， induce the visual rule {check_prompt}, apply this rule to locate the corresponding object in the Target Image.
Finally,  provide the bounding box  and a 1-2 word  noun phrase for the object in the target image. 
Output strictly in the following format: <think>[Your step-by-step analysis and reasoning]</think>  {check_answer} <bbox>[x3, y3, x4, y4]</bbox> <answer>concise noun phrase for target object</answer>
     """

def apply_white_transparent_overlay(image, mask, alpha=0.3):
    """
    Apply white transparent overlay to background regions (mask == 0)
    
    Args:
        image: RGB uint8 array (H, W, 3)
        mask: Single channel mask (H, W), non-zero indicates foreground
        alpha: White layer transparency (0~1), higher value makes background whiter
    
    Returns:
        result: Processed image
    """
    overlay = np.full_like(image, 255, dtype=np.uint8)  # Pure white layer
    mask_bool = (mask > 0).astype(np.uint8)             # Binarized foreground mask
    # Expand to 3 channels for pixel-wise selection
    mask_3ch = np.stack([mask_bool] * 3, axis=-1)
    # Blend white in background regions, keep foreground unchanged
    result = np.where(mask_3ch, image, (1 - alpha) * image + alpha * overlay)
    return result.astype(np.uint8)


def plot_single_img(img, mask, white_alpha=0.3):
    """
    Draw colored masks and add white semi-transparent overlay to background
    
    Args:
        img: Original image
        mask: Segmentation mask
        white_alpha: Background white layer intensity (0 = no effect, 1 = full white)
    
    Returns:
        masked_frame: Visualized image with masks
    """
    img = np.array(img)
    annot_masks = [np.array(mask)]  # Convert to numpy array
    original_img = img

    # Draw colored masks (foreground)
    masked_frame = utils.draw_masks_to_frame(
        frame=original_img, masks=annot_masks, colors=COLORS[:len(annot_masks)]
    )
    # Add white semi-transparent overlay to background regions
    masked_frame = apply_white_transparent_overlay(masked_frame, annot_masks[0], alpha=white_alpha)
    return masked_frame


def load_image(path, max_pixels=360000):
    """
    Load and preprocess image
    
    Args:
        path: Image path
        max_pixels: Maximum number of pixels
    
    Returns:
        sam_image: Original size image (for SAM)
        resized_image: Resized image (for model input)
    """
    image = Image.open(path).convert("RGB")
    sam_image = image.copy()  # Keep original image for SAM
    # Calculate target size (square, side length = sqrt(max_pixels))
    target_size = int(max_pixels ** 0.5)
    resized_image = image.resize((target_size, target_size))
    return sam_image, resized_image


def build_prompt_data(ref_path, question, bbox=None, template_type="cot"):
    """
    Build inference prompt data
    
    Args:
        ref_path: Reference image path (can be None)
        question: Question description
        bbox: Bounding box information (optional)
        template_type: Prompt template type
    
    Returns:
        problem: Formatted problem
    """
    check_prompt = ""
    if ref_path is None:
        ref_bboxs = ": N/A"
        check_answer = "<rule>Visual rule</rule>"
    elif bbox is not None:        
        ref_bboxs = ": Bounding boxes at red-marked boxs"
        check_answer = "<rule>Visual rule of the reference targets in the reference image</rule>"
    else:
        ref_bboxs = f": Bounding boxes at {bbox}"
        check_answer = "<rule>Visual rule of the reference targets in the reference image</rule>"
    
    problem = question_template.format(
        ref_bboxes=ref_bboxs, 
        question=question, 
        check_prompt=check_prompt,
        check_answer=check_answer
    )
    return problem


def sam3_inference(model, sam_img, prompt, max_pixels=360000):
    """
    SAM3 model inference
    
    Args:
        model: Loaded model
        sam_img: SAM input image
        prompt: Text prompt
        max_pixels: Maximum number of pixels
    
    Returns:
        mask: Segmentation mask
        confidence: Confidence score
    """
    inference_state = model.sam.set_image(sam_img)
    sam_out = model.sam.set_text_prompt(state=inference_state, prompt=[prompt, None])
    
    # Calculate target size
    target_size = int(max_pixels ** 0.5)
    mask = model.postprocess_masks(sam_out["semantic_seg"], orig_hw=(target_size, target_size))
    mask = (mask[:, 0] > 0).int()
    confidence = sam_out["presence_score"].item()
    
    return mask, confidence


def mllm_inference(model, processor, messages, image_prompt, sam_img, dtype=torch.bfloat16, max_pixels=360000):
    """
    Multimodal Large Language Model inference
    
    Args:
        model: Loaded model
        processor: Processor
        messages: Message list
        image_prompt: Image prompt
        sam_img: SAM input image
        dtype: Data type
        max_pixels: Maximum number of pixels
    
    Returns:
        pred_masks: Predicted masks
    """
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    inputs = processor(text=texts, images=image_prompt, padding=True, return_tensors="pt").to(
        device="cuda", dtype=dtype
    )

    # Remove potential temperature parameter from inputs to avoid warnings
    generate_kwargs = {
        "max_length": 2048+768, 
        "use_cache": True, 
        "do_sample": False
    }
    
    # Only pass necessary parameters to generate function
    llm_out = model.generate(**generate_kwargs, **{k: v for k, v in inputs.items() if k not in ["temperature"]})
    prompt_length = inputs['input_ids'].shape[1]
    completion_text = processor.batch_decode(llm_out[:, prompt_length:], skip_special_tokens=True)
    print(completion_text)
    # Extract SAM text prompt
    out_text = [re.search(r'<answer>(.*?)</answer>', text) for text in completion_text]
    sam_text_prompt = [text.group(1).strip() if text is not None else "" for text in out_text]

    # Update input parameters
    new_attention_mask = torch.ones_like(llm_out, dtype=torch.int64)
    pos = torch.where(llm_out == processor.tokenizer.pad_token_id)
    new_attention_mask[pos] = 0
    
    inputs.update({
        "input_ids": llm_out, 
        "attention_mask": new_attention_mask,
        "sam_images": [sam_img],
        "sam_text_prompt": sam_text_prompt,
        "prompt_length": prompt_length
    })

    _, low_res_masks = model(output_hidden_states=True, use_learnable_query=True, **inputs)
    
    # Calculate target size and resize image
    target_size = int(max_pixels ** 0.5)
    sam_img = sam_img.resize((target_size, target_size))
    pred_masks = model.postprocess_masks(low_res_masks[0], orig_hw=(target_size, target_size))
    pred_masks = (pred_masks[:, 0] > 0).int()
    
    return pred_masks


def conceptseg_r1_inference(model, processor, problem, messages, image_prompt, sam_img, dtype=torch.bfloat16, max_pixels=360000):
    """
    ConceptSeg-R1 model inference (includes routing logic)
    
    Args:
        model: Loaded model
        processor: Processor
        problem: Problem description
        messages: Message list
        image_prompt: Image prompt
        sam_img: SAM input image
        dtype: Data type
        max_pixels: Maximum number of pixels
    
    Returns:
        mask: Final segmentation mask
    """
    # Clean problem text
    # problem = re.sub(r'\([^)]*\)', '', problem).strip()
    length = len(problem.split())
    confidence_limit = 0.1 * (pow(2, length - 1))
    # ShortCut Router logic: if confidence is high enough, use SAM3 direct inference
    if confidence_limit < 1:
        mask, confidence = sam3_inference(model, sam_img, problem, max_pixels)
        if confidence > confidence_limit:
            print("============using sam3 inference===============")
            return mask
    
    # Otherwise use MLLM inference
    print("============using MLLM inference===============")
    mask = mllm_inference(model, processor, messages, image_prompt, sam_img, dtype, max_pixels)
    return mask


def post_process_results(ref_img, sam_img, pred_masks, output_path, max_pixels=360000):
    """
    Post-process and save results
    
    Args:
        ref_img: Reference image (can be None)
        sam_img: Inference image
        pred_masks: Predicted masks
        output_path: Output path
        max_pixels: Maximum number of pixels
    
    Returns:
        combined: Combined image
    """
    # Calculate target size
    target_size = int(max_pixels ** 0.5)
    
    # Resize inference image to same size as masks
    resized_sam_img = sam_img.resize((target_size, target_size))
    
    # Convert to PIL image and generate visualization results
    pred_masks_pil = Image.fromarray(pred_masks[0].cpu().numpy().astype(np.uint8) * 255)
    masked_frame = Image.fromarray(plot_single_img(resized_sam_img, pred_masks_pil))
    
    # Always use three-column layout: Reference Image | Inference Image | Output Results
    total_width = resized_sam_img.width * 3
    combined = Image.new('RGB', (total_width, resized_sam_img.height), color='black')
    
    # If reference image exists, resize and paste it, otherwise leave black
    if ref_img is not None:
        resized_ref_img = ref_img.resize((target_size, target_size))
        combined.paste(resized_ref_img, (0, 0))
    
    # Paste inference image and output results
    combined.paste(resized_sam_img, (resized_sam_img.width, 0))
    combined.paste(masked_frame, (resized_sam_img.width * 2, 0))
    
    # Ensure output directory exists and save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined.save(output_path)
    
    return combined


def build_model_and_processor(model_path, dtype=torch.bfloat16):
    """
    Build model and processor
    
    Args:
        model_path: Model path
        dtype: Data type
    
    Returns:
        model: Loaded model
        processor: Processor
    """
    model = ConceptSegR1ForConditionalGeneration_qwen2p5.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation="flash_attention_2"
    ).cuda()
    model.init_sam()
    processor = AutoProcessor.from_pretrained(model_path)
    
    return model, processor


def prepare_input_data(ref_path, infer_path, question, bbox=None, template_type="cot", max_pixels=360000):
    """
    Prepare input data
    
    Args:
        ref_path: Reference image path (can be None)
        infer_path: Inference image path
        question: Question description
        bbox: Bounding box information (optional)
        template_type: Prompt template type
        max_pixels: Maximum number of pixels
    
    Returns:
        problem: Formatted problem
        messages: Message list
        image_prompt: Image prompt
        sam_img: SAM input image
    """
    # Build problem
    problem = build_prompt_data(ref_path, question, bbox, template_type)
    
    # Build system prompt
    system_prompt_template = system_prompt_registry['default']
    messages = [build_prompt(2 if ref_path is not None else 1, problem, system_prompt_template)]
    
    # Load images
    sam_img, infer_img = load_image(infer_path, max_pixels)
    
    # Build image prompt based on whether reference image exists
    if ref_path is not None:
        _, ref_img = load_image(ref_path, max_pixels)
        if bbox is not None:
            print("Drawing bounding boxes on reference image...")
            draw_bboxes(ref_img, bbox)
        image_prompt = [[ref_img, infer_img]]
    else:
        image_prompt = [[infer_img]]
    
    return question, messages, image_prompt, sam_img


def main(args):
    """Main function: Execute inference pipeline"""
    # Prepare input data
    problem, messages, image_prompt, sam_img = prepare_input_data(
        args.ref_path, args.infer_path, args.question, args.bbox, args.template, args.max_pixels
    )

    # Get reference image (if exists) - get the image with bounding boxes from image_prompt
    ref_img = None
    if args.ref_path is not None:
        # Get the reference image with bounding boxes from image_prompt
        ref_img = image_prompt[0][0]  # image_prompt structure is [[ref_img, infer_img]]
    
    # Build model and processor
    model, processor = build_model_and_processor(args.model_path, args.dtype)
    
    # Execute inference
    with torch.inference_mode():
        mask = conceptseg_r1_inference(model, processor, problem, messages, image_prompt, sam_img, args.dtype, args.max_pixels)
    
    # Post-process and save results
    output_path = args.infer_path.replace(".jpg", ".png") if args.output_path is None else args.output_path
    combined_image = post_process_results(ref_img, sam_img, mask, output_path, args.max_pixels)
    
    print(f"Inference completed! Results saved to: {output_path}")
    return combined_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ConceptSeg-R1 Single Example Inference Script")
    
    # Required parameters
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--infer_path", type=str, required=True, help="Inference image path")
    parser.add_argument("--question", type=str, required=True, help="Question description")
    
    # Optional parameters
    parser.add_argument("--ref_path", type=str, default=None, help="Reference image path (can be empty)")
    parser.add_argument("--output_path", type=str, default=None, help="Output path (default based on inference image path)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"], 
                       help="Data type")
    parser.add_argument("--template", type=str, default="infer_cot", help="Prompt template type")
    parser.add_argument("--bbox", type=str, default=None, help="Reference Bounding box information (format: 'x1,y1,x2,y2;x1,y1,x2,y2') under max_pixels resolution (e.g., 360000=600x600), can be empty")
    parser.add_argument("--max_pixels", type=int, default=360000, help="Maximum number of pixels (default 360000, corresponds to 600x600)")
    
    args = parser.parse_args()
    
    # Convert data type
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16
    }
    args.dtype = dtype_map.get(args.dtype, torch.bfloat16)
    
    # Parse bounding boxes
    if args.bbox:
        try:
            args.bbox = [[int(x) for x in box.split(',')] for box in args.bbox.split(';')]
        except:
            print("Warning: Bounding box format error, using default value")
            args.bbox = None
    print(args)
    # Execute main function
    main(args)
