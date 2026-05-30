#!/bin/bash

# ConceptSeg-R1 Robustness Test Script
# Covers various usage scenarios: single image inference, dual image inference, inference with bounding boxes, etc.

# Set base paths
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${BASE_DIR}/ConceptSeg-R1-7B"
INFER_PATH="${BASE_DIR}/example_images/infer.jpg"
OUTPUT_DIR="${BASE_DIR}/example_images/outputs"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "=== ConceptSeg-R1 Inference Started ==="
echo "Model Path: ${MODEL_PATH}"
echo "Output Directory: ${OUTPUT_DIR}"
echo ""

# Test Case 1: Single Image Inference (No Reference Image) Simple Instruction
echo "--- Test Case 1: Single Image Inference (No Reference Image) Simple Instruction ---"
python "${BASE_DIR}/src/eval/inference_single_example.py" \
    --model_path "${MODEL_PATH}" \
    --infer_path "${INFER_PATH}" \
    --question "Flower Pot" \
    --output_path "${OUTPUT_DIR}/test1_simple_instruct_no_ref.png"
echo ""

# Test Case 2: Single Image Inference (No Reference Image) Complex Instruction
echo "--- Test Case 2: Single Image Inference (No Reference Image) Complex Instruction ---"
python "${BASE_DIR}/src/eval/inference_single_example.py" \
    --model_path "${MODEL_PATH}" \
    --infer_path "${INFER_PATH}" \
    --question "Please segment the support structure that keeps the bird cage standing" \
    --output_path "${OUTPUT_DIR}/test2_complex_instruct_no_ref.png"
echo ""

# Test Case 3: Dual Image Inference (With Reference Image)
echo "--- Test Case 3: Dual Image Inference (With Reference Image) ---"
python "${BASE_DIR}/src/eval/inference_single_example.py" \
    --model_path "${MODEL_PATH}" \
    --ref_path "${BASE_DIR}/example_images/reasoning_ref.jpg" \
    --infer_path "${INFER_PATH}" \
    --question "Which component in Figure 2 can resolve the issue shown in Figure 1?" \
    --output_path "${OUTPUT_DIR}/test3_dual_image_withref.png"
echo ""

# Test Case 4: Dual Image Inference with Bounding Boxes
echo "--- Test Case 4: Dual Image Inference (With Reference Image and Bounding Boxes) ---"
python "${BASE_DIR}/src/eval/inference_single_example.py" \
    --model_path "${MODEL_PATH}" \
    --ref_path "${BASE_DIR}/example_images/cod_ref.png" \
    --infer_path "${INFER_PATH}" \
    --bbox "325, 64, 570, 270;146, 420, 184, 491;313, 434, 516, 561" \
    --question "camouflaged object segmentation" \
    --output_path "${OUTPUT_DIR}/test4_dual_image_withref_withbox.png"
echo ""
 

echo "=== Inference Completed ==="
echo "All test results saved to: ${OUTPUT_DIR}"
echo ""
echo "Test Case Summary:"
echo "1. Single Image Inference - Simple Instruction (No Reference Image)"
echo "2. Single Image Inference - Complex Instruction (No Reference Image)"
echo "3. Dual Image Inference (With Reference Image)"
echo "4. Dual Image Inference with Bounding Boxes"