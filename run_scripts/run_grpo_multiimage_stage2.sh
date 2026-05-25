PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}" # TODO: change this to your own
echo "REPO_HOME: $REPO_HOME"
# on remote
model_path="Qwen/Qwen2.5-VL-3B-Instruct"
image_folders="/home/ubuntun/disk7T/zy/data/ConceptSeg-Benchmark"
data_file_paths="/home/ubuntun/disk7T/zy/code/MLLM_GRPO/ConceptSeg-R1/data.json"
is_reward_customized_from_vlm_module=False
reward_methods="all_match"
echo "data_file_paths: $data_file_paths"
echo "image_folders: $image_folders"

export EXP_NAME="multi-image-stage2-3B" # TODO: change this to your own experiment name
TASK_TYPE=multiimage
cd ${REPO_HOME}/src/open-r1-multimodal

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"

export WANDB_MODE=online #online


 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="15999" \
 src/open_r1/grpo_multi_image.py \
    --use_vllm False \
    --output_dir ${REPO_HOME}/checkpoints/rl/${EXP_NAME} \
    --resume_from_checkpoint True \
    --model_name_or_path $model_path \
    --data_file_paths  $data_file_paths \
    --image_folders $image_folders \
    --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
    --reward_method $reward_methods \
    --task_type $TASK_TYPE \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --save_steps 10 \
    --is_grpo_train True \
     --save_total_limit 4 \
    --num_generations 8 \
    --save_only_model True \
    --question_template cot \
    --max_completion_length 2048 \
    --max_pixels 360000 \
    --reward_funcs format   iou \
    --beta 0.04 \
    --report_to wandb \
    --dataset-name not_used \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero3.json \
    --learning_rate 1e-6
