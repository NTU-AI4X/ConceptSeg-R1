export PYTHONPATH=$PYTHONPATH:$(pwd)

model=/paht/to/model

###### evalution reasonseg val ########
echo "----------------------evalution Reasoning Segmentation  Val start ----------------------"
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.run   --nproc_per_node 1 --standalone  \
--master_port="25999" src/eval/evaluate_reasonseg.py \
--model_path  $model --cot

echo "----------------------evalution Reasoning Segmentation  Val start ----------------------"


echo "----------------------evalution Reasoning Segmentation  Test start ----------------------"
###### evalution reasonseg test ########
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.run   --nproc_per_node 1 --standalone  \
--master_port="25999" src/eval/evaluate_reasonseg.py \
--model_path  $model
echo "----------------------evalution Reasoning Segmentation  Test start ----------------------"






