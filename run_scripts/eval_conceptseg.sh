export PYTHONPATH=$PYTHONPATH:$(pwd)

#sleep 5400

model=/path/to/model
image_folders="/path/to/ConceptSegDatasets"
data_file_paths="/path/to/all_meta.json"
batch_size=1
save_name=evalution


###### evalution CI concepts start ########
echo "----------------------evalution CI concepts start ----------------------"
CUDA_VISIBLE_DEVICES=7 python  src/eval/evaluate_conceptseg.py \
 --model_path $model --vis --dataset_names rare --batch_size $batch_size  --save_name $save_name --data_files $data_file_paths --image_folders $image_folders

CUDA_VISIBLE_DEVICES=7 python  src/eval/evaluate_conceptseg.py \
 --model_path $model --vis --dataset_names ultra_rare --batch_size $batch_size  --save_name $save_name --data_files $data_file_paths --image_folders $image_folders

CUDA_VISIBLE_DEVICES=6 python  src/eval/evaluate_conceptseg.py \
 --model_path $model --vis --dataset_names coco2014_Artifact --batch_size $batch_size  --save_name $save_name --data_files $data_file_paths --image_folders $image_folders

CUDA_VISIBLE_DEVICES=6 python  src/eval/evaluate_conceptseg.py \
 --model_path $model --vis --dataset_names coco2014_Living --batch_size $batch_size  --save_name $save_name --data_files $data_file_paths --image_folders $image_folders
echo "----------------------evalution CI concepts finish ----------------------"






###### evalution CI concepts ########
echo "----------------------evalution CD concepts start ----------------------"

CUDA_VISIBLE_DEVICES=7 python  src/eval/evaluate_conceptseg.py \
--model_path $model --vis --dataset_names DUTS --batch_size $batch_size  --save_name $save_name --data_files $data_file_paths --image_folders $image_folders

CUDA_VISIBLE_DEVICES=7 python  src/eval/evaluate_conceptseg.py \
 --model_path $model --vis --dataset_names isic2018 --batch_size $batch_size  --save_name $save_name --data_files $data_file_paths --image_folders $image_folders

CUDA_VISIBLE_DEVICES=7 python  src/eval/evaluate_conceptseg.py \
 --model_path $model --vis --dataset_names Breast_Tumor --batch_size $batch_size  --save_name $save_name --data_files $data_file_paths --image_folders $image_folders


CUDA_VISIBLE_DEVICES=6 python  src/eval/evaluate_conceptseg.py \
 --model_path $model --vis --dataset_names ESDIDefects --batch_size $batch_size  --save_name $save_name --data_files $data_file_paths --image_folders $image_folders


CUDA_VISIBLE_DEVICES=3 python  src/eval/evaluate_conceptseg.py \
 --model_path $model --vis --dataset_names COD10K1024 --batch_size $batch_size  --save_name $save_name --data_files $data_file_paths --image_folders $image_folders
CUDA_VISIBLE_DEVICES=6 python  src/eval/evaluate_conceptseg.py \
 --model_path $model --vis --dataset_names Polyp --batch_size $batch_size  --save_name $save_name --data_files $data_file_paths --image_folders $image_folders

CUDA_VISIBLE_DEVICES=3 python  src/eval/evaluate_conceptseg.py \
 --model_path $model --vis --dataset_names transparent1024 --batch_size $batch_size  --save_name $save_name --data_files $data_file_paths --image_folders $image_folders
CUDA_VISIBLE_DEVICES=3 python  src/eval/evaluate_conceptseg.py \
 --model_path $model --vis --dataset_names Shadow_detection --batch_size $batch_size  --save_name $save_name --data_files $data_file_paths --image_folders $image_folders


echo "----------------------evalution CD concepts finish ----------------------"



###### evalution CI concepts ########
echo "----------------------evalution CR concepts start ----------------------"

gpu=3
datasets=("MIG_correspondence" "MIG_object_tracking" "MIG_refer_grounding" "MIG_reasoning" "MIG_diff" "MIG_common_object" "MIG_multi_view" "MIG_view_diff")

for dataset in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu python src/eval/evaluate_conceptseg.py \
        --model_path $model --vis --dataset_names $dataset \
        --batch_size $batch_size --save_name $save_name --data_files $data_file_paths --image_folders $image_folders
done

echo "----------------------evalution CR concepts finish ----------------------"



