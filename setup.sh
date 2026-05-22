

echo "-----------------unzip sam3-main start-------------"
unzip sam3-main.zip
echo "-----------------unzip sam3-main finish-------------"


echo "-----------------install sam3 install-------------"
cd sam3-main
pip install -e .
cd ../
echo "-----------------install sam3 finish-------------"


echo "-----------------unzip all_meta.json start-------------"
unzip all_meta.json.zip
echo "-----------------unzip all_meta.json finish-------------"



# Install the packages in open-r1-multimodal .
cd src/open-r1-multimodal # We edit the grpo.py and grpo_trainer.py in open-r1 repo.
pip install -e ".[dev]"
# Addtional modules
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils torchvision
pip install flash-attn --no-build-isolation
pip install babel
pip install python-Levenshtein
pip install matplotlib
pip install pycocotools
pip install openai
pip install httpx[socks]
pip install json_repair
pip install opencv-python
pip install decord

