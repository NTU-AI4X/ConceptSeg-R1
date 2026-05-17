<div align="center">
 
## ConceptSeg-R1: Segment any concept by meta-reinforcement learning


### Arxiv 2026 
[![arXiv](https://img.shields.io/badge/arXiv-2509.25934-b31b1b.svg?style=plastic)]([https://arxiv.org/abs/2509.25934](https://arxiv.org))

</div>

PyTorch Implementation of ConceptSeg-R1

- **If you find this work useful, please star ⭐ the repo!**

- This project is released under **Apache-2.0 License**. 

## News
 
 - _05.2026_: We have released the arXiv paper of ConceptSeg-R1.😛


## 🚀 TODO
- [x] Release arXiv paper
- [x] open training code
- [x] open testing code
- [x] release MMAD  Datasets
- [x] release pre-trained ConceptSeg-R1 7B Weights
- [ ] release larger pre-trained ConceptSeg-R1  Weights


<div align="center">
<img src="./assets/Concept_Tree.png" width="1300"/>
</div>

## Introduction 💡

ConceptSeg-R1 addresses the limitations of existing segmentation models by shifting visual perception from object-level localization toward **concept-level understanding**.

### Key Contributions:

* **Three-level Concept Taxonomy**: Formally categorizes concepts into **Context-Independent (CI)**, **Context-Dependent (CD)**, and **Context-Reasoning (CR)** based on cognitive complexity.

* **Meta-GRPO**: A meta-reinforcement learning mechanism that induces transferable task rules from visual demonstrations and verifies them through proxy reasoning.

* **Concept Translation Module (CTM)**: A lightweight module that maps MLLM reasoning states into implicit concept groups, eliminating the semantic bottleneck between reasoning and mask execution.

* **Shortcut Router**: An adaptive gate that preserves the efficiency of SAM 3 for simple cases while activating full reasoning only for complex CD/CR concepts.


<div align="center">
<img src="./assets/Architecture.png" width="750"/>
</div>



## Get Started 



### Environment 

```bash
conda create -n conceptseg-r1 python=3.10
conda activate conceptseg-r1
bash setup.sh
```





### Train

1. **Prepare Data**: Download the dataset and define your root image_folders path in the sh files.

2. **Launch Srage1 SFT Training**:

```bash
bash run_grpo_multiimage_stage1.sh 
```

3. **Launch Srage2 GRPO Training**:

```bash
bash run_grpo_multiimage_stage2.sh 
```



### Test
#### Test Concept Segmentation 

1. **Model Weights**: Download the trained model weights and provide the model path to eval_conceptseg.sh.

```bash
bash eval_conceptseg.sh
```
Note: You can configure specific task for  testing in eval_conceptseg.sh.
#### Test Reasoning Segmentation 
2. **Model Weights**: Download the trained model weights and provide the model path to eval_reasonseg.sh.

```bash
bash  eval_reasonseg.sh
```


### Data



```

root (image_folders)
|-- isic2018
|-- rare
|-- Breast_Tumor
|-- transparent1024
|-- MGrounding-630k
|-- Polyp
|-- Shadow_detection
|-- MIG-Bench
|-- coco2014_Living
|-- CoSOD3k1024
|-- ultra_rare
|-- coco2014_Artifact
|-- fewshot1000
|-- DUTS
|-- SASilver_inaturalist
|-- ESDIDefects
|-- COD10K1024
```




## Metric
We use the evaluation toolkit from [PySegMetric_EvalToolkit](https://github.com/Xiaoqi-Zhao-DLUT/PySegMetric_EvalToolkit).

## 📁 datasets & checkpoints Links


| Item          | Datasets | ConceptSeg-R1-7B Weights                              |
|---------------|----------|-----------------------------------------------------------------|
| ConceptSeg-R1 |  [Download](https://huggingface.co/datasets/zhaoyuan666/ConceptSeg-Benchmark) | [Download](https://huggingface.co/zhaoyuan666/ConceptSeg-R1-7B) |



## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yuanzhao-CVLAB/ConceptSeg-R1&type=Timeline)](https://www.star-history.com/#yuanzhao-CVLAB/ConceptSeg-R1&Timeline)****
