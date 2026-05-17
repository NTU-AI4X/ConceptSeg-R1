<div align="center">

<h1>ConceptSeg-R1</h1>

**ConceptSeg-R1: Segment Any Concept via Meta-Reinforcement Learning**

[![arXiv](https://img.shields.io/badge/arXiv-2026-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org)
[![HuggingFace](https://img.shields.io/badge/🤗%20Model-7B%20Weights-ffd21e?style=flat-square)](https://huggingface.co/zhaoyuan666/ConceptSeg-R1-7B)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-ConceptSeg--Benchmark-ffd21e?style=flat-square)](https://huggingface.co/datasets/zhaoyuan666/ConceptSeg-Benchmark)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)](LICENSE)
[![Stars](https://img.shields.io/github/stars/yuanzhao-CVLAB/ConceptSeg-R1?style=flat-square)](https://github.com/yuanzhao-CVLAB/ConceptSeg-R1/stargazers)

<p>
  <a href="#introduction">Introduction</a> •
  <a href="#get-started">Get Started</a> •
  <a href="#data">Data</a> •
  <a href="#datasets--checkpoints">Checkpoints</a>
</p>

<img src="./assets/Concept_Tree.png" width="90%"/>

</div>

---

## 📰 News

- **May 2026** — arXiv paper released 🎉

## 🗺️ Roadmap

| Status | Item |
|:------:|------|
| ✅ | arXiv paper |
| ✅ | Training code |
| ✅ | Testing code |
| ✅ | MMAD datasets |
| ✅ | ConceptSeg-R1 7B weights |
| ⬜ | Support larger MLLM backbones, e.g., Gemini 2.5 Pro|

---

## Introduction

ConceptSeg-R1 shifts visual perception from **object-level localization** toward **concept-level understanding**, addressing core limitations of existing segmentation models.

<div align="center">
<img src="./assets/Architecture.png" width="70%"/>
</div>

<br>

### Key Contributions

**🌳 Three-level Concept Taxonomy**
Formally categorizes concepts into **Context-Independent (CI)**, **Context-Dependent (CD)**, and **Context-Reasoning (CR)** tiers based on cognitive complexity.

**🔁 Meta-GRPO**
A meta-reinforcement learning mechanism that induces transferable task rules from visual demonstrations and verifies them through proxy reasoning.

**🔗 Concept Translation Module (CTM)**
A lightweight module mapping MLLM reasoning states into implicit concept groups, eliminating the semantic bottleneck between reasoning and mask execution.

**⚡ Shortcut Router**
An adaptive gate preserving SAM 3 efficiency for simple cases while activating full reasoning for complex CD/CR concepts.

---

## Get Started

### 1. Environment Setup

```bash
conda create -n conceptseg-r1 python=3.10
conda activate conceptseg-r1
bash setup.sh
```

### 2. Training

**Prepare data** — Download the dataset and set your `image_folders` path in the shell scripts.

```bash
# Stage 1: SFT Training
bash run_grpo_multiimage_stage1.sh

# Stage 2: GRPO Training
bash run_grpo_multiimage_stage2.sh
```

### 3. Evaluation

**Concept Segmentation** — Download weights, set the model path in `eval_conceptseg.sh`, then run:

```bash
bash eval_conceptseg.sh
```

> **Tip:** Configure specific tasks for testing inside `eval_conceptseg.sh`.

**Reasoning Segmentation** — Download weights, set the model path in `eval_reasonseg.sh`, then run:

```bash
bash eval_reasonseg.sh
```

---

## Data

Place datasets under a shared root directory (`image_folders`):

```
root/
├── isic2018/
├── rare/
├── Breast_Tumor/
├── transparent1024/
├── MGrounding-630k/
├── Polyp/
├── Shadow_detection/
├── MIG-Bench/
├── coco2014_Living/
├── CoSOD3k1024/
├── ultra_rare/
├── coco2014_Artifact/
├── fewshot1000/
├── DUTS/
├── SASilver_inaturalist/
├── ESDIDefects/
└── COD10K1024/
```

---

## Metric

Evaluation uses the [PySegMetric_EvalToolkit](https://github.com/Xiaoqi-Zhao-DLUT/PySegMetric_EvalToolkit).

---

## Datasets & Checkpoints

| Resource | Link |
|----------|------|
| 📦 ConceptSeg-Benchmark Dataset | [Download on HuggingFace](https://huggingface.co/datasets/zhaoyuan666/ConceptSeg-Benchmark) |
| 🤖 ConceptSeg-R1-7B Weights | [Download on HuggingFace](https://huggingface.co/zhaoyuan666/ConceptSeg-R1-7B) |

