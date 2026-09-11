# FinVLM-Reliability

> **Beyond Raw Accuracy: Diagnostic Benchmarking & Robust Alignment for Financial VLMs**  
> *中文金融视觉大模型的可靠性诊断与防泄漏微调基准*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Base Model](https://img.shields.io/badge/Base%20Model-Qwen3--VL--8B-00A67E.svg)](https://github.com/QwenLM/Qwen3-VL)
[![Training Engine](https://img.shields.io/badge/Engine-ms--swift%204.6-FF6F00.svg)](https://github.com/modelscope/ms-swift)
[![Benchmark](https://img.shields.io/badge/Benchmark-VisFinEval%20(EMNLP'25)-792EE5.svg)](https://github.com/SUFE-AIFLM-Lab/VisFinEval)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## 📢 News

- **[2026-09]** 发布训练后评测报告：通过标签均衡 LoRA 微调，判断题少数类召回率实现 **+30.1pp** 的重大突破（39.7% → 69.8%）。
- **[2026-09]** 建立 0 泄漏研报分组切分体系（70/15/15），彻底规避原基准 78% 的数据跨集记忆风险。
- **[2026-09]** 完成 Qwen3-VL-8B-Instruct 全量 19,208 条零样本基准评测及 6 组 Prompt 鲁棒性压力测试。

---

## 🌟 Key Highlights

- **0 泄漏数据切分协议 (0-Leakage Protocol)**  
  VisFinEval 原生无训练切分，单份研报最多贡献 161 道 QA。随机切分会导致 78% 研报跨集泄漏。本项目通过 `doc_key` 实现研报级分组隔离，达成严格的 0 泄漏契约。
- **穿透单一准确率假象 (Beyond Raw Accuracy)**  
  Qwen3-VL-8B 在判断题上 Raw Accuracy 达 77.46%，但少数类「否」召回率仅 **39.7%**（显著低于随机抛硬币 50%）。本项目引入常数及格线、Macro-Recall 与 95% Wilson 置信区间，量化虚假繁荣。
- **长多图推理崩溃边界 (Multi-Image Breakdown Curve)**  
  量化模型在多图上下文下的退化曲线：图数 ≥8 时能力崩塌，10 张图题目的准确率降至 29.5%，相对常数基线净跌 29.5pp。
- **配对闭环微调 (Rigorous Paired Alignment)**  
  基于 ms-swift 打造多卡 LoRA 流水线，对比 Natural 原始分布与 Balanced 均衡上采样。在固化 UID 的 2,889 条测试集上实现逐样本 McNemar 配对检验与指标跃升。

---

## 🏗️ System Architecture

```
Raw Brokerage Reports (3,318 Docs, 19,208 QAs)
                      |
                      v
      [0-Leakage Report-Level Split]
      ├── Train (13,445) ── doc_key Grouping
      ├── Val    (2,874) ── 0 Overlap Assertion
      └── Test   (2,889) ── Fixed Global UID
                      |
        +-------------+-------------+
        |                           |
        v                           v
  [LoRA Natural]              [LoRA Balanced]
  Original Distribution       Upsampled Minority
  (422 steps / 1.6h)          (1,262 steps / 4.9h)
        |                           |
        +-------------+-------------+
                      |
                      v
  [Rigorous Paired Evaluation Harness]
  • Raw Accuracy vs Constant Baseline
  • Macro-Recall & Wilson 95% CI
  • Paired McNemar Exact Significance Test
```

---

## 📊 Benchmark & Experimental Results

所有条件均在完全相同的一批无泄漏测试集（Test Split: n=2,889）上执行，推理采用 Greedy 解码，对齐全局 UID。

### 1. 训练前后主结果对比 (Leaderboard)

| 实验条件 | 多选题 Raw | 多选多图题 Raw | 判断题 Raw | 判断题 Macro | 「否」召回率 (95% Wilson CI) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **常数及格线** (多数类盲猜) | 58.61% | -- | 73.87% | -- | -- |
| **Zero-shot** (基座模型) | 73.70% | 62.83% | 75.23% | 63.73% | 39.7% [31.2, 48.8] |
| **LoRA Natural** (原始分布) | **84.13%** | **72.77%** | **84.01%** | 79.15% | 69.0% [60.1, 76.7] |
| **LoRA Balanced** (均衡上采样) | 82.78% | 68.06% | **84.01%** | **79.43%** | **69.8% [60.9, 77.4]** |
| **净增益 (Net Gain)** | **+10.43pp** | **+9.94pp** | **+8.78pp** | **+15.70pp** | **+30.1pp** |

> **关键判读**：
> 1. 微调使少数类「否」的召回率从 **39.7% 跃迁至 69.8%**，彻底修复了基座模型在判断题上退化为“盲选是”的结构性缺陷。
> 2. Balanced 条件在少数类召回上达到最优（69.8%），但多选题整体准确率轻微下降 1.35pp，清晰量化了类别均衡带来的 Pareto 权衡。

---

## 🔬 Core Diagnostic Findings

在全量 19,208 条 VisFinEval 数据上的零样本诊断揭示了 4 个关键科学事实：

### 发现 1 — Raw Accuracy 掩盖了判断题的少数类崩溃
判断题表面 Raw 准确率高达 77.46%，但模型在「否」类别的召回率仅 42.5%，落后抛硬币。模型在隐空间更倾向于顺应语言先验输出肯定答复。

### 发现 2 — 多图复合推理的清晰退化曲线

```
   图表数      样本数 (n)      Raw Accuracy      常数基线      相对基线净增益
     2            666            75.5%            45.8%          +29.7pp
     4            227            61.7%            29.5%          +32.2pp
     8             60            50.0%            56.7%           -6.7pp  (跌破基线)
    10             78            29.5%            59.0%          -29.5pp  (崩溃)
```

10 张图的复合题目中，模型准确率落后“全选 A”达 29.5 个百分点，暴露出视觉 Transformer 跨图表注意力随序列增长的瓶颈。

### 发现 3 — 题型对 Prompt 措辞的敏感度差异巨大
在 6 组措辞变体（各 n=2,804）测试中发现：
- **多选题极其鲁棒**：措辞调整导致的指标摆动 <1.5pp。
- **判断题极其脆弱**：仅将“是/否”改为“正确/错误”，少数类召回率直接由 **42.5% 跃升至 65.6%**（Macro 提升 6.7pp）。
- **显式去偏失效**：加入 `nobias`（提示“不要默认选是”）后净翻错 23 条，说明该缺陷源于能力上限而非指令理解。

### 发现 4 — 基准数据集本身的格式矛盾
定位出 55 条模型输出多个字母被规则判错的题目，全部为「以下哪些……」句式（富集度达 10.8 倍），但标注真值仅为单字母，属于基准自身的标注缺陷。

---

## 🚀 Quick Start

### 1. 环境准备 (约 10 分钟)
```bash
git clone https://github.com/gavinzsmeng/visfineval-reliability.git
cd visfineval-reliability

# 配置 conda 环境并安装 ms-swift 4.6
bash scripts/setup_env.sh
```

### 2. 下载数据与模型权重
```bash
# 下载基准数据集与 Qwen3-VL 权重
bash scripts/hf.sh download SUFE-AIFLM-Lab/VisFinEval --repo-type dataset --local-dir data/VisFinEval
bash scripts/hf.sh download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct

# 解压图表资产 (约 1.7 GB)
7z x data/VisFinEval/data.7z -odata/VisFinEval/
```

### 3. 数据无泄漏切分
```bash
# 生成自然分布与标签均衡分布切分 (包含 0 泄漏断言检查)
python scripts/prepare_data.py --out data/swift     --balance natural
python scripts/prepare_data.py --out data/swift_bal --balance balanced
```

### 4. 启动多卡 LoRA 微调
```bash
# 使用 4×RTX 4090 并行训练
bash scripts/run_lora.sh natural     # 原始分布微调 (~1.6h)
bash scripts/run_lora.sh balanced    # 标签均衡微调 (~4.9h)
```

### 5. 训练后自动化配对评测
```bash
# 在固定测试集上自动化运行三条件推理并生成配对比较
bash scripts/run_posttrain_eval.sh
```

---

## 🛠️ Codebase Structure

```
├── configs/               # 分布式与微调配置 (DeepSpeed ZeRO-2 等)
├── data/                  # 数据切分产物 (已做研报级隔离，不托管图像原图)
├── models/                # 基座模型存放目录 (Qwen3-VL-8B-Instruct)
├── output/                # 训练 Checkpoints 与 TensorBoard 日志
├── results/               # 评测原始预测 (jsonl) 与全景指标汇总
│   ├── RESULTS.md         # 详细实验记录与消融报告
│   ├── zeroshot/          # 零样本基线预测
│   ├── posttrain_*/       # 微调模型评估结果
│   └── sensitivity/       # Prompt 敏感性测试结果
├── scripts/               # 核心流水线脚本
│   ├── prepare_data.py    # 防泄漏切分与 ms-swift 格式化
│   ├── eval_baseline.py   # 多卡评测 Harness 与指标计算
│   ├── run_lora.sh        # LoRA 微调启动器
│   ├── run_posttrain_eval.sh # 训练后评测编排脚本
│   └── compare_conditions.py # McNemar 配对检验与 Wilson 置信区间
└── requirements.lock.txt  # 严密锁定的环境依赖清单
```

---

## 📜 Copyright & Legal Notice

⚠ **本仓库不重新分发 VisFinEval 的图像资产。**  
测试图片源于各大券商公开研究报告，版权归原作者所有。请根据学术许可自官方通道获取：
- HuggingFace: [SUFE-AIFLM-Lab/VisFinEval](https://huggingface.co/datasets/SUFE-AIFLM-Lab/VisFinEval)
- GitHub: [SUFE-AIFLM-Lab/VisFinEval](https://github.com/SUFE-AIFLM-Lab/VisFinEval)

---

## 📑 Citation

如果本项目的方法论或评测工具链对您的研究/工程有所启发，请引用：

```bibtex
@misc{meng2026finvlmreliability,
  title  = {FinVLM-Reliability: Beyond Raw Accuracy in Financial Multimodal Understanding},
  author = {Meng, Gavin C.},
  year   = {2026},
  url    = {https://github.com/gavinzsmeng/visfineval-reliability}
}
```

底层 VisFinEval 基准请引用原论文：

```bibtex
@inproceedings{visfineval2025,
  title     = {VisFinEval: A Scenario-Driven Chinese Multimodal Benchmark for Holistic Financial Understanding},
  booktitle = {EMNLP},
  year      = {2025}
}
```

---

## 📄 License

本项目代码遵循 [Apache-2.0](LICENSE) 开源协议。
