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

- **[2026-09]** 发布微调配对评测结果：LoRA 微调使判断题少数类召回率实现 **+29.3pp ~ +30.1pp** 的重大跃升（39.7% → 69.0% / 69.8%），逐样本 McNemar 检验达成 p < 0.001 显著性。
- **[2026-09]** 建立 0 泄漏研报分组切分体系（70/15/15），彻底消除原基准 78% 的跨集数据记忆风险。
- **[2026-09]** 完成 Qwen3-VL-8B-Instruct 全量 19,208 条零样本评测、视觉 Token 预算扫描及 6 组 Prompt 鲁棒性压力测试。

---

## 🌟 Key Highlights

- **0 泄漏数据切分协议 (0-Leakage Protocol)**  
  VisFinEval 原生无训练集划分，单份研报最多衍生 161 道题目。若随机切分，78% 的测试样本其来源研报会提前出现在训练集。本项目基于 `doc_key` 实现研报级分组隔离，达成严格的 0 泄漏契约。
- **穿透单一准确率假象 (Beyond Raw Accuracy)**  
  Qwen3-VL-8B 在判断题上 Raw Accuracy 达 77.46%，但少数类「否」的召回率仅 **39.7%**（显著低于随机抛硬币 50%）。本项目强制同时报告常数及格线、Macro-Recall 与 95% Wilson 置信区间。
- **长多图推理崩溃边界 (Multi-Image Breakdown Curve)**  
  量化模型在多图上下文下的退化曲线：图数 ≥8 时能力崩塌，10 张图题目的准确率降至 29.5%，相对常数基线净跌 29.5pp；并通过分辨率消融证明其瓶颈在于跨图注意力整合而非视觉 Token 预算。
- **配对闭环微调 (Rigorous Paired Alignment)**  
  基于 ms-swift 打造多卡 LoRA 流水线，在完全相同的 2,889 条测试样本上验证微调增益，通过逐样本 McNemar 精确检验，证实微调增益 95% 集中于少数类。

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

所有条件均在完全相同的一批无泄漏测试集（Test Split: n=2,889）上执行，采用 Greedy 解码，对齐全局 UID。

| 实验条件 | 多选题 Raw | 多选多图题 Raw | 判断题 Raw | 判断题 Macro | 「否」召回率 (95% Wilson CI) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **常数及格线** (多数类盲猜) | 58.61% | -- | 73.87% | -- | -- |
| **Zero-shot** (基座模型) | 73.70% | 62.83% | 75.23% | 63.73% | 39.7% [31.2, 48.8] |
| **LoRA Natural** (原始分布) | **84.13%** | **72.77%** | **84.01%** | 79.15% | 69.0% [60.1, 76.7] |
| **LoRA Balanced** (均衡上采样) | 82.78% | 68.06% | **84.01%** | **79.43%** | **69.8% [60.9, 77.4]** |
| **净增益 (Net Gain)** | **+10.43pp** | **+9.94pp** | **+8.78pp** | **+15.70pp** | **+30.1pp** |

---

## 🔬 Core Diagnostic Findings

在全量 19,208 条样本与测试子集上的系统性实证诊断：

### Finding 1 — 答案分布导致 Raw Accuracy 严重失真

<p align="center">
  <img src="assets/1.png" width="95%" alt="Severely Skewed Answer Distribution in Multiple-Choice Questions">
</p>

在 VisFinEval 全量 **16,404** 道多选题中，**58.5% 的真实答案是 "A"**。  
一个完全不看图、不做任何推理的模型，只要永远输出 `A`，就能拿到 **58.5% 的准确率**。  
因此，「常数基线」不是地板，而是**及格线**。任何汇报的成绩都必须对照及格线阅读：

| 指标 | 核心内涵 |
| :--- | :--- |
| **Raw Accuracy** | 传统论文常报指标（易受答案分布倾斜严重污染） |
| **Constant Baseline** | 永远输出多数类答案的基线 —— 及格线而非零分线 |
| **Macro-Recall** | 各类别独立召回率的均值 —— 剔除分布污染的客观能力 |

---

### Finding 2 — 模型在少数类上的表现显著低于抛硬币

<p align="center">
  <img src="assets/2.png" width="95%" alt="Few-shot classification is close to zero-shot; fine-tuning improves it significantly">
</p>

全量判断题看起来是最好的题型（Raw Accuracy 77.46%），但拆开看：

```
  真值     样本量(n)     占比        召回率
   是       2,063       73.6%       90.0%
   否         741       26.4%       42.5%    [95% CI 39.0–46.1]   ← 显著低于随机猜(50%)
```

Raw 与 Macro-Recall 相差 **11.20 个百分点**（全数据集最大）。模型在判断题上高度趋向退化为“全选是”（答了 520 次「否」，仅对 315 次）。

---

### Finding 3 — 多图推理崩溃是跨图整合上限，而非 Token 预算不足

<p align="center">
  <img src="assets/3.png" width="95%" alt="Performance Drop with Multiple Images is Not a Token Budget Problem">
</p>

准确率随题目包含的图表数量呈现单调剧烈下滑：

```
   图表数     样本数 (n)      Raw Accuracy      常数基线      相对基线净增益
     2           666            75.5%            45.8%          +29.7pp
     4           227            61.7%            29.5%          +32.2pp
     8            60            50.0%            56.7%           -6.7pp  (跌破基线)
    10            78            29.5%            59.0%          -29.5pp  (崩溃)
```

在 10 张图的题目中，模型得分比“盲选 A”低 29.5 个百分点。多期报表对比是金融分析的核心场景，该瓶颈极为关键。

**直觉假设（显存或视觉 Token 不足导致的截断）被实验否定。**  
我们通过调节处理器 `longest_edge` 对视觉 Token 预算进行了 4 档扫描（2,939 样本 × 4 档）：

```
   Token 预算     tok/img    1图     2图    3-4图   5-6图   8+图    整体表现
   无上限 (no cap)  4281    74.7%   75.5%   62.1%   85.2%   38.8%    70.7%
   上限 1.0M         979    74.5%   74.8%   59.6%   70.4%   43.2%    70.0%
   上限 0.5M         461    72.5%   74.9%   56.3%   66.7%   43.9%    68.3%
   上限 0.25M        243    70.1%   69.7%   52.7%   59.3%   35.3%    64.6%
```

在 8+ 张图区间，压缩 Token 预算后准确率反而略微提升（38.8% → 43.9%），且 4 档预算下的退化曲线形态高度一致（相对单图落后 −35.9 / −31.4 / −28.6 / −34.8 pp）。  
**结论**：多图崩溃的根因在于模型跨图注意力表征能力的极限，而非输入 Token 数量。  
**工程参考**：`longest_edge = 1M` 是最佳操作点，在仅损失 0.7pp 精度下减少了 4.4 倍视觉 Token。

---

### Finding 4 — 评测结论极度依赖措辞，Raw 指标甚至给出错误导向

在 4 种 Prompt 变体下重复判断题实验（各 n = 2,804）：

```
  变体                                     Raw     常数基线    Macro     「否」召回率
  base (基线默认措辞)                     77.5%     73.6%     66.3%        42.5%
  nobias (显式提示“不要默认选是”)         76.6%     73.6%     67.9%        49.3%
  correct_wrong (将“是/否”改为“正确/错误”) 76.5%     73.6%     73.0%        65.6%
  ab_format (转换为 A/B 选项题型)         77.5%     73.6%     72.3%        61.3%
```

1. **少数类崩溃是固有属性**：4 个变体下「否」召回率均低于「是」22~48 个百分点。
2. **措辞可大幅改变结论**：仅将“是/否”替换为“正确/错误”，少数类召回率即从 **42.5% 跃升至 65.6%**。
3. **Raw 与 Macro 方向相反**：所有变体的 Raw 均不优于 base（净翻对/错为 −27 / −23 / 0），但 Macro-Recall 最大提升 6.7pp。**若仅以 Raw Accuracy 为导向优化 Prompt，会选出对少数类最差的版本。**
4. **显式提示失效**：`nobias` 指令反而带来净变化 −23，证明模型并非不知道要平衡，而是受限于隐空间计算能力。  
   多选题对措辞极不敏感（<1.5pp），两类题型的评测鲁棒性存在本质差异。

---

### Finding 5 — 微调能够修复少数类崩溃，且增益几乎全部集中于少数类

<p align="center">
  <img src="assets/4.png" width="95%" alt="Fine-tuning brings large gains on the minority class, with minimal change on the majority class">
</p>

```
  实验条件              多选 Raw    判断 Raw    判断 Macro    「否」召回率 [95% CI]
  Zero-shot (基座)      73.70%      75.23%      63.73%       39.7% [31.2, 48.8]
  LoRA Natural          84.13%      84.01%      79.15%       69.0% [60.1, 76.7]
  LoRA Balanced         82.78%      84.01%      79.43%       69.8% [60.9, 77.4]
```

逐样本配对比较（基于全局统一 `uid`）：

```
  Zero-shot -> LoRA Natural     翻对 430 条   翻错 136 条   净变化 +294 条   McNemar p < 0.001 (极显著)
  Zero-shot -> LoRA Balanced    翻对 428 条   翻错 167 条   净变化 +261 条   McNemar p < 0.001 (极显著)
```

**增益高度集中于少数类（增益相差 20 倍）：**

```
  「是」召回率 (n=328)    87.8%  ->  89.3%     +1.5pp
  「否」召回率 (n=116)    39.7%  ->  69.0%    +29.3pp      ← 20 倍增益集中度
```

**Negative Finding 消融结论**：标签均衡重采样（Balanced）并未带来额外显著收益 —— 多选 Raw Accuracy 下降了 1.35pp（p = 0.024），少数类召回率仅提升 0.8pp。说明少数类崩溃并非由训练样本不均衡引起，常规 Natural 分布下的监督微调已足够激活模型的图表辨别能力。

---

## 🚀 Quick Start

### 1. 环境准备 (~10 分钟)
```bash
git clone https://github.com/gavinzsmeng/visfineval-reliability.git
cd visfineval-reliability
bash scripts/setup_env.sh
```

### 2. 下载数据与基座权重
```bash
bash scripts/hf.sh download SUFE-AIFLM-Lab/VisFinEval --repo-type dataset --local-dir data/VisFinEval
bash scripts/hf.sh download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct
7z x data/VisFinEval/data.7z -odata/VisFinEval/
```

### 3. 数据无泄漏切分
```bash
python scripts/prepare_data.py --out data/swift     --balance natural
python scripts/prepare_data.py --out data/swift_bal --balance balanced
```

### 4. 启动多卡 LoRA 微调
```bash
bash scripts/run_lora.sh natural     # 原始分布微调 (~1.6h)
bash scripts/run_lora.sh balanced    # 标签均衡微调 (~4.9h)
```

### 5. 训练后自动化配对评测
```bash
bash scripts/run_posttrain_eval.sh
```

---

## 🛠️ Codebase Structure

```
├── configs/                  # 分布式与微调配置 (DeepSpeed ZeRO-2 等)
├── data/                     # 数据切分产物 (研报级隔离，不托管图像原图)
├── models/                   # 基座模型存放目录 (Qwen3-VL-8B-Instruct)
├── output/                   # 训练 Checkpoints 与 TensorBoard 日志
├── results/                  # 评测原始预测与全景指标汇总
│   ├── RESULTS.md            # 详细实验记录与消融报告
│   ├── zeroshot/             # 零样本基线预测
│   ├── posttrain_*/          # 微调模型评测产物
│   └── sensitivity/          # Prompt 敏感性测试产物
├── scripts/                  # 核心流水线脚本
│   ├── prepare_data.py       # 防泄漏分组切分
│   ├── eval_baseline.py      # 多卡评测 Harness
│   ├── run_lora.sh           # LoRA 微调启动器
│   ├── run_posttrain_eval.sh # 训练后评测编排脚本
│   └── compare_conditions.py # McNemar 配对检验与 Wilson 置信区间
└── requirements.lock.txt     # 锁定的环境依赖清单
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
@misc{meng2026visfinevalrel,
  title  = {VisFinEval Reliability Audit: what reported accuracy hides},
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
