# VisFinEval 可靠性诊断

> 中文金融图表 VLM 的可复现评测：不只看准确率，还看准确率**掩盖**了什么。

[VisFinEval](https://github.com/SUFE-AIFLM-Lab/VisFinEval)（EMNLP 2025）是一个
中文金融多模态基准，报告 SOTA 模型 Qwen-VL-max 达到 76.3%。但 **raw accuracy
是一个会被答案分布污染的数字** —— 本仓库量化这件事，并给出可复现的工具链。

## 核心发现

在 **Qwen3-VL-8B-Instruct**（开源 8B）上跑全量 19,208 条 zero-shot：

```
                         n         raw      常数基线   macro-recall
  多选题 · 全部      16,404     73.82%      58.54%      71.03%
  多选题 · 单图      14,965     74.53%      59.90%      72.32%
  多选题 · 多图       1,439     66.44%      44.41%      65.53%
  判断题 · 全部       2,804     77.46%      73.57%      66.26%
  ─────────────────────────────────────────────────────────────
  总体               19,208     74.35%      49.99%      69.44%
```

**「常数基线」= 永远输出该题型的多数类**（多选题选 A / 判断题选「是」）。
它不是零分线，而是及格线。

### 发现 1 — raw accuracy 掩盖了判断题的少数类崩溃

判断题看起来最好（raw 77.46%），拆开看：

```
  真值     n     占比     召回率
  是    2,063   73.6%    90.0%
  否      741   26.4%    42.5%   [95%CI 39.0–46.1]  ← 显著低于随机猜(50%)
```

raw 与 macro-recall 相差 **11.20 个百分点**（全数据集最大）。
模型在少数类「否」上的召回率比抛硬币还差。

### 发现 2 — 多图题存在清晰的崩溃曲线

```
   图数     n       raw      常数基线      增益
    2     666     75.5%      45.8%      +29.7pp
    4     227     61.7%      29.5%      +32.2pp
    8      60     50.0%      56.7%       -6.7pp   ← 低于常数基线
   10      78     29.5%      59.0%      -29.5pp   ← 崩溃
```

**10 张图的题，模型比「无脑全选 A」低 29.5 个百分点**（n=78）。
金融分析的真实场景就是多期对比，这个上限很关键。

**这不是 token 预算问题。** 做了 4 档分辨率的双因子实验（2,939 条 × 4）：

```
               单图tok    1图     2图    3-4图   5-6图   8图+    全部
无上限           4281   74.7%  75.5%  62.1%  85.2%  38.8%   70.7%
cap 1.0M         979   74.5%  74.8%  59.6%  70.4%  43.2%   70.0%
cap 0.5M         461   72.5%  74.9%  56.3%  66.7%  43.9%   68.3%
cap 0.25M        243   70.1%  69.7%  52.7%  59.3%  35.3%   64.6%
```

「8图+」档**收紧预算后反而变好**（38.8% → 43.9%），曲线形状也不随预算变化 ——
说明 8+ 图崩溃是**能力上限**，调分辨率解决不了。

但实验带来一个实用结论：**`longest_edge=1M` 是安全的工作点** ——
视觉 token 省 4.4 倍，精度只掉 0.7pp；低于 0.5M 后代价迅速上升。

### 发现 3 — 换个 prompt 措辞，结论就变了

用 4 个措辞变体重复判断题实验（各 n=2,804）：

```
变体                                     raw      常数基线    macro     否召回
base (baseline 措辞)                  77.5%     73.6%     66.3%    42.5%
nobias (提示「不要默认选是」)             76.6%     73.6%     67.9%    49.3%
correct_wrong (用「正确/错误」)         76.5%     73.6%     73.0%    65.6%
ab_format (转成 A/B 选项)              77.5%     73.6%     72.3%    61.3%
```

- **少数类崩溃是模型性质**：4 个变体下都复现（「否」召回率全部低于「是」22–48pp）
- **但 prompt 的影响大到足以推翻结论**：仅把「是/否」换成「正确/错误」，
  少数类召回率就从 **42.5% → 65.6%**
- **raw 与 macro 方向相反**：所有变体的 raw 都不优于 base（净变化 −27 / −23 / 0），
  但 macro-recall 提升最多 6.7pp。**只按 raw accuracy 调 prompt，会选出最差的版本**
- **`nobias` 失效**：显式告诉模型别默认选「是」，净变化 −23，反而更差 ——
  这不是「模型不知道要平衡」，而是**它做不到**

多选题对措辞不敏感（<1.5pp）。**两类题型的评测可信度不同。**

### 发现 4 — 数据集本身存在标注缺陷

模型输出多个字母（`ABCD`/`ABC`）被现有规则判错的 55 条，逐条检查发现：

- 全部是「以下**哪些**……」句式
- 「哪些」在这批中占 87%，全量数据中仅占 8.10% —— **富集 10.8 倍**
- 但 gold 只标注了单个字母

**这不是模型的错误，是标注格式与问题语义矛盾。**

### 发现 5 — 微调**能**修复少数类崩溃，而且不需要类别均衡

在 2,889 条 held-out test 样本上（按来源研报切分，与训练集零重叠）：

```
条件                      多选raw    判断raw   判断macro   「否」召回 [95%CI]
zero-shot (基座)         73.70%    75.23%    63.73%    39.7% [31.2, 48.8]
LoRA natural            84.13%    84.01%    79.15%    69.0% [60.1, 76.7]
LoRA balanced           82.78%    84.01%    79.43%    69.8% [60.9, 77.4]
```

逐样本配对比较（同一 uid）：

```
zero-shot -> LoRA natural     翻对 430  翻错 136   净 +294   p < 0.001
zero-shot -> LoRA balanced    翻对 428  翻错 167   净 +261   p < 0.001
```

**增益几乎全部落在少数类上：**

```
                     zero-shot -> LoRA natural
「是」召回 (n=328)      87.8%  ->  89.3%     +1.5pp
「否」召回 (n=116)      39.7%  ->  69.0%    +29.3pp   ← 差 20 倍
```

**负结果：类别均衡采样没有帮助。** `balanced` 条件把 6 个（题型 × 标签）
组合各上采样到 6,725 条，结果多选 raw 反而低 1.35pp（p=0.024），
「否」召回差异在噪声内。

→ **少数类崩溃不是「训练数据标签不平衡」造成的**，普通 SFT 就能修复它。

完整实验细节见 [`results/RESULTS.md`](results/RESULTS.md)。

## 快速开始

```bash
# 1. 环境（约 10 分钟）
bash scripts/setup_env.sh

# 2. 下载数据与权重
bash scripts/hf.sh download SUFE-AIFLM-Lab/VisFinEval --repo-type dataset --local-dir data/VisFinEval
bash scripts/hf.sh download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct
7z x data/VisFinEval/data.7z -odata/VisFinEval/

# 3. zero-shot 评测（4 卡约 35 分钟）
bash scripts/run_eval.sh

# 4. 分析
python scripts/analyze_results.py --dir results/zeroshot
```

> `scripts/hf.sh` 是一个 wrapper，用于绕过本机 SOCKS 代理与 httpx 的兼容问题。
> 你自己的环境可能不需要它。

## 复现完整实验

```bash
# prompt 敏感性（6 变体，约 35 分钟）
bash scripts/run_prompt_sensitivity.sh
python scripts/analyze_sensitivity.py

# 数据准备（按来源研报切分，无泄漏）
python scripts/prepare_data.py --out data/swift      --balance natural
python scripts/prepare_data.py --out data/swift_bal  --balance balanced

# LoRA 微调（4 卡，natural ≈1.6h / balanced ≈4.9h）
bash scripts/run_lora.sh natural
bash scripts/run_lora.sh balanced

# 训练后评测 + 配对比较
bash scripts/run_posttrain_eval.sh
```

## 方法说明

### 为什么必须按「来源研报」切分

VisFinEval 是 benchmark，**没有 train split**。19,208 条样本来自 3,318 份券商研报，
平均每份 5.8 条 QA，最大的一份贡献 161 条。

若按样本随机切分，**78% 的训练样本其来源研报会同时出现在测试集** ——
模型可能只是记住那份研报的排版和数值，而非学会读图。

本仓库按 `doc_key` 分组切分（70/15/15），保证同一研报的所有 QA 落在同一侧，
并自动做泄漏自检。

### 为什么报告 macro-recall 和常数基线

```
raw accuracy   会被答案分布污染。极端例：真值 90% 是 A，全选 A 就有 90%
macro-recall   每类召回率的平均，不受分布影响
常数基线       及格线，不是零分线
```

三者一起报，才能区分「模型有真实能力」和「模型学会了猜多数类」。

### 统计检验

少数类样本量小（test split 中判断题「否」仅 115 条），因此：

- 比例估计用 **Wilson 置信区间**（小样本/极端比例下比正态近似更稳）
- 条件间比较用 **McNemar 精确检验**（配对设计）

## 已知限制

- 只测了 Qwen3-VL-8B-Instruct，未横向对比其他模型
- 单一 prompt 模板（敏感性实验除外），未测采样方差（greedy 解码）
- 多图题未做图像顺序打乱的鲁棒性检验
- `L2_Q1.tsv` 中 1,032 行 `answer` 为空已剔除；`L3_Q4.tsv` 字段结构不同未纳入
- 切分为本项目自建，**论文/SOTA 数字不可直接比较**，引用时必须声明

## 数据版权

⚠ **本仓库不托管 VisFinEval 的图像数据。**
其图片来自券商研究报告，有版权。请通过官方渠道自行获取：

- HuggingFace: [`SUFE-AIFLM-Lab/VisFinEval`](https://huggingface.co/datasets/SUFE-AIFLM-Lab/VisFinEval)
- GitHub: [SUFE-AIFLM-Lab/VisFinEval](https://github.com/SUFE-AIFLM-Lab/VisFinEval)

本仓库只发布**代码、切分逻辑与评测结果**。

## 致谢

- [VisFinEval](https://aclanthology.org/2025.emnlp-main.1229/) — 上海财经大学，EMNLP 2025
- [ms-swift](https://github.com/modelscope/ms-swift) — 训练框架
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) — 基座模型

## 许可证

[Apache-2.0](LICENSE) © 2026 Gavin C. Meng

## 引用

如果这个仓库对你的工作有帮助：

```bibtex
@misc{meng2026visfinevalrel,
  title  = {VisFinEval 可靠性诊断：中文金融图表 VLM 的准确率掩盖了什么},
  author = {Meng, Gavin C.},
  year   = {2026},
  url    = {https://github.com/gavinzsmeng/visfineval-reliability}
}
```

底层基准请引用原论文：

```bibtex
@inproceedings{visfineval2025,
  title     = {VisFinEval: A Scenario-Driven Chinese Multimodal Benchmark
               for Holistic Financial Understanding},
  booktitle = {EMNLP},
  year      = {2025}
}
```
