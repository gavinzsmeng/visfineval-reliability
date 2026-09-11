# VisFinEval 可靠性诊断

[English](README.md) · **中文**

> 对 [VisFinEval](https://github.com/SUFE-AIFLM-Lab/VisFinEval)（中文金融多模态基准）
> 的可复现可靠性审查。它回答的不是「这个模型多准」，而是
> **「报出来的准确率，掩盖了什么？」**

VisFinEval（EMNLP 2025）报告其最优模型 Qwen-VL-max 达到 **76.3%**。这个数字是在一个
答案分布严重偏斜的多选题任务上算出的 raw accuracy。本仓库量化这个偏斜、展示它掩盖了什么，
并提供能复现每个数字的工具链。

---

## 发现 1 — 答案分布让 raw accuracy 几乎失去意义

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/fig1_answer_distribution_dark.png">
  <img alt="多选题答案分布：A 58.5%、B 26.6%、C 10.7%、D 4.1%" src="assets/fig1_answer_distribution.png">
</picture>

VisFinEval 全部 **16,404** 道多选题中，**58.5% 的正确答案是「A」**。

一个永远输出 `A` 的模型——不看图、不读题、不推理——就能拿 **58.5%**。
这不是零分线，**这是及格线**。任何报告出来的分数都必须对照它来读。

因此本仓库对每个结果都报三个数字：

| 指标 | 含义 |
|---|---|
| **raw accuracy** | 论文里报的那个 |
| **常数基线** | 永远输出多数类，即及格线 |
| **macro-recall** | 各类召回率的平均，不受答案分布污染 |

---

## 发现 2 — 模型在少数类上比随机猜还差

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/fig2_minority_recall_dark.png">
  <img alt="判断题召回率：zero-shot 少数类 42.5%，LoRA 后 69.0%" src="assets/fig2_minority_recall.png">
</picture>

判断题看起来是表现**最好**的一类（raw 77.46%），但按标签拆开：

```
  真值      n     占比     召回率
  是     2,063   73.6%    90.0%
  否       741   26.4%    42.5%    [95% CI 39.0–46.1]   ← 低于随机猜（50%）
```

raw accuracy 与 macro-recall 相差 **11.20 个百分点**——全数据集最大。
模型退化到了多数类上，但又不是完全退化：它答了 520 次「否」，只对 315 次。

---

## 发现 3 — 多图崩溃是能力上限，不是 token 预算问题

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/fig3_multi_image_curve_dark.png">
  <img alt="准确率随图片数量的变化，四条线对应四档视觉 token 预算" src="assets/fig3_multi_image_curve.png">
</picture>

准确率随单题图片数单调下降：

```
  图数     n      raw     常数基线      增益
    2     666    75.5%     45.8%     +29.7pp
    4     227    61.7%     29.5%     +32.2pp
    8      60    50.0%     56.7%      -6.7pp
   10      78    29.5%     59.0%     -29.5pp   ← 比无脑选「A」还低
```

10 张图的题，模型比什么都不做的基线**低 29.5 个百分点**。
多期对比正是金融分析的真实工作流，这个上限很关键。

**最直觉的假设——视觉 token 预算耗尽——是错的。**
我们把 processor 的 `longest_edge` 扫了四档（2,939 条 × 4）：

```
   预算       单图tok   1图     2图    3-4图  5-6图  8图+   整体
   无上限        4281   74.7%  75.5%  62.1%  85.2%  38.8%  70.7%
   cap 1.0M       979   74.5%  74.8%  59.6%  70.4%  43.2%  70.0%
   cap 0.5M       461   72.5%  74.9%  56.3%  66.7%  43.9%  68.3%
   cap 0.25M      243   70.1%  69.7%  52.7%  59.3%  35.3%  64.6%
```

**「8图+」档收紧预算后反而变好**（38.8% → 43.9%），且曲线形状在四档下稳定
（相对单图：−35.9 / −31.4 / −28.6 / −34.8 pp）。所以瓶颈是模型整合多张图的能力，
不是它收到多少 token。

**一个实用的副产物：** `longest_edge = 1M` 是安全的工作点——
视觉 token 省 4.4 倍，精度只掉 0.7pp。低于 0.5M 后代价迅速上升。

> 一个值得知道的配置陷阱：`MAX_PIXELS` 是 **ms-swift** 的约定，
> transformers 的 processor **完全不读它**。processor 默认
> `longest_edge=16777216`，等于没有上限。如果你以为设了那个环境变量就限制了分辨率，
> 那大概率没有生效。

---

## 发现 4 — 换个措辞结论就变，而 raw accuracy 指向错误的方向

我们在 4 种提问措辞下重跑判断题（各 n = 2,804）：

```
  变体                                    raw    常数基线   macro   「否」召回
  base                                 77.5%     73.6%    66.3%     42.5%
  nobias（显式提示「不要默认答是」）        76.6%     73.6%    67.9%     49.3%
  correct_wrong（是/否 → 正确/错误）      76.5%     73.6%    73.0%     65.6%
  ab_format（改为 A/B 选项）             77.5%     73.6%    72.3%     61.3%
```

三条结论：

1. **少数类崩溃是模型的性质**——四种措辞下全部复现（「否」召回率都低于「是」22–48 pp）。
2. **但措辞的影响大到足以改变结论。** 仅把「是/否」换成「正确/错误」，
   少数类召回率就从 **42.5% 升到 65.6%**。
3. **raw accuracy 与 macro-recall 方向相反。** *每一个*变体的 raw 都不优于 base
   （配对净变化 −27 / −23 / 0），而 macro-recall 最多涨 6.7 pp。
   **按 raw accuracy 调 prompt，会选出最差的那一版。**

显式的去偏提示（`nobias`）反而让结果**更差**（净 −23）——
模型不是因为「不知道应该平衡」，而是**做不到**。

多选题对措辞不敏感（<1.5 pp）。**两类题型的评测可信度并不相同。**

---

## 发现 5 — 微调确实能修复少数类崩溃（而类别均衡没有帮助）

在 2,889 条 held-out test 样本上（按来源研报切分，与训练集零重叠）：

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/fig4_gain_concentration_dark.png">
  <img alt="增益分布：多数类 +1.5pp，少数类 +29.3pp" src="assets/fig4_gain_concentration.png">
</picture>

```
  条件                  多选raw   判断raw  判断macro   「否」召回 [95% CI]
  zero-shot（基座）     73.70%   75.23%    63.73%    39.7% [31.2, 48.8]
  LoRA natural         84.13%   84.01%    79.15%    69.0% [60.1, 76.7]
  LoRA balanced        82.78%   84.01%    79.43%    69.8% [60.9, 77.4]
```

逐样本配对比较（同一 `uid`）：

```
  zero-shot -> LoRA natural      翻对 430  翻错 136   净 +294   McNemar p < 0.001
  zero-shot -> LoRA balanced     翻对 428  翻错 167   净 +261   McNemar p < 0.001
```

**增益几乎全部落在少数类上：**

```
  「是」召回 (n=328)    87.8%  ->  89.3%     +1.5pp
  「否」召回 (n=116)    39.7%  ->  69.0%    +29.3pp      ← 差 20 倍
```

**负结果：** 类别均衡重采样（6 个「题型 × 标签」组合各上采样到 6,725 条）
**没有带来任何收益**——多选 raw 反而降了 1.35 pp（p = 0.024），少数类召回差异在噪声内。
这个崩溃**不是**训练数据标签不平衡造成的；按原始分布的普通 SFT 就能修好它。

---

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
python scripts/analyze_results.py --dir results/zeroshot
```

> `scripts/hf.sh` 是一个 wrapper，用来绕过我们机器上 SOCKS 代理与 httpx 的兼容问题。
> 你自己的环境多半不需要。

完整复现步骤见 `results/RESULTS.md`，其中记录了每个实验的细节，
包括 prompt 敏感性扫描、LoRA 两种条件、以及 token 预算扫描。

---

## 方法说明

### 为什么必须按来源研报切分

VisFinEval 是 benchmark，**没有 train split。** 它的 19,208 道题来自 3,318 份券商研报
（平均每份 5.8 条，最大的一份贡献 161 条）。若按样本随机切分，
**78%** 的训练样本其来源研报会同时出现在测试集——模型可能只是记住那份研报的排版和数值，
而不是学会读图。

我们按 `doc_key` 做 70/15/15 分组切分，保证同一研报的所有题落在同一侧，
并在管线里自动做泄漏自检。

### 为什么报三个指标，以及为什么用这些统计方法

```
raw accuracy    会被答案分布污染
macro-recall    各类召回率的平均，与分布无关
常数基线        及格线，不是零分线
```

少数类样本量小（test split 中「否」只有 116 条），因此比例估计用
**Wilson 置信区间**，条件间比较用 **McNemar 精确检验**（配对设计）。

---

## 已知限制

- 目前只测了 Qwen 系列；跨家族对比正在进行
- 单 seed，贪心解码（未估计采样方差）
- 多图题未做图像顺序打乱的鲁棒性检验
- `L2_Q1.tsv` 中 1,032 行 `answer` 为空已剔除；`L3_Q4.tsv` 字段结构不同未纳入
- **切分是我们自建的，不是 VisFinEval 官方的。** 这里的数字不能直接与论文或榜单比较，
  引用时必须声明

---

## 数据版权

⚠ **本仓库不托管 VisFinEval 的图像数据。** 它们来自券商研究报告，有版权。
请通过官方渠道获取：

- HuggingFace: [`SUFE-AIFLM-Lab/VisFinEval`](https://huggingface.co/datasets/SUFE-AIFLM-Lab/VisFinEval)
- GitHub: [SUFE-AIFLM-Lab/VisFinEval](https://github.com/SUFE-AIFLM-Lab/VisFinEval)

本仓库只发布**代码、切分逻辑与评测结果**。

---

## 许可证

[Apache-2.0](LICENSE) © 2026 Gavin C. Meng

## 引用

```bibtex
@misc{meng2026visfinevalrel,
  title  = {VisFinEval Reliability Audit: what reported accuracy hides},
  author = {Meng, Gavin C.},
  year   = {2026},
  url    = {https://github.com/gavinzsmeng/visfineval-reliability}
}
```

也请引用底层基准：

```bibtex
@inproceedings{visfineval2025,
  title     = {VisFinEval: A Scenario-Driven Chinese Multimodal Benchmark
               for Holistic Financial Understanding},
  booktitle = {EMNLP},
  year      = {2025}
}
```
