# FinVLM-Reliability

> **Beyond Raw Accuracy: Diagnostic Benchmarking & Robust Alignment for Financial VLMs**  
> *中文金融視覺大模型的可靠性診斷與防洩漏微調基準*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Base Model](https://img.shields.io/badge/Base%20Model-Qwen3--VL--8B-00A67E.svg)](https://github.com/QwenLM/Qwen3-VL)
[![Training Engine](https://img.shields.io/badge/Engine-ms--swift%204.6-FF6F00.svg)](https://github.com/modelscope/ms-swift)
[![Benchmark](https://img.shields.io/badge/Benchmark-VisFinEval%20(EMNLP'25)-792EE5.svg)](https://github.com/SUFE-AIFLM-Lab/VisFinEval)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

<p align="center">
  <a href="README.md">English</a> |
  <a href="README.zh-CN.md">简体中文</a> |
  <b>繁體中文</b> |
  <a href="README.ko.md">한국어</a> |
  <a href="README.ja.md">日本語</a>
</p>

---

## 📢 最新動態 (News)

- **[2026-09]** 發布微調配對評測結果：LoRA 微調使是非題少數類（「否」）召回率實現 **+29.3pp ~ +30.1pp** 的重大躍升（39.7% → 69.0% / 69.8%），逐樣本 McNemar 檢定達成 $p < 0.001$ 顯著性。
- **[2026-09]** 建立 0 洩漏研報分組切分體系（70/15/15），徹底消除原基準 78% 的跨集數據記憶風險。
- **[2026-09]** 完成 Qwen3-VL-8B-Instruct 全量 19,208 條零樣本評測、4 檔視覺 Token 預算掃描及 6 組 Prompt 魯棒性壓力測試。

---

## 🌟 核心亮點 (Key Highlights)

- **0 洩漏數據切分協議 (0-Leakage Protocol)**  
  VisFinEval 原生無訓練集劃分，單份研報最多衍生 161 道題目。若隨機切分，78% 的測試樣本其來源研報會提前出現在訓練集。本專案基於 `doc_key` 實現研報級分組隔離，達成嚴格的 0 洩漏契約。
- **穿透單一準確率假象 (Beyond Raw Accuracy)**  
  Qwen3-VL-8B 在是非題上 Raw Accuracy 達 77.46%，但少數類「否」的召回率僅 **39.7%**（顯著低於隨機拋硬幣 50%）。本專案強制同時報告常數及格線、Macro-Recall 與 95% Wilson 信賴區間。
- **長多圖推理崩潰邊界 (Multi-Image Breakdown Curve)**  
  量化模型在多圖上下文下的退化曲線：圖數 ≥8 時能力崩塌，10 張圖題目的準確率降至 29.5%，相對常數基準淨跌 29.5pp；並通過解析度消融證明其瓶頸在於跨圖注意力整合而非視覺 Token 預算。
- **配對閉環微調 (Rigorous Paired Alignment)**  
  基於 ms-swift 打造多卡 LoRA 管線，在完全相同的 2,889 條測試樣本上驗證微調增益，通過逐樣本 McNemar 精確檢定，證實微調增益 95% 集中於少數類。

---

## 🏗️ 系統架構 (System Architecture)

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

## 📊 基準與實驗結果 (Benchmark & Results)

所有條件均在完全相同的一批無洩漏測試集（Test Split: n=2,889）上執行，採用 Greedy 解碼，對齊全局 UID。

| 實驗條件 | 選擇題 Raw | 選擇題多圖題 Raw | 是非題 Raw | 是非題 Macro | 「否」召回率 (95% Wilson CI) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **常數及格線** (多數類盲猜) | 58.61% | -- | 73.87% | -- | -- |
| **Zero-shot** (基座模型) | 73.70% | 62.83% | 75.23% | 63.73% | 39.7% [31.2, 48.8] |
| **LoRA Natural** (原始分佈) | **84.13%** | **72.77%** | **84.01%** | 79.15% | 69.0% [60.1, 76.7] |
| **LoRA Balanced** (均衡上採樣) | 82.78% | 68.06% | **84.01%** | **79.43%** | **69.8% [60.9, 77.4]** |
| **淨增益 (Net Gain)** | **+10.43pp** | **+9.94pp** | **+8.78pp** | **+15.70pp** | **+30.1pp** |

---

## 🔬 核心診斷發現 (Core Findings)

### Finding 1 — 答案分佈導致 Raw Accuracy 嚴重失真

<p align="center">
  <img src="assets/1.png" width="95%" alt="Severely Skewed Answer Distribution in Multiple-Choice Questions">
</p>

在 VisFinEval 全量 **16,404** 道選擇題中，**58.5% 的真實答案是「A」**。  
一個完全不看圖、不做任何推理的模型，只要永遠輸出 `A`，就能拿到 **58.5% 的準確率**。  
因此，「常數基準」不是地板，而是**及格線**。任何匯報的成績都必須對照及格線閱讀：

| 指標 | 核心內涵 |
| :--- | :--- |
| **Raw Accuracy** | 傳統論文常報指標（易受答案分佈傾斜嚴重污染） |
| **Constant Baseline** | 永遠輸出多數類答案的基準 —— 及格線而非零分線 |
| **Macro-Recall** | 各類別獨立召回率的平均值 —— 剔除分佈污染的客觀能力 |

---

### Finding 2 — 模型在少數類上的表現顯著低於拋硬幣

<p align="center">
  <img src="assets/2.png" width="95%" alt="Few-shot classification is close to zero-shot; fine-tuning improves it significantly">
</p>

全量是非題看起來是最好的題型（Raw Accuracy 77.46%），但拆開看：

```
  真值     樣本量(n)     佔比        召回率
   是       2,063       73.6%       90.0%
   否         741       26.4%       42.5%    [95% CI 39.0–46.1]   ← 顯著低於隨機猜(50%)
```

Raw 與 Macro-Recall 相差 **11.20 個百分點**（全數據集最大）。模型在是非題上高度趨向退化為「全選是」（回答了 520 次「否」，僅答對 315 次）。

---

### Finding 3 — 多圖推理崩潰是跨圖整合上限，而非 Token 預算不足

<p align="center">
  <img src="assets/3.png" width="95%" alt="Performance Drop with Multiple Images is Not a Token Budget Problem">
</p>

準確率隨題目包含的圖表數量呈現單調劇烈下滑：

```
   圖表數     樣本數 (n)      Raw Accuracy      常數基準      相對基準淨增益
     2           666            75.5%            45.8%          +29.7pp
     4           227            61.7%            29.5%          +32.2pp
     8            60            50.0%            56.7%           -6.7pp  (跌破基準)
    10            78            29.5%            59.0%          -29.5pp  (崩潰)
```

在 10 張圖的題目中，模型得分比「盲選 A」低 29.5 個百分點。多期報表對比是金融分析的核心場景，該瓶頸極為關鍵。

**直覺假設（記憶體或視覺 Token 不足導致的截斷）被實驗否定。**  
我們通過調節處理器 `longest_edge` 對視覺 Token 預算進行了 4 檔掃描（2,939 樣本 × 4 檔）：

```
   Token 預算     tok/img    1圖     2圖    3-4圖   5-6圖   8+圖    整體表現
   無上限 (no cap)  4281    74.7%   75.5%   62.1%   85.2%   38.8%    70.7%
   上限 1.0M         979    74.5%   74.8%   59.6%   70.4%   43.2%    70.0%
   上限 0.5M         461    72.5%   74.9%   56.3%   66.7%   43.9%    68.3%
   上限 0.25M        243    70.1%   69.7%   52.7%   59.3%   35.3%    64.6%
```

在 8+ 張圖區間，壓縮 Token 預算後準確率反而略微提升（38.8% → 43.9%），且 4 檔預算下的退化曲線形態高度一致（相對單圖落後 −35.9 / −31.4 / −28.6 / −34.8 pp）。  
**結論**：多圖崩潰的根因在於模型跨圖注意力表徵能力的極限，而非輸入 Token 數量。  
**工程參考**：`longest_edge = 1M` 是最佳操作點，在僅損失 0.7pp 精度下減少了 4.4 倍視覺 Token。

---

### Finding 4 — 評測結論極度依賴措辭，Raw 指標甚至給出錯誤導向

在 4 種 Prompt 變體下重複是非題實驗（各 n = 2,804）：

```
  變體                                     Raw     常數基準    Macro     「否」召回率
  base (基準默認措辭)                     77.5%     73.6%     66.3%        42.5%
  nobias (顯式提示「不要默認選是」)         76.6%     73.6%     67.9%        49.3%
  correct_wrong (將「是/否」改為「正確/錯誤」) 76.5%     73.6%     73.0%        65.6%
  ab_format (轉換為 A/B 選項題型)         77.5%     73.6%     72.3%        61.3%
```

1. **少數類崩潰是固有屬性**：4 個變體下「否」召回率均低於「是」22~48 個百分點。
2. **措辭可大幅改變結論**：僅將「是/否」替換為「正確/錯誤」，少數類召回率即從 **42.5% 躍升至 65.6%**。
3. **Raw 與 Macro 方向相反**：所有變體的 Raw 均不優於 base（淨翻對/錯為 −27 / −23 / 0），但 Macro-Recall 最大提升 6.7pp。**若僅以 Raw Accuracy 為導向優化 Prompt，會選出對少數類最差的版本。**
4. **顯式提示失效**：`nobias` 指令反而帶來淨變化 −23，證明模型並非不知道要平衡，而是受限於隱空間計算能力。  
   選擇題對措辭極不敏感（<1.5pp），兩類題型的評測魯棒性存在本質差異。

---

### Finding 5 — 微調能夠修復少數類崩潰，且增益幾乎全部集中於少數類

在獨立測試集（n=2,889，按來源研報切分，與訓練集 0 重疊）上的實驗：

<p align="center">
  <img src="assets/4.png" width="95%" alt="Fine-tuning brings large gains on the minority class, with minimal change on the majority class">
</p>

```
  實驗條件              選擇題 Raw   是非題 Raw   是非題 Macro   「否」召回率 [95% CI]
  Zero-shot (基座)      73.70%      75.23%       63.73%        39.7% [31.2, 48.8]
  LoRA Natural          84.13%      84.01%       79.15%        69.0% [60.1, 76.7]
  LoRA Balanced         82.78%      84.01%       79.43%        69.8% [60.9, 77.4]
```

逐樣本配對比較（基於全局統一 `uid`）：

```
  Zero-shot -> LoRA Natural     翻對 430 條   翻錯 136 條   淨變化 +294 條   McNemar p < 0.001 (極顯著)
  Zero-shot -> LoRA Balanced    翻對 428 條   翻错 167 條   淨變化 +261 條   McNemar p < 0.001 (極顯著)
```

**增益高度集中於少數類（增益相差 20 倍）：**

```
  「是」召回率 (n=328)    87.8%  ->  89.3%     +1.5pp
  「否」召回率 (n=116)    39.7%  ->  69.0%    +29.3pp      ← 20 倍增益集中度
```

**Negative Finding 消融結論**：標籤均衡重採樣（Balanced）並未帶來額外顯著收益 —— 選擇題 Raw Accuracy 下降了 1.35pp（p = 0.024），少數類召回率僅提升 0.8pp。說明少數類崩潰並非由訓練樣本不均衡引起，常規 Natural 分佈下的監督微調已足夠激活模型的圖表辨別能力。

---

## 🚀 快速開始 (Quick Start)

### 1. 環境準備 (~10 分鐘)
```bash
git clone https://github.com/gavinzsmeng/visfineval-reliability.git
cd visfineval-reliability
bash scripts/setup_env.sh
```

### 2. 下載數據與基座權重
```bash
bash scripts/hf.sh download SUFE-AIFLM-Lab/VisFinEval --repo-type dataset --local-dir data/VisFinEval
bash scripts/hf.sh download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct
7z x data/VisFinEval/data.7z -odata/VisFinEval/
```

### 3. 數據無洩漏切分
```bash
python scripts/prepare_data.py --out data/swift     --balance natural
python scripts/prepare_data.py --out data/swift_bal --balance balanced
```

### 4. 啟動多卡 LoRA 微調
```bash
bash scripts/run_lora.sh natural     # 原始分佈微調 (~1.6h)
bash scripts/run_lora.sh balanced    # 標籤均衡微調 (~4.9h)
```

### 5. 訓練後自動化配對評測
```bash
bash scripts/run_posttrain_eval.sh
```

---

## 🛠️ 代碼結構 (Codebase Structure)

```
├── configs/                  # 分布式與微調配置 (DeepSpeed ZeRO-2 等)
├── data/                     # 數據切分產物 (研報級隔離，不託管圖像原圖)
├── models/                   # 基座模型存放目錄 (Qwen3-VL-8B-Instruct)
├── output/                   # 訓練 Checkpoints 與 TensorBoard 日誌
├── results/                  # 評測原始預測與全景指標匯總
│   ├── RESULTS.md            # 詳細實驗記錄與消融報告
│   ├── zeroshot/             # 零樣本基準預測
│   ├── posttrain_*/          # 微調模型評測產物
│   └── sensitivity/          # Prompt 敏感性測試產物
├── scripts/                  # 核心流水線腳本
│   ├── prepare_data.py       # 防洩漏分組切分
│   ├── eval_baseline.py      # 多卡評測 Harness
│   ├── run_lora.sh           # LoRA 微調啟動器
│   ├── run_posttrain_eval.sh # 訓練後評測編排腳本
│   └── compare_conditions.py # McNemar 配對檢定與 Wilson 信賴區間
└── requirements.lock.txt     # 鎖定的環境依賴清單
```

---

## 📜 版權聲明 (Copyright & Legal Notice)

⚠ **本倉庫不重新分發 VisFinEval 的圖像資產。**  
測試圖片源於各大券商公開研究報告，版權歸原作者所有。請根據學術許可自官方管道獲取：
- HuggingFace: [SUFE-AIFLM-Lab/VisFinEval](https://huggingface.co/datasets/SUFE-AIFLM-Lab/VisFinEval)
- GitHub: [SUFE-AIFLM-Lab/VisFinEval](https://github.com/SUFE-AIFLM-Lab/VisFinEval)

---

## 📑 引用 (Citation)

```bibtex
@misc{meng2026visfinevalrel,
  title  = {VisFinEval Reliability Audit: what reported accuracy hides},
  author = {Meng, Gavin C.},
  year   = {2026},
  url    = {https://github.com/gavinzsmeng/visfineval-reliability}
}
```

底層 VisFinEval 基準請引用原論文：

```bibtex
@inproceedings{visfineval2025,
  title     = {VisFinEval: A Scenario-Driven Chinese Multimodal Benchmark for Holistic Financial Understanding},
  booktitle = {EMNLP},
  year      = {2025}
}
```

---

## 📄 開源許可 (License)

本專案代碼遵循 [Apache-2.0](LICENSE) 開源協議。
