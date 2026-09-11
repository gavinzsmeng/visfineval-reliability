# FinVLM-Reliability

> **Beyond Raw Accuracy: Diagnostic Benchmarking & Robust Alignment for Financial VLMs**  
> *金融マルチモーダル基盤モデルにおける信頼性診断とデータリーク防止ファインチューニング基準*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Base Model](https://img.shields.io/badge/Base%20Model-Qwen3--VL--8B-00A67E.svg)](https://github.com/QwenLM/Qwen3-VL)
[![Training Engine](https://img.shields.io/badge/Engine-ms--swift%204.6-FF6F00.svg)](https://github.com/modelscope/ms-swift)
[![Benchmark](https://img.shields.io/badge/Benchmark-VisFinEval%20(EMNLP'25)-792EE5.svg)](https://github.com/SUFE-AIFLM-Lab/VisFinEval)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

<p align="center">
  <a href="README.md">English</a> |
  <a href="README.zh-CN.md">简体中文</a> |
  <a href="README.zh-TW.md">繁體中文</a> |
  <a href="README.ko.md">한국어</a> |
  <b>日本語</b>
</p>

---

## 📢 ニュース (News)

- **[2026-09]** 訓練後評価レポート公開：LoRA微調により真偽判定問題の少数派クラス（「いいえ」）再現率が **+29.3pp 〜 +30.1pp** 大幅向上（39.7% → 69.0% / 69.8%）、サンプル単位McNemar検定で $p < 0.001$ の統計的有意性を達成。
- **[2026-09]** リークゼロ（0-Leakage）レポート単位グループ分割体系（70/15/15）を構築、元ベンチマークの78%に及ぶ交差記憶リスクを完全排除。
- **[2026-09]** Qwen3-VL-8B-Instructによる全19,208問ゼロショット評価、4段階視覚トークン予算スキャン、および6通りのプロンプト堅牢性ストレステストを完遂。

---

## 🌟 主なハイライト (Key Highlights)

- **リークゼロ・データ分割プロトコル (0-Leakage Protocol)**  
  VisFinEvalには訓練用分割がなく、全19,208問が3,318件の証券会社レポートから構成されています。ランダム分割ではテストサンプルの78%が事前に学習されてしまうため、`doc_key`に基づくレポート単位の物理隔離分割を導入。
- **見かけの正解率を超えた厳密な診断 (Beyond Raw Accuracy)**  
  Qwen3-VL-8Bは真偽判定問題で見かけの正解率（Raw Accuracy）77.46%を記録しますが、少数派クラス（「いいえ」）に対する再現率はわずか **39.7%**（コイントス50%を有意に下回る）。本プロジェクトでは多数派定数基準、マクロ再現率（Macro-Recall）、95% Wilson信頼区間を併せて報告。
- **複数画像推論の崩壊境界 (Multi-Image Breakdown Curve)**  
  画像数の増加に伴う急激な性能低下曲線を定量化：画像数8枚以上で能力が崩壊し、10枚の問題では正解率29.5%（多数派当てずっぽう基準より29.5pp低下）。解像度アブレーションにより、このボトルネックがトークン予算不足ではなく画像間クロスアテンションの統合限界であることを実証。
- **厳密な対応のあるファインチューニング (Rigorous Paired Alignment)**  
  ms-swiftを用いたマルチGPU LoRAパイプラインを構築。完全に同一の固定テストセット（2,889問）においてMcNemar検定を実施し、微調による利得の95%以上が崩壊していた少数派クラスの改善に集中していることを証明。

---

## 🏗️ システムアーキテクチャ (System Architecture)

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

## 📊 ベンチマークおよび実験結果 (Benchmark & Results)

全条件は同一の独立テストセット（n=2,889）、Greedyデコード、グローバルUIDで評価されています。

| 実験条件 | 選択式 Raw | 選択式(複数画像) | 真偽判定 Raw | 真偽判定 Macro | 「いいえ」再現率 (95% Wilson CI) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **定数基準** (多数派推測) | 58.61% | -- | 73.87% | -- | -- |
| **Zero-shot** (ベースモデル) | 73.70% | 62.83% | 75.23% | 63.73% | 39.7% [31.2, 48.8] |
| **LoRA Natural** (自然分布) | **84.13%** | **72.77%** | **84.01%** | 79.15% | 69.0% [60.1, 76.7] |
| **LoRA Balanced** (均衡サンプリング) | 82.78% | 68.06% | **84.01%** | **79.43%** | **69.8% [60.9, 77.4]** |
| **純向上 (Net Gain)** | **+10.43pp** | **+9.94pp** | **+8.78pp** | **+15.70pp** | **+30.1pp** |

---

## 🔬 主な診断結果 (Core Diagnostic Findings)

### Finding 1 — 回答分布の偏りによる正解率の歪み

<p align="center">
  <img src="assets/1.png" width="95%" alt="Severely Skewed Answer Distribution in Multiple-Choice Questions">
</p>

VisFinEvalの全 **16,404** 問の選択問題において、**58.5% の正解が「A」**です。  
図を見ず問題も読まず、単に `A` を出力し続けるだけのモデルでも **58.5% の正解率**を獲得できます。  
したがって、定数基準は最低点ではなく**合格基準点（Passing Grade）**です。

| 指標 | 定義・本質 |
| :--- | :--- |
| **Raw Accuracy** | 従来のベンチマークが報告する正解率（偏りに大きく汚染される） |
| **Constant Baseline** | 常に多数派クラスを出力する基準値（最低合格ライン） |
| **Macro-Recall** | クラス別再現率の平均値（分布の偏りに影響されない客観的能力） |

---

### Finding 2 — 少数派クラスにおけるコイントス以下の性能崩壊

<p align="center">
  <img src="assets/2.png" width="95%" alt="Few-shot classification is close to zero-shot; fine-tuning improves it significantly">
</p>

真偽判定問題は一見最も高い正解率（Raw 77.46%）を示しますが、ラベル別に分解すると：

```
  正解      サンプル数(n)     割合        再現率
  はい         2,063        73.6%       90.0%
いいえ           741        26.4%       42.5%    [95% CI 39.0–46.1]   ← コイントス(50%)を有意に下回る
```

Raw正解率とMacro再現率の乖離は **11.20pp**（全データセット中最大）に達し、モデルが多数派クラス（「はい」）に極度に退行している実態が明らかになりました。

---

### Finding 3 — 複数画像の性能崩壊はトークン制限ではなくモデルの統合限界

<p align="center">
  <img src="assets/3.png" width="95%" alt="Performance Drop with Multiple Images is Not a Token Budget Problem">
</p>

含まれる画像数の増加に伴い、正解率は単調かつ急激に低下します：

```
   画像数     サンプル数 (n)     Raw Accuracy     定数ベースライン     純向上マージン
     2           666            75.5%             45.8%            +29.7pp
     4           227            61.7%             29.5%            +32.2pp
     8            60            50.0%             56.7%             -6.7pp (基準割れ)
    10            78            29.5%             59.0%            -29.5pp (崩壊)
```

10枚の複合画像問題では、正解率は「常にAと答える」より29.5pp低くなります。

**視覚トークン予算不足という直感的仮説は実験により否定されました。**  
プロセッサの `longest_edge` を用いて4段階のトークン予算スキャン（2,939問 × 4段階）を実施：

```
   トークン予算       tok/画像    1枚     2枚    3-4枚   5-6枚   8枚+    全体
   無制限 (no cap)     4281    74.7%   75.5%   62.1%   85.2%   38.8%   70.7%
   上限 1.0M            979    74.5%   74.8%   59.6%   70.4%   43.2%   70.0%
   上限 0.5M            461    72.5%   74.9%   56.3%   66.7%   43.9%   68.3%
   上限 0.25M           243    70.1%   69.7%   52.7%   59.3%   35.3%   64.6%
```

8枚以上の区間では、予算を絞った方がむしろ正解率が向上（38.8% → 43.9%）し、4本の曲線は平行に下降します。  
**結論**：崩壊の本質はトークン数ではなく、複数画像を横断するクロスアテンション統合能力の限界です。  
`longest_edge = 1M` はわずか0.7ppの低下でトークンを4.4倍節約できる最適動作点です。

---

### Finding 4 — プロンプト言い回しへの過敏性とRaw正解率の誤導

4種類のプロンプト表現で真偽判定問題を再評価した結果（各 n = 2,804）：

```
  バリアント                               Raw     定数基準    Macro    「いいえ」再現率
  base (デフォルト表現)                  77.5%     73.6%     66.3%        42.5%
  nobias (「安易にはいを選ばない」明示)    76.6%     73.6%     67.9%        49.3%
  correct_wrong (「はい/いいえ」→「正/誤」) 76.5%     73.6%     73.0%        65.6%
  ab_format (A/B選択肢形式に変換)         77.5%     73.6%     72.3%        61.3%
```

単に「はい/いいえ」を「正/誤」に置き換えるだけで、少数派再現率は **42.5% から 65.6% へ急増**します。  
見かけのRaw正解率だけを基準にプロンプトを最適化すると、少数派クラスの性能が最も低いバージョンを選んでしまう危険があります。

---

### Finding 5 — ファインチューニングによる少数派崩壊の修復と効果の集中度

独立テストセット（n=2,889）での評価結果：

<p align="center">
  <img src="assets/4.png" width="95%" alt="Fine-tuning brings large gains on the minority class, with minimal change on the majority class">
</p>

```
  実験条件              選択式 Raw   真偽判定 Raw   真偽判定 Macro   「いいえ」再現率 [95% CI]
  Zero-shot (ベース)    73.70%       75.23%         63.73%          39.7% [31.2, 48.8]
  LoRA Natural          84.13%       84.01%         79.15%          69.0% [60.1, 76.7]
  LoRA Balanced         82.78%       84.01%         79.43%          69.8% [60.9, 77.4]
```

サンプル単位McNemar対応のある検定：
- Zero-shot → LoRA Natural: 不正解から正解へ 430件、正解から不正解へ 136件、純改善 **+294件** ($p < 0.001$, 極めて有意)
- 多数派（「はい」）再現率: 87.8% → 89.3% (+1.5pp)
- 少数派（「いいえ」）再現率: 39.7% → 69.0% (**+29.3pp, 20倍の集中度**)

**Negative Finding**: クラス均衡リサンプリング（Balanced）は追加の優位性をもたらしませんでした（選択式Rawが1.35pp低下）。自然分布での一般的な教師ありファインチューニングで十分であることが示されました。

---

## 🚀 クイックスタート (Quick Start)

```bash
# 1. 環境構築 (~10分)
git clone https://github.com/gavinzsmeng/visfineval-reliability.git
cd visfineval-reliability
bash scripts/setup_env.sh

# 2. データと重みのダウンロード
bash scripts/hf.sh download SUFE-AIFLM-Lab/VisFinEval --repo-type dataset --local-dir data/VisFinEval
bash scripts/hf.sh download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct
7z x data/VisFinEval/data.7z -odata/VisFinEval/

# 3. リークゼロ・データセット分割
python scripts/prepare_data.py --out data/swift     --balance natural
python scripts/prepare_data.py --out data/swift_bal --balance balanced

# 4. マルチGPU LoRA微調の実行
bash scripts/run_lora.sh natural
bash scripts/run_lora.sh balanced

# 5. 事後評価とペア検定の自動実行
bash scripts/run_posttrain_eval.sh
```

---

## 🛠️ コード構成 (Codebase Structure)

```
├── configs/                  # 分散訓練およびDeepSpeed設定ファイル
├── data/                     # レポート単位の隔離データセット
├── models/                   # ベースモデル格納先 (Qwen3-VL-8B-Instruct)
├── output/                   # 学習済みチェックポイントとログ
├── results/                  # 推論予測結果 (jsonl) およびメトリクス集計
│   ├── RESULTS.md            # 詳細な実験記録とアブレーション結果
│   ├── zeroshot/             # ゼロショット予測
│   ├── posttrain_*/          # 微調後モデル評価結果
│   └── sensitivity/          # プロンプト頑健性テスト結果
├── scripts/                  # 実行パイプラインスクリプト
│   ├── prepare_data.py       # リーク防止分割・フォーマット
│   ├── eval_baseline.py      # マルチGPU評価ハーネス
│   ├── run_lora.sh           # LoRA実行スクリプト
│   ├── run_posttrain_eval.sh # 訓練後評価オーケストレータ
│   └── compare_conditions.py # McNemar検定およびWilson信頼区間
└── requirements.lock.txt     # 固定された依存関係リスト
```

---

## 📜 著作権および利用規約 (Copyright & Notice)

⚠ **本リポジトリはVisFinEvalの元画像ファイルを直接配布しません。**  
画像は各証券会社の公開レポートに由来し、著作権は各著作者に帰属します。公式チャンネルより取得してください：
- HuggingFace: [SUFE-AIFLM-Lab/VisFinEval](https://huggingface.co/datasets/SUFE-AIFLM-Lab/VisFinEval)
- GitHub: [SUFE-AIFLM-Lab/VisFinEval](https://github.com/SUFE-AIFLM-Lab/VisFinEval)

本リポジトリは **コード、データ分割ロジック、および評価結果のみを公開します。**

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

元文献のVisFinEval引用：

```bibtex
@inproceedings{visfineval2025,
  title     = {VisFinEval: A Scenario-Driven Chinese Multimodal Benchmark for Holistic Financial Understanding},
  booktitle = {EMNLP},
  year      = {2025}
}
```

---

## 📄 ライセンス (License)

本プロジェクトは [Apache-2.0](LICENSE) ライセンスの下で公開されています。
