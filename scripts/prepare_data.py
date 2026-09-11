#!/usr/bin/env python
"""
VisFinEval -> ms-swift 训练数据
════════════════════════════════════════════════════════════════════

## 为什么必须按「来源研报」切分

VisFinEval 是 benchmark，**没有 train split**。19,208 条样本来自 3,318 份券商研报，
平均每份 5.8 条 QA，最大的一份贡献 161 条。

若按样本随机切分：
    78% 的训练样本，其来源研报会同时出现在测试集
    -> 模型可能只是记住该研报的排版/数值，而非学会读图（严重泄漏）

所以按 doc_key 分组切分，保证同一研报的所有 QA 落在同一侧。

## 两种训练条件（对应项目的核心问题）

  natural  按原始分布训练。真值 A 占 58.5% -> 预期会加剧少数类崩溃
  balanced 按标签均衡采样。用 raw accuracy 换 macro-recall
           这一组是「微调能否修复少数类崩溃」的直接检验

## 输出（ms-swift 标准格式）

  {"messages": [{"role":"user","content":"<image>...问题..."},
                {"role":"assistant","content":"A"}],
   "images": ["/abs/path.jpg"]}

多图题：content 里按顺序放 N 个 <image>，images 列表对应 N 个路径。

用法:
  python scripts/prepare_data.py --out data/swift --balance natural
  python scripts/prepare_data.py --out data/swift_bal --balance balanced
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_baseline import QTYPE_MC, QTYPE_TF, load_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def doc_key(image_rel: str) -> str:
    """从图片路径提取「来源研报」标识。

    形如: 2024-07-03--联储证券-2024年中宏观展望：固本培元，如日之升_page8_图12.jpg
      ->  2024-07-03--联储证券-2024年中宏观展望：固本培元，如日之升
    也处理无日期前缀的: 酿酒行业-今世缘-2万吨产能..._page2_财务预测与估值.jpg
    """
    name = image_rel.split("/")[-1]
    name = re.sub(r"\.(jpg|jpeg|png|JPG|JPEG|PNG)$", "", name)
    name = re.sub(r"_page\d+.*$", "", name)
    return name


def build_prompt(row) -> str:
    """与评测时**完全一致**的 prompt —— 训练/评测口径必须对齐。"""
    if row["qtype"] == QTYPE_MC:
        o = row["options"]
        return (
            "请仔细阅读图片，回答问题，只输出正确选项的字母（A/B/C/D），不要解释。\n\n"
            f"问题：{row['question']}\n"
            f"A. {o.get('A') or '-'}\nB. {o.get('B') or '-'}\n"
            f"C. {o.get('C') or '-'}\nD. {o.get('D') or '-'}\n\n答案："
        )
    return (
        "请仔细阅读图片，回答问题，只回答「是」或「否」，不要解释。\n\n"
        f"问题：{row['question']}\n\n答案："
    )


def to_swift_record(row) -> dict:
    """转成 ms-swift 标准格式。<image> 占位符按原顺序，与 images 列表一一对应。"""
    n = len(row["abs_images"])
    content = "<image>" * n + build_prompt(row)
    return {
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": row["gold"]},
        ],
        "images": list(row["abs_images"]),
        "_meta": {
            "uid": int(row["uid"]), "qtype": row["qtype"],
            "src": row["src"], "doc": row["doc"], "num_images": int(row["num_images"]),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/swift", help="输出目录")
    ap.add_argument("--ratio", default="0.70,0.15,0.15", help="train,val,test 比例")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--balance", default="natural", choices=["natural", "balanced"],
                    help="natural=原始分布; balanced=按标签均衡采样（只用 train）")
    args = ap.parse_args()

    r_tr, r_va, r_te = [float(x) for x in args.ratio.split(",")]
    assert abs(r_tr + r_va + r_te - 1.0) < 1e-6, "比例之和必须为 1"

    print("═" * 64)
    print(" VisFinEval -> ms-swift")
    print("═" * 64)

    # uid 由 load_dataset() 统一分配（load 顺序）—— 不要在这里重编，
    # 否则会与 eval_baseline.py 的 uid 不一致，导致 test split 筛错样本。
    df = load_dataset()
    df["doc"] = df["images"].apply(lambda ps: doc_key(ps[0]))
    print(f"\n  样本 {len(df)}  来源研报 {df['doc'].nunique()}  条件 {args.balance}")

    # ── 按 doc 分组切分 ──────────────────────────────────
    docs = np.array(sorted(df["doc"].unique()))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(docs)
    n = len(docs)
    i1, i2 = int(n * r_tr), int(n * (r_tr + r_va))
    split_of = {}
    for i, d in enumerate(docs):
        split_of[d] = "train" if i < i1 else ("val" if i < i2 else "test")

    df["split"] = df["doc"].map(split_of)

    # ── 泄漏自检（这是本脚本存在的主要理由）────────────────
    print("\n── 泄漏自检 ──")
    tr_docs = set(df[df.split == "train"]["doc"])
    va_docs = set(df[df.split == "val"]["doc"])
    te_docs = set(df[df.split == "test"]["doc"])
    leaks = (tr_docs & te_docs) | (tr_docs & va_docs) | (va_docs & te_docs)
    if leaks:
        sys.exit(f"  ✗ 发现 {len(leaks)} 份研报跨 split，切分逻辑有误")
    print(f"  ✓ 研报无重叠: train {len(tr_docs)} / val {len(va_docs)} / test {len(te_docs)}")
    print("  ✓ 同一研报的所有 QA 都落在同一侧")

    # ── balanced 条件：train 上按标签均衡采样 ──────────────
    train = df[df.split == "train"]
    if args.balance == "balanced":
        # 以「(题型, 标签)」为单元向上采样到该类组合的最大值
        counts = train.groupby(["qtype", "gold"]).size()
        target = counts.max()
        parts = []
        for (qt, g), grp in train.groupby(["qtype", "gold"]):
            parts.append(grp.sample(n=target, replace=True,
                                    random_state=args.seed) if len(grp) < target else grp)
        train = pd.concat(parts, ignore_index=True)
        print(f"\n  balanced: train {len(df[df.split=='train'])} -> {len(train)} 条")
        print(f"    每类目标 {target} 条，少数类向上采样 (replace=True)")

    # ── 写出 ────────────────────────────────────────────
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split, sub in [("train", train), ("val", df[df.split == "val"]),
                       ("test", df[df.split == "test"])]:
        path = out / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for _, row in sub.iterrows():
                f.write(json.dumps(to_swift_record(row), ensure_ascii=False) + "\n")
        counts[split] = len(sub)
        dist = Counter(zip(sub["qtype"], sub["gold"]))
        top = "  ".join(f"{a}/{b}:{c}" for (a, b), c in dist.most_common(6))
        print(f"  {split:<6} {len(sub):>6} 条 -> {path.name}")
        print(f"         {top}")

    # ── 落盘切分信息（可复现）────────────────────────────
    meta = {
        "seed": args.seed, "ratio": args.ratio, "balance": args.balance,
        "n_samples": len(df), "n_docs": int(df["doc"].nunique()),
        "counts": counts,
        "doc_split_counts": {"train": len(tr_docs), "val": len(va_docs), "test": len(te_docs)},
        "note": "按来源研报分组切分，同一研报的 QA 不跨 split。"
                "VisFinEval 无官方 train split，此切分为本项目自建，报告结果时必须声明。",
    }
    (out / "split_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 单独落盘 test 的 uid 列表（全局 uid，跨条件一致）。
    # 用途：训练后评测按同一批样本对比，能做逐样本配对检验，
    #       而不是比较两个不同子集的均值。
    (out / "test_uids.json").write_text(
        json.dumps(sorted(int(x) for x in df[df.split == "test"]["uid"].tolist())),
        encoding="utf-8")
    print(f"  ✓ test uid 列表 -> {out/'test_uids.json'}  ({len(df[df.split=='test'])} 个)")

    print(f"\n  ✓ 切分信息 -> {out/'split_meta.json'}")
    print(f"\n  ⚠ 提醒: VisFinEval 无官方 train split，此为自建切分。")
    print(f"     论文/SOTA 数字不可直接比较，必须声明切分方式。")


if __name__ == "__main__":
    main()
