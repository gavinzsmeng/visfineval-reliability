#!/usr/bin/env python
"""
结果分析 —— 分选项拆解 + 偏差诊断
════════════════════════════════════════════════════════════════════

为什么需要这个脚本
──────────────────
raw accuracy 会被答案分布严重污染。举个极端例子：
  真值 90% 是 A，模型全选 A -> raw accuracy 90%，但它什么都没学会。

所以本脚本额外报告：
  1. **平衡准确率 (macro-recall)** —— 每类召回率的平均，不受分布影响
  2. **常数基线** —— 及格线
  3. **预测分布 vs 真值分布** —— 看模型是否在"逆着偏差走"

用法
────
  python scripts/analyze_results.py --dir results/zeroshot
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def load(dirpath: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(dirpath / "predictions.shard*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"✗ {dirpath} 下没有预测文件")
    df = pd.DataFrame(rows)
    # 去重（uid 优先，兼容旧文件用 文件名:idx）
    if "uid" not in df.columns:
        df["uid"] = df.index.astype(str)
    df = df.drop_duplicates(subset=["uid"], keep="first").reset_index(drop=True)
    return df


def report(tag: str, sub: pd.DataFrame, labels: str):
    """打印一个子集的完整诊断。"""
    if len(sub) == 0:
        return
    n = len(sub)
    acc = float((sub["pred"] == sub["gold"]).mean())

    print(f"══ {tag}  (n={n}) ══")
    print(f"  raw accuracy = {acc*100:.2f}%")
    print()
    print(f"  {'真值':<4}{'n':>7}{'占比':>8}{'召回率':>9}    预测去向")
    print(f"  {'-'*58}")

    recalls = []
    for lb in labels:
        cls = sub[sub["gold"] == lb]
        if len(cls) == 0:
            continue
        rec = float((cls["pred"] == lb).mean())
        recalls.append(rec)
        dest = Counter(cls["pred"])
        s = "  ".join(f"{k}:{v}" for k, v in dest.most_common(4))
        print(f"  {lb:<4}{len(cls):>7}{len(cls)/n*100:>7.1f}%{rec*100:>8.1f}%    {s}")

    bal = sum(recalls) / len(recalls) if recalls else 0.0
    print(f"  {'-'*58}")
    print(f"  平衡准确率 (macro-recall) = {bal*100:.2f}%   <- 不受答案分布污染")
    print(f"  raw - 平衡 = {(acc-bal)*100:+.2f} pp   <- 差值越大，说明越靠猜多数类")

    gd = Counter(sub["gold"])
    pd_ = Counter(sub["pred"])
    print()
    print("  真值分布: " + "  ".join(f"{lb}:{gd.get(lb,0)/n*100:5.1f}%" for lb in labels))
    print("  预测分布: " + "  ".join(f"{lb}:{pd_.get(lb,0)/n*100:5.1f}%" for lb in labels))

    # 偏差方向：模型给多数类的比例 vs 真值给多数类的比例
    maj = max(labels, key=lambda lb: gd.get(lb, 0))
    delta = (pd_.get(maj, 0) - gd.get(maj, 0)) / n * 100
    direction = "逆着偏差走" if delta < 0 else "顺着偏差走"
    print(f"  偏差诊断: 多数类「{maj}」 真值{gd.get(maj,0)/n*100:.1f}% "
          f"-> 模型{pd_.get(maj,0)/n*100:.1f}%  ({delta:+.1f}pp, {direction})")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/zeroshot")
    args = ap.parse_args()

    df = load(Path(args.dir))
    print(f"载入 {len(df)} 条预测\n")

    mc = df[df["qtype"] == "mc"]
    tf = df[df["qtype"] == "tf"]

    report("多选题 · 全部", mc, "ABCD")
    report("多选题 · 单图", mc[mc["num_images"] == 1], "ABCD")
    report("多选题 · 多图", mc[mc["num_images"] > 1], "ABCD")
    report("判断题 · 全部", tf, "是否")
    report("总体", df, "ABCD是否")

    # 多图题按图数量拆
    multi = mc[mc["num_images"] > 1]
    if len(multi):
        print("══ 多图题：图数量 vs 准确率 ══")
        print(f"  {'图数':>4}{'n':>7}{'raw':>9}{'常数基线':>11}{'增益':>10}")
        for k in sorted(multi["num_images"].unique()):
            s = multi[multi["num_images"] == k]
            a = float((s["pred"] == s["gold"]).mean())
            b = Counter(s["gold"]).most_common(1)[0][1] / len(s)
            print(f"  {k:>4}{len(s):>7}{a*100:>8.1f}%{b*100:>10.1f}%{(a-b)*100:>+9.1f}pp")
        print()

    # 无法解析的输出
    bad = mc[mc["pred"] == "?"]
    if len(bad):
        print(f"══ 无法解析的输出 (n={len(bad)}, {len(bad)/len(mc)*100:.2f}%) ══")
        for k, v in Counter(bad["raw_output"].str.strip().str[:40]).most_common(8):
            print(f"  {v:>4}x  {k!r}")
        print()


if __name__ == "__main__":
    main()
