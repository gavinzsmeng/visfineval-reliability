#!/usr/bin/env python
"""
Prompt 敏感性分析
════════════════════════════════════════════════════════════════════

回答一个问题:
  baseline 上观察到的失败模式（判断题少数类「否」召回率仅 42.5%）
  是模型的性质，还是单一 prompt 措辞造成的假象？

判据:
  • 所有变体下失败模式都复现  -> 是模型性质，baseline 结论成立
  • 某个变体大幅改善          -> 之前的结论是 prompt artifact，必须修正

做法:
  1. 对每个变体算 raw / 常数基线 / macro-recall / 各标签召回率
  2. 用 uid 做**逐样本配对比较**（同一样本在不同 prompt 下是否翻盘）
     —— 这比比较总体均值更严格，能排除样本难度差异
"""
from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SENS = ROOT / "results" / "sensitivity"

VARIANT_LABEL = {
    "base": "base (baseline 措辞)",
    "nobias": "nobias (显式提示不要默认选某选项)",
    "terse": "terse (极简措辞)",
    "correct_wrong": "correct_wrong (用「正确/错误」代替「是/否」)",
    "ab_format": "ab_format (转成 A/B 选项格式)",
}


def load(dirpath: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(dirpath / "predictions.shard*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["uid"], keep="first").reset_index(drop=True)
    return df


def stats(sub: pd.DataFrame, labels: str) -> dict:
    n = len(sub)
    if n == 0:
        return {}
    recalls = {}
    for lb in labels:
        cls = sub[sub["gold"] == lb]
        if len(cls):
            recalls[lb] = float((cls["pred"] == lb).mean())
    gd = Counter(sub["gold"])
    maj = max(labels, key=lambda lb: gd.get(lb, 0))
    return {
        "n": n,
        "raw": float((sub["pred"] == sub["gold"]).mean()),
        "const_base": gd.get(maj, 0) / n,
        "const_label": maj,
        "macro": sum(recalls.values()) / len(recalls) if recalls else 0.0,
        "recalls": recalls,
        "pred_dist": {k: v / n for k, v in Counter(sub["pred"]).items()},
    }


def report(qtype: str, labels: str):
    print(f"\n{'═'*72}")
    print(f" {qtype.upper()}  —  {labels}")
    print(f"{'═'*72}")

    variants = sorted(d.name.split("_", 1)[1] for d in SENS.glob(f"{qtype}_*"))
    if not variants:
        print("  (无数据)")
        return

    data = {}
    for v in variants:
        df = load(SENS / f"{qtype}_{v}")
        if len(df):
            data[v] = df

    # ── 汇总表 ──
    print(f"\n  {'变体':<34}{'n':>6}{'raw':>8}{'常数基线':>10}{'macro':>8}{'raw-常数':>10}")
    print(f"  {'-'*70}")
    for v in variants:
        if v not in data:
            continue
        s = stats(data[v], labels)
        print(f"  {VARIANT_LABEL.get(v,v):<34}{s['n']:>6}{s['raw']*100:>7.1f}%"
              f"{s['const_base']*100:>9.1f}%{s['macro']*100:>7.1f}%"
              f"{(s['raw']-s['const_base'])*100:>+9.1f}pp")

    # ── 各标签召回率 —— 关键看少数类 ──
    print(f"\n  各标签召回率（关键：少数类的表现）")
    print(f"  {'变体':<34}" + "".join(f"{lb:>10}" for lb in labels))
    print(f"  {'-'*70}")
    for v in variants:
        if v not in data:
            continue
        s = stats(data[v], labels)
        cells = "".join(
            f"{s['recalls'].get(lb,float('nan'))*100:>9.1f}%" for lb in labels
        )
        print(f"  {VARIANT_LABEL.get(v,v):<34}{cells}")

    # ── 配对比较：以 base 为基准 ──
    if "base" in data and len(data) > 1:
        base = data["base"].set_index("uid")
        print(f"\n  逐样本配对比较（相对 base，同一 uid）")
        print(f"  {'变体':<34}{'翻对':>8}{'翻错':>8}{'净变化':>10}")
        print(f"  {'-'*70}")
        for v in variants:
            if v == "base" or v not in data:
                continue
            other = data[v].set_index("uid")
            common = base.index.intersection(other.index)
            b = base.loc[common]
            o = other.loc[common]
            b_ok = b["pred"] == b["gold"]
            o_ok = o["pred"] == o["gold"]
            fixed = int((~b_ok & o_ok).sum())
            broke = int((b_ok & ~o_ok).sum())
            print(f"  {VARIANT_LABEL.get(v,v):<34}{fixed:>8}{broke:>8}{fixed-broke:>+10}")


def main():
    tf_labels = "是否"
    report("tf", tf_labels)
    report("mc", "ABCD")

    print(f"\n{'═'*72}")
    print(" 判读")
    print(f"{'═'*72}")
    print("""
  • 若所有 TF 变体的「否」召回率都显著低于「是」-> 少数类崩溃是模型性质
  • 若 ab_format 变体把差距抹平 -> 说明只是「是」这个 token 的语言先验
  • 若 nobias 变体没有改善 -> 说明不是"模型不知道要平衡"，而是它做不到
""")


if __name__ == "__main__":
    main()
