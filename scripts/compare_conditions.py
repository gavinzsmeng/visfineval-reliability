#!/usr/bin/env python
"""
训练前后配对比较
════════════════════════════════════════════════════════════════════

回答项目的核心问题：

    微调能否修复 zero-shot baseline 上发现的少数类崩溃？
    （判断题「否」的召回率仅 42.5%，低于随机猜的 50%）

## 为什么必须做配对比较

三个条件跑的是**同一批 test 样本**（按全局 uid 对齐）。
比较总体均值会混入样本难度差异；逐样本配对才能说
"这一条样本因为微调而从错变对"。

    net = (zero-shot 错 -> LoRA 对) - (zero-shot 对 -> LoRA 错)

## 关键判读

  • natural 条件（原始分布，真值 A 占 58.5%）
      预期 raw 可能涨，但「否」召回率可能不涨甚至跌
      -> 若成立，说明「微调提升准确率」与「微调提升可靠性」是两件事

  • balanced 条件（6 类各 6725 条）
      预期「否」召回率显著提升，但 raw 可能下降
      -> raw 与 macro 的 trade-off 直接被量化

用法:
  python scripts/compare_conditions.py
"""
from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

import math

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


# ════════════════════════════════════════════════════════════
# 统计工具
#
# 为什么需要：test split 里判断题少数类「否」只有 115 条，
# 单条约值 0.9 个百分点。直接报点估计会被质疑统计功效不足。
# ════════════════════════════════════════════════════════════

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 置信区间（比正态近似在极端比例/小样本下更稳）。"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(b: int, c: int) -> float:
    """McNemar 精确检验（二项检验）双尾 p 值。

    b = 从错变对的样本数，c = 从对变错的样本数。
    只在这两者不相等时才可能显著 —— 这是配对设计的标准做法。
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # 双尾：P(X <= k) * 2，X ~ Binomial(n, 0.5)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def fmt_ci(k: int, n: int) -> str:
    lo, hi = wilson_ci(k, n)
    return f"{k/n*100:5.1f}% [{lo*100:.1f}, {hi*100:.1f}]" if n else "  n/a"

CONDITIONS = [
    ("posttrain_zeroshot", "zero-shot (基座)", ""),
    ("posttrain_lora_natural", "LoRA natural", "按原始分布微调"),
    ("posttrain_lora_balanced", "LoRA balanced", "按标签均衡微调"),
]


def load(tag: str) -> pd.DataFrame:
    d = ROOT / "results" / tag
    rows = []
    for f in sorted(glob.glob(str(d / "predictions.shard*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    # 跨条件 join 用的键。
    #   新结果有 uid（load_dataset 统一分配）-> 直接用，可跨条件配对
    #   早期结果（加 uid 之前生成的，如 results/zeroshot）没有 uid，
    #   也没有 question 字段，无法与其它条件 join -> 返回不带索引的 df，
    #   调用方会跳过配对比较（它只用于全量 headline，不参与本节比较）。
    if "uid" not in df.columns:
        return df
    df = df.assign(__key=df["uid"].astype(str))
    return df.drop_duplicates(subset=["__key"], keep="first").set_index("__key")


def recall(df: pd.DataFrame, qtype: str, label: str) -> tuple[float, int]:
    sub = df[(df["qtype"] == qtype) & (df["gold"] == label)]
    if len(sub) == 0:
        return float("nan"), 0
    return float((sub["pred"] == label).mean()), len(sub)


def summarize(name: str, note: str, df: pd.DataFrame) -> dict:
    if len(df) == 0:
        print(f"\n── {name} ──   (无数据，跳过)")
        return {}
    tf = df[df["qtype"] == "tf"]
    mc = df[df["qtype"] == "mc"]

    r_yes, n_yes = recall(df, "tf", "是")
    r_no, n_no = recall(df, "tf", "否")

    # TF 的 raw / 常数基线 / macro
    tf_raw = float((tf["pred"] == tf["gold"]).mean()) if len(tf) else float("nan")
    gd = Counter(tf["gold"])
    tf_const = gd.most_common(1)[0][1] / len(tf) if len(tf) else float("nan")
    tf_macro = (r_yes + r_no) / 2 if len(tf) else float("nan")

    mc_raw = float((mc["pred"] == mc["gold"]).mean()) if len(mc) else float("nan")
    mc_gd = Counter(mc["gold"])
    mc_const = mc_gd.most_common(1)[0][1] / len(mc) if len(mc) else float("nan")

    print(f"\n── {name} ──  {note}")
    print(f"   总样本 {len(df)}   多选 {len(mc)}   判断 {len(tf)}")
    print(f"   多选题  raw {mc_raw*100:6.2f}%   常数基线 {mc_const*100:6.2f}%")
    print(f"   判断题  raw {tf_raw*100:6.2f}%   常数基线 {tf_const*100:6.2f}%")
    print(f"   判断题  macro-recall {tf_macro*100:6.2f}%")
    k_yes = int((tf[(tf['gold'] == '是')]['pred'] == '是').sum()) if len(tf) else 0
    k_no = int((tf[(tf['gold'] == '否')]['pred'] == '否').sum()) if len(tf) else 0
    print(f"     ├─ 「是」召回 {fmt_ci(k_yes, n_yes)}  (n={n_yes})")
    print(f"     └─ 「否」召回 {fmt_ci(k_no, n_no)}  (n={n_no})  ← 核心指标")
    print(f"        中括号为 95% Wilson 置信区间")

    return {
        "name": name, "df": df,
        "tf_raw": tf_raw, "tf_const": tf_const, "tf_macro": tf_macro,
        "r_yes": r_yes, "r_no": r_no,
        "mc_raw": mc_raw, "mc_const": mc_const,
    }


def paired(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str):
    """逐样本配对比较 a -> b。"""
    common = a.index.intersection(b.index)
    if len(common) == 0:
        print(f"   {name_a} vs {name_b}: 无共同 uid")
        return
    A, B = a.loc[common], b.loc[common]
    a_ok = A["pred"] == A["gold"]
    b_ok = B["pred"] == B["gold"]
    fixed = int((~a_ok & b_ok).sum())
    broke = int((a_ok & ~b_ok).sum())
    p = mcnemar_exact(fixed, broke)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
    print(f"   {name_a:<20} -> {name_b:<20} 翻对 {fixed:>4}  翻错 {broke:>4}"
          f"  净 {fixed-broke:+5}   McNemar p={p:.3f} {sig}")


def main():
    print("═" * 74)
    print(" 训练前后配对比较  (同一批 test 样本，按全局 uid 对齐)")
    print("═" * 74)

    stats, dfs = {}, {}
    for tag, name, note in CONDITIONS:
        df = load(tag)
        s = summarize(name, note, df)
        if s:
            stats[tag] = s
            dfs[tag] = df

    if len(dfs) < 2:
        print("\n⚠ 可比较的条件不足 2 个，无法做配对检验。")
        print("  先跑 scripts/run_posttrain_eval.sh")
        return

    # ── 一致性自检（可选）──
    # results/zeroshot 是加 uid 字段之前生成的，没有 uid / question，
    # 无法与本节三个条件 join。仅当它能对上时才做检查。
    orig = load("zeroshot")
    if len(orig) and "uid" in orig.columns and "posttrain_zeroshot" in dfs:
        common = orig.index.intersection(dfs["posttrain_zeroshot"].index)
        if len(common):
            a = orig.loc[common]["pred"]
            b = dfs["posttrain_zeroshot"].loc[common]["pred"]
            agree = float((a == b).mean())
            flag = "✓" if agree > 0.999 else "⚠"
            print(f"\n── 一致性自检 ──")
            print(f"   {flag} 重跑的 zero-shot 与原始 baseline 在 {len(common)} 条共同样本上")
            print(f"      预测一致率 {agree*100:.2f}%   (greedy 解码，应接近 100%)")
    else:
        print(f"\n── 一致性自检 ──")
        print(f"   跳过：results/zeroshot 生成于加 uid 字段之前，无法与本节条件 join。")
        print(f"   （它只用于全量 headline 数字；本节的 zero-shot 是新跑的、可配对的版本）")

    # ── 配对比较 ──
    print(f"\n{'─'*74}")
    print(" 逐样本配对比较（净变化 > 0 表示后者更好）")
    print(f"{'─'*74}")
    tags = list(dfs.keys())
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            paired(dfs[tags[i]], dfs[tags[j]],
                   stats[tags[i]]["name"], stats[tags[j]]["name"])

    # ── 冻结在少数类上的配对比较（最关键）──
    print(f"\n{'─'*74}")
    print(" 「否」样本子集上的配对比较（少数类，n 很小但最说明问题）")
    print(f"{'─'*74}")
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            A, B = dfs[tags[i]], dfs[tags[j]]
            common = A.index.intersection(B.index)
            A, B = A.loc[common], B.loc[common]
            m = (A["qtype"] == "tf") & (A["gold"] == "否")
            if m.sum() == 0:
                continue
            a_ok = (A.loc[m, "pred"] == A.loc[m, "gold"])
            b_ok = (B.loc[m, "pred"] == B.loc[m, "gold"])
            print(f"   {stats[tags[i]]['name']:<22} -> {stats[tags[j]]['name']:<22}"
                  f" 翻对 {int((~a_ok & b_ok).sum()):>3}  翻错 {int((a_ok & ~b_ok).sum()):>3}"
                  f"  净 {int((~a_ok & b_ok).sum()) - int((a_ok & ~b_ok).sum()):+4}")

    # ── 结论 ──
    print(f"\n{'═'*74}")
    print(" 核心问题：微调能否修复少数类崩溃？")
    print(f"{'═'*74}")
    print(f"   {'条件':<24}{'多选raw':>9}{'判断raw':>9}{'判断macro':>11}   {'否召回 95%CI':<20}")
    print(f"   {'-'*76}")
    for tag, s in stats.items():
        df = s["df"]
        tf = df[df["qtype"] == "tf"]
        n_no = int((tf["gold"] == "否").sum())
        k_no = int(((tf["gold"] == "否") & (tf["pred"] == "否")).sum())
        ci = fmt_ci(k_no, n_no) if n_no else "n/a"
        print(f"   {s['name']:<24}{s['mc_raw']*100:>8.2f}%{s['tf_raw']*100:>8.2f}%"
              f"{s['tf_macro']*100:>10.2f}%   {ci:<20}")
    print()
    print("   判读要点：")
    print("   • 若 natural 的「否」召回没涨而 raw 涨了 -> 微调只学会了更强的多数类偏置")
    print("   • 若 balanced 的「否」召回大涨但 raw 跌了 -> raw 与 macro 的 trade-off 被量化")
    print("   • 若两者「否」召回都不动 -> 该失败模式不是 SFT 能修的，需要改训练目标")


if __name__ == "__main__":
    main()
