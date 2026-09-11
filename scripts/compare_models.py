#!/usr/bin/env python
"""
多模型横向对比
════════════════════════════════════════════════════════════════════

回答的问题
  「少数类崩溃」是 Qwen3-VL 一家的怪癖，还是 benchmark 的系统性问题？

判读
  · 若多个家族都出现「少数类召回 < 50%」且常数基线同样高
    -> benchmark 的系统性缺陷，不是某个模型的锅
  · 若只有个别模型崩
    -> 是模型性质，benchmark 没问题

所有模型跑的是同一批 2,889 条 test 样本、同一个 prompt、同一套指标口径，
所以可以直接横排比较。

用法:
  python scripts/compare_models.py
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from compare_conditions import wilson_ci  # noqa: E402

MODELS = [
    # Qwen3-VL 直接复用 posttrain_zeroshot：它的配置与本脚本会产生的完全一致
    # （同 test uids、同 base prompt、无 adapter、不设 max_pixels），
    # 重跑一遍只是浪费时间，不构成口径差异。
    ("posttrain_zeroshot", "Qwen3-VL-8B-Instruct"),
    ("model_qwen25vl-7b", "Qwen2.5-VL-7B-Instruct"),
]

# 参考：微调后的同一模型，用来对比「换个模型」和「微调同一个模型」哪个更有效
EXTRA = [("posttrain_lora_natural", "Qwen3-VL-8B + LoRA (参考)")]


def load(tag: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(ROOT / "results" / tag / "predictions.shard*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["uid"], keep="first")


def stats(df: pd.DataFrame) -> dict:
    mc = df[df["qtype"] == "mc"]
    tf = df[df["qtype"] == "tf"]
    out = {"n": len(df)}

    if len(mc):
        out["mc_raw"] = float((mc["pred"] == mc["gold"]).mean())
        gd = Counter(mc["gold"])
        out["mc_const"] = gd.most_common(1)[0][1] / len(mc)
    if len(tf):
        out["tf_raw"] = float((tf["pred"] == tf["gold"]).mean())
        gd = Counter(tf["gold"])
        out["tf_const"] = gd.most_common(1)[0][1] / len(tf)
        rec = {}
        for lb in ("是", "否"):
            sub = tf[tf["gold"] == lb]
            k = int((sub["pred"] == lb).sum())
            rec[lb] = (k / len(sub), k, len(sub))
        out["rec"] = rec
        out["tf_macro"] = (rec["是"][0] + rec["否"][0]) / 2
    return out


def main():
    rows = []
    for tag, name in MODELS:
        df = load(tag)
        if len(df):
            rows.append((name, stats(df)))
    for tag, name in EXTRA:
        df = load(tag)
        if len(df):
            rows.append((name, stats(df)))

    if not rows:
        print("没有找到任何模型结果。先跑 scripts/run_model_comparison.sh")
        return

    print("═" * 92)
    print(" 多模型横向对比（同一批 test 样本，同一 prompt，同一指标口径）")
    print("═" * 92)
    print()
    print(f"  {'模型':<28}{'多选raw':>9}{'常数基线':>10}{'判断raw':>9}{'判断macro':>11}"
          f"{'「是」召回':>11}{'「否」召回':>11}")
    print("  " + "-" * 87)
    for name, s in rows:
        r = s.get("rec", {})
        yes = f"{r['是'][0]*100:.1f}%" if "是" in r else "n/a"
        no = f"{r['否'][0]*100:.1f}%" if "否" in r else "n/a"
        print(f"  {name:<28}{s.get('mc_raw',float('nan'))*100:>8.1f}%"
              f"{s.get('mc_const',float('nan'))*100:>9.1f}%"
              f"{s.get('tf_raw',float('nan'))*100:>8.1f}%"
              f"{s.get('tf_macro',float('nan'))*100:>10.1f}%"
              f"{yes:>11}{no:>11}")

    # 少数类的置信区间 —— 样本量小，点估计不够
    print()
    print("  「否」召回率及 95% Wilson 置信区间")
    print("  " + "-" * 60)
    for name, s in rows:
        r = s.get("rec", {}).get("否")
        if not r:
            continue
        p, k, n = r
        lo, hi = wilson_ci(k, n)
        below = " ← 低于随机猜(50%)" if hi < 0.5 else ""
        print(f"  {name:<28}{p*100:>6.1f}%  [{lo*100:.1f}, {hi*100:.1f}]  (n={n}){below}")

    # 判读
    print()
    print("═" * 92)
    print(" 判读")
    print("═" * 92)
    base_models = [(n, s) for n, s in rows if "LoRA" not in n]
    below_chance = 0
    for name, s in base_models:
        r = s.get("rec", {}).get("否")
        if r:
            lo, hi = wilson_ci(r[1], r[2])
            if hi < 0.5:
                below_chance += 1
    print(f"\n  共 {len(base_models)} 个未微调模型，其中 {below_chance} 个的「否」召回率")
    print(f"  置信区间上限低于 50%（即显著低于随机猜）。\n")
    if below_chance == len(base_models) and len(base_models) > 1:
        print("  → 所有被测模型都出现少数类崩溃。")
        print("    **这是 benchmark 的系统性问题，不是某个模型的怪癖。**")
    elif below_chance == 0:
        print("  → 没有模型显著低于随机猜。少数类崩溃可能是特定模型的性质。")
    else:
        print("  → 部分模型出现、部分没有。需要更多模型才能判断是普遍现象还是模型差异。")


if __name__ == "__main__":
    main()
