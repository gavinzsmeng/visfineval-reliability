#!/usr/bin/env python
"""
视觉 token 预算 × 图数量  双因子分析
════════════════════════════════════════════════════════════════════

要回答的问题
  发现 2 观测到「10 图题 raw 29.5%，比无脑全选 A 低 29.5pp」。
  机制是哪个？

    假设 A  视觉 token 预算耗尽
            财报原图最大 5.8M 像素，单图最多 4281 token，10 图 4 万+。
            若砍掉每图预算，高图数题应跌得更狠。

    假设 B  能力上限，与 token 数无关
            若曲线形状不随预算变化，说明模型就是做不了多路视觉整合。

判读表
  预算↓ 时                         结论
  ─────────────────────────────────────────────
  高图数题跌得比单图题更狠        支持 A（预算受限）
  所有题等比例下跌                预算不是特异因素
  曲线形状完全不变                支持 B（能力上限）

用法
  python scripts/analyze_token_budget.py
"""
from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# (目录后缀, 展示名, longest_edge)
CONDS = [
    ("nocaq",   "无上限",    0),
    ("cap1m",   "cap 1.0M",  1003520),
    ("cap500k", "cap 0.5M",  501760),
    ("cap250k", "cap 0.25M", 250880),
]

# 实测的单图视觉 token 数（见 RESULTS.md，由 scripts 里的实测得出）
TOK_PER_IMG = {0: 4281, 1003520: 979, 501760: 461, 250880: 243}


def load(tag: str) -> pd.DataFrame:
    d = ROOT / "results" / f"tokenbudget_{tag}"
    rows = []
    for f in sorted(glob.glob(str(d / "predictions.shard*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset=["uid"], keep="first")
    return df


def bracket(n: int) -> str:
    if n == 1:
        return "1图"
    if n <= 2:
        return "2图"
    if n <= 4:
        return "3-4图"
    if n <= 6:
        return "5-6图"
    return "8图+"


ORDER = ["1图", "2图", "3-4图", "5-6图", "8图+"]


def main():
    data = {}
    for tag, name, cap in CONDS:
        df = load(tag)
        if len(df):
            data[tag] = (name, cap, df)

    if not data:
        print("✗ 没有找到结果，先跑 scripts/run_token_budget_sweep.sh")
        return

    print("═" * 96)
    print(" 视觉 token 预算 × 图数量")
    print("═" * 96)
    print(f"  样本: {len(next(iter(data.values()))[2])} 条/条件")
    print()

    # ── 主表：每个预算下，各图数区间的准确率 ──
    print("  【准确率】")
    hdr = f"  {'预算':<10}{'单图token':>10}" + "".join(f"{b:>10}" for b in ORDER) + f"{'全部':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    all_rows = {}
    for tag, (name, cap, df) in data.items():
        df = df.copy()
        df["br"] = df["num_images"].map(bracket)
        cells = []
        row = {}
        for b in ORDER:
            sub = df[df["br"] == b]
            if len(sub) == 0:
                cells.append("     n/a"); row[b] = None; continue
            acc = float((sub["pred"] == sub["gold"]).mean())
            row[b] = (acc, len(sub))
            cells.append(f"{acc*100:>9.1f}%")
        acc_all = float((df["pred"] == df["gold"]).mean())
        row["_all"] = acc_all
        all_rows[tag] = row
        print(f"  {name:<10}{TOK_PER_IMG.get(cap,0):>10}" + "".join(cells) + f"{acc_all*100:>9.1f}%")

    # 样本量
    ref = next(iter(data.values()))[2]
    ref = ref.copy(); ref["br"] = ref["num_images"].map(bracket)
    print(f"\n  {'样本量':<10}{'':>10}" + "".join(
        f"{len(ref[ref['br']==b]):>10}" for b in ORDER) + f"{len(ref):>10}")

    # ── 关键判读 ──
    print()
    print("═" * 96)
    print(" 判读：假设 A（预算耗尽） vs 假设 B（能力上限）")
    print("═" * 96)

    if "nocaq" in all_rows and "cap250k" in all_rows:
        base, tight = all_rows["nocaq"], all_rows["cap250k"]
        print(f"\n  对比「无上限」(单图 {TOK_PER_IMG[0]} tok) "
              f"-> 「cap 0.25M」(单图 {TOK_PER_IMG[250880]} tok)")
        print(f"  预算压缩 {TOK_PER_IMG[0]/TOK_PER_IMG[250880]:.1f} 倍\n")
        print(f"  {'图数':<8}{'无上限':>10}{'cap0.25M':>11}{'变化':>10}")
        print("  " + "-" * 40)
        drops = []
        for b in ORDER:
            a = base.get(b); c = tight.get(b)
            if not a or not c:
                continue
            d = (c[0] - a[0]) * 100
            drops.append((b, d))
            print(f"  {b:<8}{a[0]*100:>9.1f}%{c[0]*100:>10.1f}%{d:>+9.1f}pp")
        print()
        if drops:
            single_drop = dict(drops).get("1图", 0)
            # ★ 必须按样本量加权：5-6图 只有 27 条，它的百分比波动极大，
            #   不加权会让噪声主导结论（第一版就踩了这个坑）。
            pairs = [(d, base[b][1]) for b, d in drops if b != "1图"]
            wsum = sum(n for _, n in pairs)
            avg_multi = sum(d * n for d, n in pairs) / wsum if wsum else 0
            print(f"  单图题变化  : {single_drop:+.1f}pp   (n={base['1图'][1]})")
            print(f"  多图题变化  : {avg_multi:+.1f}pp   (按样本量加权, n={wsum})")
            print()
            diff = avg_multi - single_drop
            if diff < -3:
                print("  → 多图题跌得更狠（差 %.1fpp）。**支持假设 A：预算是多图瓶颈之一。**" % diff)
            elif abs(diff) <= 3:
                print("  → 单图与多图跌幅相当（差 %.1fpp）。" % diff)
                print("    **不支持假设 A** —— 预算压缩是普遍性伤害，不是多图特有。")
            else:
                print("  → 多图题反而更抗压缩（差 %+.1fpp）。**不支持假设 A。**" % diff)

            # 逐档看：若某档在「降低预算」后反而变好，说明该档本来就不受预算限制
            print()
            print("  逐档细看（cap 越紧 token 越少）：")
            for b in ORDER:
                vals = [all_rows[t].get(b) for t in data if all_rows[t].get(b)]
                if len(vals) < 2:
                    continue
                arr = [v[0] for v in vals]     # 从宽到紧
                best = max(range(len(arr)), key=lambda i: arr[i])
                mark = "  <- 最紧预算下最好" if best == len(arr) - 1 else ""
                nonmono = arr[1] > arr[0]      # 收紧第一档反而变好
                flag = "  ⚠ 收紧后反而更好，该档不受预算限制" if nonmono else ""
                print(f"    {b:<8}" + "  ".join(f"{a*100:5.1f}%" for a in arr) + flag + mark)

    # ── 补充：所有预算下的曲线形状是否一致 ──
    print()
    print("═" * 96)
    print(" 曲线形状稳定性检查")
    print("═" * 96)
    print("\n  若各预算下「相对单图的跌幅」形状一致 -> 支持假设 B（能力上限）\n")
    print(f"  {'预算':<10}" + "".join(f"{b:>10}" for b in ORDER[1:]))
    print("  " + "-" * (10 + 10 * (len(ORDER) - 1)))
    for tag, (name, cap, df) in data.items():
        row = all_rows.get(tag, {})
        base_acc = row.get("1图")
        if not base_acc:
            continue
        cells = []
        for b in ORDER[1:]:
            v = row.get(b)
            cells.append(f"{(v[0]-base_acc[0])*100:>+9.1f}" if v else "       n/a")
        print(f"  {name:<10}" + "".join(cells))
    print("\n  （数值 = 相对同预算下「1图」准确率的差值，单位 pp）")


if __name__ == "__main__":
    main()
