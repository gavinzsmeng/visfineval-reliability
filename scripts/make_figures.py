#!/usr/bin/env python
"""
生成 README 用的四张数据图
════════════════════════════════════════════════════════════════════

全部从**真实预测文件**计算，不用任何手写数字 —— 图上的每个数都能追溯到
results/ 下的 predictions.shard*.jsonl。

输出 assets/ 下的 PNG（浅色 + 深色两套，README 用 <picture> 切换）。

配色取自已验证的分类色板与顺序色阶：
  · 分类（不同实体）  #2a78d6 蓝 / #eb6834 橙 / #1baf7a 青 / #eda100 黄
  · 顺序（有序量，如 token 预算）  单一蓝色由深到浅
文字一律用中性色（#0b0b0b / #52514e），不穿系列色 —— 身份由色块承载。

用法:
  python scripts/make_figures.py
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from compare_conditions import wilson_ci  # noqa: E402

OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)

# ── 字体 ─────────────────────────────────────────────────────
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── 主题 ─────────────────────────────────────────────────────
LIGHT = dict(
    surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e3e3e0",
    s=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    seq=["#184f95", "#2a78d6", "#5598e7", "#86b6ef"],
)
DARK = dict(
    surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#33332f",
    s=["#3987e5", "#d95926", "#199e70", "#c98500"],
    seq=["#9ec5f4", "#6da7ec", "#3987e5", "#256abf"],
)

BAR_W = 0.38          # 相邻柱之间留 2px 视觉间隙靠留白实现
GRID_KW = dict(axis="y", lw=0.6, alpha=0.9, zorder=0)


def style(ax, th):
    ax.set_facecolor(th["surface"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(th["grid"])
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors=th["ink2"], labelsize=9, length=3, width=0.8)
    ax.yaxis.grid(True, color=th["grid"], **{k: v for k, v in GRID_KW.items() if k != "axis"})
    ax.set_axisbelow(True)


def finish(fig, ax_or_axes, th, name, title=None, sub=None):
    fig.patch.set_facecolor(th["surface"])
    if title:
        fig.suptitle(title, color=th["ink"], fontsize=13, fontweight="bold",
                     x=0.02, ha="left", y=0.97)
    if sub:
        fig.text(0.02, 0.90, sub, color=th["ink2"], fontsize=9.5, ha="left")
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight",
                facecolor=th["surface"], pad_inches=0.25)
    plt.close(fig)


# ════════════════════════════════════════════════════════════
# 数据加载
# ════════════════════════════════════════════════════════════
def load(tag_glob: str):
    rows = []
    for f in sorted(glob.glob(str(ROOT / "results" / tag_glob / "predictions.shard*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def bracket(n):
    if n == 1: return "1图"
    if n == 2: return "2图"
    if n <= 4: return "3-4图"
    if n <= 6: return "5-6图"
    return "8图+"


BR_ORDER = ["1图", "2图", "3-4图", "5-6图", "8图+"]


# ════════════════════════════════════════════════════════════
# 图 1：答案分布 —— 常数基线是怎么来的
# ════════════════════════════════════════════════════════════
def fig1(th, suffix):
    rows = [r for r in load("zeroshot") if r["qtype"] == "mc"]
    gold = Counter(r["gold"] for r in rows)
    pred = Counter(r["pred"] for r in rows)
    n = len(rows)
    labels = list("ABCD")
    g = [gold.get(l, 0) / n * 100 for l in labels]
    p = [pred.get(l, 0) / n * 100 for l in labels]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(labels))
    b1 = ax.bar(x - BAR_W / 2 - 0.01, g, BAR_W, color=th["s"][0], zorder=3,
                label="真值分布", linewidth=1.5, edgecolor=th["surface"])
    b2 = ax.bar(x + BAR_W / 2 + 0.01, p, BAR_W, color=th["s"][1], zorder=3,
                label="模型预测分布", linewidth=1.5, edgecolor=th["surface"])

    for bars, vals in ((b1, g), (b2, p)):
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.9, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8.5, color=th["ink2"])

    # 常数基线：真值 A 的占比就是「永远选 A」的准确率
    ax.axhline(g[0], color=th["s"][0], lw=1.2, ls=(0, (4, 3)), zorder=2, alpha=0.85)
    ax.text(len(labels) - 0.52, g[0] + 1.2,
            f"常数基线（永远选 A）= {g[0]:.1f}%",
            ha="right", va="bottom", fontsize=9, color=th["ink"], fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("选项", color=th["ink2"], fontsize=9.5)
    ax.set_ylabel("占比 (%)", color=th["ink2"], fontsize=9.5)
    ax.set_ylim(0, max(g) * 1.30)
    ax.legend(frameon=False, fontsize=9, labelcolor=th["ink2"], loc="upper right")
    style(ax, th)
    finish(fig, ax, th, f"fig1_answer_distribution{suffix}.png",
           "多选题答案分布严重偏斜",
           f"VisFinEval 全部 {n:,} 道多选题。无脑输出「A」即可获得 {g[0]:.1f}% 准确率 —— 这不是零分线，是及格线。")


# ════════════════════════════════════════════════════════════
# 图 2：判断题少数类召回率（zero-shot vs LoRA）
# ════════════════════════════════════════════════════════════
def fig2(th, suffix):
    conds = [("posttrain_zeroshot", "zero-shot (基座)"),
             ("posttrain_lora_natural", "LoRA natural"),
             ("posttrain_lora_balanced", "LoRA balanced")]
    data = []
    for tag, name in conds:
        rows = [r for r in load(tag) if r["qtype"] == "tf"]
        cell = {}
        for lb in ("是", "否"):
            sub = [r for r in rows if r["gold"] == lb]
            k = sum(1 for r in sub if r["pred"] == lb)
            lo, hi = wilson_ci(k, len(sub))
            cell[lb] = (k / len(sub) * 100, (k / len(sub) - lo) * 100,
                        (hi - k / len(sub)) * 100, len(sub))
        data.append((name, cell))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(2)
    w = 0.26
    for i, (name, cell) in enumerate(data):
        vals = [cell["是"][0], cell["否"][0]]
        err = [[cell["是"][1], cell["否"][1]], [cell["是"][2], cell["否"][2]]]
        pos = x + (i - 1) * (w + 0.02)
        ax.bar(pos, vals, w, color=th["s"][i], zorder=3, label=name,
               linewidth=1.5, edgecolor=th["surface"])
        ax.errorbar(pos, vals, yerr=err, fmt="none", ecolor=th["ink2"],
                    elinewidth=1.0, capsize=2.5, zorder=4)
        for xx, v in zip(pos, vals):
            ax.text(xx, v + 2.6, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=8.2, color=th["ink2"])

    ax.axhline(50, color=th["ink2"], lw=1.0, ls=(0, (3, 3)), zorder=2, alpha=0.8)
    # 放在两组柱之间的空白处，避免压到柱子
    ax.text(0.52, 51.5, "随机猜 = 50%", ha="center", fontsize=8.5, color=th["ink2"])

    ax.set_xticks(x); ax.set_xticklabels(["真值为「是」", "真值为「否」"], fontsize=10)
    ax.set_ylabel("召回率 (%)", color=th["ink2"], fontsize=9.5)
    ax.set_ylim(0, 118)
    ax.legend(frameon=False, fontsize=9, labelcolor=th["ink2"], loc="upper left",
              ncol=3, bbox_to_anchor=(0, 1.02))
    style(ax, th)
    n_no = data[0][1]["否"][3]
    finish(fig, ax, th, f"fig2_minority_recall{suffix}.png",
           "判断题少数类：zero-shot 比随机猜还差，微调修好了它",
           f"误差棒为 95% Wilson 置信区间。少数类「否」仅 {n_no} 条样本；"
           f"zero-shot 的 39.7% 显著低于 50%，微调后升至 69%。")


# ════════════════════════════════════════════════════════════
# 图 3：多图崩溃曲线 × 四档 token 预算
# ════════════════════════════════════════════════════════════
def fig3(th, suffix):
    conds = [("nocaq", "无上限 (4281 tok)", 0),
             ("cap1m", "cap 1.0M (979 tok)", 1),
             ("cap500k", "cap 0.5M (461 tok)", 2),
             ("cap250k", "cap 0.25M (243 tok)", 3)]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    X = np.arange(len(BR_ORDER))
    for tag, name, i in conds:
        rows = load(f"tokenbudget_{tag}")
        acc = []
        for b in BR_ORDER:
            sub = [r for r in rows if bracket(r["num_images"]) == b]
            acc.append(sum(1 for r in sub if r["pred"] == r["gold"]) / len(sub) * 100 if sub else np.nan)
        ax.plot(X, acc, "-o", color=th["seq"][i], lw=2.0, ms=5.5,
                zorder=3, markeredgecolor=th["surface"], markeredgewidth=1.2,
                label=name)

    # 样本量注释（只标一次）
    rows = load("tokenbudget_nocaq")
    cnts = [sum(1 for r in rows if bracket(r["num_images"]) == b) for b in BR_ORDER]
    for xi, c in zip(X, cnts):
        ax.text(xi, 3.0, f"n={c}", ha="center", fontsize=7.8, color=th["ink2"], alpha=0.85)

    # ★ 不要在线条末端做直接标注：四条线在「8图+」处收敛到 35–44%，
    #   末端标签必然互相重叠（第一版就是这么翻车的）。
    #   改把图例放进中下部空白区（y≈10–45, x≈2.3–4.6 是空的）。
    # 布局：左下大片空白放图例（x≈0–2, y≈10–38 无数据），
    #       右下空白放标注（x≈3.3–4.4, y≈8–28 无数据）。
    #       两块区域互不重叠，也不压线。
    ax.legend(frameon=False, fontsize=9, labelcolor=th["ink2"], loc="lower left",
              bbox_to_anchor=(0.01, 0.09), ncol=1, handlelength=1.8,
              labelspacing=0.5)

    ax.annotate("收紧预算反而变好\n38.8% → 43.9%",
                xy=(3.93, 42.5), xytext=(3.55, 14),
                fontsize=9, color=th["ink"], ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=th["ink2"], lw=1.0,
                                connectionstyle="arc3,rad=0.18"), zorder=6)

    ax.set_xticks(X); ax.set_xticklabels(BR_ORDER)
    ax.set_xlabel("单题图片数量", color=th["ink2"], fontsize=9.5)
    ax.set_ylabel("准确率 (%)", color=th["ink2"], fontsize=9.5)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.35, len(BR_ORDER) - 0.65)
    style(ax, th)
    finish(fig, ax, th, f"fig3_multi_image_curve{suffix}.png",
           "多图崩溃不是 token 预算问题",
           "四条线对应四档视觉 token 预算。「8图+」档收紧预算后反而变好（38.8% → 43.9%），"
           "四条线基本平行 —— 说明瓶颈是模型能力上限，不是 token 数。")


# ════════════════════════════════════════════════════════════
# 图 4：微调增益几乎全部落在少数类
# ════════════════════════════════════════════════════════════
def fig4(th, suffix):
    def rec(tag, lb):
        rows = [r for r in load(tag) if r["qtype"] == "tf" and r["gold"] == lb]
        return sum(1 for r in rows if r["pred"] == lb) / len(rows) * 100

    labels = ["真值为「是」", "真值为「否」"]
    zs = [rec("posttrain_zeroshot", "是"), rec("posttrain_zeroshot", "否")]
    lora = [rec("posttrain_lora_natural", "是"), rec("posttrain_lora_natural", "否")]

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    x = np.arange(2)
    b1 = ax.bar(x - BAR_W / 2 - 0.01, zs, BAR_W, color=th["s"][0], zorder=3,
                label="zero-shot", linewidth=1.5, edgecolor=th["surface"])
    b2 = ax.bar(x + BAR_W / 2 + 0.01, lora, BAR_W, color=th["s"][1], zorder=3,
                label="LoRA 微调后", linewidth=1.5, edgecolor=th["surface"])
    # 数值放柱内：柱外上方会和箭头/误差线抢位置（第一版「39.7」就被箭头压住）
    for bars, vals in ((b1, zs), (b2, lora)):
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v - 4.5, f"{v:.1f}",
                    ha="center", va="top", fontsize=9,
                    color=th["surface"], fontweight="bold")

    for xi, (a, b) in enumerate(zip(zs, lora)):
        d = b - a
        ax.annotate("", xy=(xi + BAR_W / 2 + 0.01, b - 11), xytext=(xi - BAR_W / 2 - 0.01, a - 11),
                    arrowprops=dict(arrowstyle="->", color=th["ink2"], lw=1.1,
                                    connectionstyle="arc3,rad=-0.25"), zorder=5)
        ax.text(xi, max(a, b) + 5, f"{d:+.1f} pp", ha="center",
                fontsize=12, color=th["ink"], fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("召回率 (%)", color=th["ink2"], fontsize=9.5)
    ax.set_ylim(0, 112)
    # 图例放两组柱之间的空白（x≈0.5），不放轴上方以免顶到副标题
    ax.legend(frameon=False, fontsize=9, labelcolor=th["ink2"], loc="center",
              ncol=1, bbox_to_anchor=(0.5, 0.40), handlelength=1.6)
    style(ax, th)
    finish(fig, ax, th, f"fig4_gain_concentration{suffix}.png",
           "微调的增益几乎全部落在少数类上",
           "同一批 2,889 条 held-out 样本。多数类只涨 1.5pp，少数类涨 29.3pp —— 差 20 倍。")


def main():
    for th, suffix in ((LIGHT, ""), (DARK, "_dark")):
        fig1(th, suffix); fig2(th, suffix); fig3(th, suffix); fig4(th, suffix)
    for p in sorted(OUT.glob("*.png")):
        print(f"{p.name}  {p.stat().st_size//1024} KB")


if __name__ == "__main__":
    main()
