#!/usr/bin/env python
"""
VLM-FinChart 评测 harness —— Qwen3-VL zero-shot baseline
════════════════════════════════════════════════════════════════════

设计原则
────────
1. 题型必须分开算
   VisFinEval 的 19 个 TSV 里混了三种格式，混在一起算准确率是错的：
     多选题   A/B/C/D 列齐全，answer 是字母        -> 单独评分
     判断题   L1_Q4，answer 是「是/否」，ABCD 全空  -> 单独评分
     多轮案例 L3_Q4，字段结构不同                   -> 本 harness 跳过

2. 必须报告常数基线
   实测多选题答案分布 A=58.5% / B=26.6% / C=10.7% / D=4.1%。
   只报 raw accuracy 会严重误导 —— 永远输出 "A" 就能拿 58.5%。
   所以本 harness 强制同时输出 raw / 常数基线 / 相对增益。

3. 保存 raw output
   答案提取规则会显著影响分数（模型会输出 'A,B,C,D' 这种）。
   保存原文，规则可以事后重算，不用重跑推理。

4. 可分片续跑
   --shard-id / --num-shards 支持多卡并行，结果按 shard 增量落盘。

用法
────
  # 单卡小样本自测
  CUDA_VISIBLE_DEVICES=4 python scripts/eval_baseline.py --limit 100

  # 4 卡全量（见 scripts/run_eval.sh）
  python scripts/eval_baseline.py --shard-id 0 --num-shards 4

  # 汇总
  python scripts/eval_baseline.py --merge results/zeroshot
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Qwen3-VL-8B-Instruct"
DATA_DIR = ROOT / "data" / "VisFinEval" / "data"

# ── 题型定义 ────────────────────────────────────────────────
QTYPE_MC = "mc"   # 多选题
QTYPE_TF = "tf"   # 判断题（是/否）

# L3_Q4 是「多轮案例」，字段是 index/image/information/round/question/answer，
# 和其它 TSV 完全不同，本版本不处理。
EXCLUDE_FILES = {"L3_Q4.tsv"}

# ── Prompt 变体 ─────────────────────────────────────────────
# 用途：验证 baseline 上观察到的失败模式（判断题少数类召回率仅 42.5%、
#      多图题崩溃）是不是单一 prompt 造成的假象。
# 只有当多个措辞变体下失败模式都复现时，才能说它是模型的性质。
PROMPT_VARIANTS = {
    # 多选题
    ("mc", "base"): """请仔细阅读图片，回答问题，只输出正确选项的字母（A/B/C/D），不要解释。

问题：{question}
A. {A}
B. {B}
C. {C}
D. {D}

答案：""",

    ("mc", "nobias"): """请仔细阅读图片，回答问题。注意：正确答案不一定是 A，请独立判断，不要默认选择 A。
只输出正确选项的字母（A/B/C/D），不要解释。

问题：{question}
A. {A}
B. {B}
C. {C}
D. {D}

答案：""",

    ("mc", "terse"): """看图答题，只输出字母。

{question}
A. {A}
B. {B}
C. {C}
D. {D}
答案：""",

    # 判断题
    ("tf", "base"): """请仔细阅读图片，回答问题，只回答「是」或「否」，不要解释。

问题：{question}

答案：""",

    ("tf", "nobias"): """请仔细阅读图片，回答问题。注意：正确答案不一定是「是」，请独立判断，不要默认回答「是」。
只回答「是」或「否」，不要解释。

问题：{question}

答案：""",

    # 换用「正确/错误」措辞，看是否只是「是」这个 token 的先验问题
    ("tf", "correct_wrong"): """请仔细阅读图片，判断下列说法是否正确。只回答「正确」或「错误」，不要解释。

说法：{question}

答案：""",

    # 转成 A/B 选项格式，与多选题一致，排除「是/否」二字本身的影响
    ("tf", "ab_format"): """请仔细阅读图片，回答问题，只输出选项字母，不要解释。

问题：{question}
A. 是
B. 否

答案：""",
}

# 变体名 -> 判断题的答案标签映射（用于评分时归一化）
TF_LABEL_MAP = {
    "correct_wrong": {"正确": "是", "错误": "否"},
    "ab_format": {"A": "是", "B": "否"},
}

PROMPT_MC = PROMPT_VARIANTS[("mc", "base")]
PROMPT_TF = PROMPT_VARIANTS[("tf", "base")]


# ════════════════════════════════════════════════════════════
# 答案提取
# ════════════════════════════════════════════════════════════
def extract_mc(text: str) -> str:
    """从生成文本里抠多选题字母。

    优先级：
      1. 显式标记「答案：X」—— 模型听话时走这条
      2. 首个「独立」字母 —— 排除单词内部的字母
    注意：模型输出 'A,B,C,D' 时按首个字母算，这是已知歧义，
          raw output 已保存，可事后重算。
    """
    t = text.strip().upper()
    m = re.search(r"答\s*案\s*[:：]?\s*\**\s*([ABCD])", t)
    if m:
        return m.group(1)
    for m in re.finditer(r"[ABCD]", t):
        i = m.start()
        prev = t[i - 1] if i > 0 else " "
        nxt = t[i + 1] if i + 1 < len(t) else " "
        if prev not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and nxt not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return m.group(0)
    return "?"


def extract_tf(text: str, variant: str = "base") -> str:
    """从生成文本里抠判断题答案，一律归一化为「是」/「否」。

    注意「正确/错误」里「不正确」包含「正确」，所以先查否定词再查肯定词，
    否则「不正确」会被误判成「正确」。
    """
    t = text.strip()

    # ab_format: 模型输出 A/B
    if variant == "ab_format":
        for ch in t.upper():
            if ch == "A":
                return "是"
            if ch == "B":
                return "否"
        return "?"

    head = t[:20]
    # 先查否定（长词优先，"不正确"/"错误" 要先于 "正确" 匹配）
    for k in ("不正确", "错误", "否", "不对", "错"):
        if k in head:
            return "否"
    for k in ("正确", "是", "对"):
        if k in head:
            return "是"
    return "?"


# ════════════════════════════════════════════════════════════
# 数据加载
# ════════════════════════════════════════════════════════════
def load_dataset() -> pd.DataFrame:
    """加载全部 TSV，统一成标准 schema，并标注题型。

    返回列: qtype / src / question / options / gold / image_path
    """
    records = []
    stats = Counter()

    for tsv in sorted(DATA_DIR.glob("L*_Q*.tsv")):
        if tsv.name in EXCLUDE_FILES:
            stats["跳过文件(L3_Q4 多轮案例)"] += 1
            continue
        try:
            df = pd.read_csv(tsv, sep="\t", dtype=str)
        except Exception as e:
            print(f"  ⚠ 读取失败 {tsv.name}: {e}")
            stats["读取失败"] += 1
            continue

        cols = set(df.columns)
        n0 = len(df)

        # 图片列名不统一：多数 TSV 用 image，L3_Q1 用 image_path
        img_col = "image" if "image" in cols else ("image_path" if "image_path" in cols else None)

        if img_col and {"question", "answer"} <= cols and {"A", "B", "C", "D"} <= cols:
            # 多选题 or 判断题：靠 answer 取值区分
            ans = df["answer"].astype(str).str.strip()
            is_tf = ans.isin(["是", "否"])
            is_mc = ans.str.upper().str[:1].isin(list("ABCD"))

            # 关键：image 单元格可能是「逗号分隔的多张图」
            # 例: data/figure/fs/A.jpg,data/figure/fs/B.jpg,data/figure/fs/C.jpg
            # L2_Q2 每题最多 11 张 —— 这是多图推理题，不是脏数据，不能丢
            def split_imgs(v):
                if pd.isna(v):
                    return []
                return [s.strip() for s in str(v).split(",") if s.strip()]

            for _, r in df[is_mc].iterrows():
                records.append({
                    "qtype": QTYPE_MC, "src": tsv.name,
                    "question": str(r["question"]).strip(),
                    "options": {k: (str(r[k]).strip() if pd.notna(r[k]) else "") for k in "ABCD"},
                    "gold": str(r["answer"]).strip().upper()[:1],
                    "images": split_imgs(r[img_col]),
                })
            for _, r in df[is_tf].iterrows():
                records.append({
                    "qtype": QTYPE_TF, "src": tsv.name,
                    "question": str(r["question"]).strip(),
                    "options": {},
                    "gold": "是" if str(r["answer"]).strip() == "是" else "否",
                    "images": split_imgs(r[img_col]),
                })
            stats["多选题"] += int(is_mc.sum())
            stats["判断题"] += int(is_tf.sum())
            stats["丢弃(答案为空/格式不明)"] += int(n0 - is_mc.sum() - is_tf.sum())
        else:
            stats["跳过(列结构不匹配)"] += n0
            stats[f"  结构: {tsv.name}"] = sorted(cols)

    df = pd.DataFrame(records)
    base = DATA_DIR.parent
    df["abs_images"] = df["images"].apply(lambda ps: [str(base / p) for p in ps])
    df["num_images"] = df["abs_images"].apply(len)

    # 丢掉任一图片不存在的
    ok = df["abs_images"].apply(lambda ps: len(ps) > 0 and all(Path(p).exists() for p in ps))
    stats["丢弃(图片缺失或为空)"] = int((~ok).sum())
    df = df[ok].reset_index(drop=True)

    # ★★ uid 必须在这里（load 顺序）统一分配，不能由调用方各自分配。
    #
    # 踩过的坑：prepare_data.py 在 load 顺序上编号，eval_baseline.py 在
    # shuffle 之后编号 —— 两套编号完全不同，导致 --uids-file 筛出来的
    # 根本不是 test split（很可能筛到训练样本，冒烟测试准确率 100% 就是这个原因）。
    #
    # 现在 uid 唯一的定义点就是这里：同一份数据，任何脚本拿到的 uid 都一致。
    df = df.reset_index(drop=True)
    df["uid"] = range(len(df))

    print("── 数据加载 ──")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  可用总计: {len(df)}")
    for qt in (QTYPE_MC, QTYPE_TF):
        sub = df[df["qtype"] == qt]
        if len(sub):
            single = int((sub["num_images"] == 1).sum())
            multi = int((sub["num_images"] > 1).sum())
            print(f"    {qt}: {len(sub)}   (单图 {single} / 多图 {multi})")
    return df


def build_prompt(row, variant: str = "base") -> str:
    qt = row["qtype"]
    tmpl = PROMPT_VARIANTS.get((qt, variant)) or PROMPT_VARIANTS[(qt, "base")]
    if qt == QTYPE_MC:
        o = row["options"]
        return tmpl.format(
            question=row["question"],
            A=o.get("A") or "-", B=o.get("B") or "-",
            C=o.get("C") or "-", D=o.get("D") or "-",
        )
    return tmpl.format(question=row["question"])


# ════════════════════════════════════════════════════════════
# 指标
# ════════════════════════════════════════════════════════════
def _stats(sub: pd.DataFrame) -> dict:
    """对任意子集算 raw accuracy + 常数基线 + 相对增益。"""
    n = len(sub)
    if n == 0:
        return {}
    correct = int((sub["pred"] == sub["gold"]).sum())
    raw = correct / n

    dist = sub["gold"].value_counts()
    maj_label, maj_n = dist.index[0], int(dist.iloc[0])
    const_base = maj_n / n
    k = len(dist)

    return {
        "n": n,
        "correct": correct,
        "raw_accuracy": round(raw, 4),
        "constant_baseline": round(const_base, 4),
        "constant_label": maj_label,
        "random_baseline": round(1.0 / k, 4),
        "lift_vs_constant": round(raw - const_base, 4),
        "lift_vs_random": round(raw - 1.0 / k, 4),
        "unparsed": int((sub["pred"] == "?").sum()),
        "gold_distribution": {str(a): int(b) for a, b in dist.items()},
        "pred_distribution": {str(a): int(b) for a, b in sub["pred"].value_counts().items()},
    }


def compute_metrics(df: pd.DataFrame) -> dict:
    """分题型 + 分单图/多图 输出指标。

    多图题（最多 11 张）是「跨图表推理」，和单图题是两种能力，必须分开报。
    """
    out = {}
    for qt in (QTYPE_MC, QTYPE_TF):
        sub = df[df["qtype"] == qt]
        if len(sub) == 0:
            continue
        out[qt] = _stats(sub)

        # 单图 / 多图拆分
        s1 = sub[sub["num_images"] == 1]
        sm = sub[sub["num_images"] > 1]
        if len(s1):
            out[f"{qt}_single"] = _stats(s1)
        if len(sm):
            out[f"{qt}_multi"] = _stats(sm)

    out["_overall"] = _stats(df)
    return out


def print_metrics(m: dict, title: str = ""):
    if title:
        print(f"\n{'═'*70}\n {title}\n{'═'*70}")
    names = {
        "mc": "多选题 · 全部", "mc_single": "多选题 · 单图",
        "mc_multi": "多选题 · 多图(跨图表推理)",
        "tf": "判断题 · 全部", "tf_single": "判断题 · 单图",
        "tf_multi": "判断题 · 多图",
        "_overall": "总体",
    }
    for k, s in m.items():
        if not s:
            continue
        name = names.get(k, k)
        print(f"\n── {name}  (n={s['n']}) ──")
        print(f"  raw accuracy          : {s['raw_accuracy']*100:6.2f}%   ({s['correct']}/{s['n']})")
        print(f"  常数基线 (全选「{s['constant_label']}」): {s['constant_baseline']*100:6.2f}%   ★ 及格线")
        print(f"  随机基线              : {s['random_baseline']*100:6.2f}%")
        print(f"  ───────────────────────────────────────")
        print(f"  相对常数基线增益      : {s['lift_vs_constant']*100:+6.2f} 个百分点")
        print(f"  相对随机增益          : {s['lift_vs_random']*100:+6.2f} 个百分点")
        if s["unparsed"]:
            print(f"  ⚠ 无法解析的输出      : {s['unparsed']} 条 ({s['unparsed']/s['n']*100:.1f}%)")
        print(f"  预测分布              : {s['pred_distribution']}")
        print(f"  真值分布              : {s['gold_distribution']}")


# ════════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════════
def run_inference(df: pd.DataFrame, out_path: Path, max_new_tokens: int,
                  variant: str = "base", adapter: str = ""):
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print("\n── 环境 ──")
    print(f"  torch        : {torch.__version__}")
    print(f"  transformers : {transformers.__version__}")
    print(f"  device       : {torch.cuda.get_device_name(0)}")
    print(f"  样本数       : {len(df)}")

    print("\n── 加载模型 ──")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL_DIR), dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()

    # LoRA adapter（微调后评测用）。
    # 用 merge_and_unload() 把 adapter 合进基座：
    #   - 推理更快（无额外 kernel launch）
    #   - 与 zero-shot baseline 走完全相同的代码路径，排除"加载方式"这个混淆变量
    if adapter:
        from peft import PeftModel
        print(f"  加载 LoRA adapter: {adapter}")
        model = PeftModel.from_pretrained(model, str(adapter))
        model = model.merge_and_unload()
        print("  ✓ adapter 已合并进基座")
    print(f"  ✓ {time.time()-t0:.1f}s")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    if out_path.exists():
        with open(out_path) as f:
            done = sum(1 for _ in f)
        print(f"  续跑：已完成 {done} 条，跳过")

    t_start = time.time()
    with open(out_path, "a", encoding="utf-8") as fout:
        for i, row in df.iloc[done:].iterrows():
            try:
                # 多图题：每张图一个 image content，最后接文本
                content = []
                for p in row["abs_images"]:
                    content.append({"type": "image", "image": Image.open(p).convert("RGB")})
                content.append({"type": "text", "text": build_prompt(row, variant)})

                messages = [{"role": "user", "content": content}]
                inputs = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                ).to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                gen = processor.batch_decode(
                    out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )[0]
            except Exception as e:
                gen = f"__ERROR__ {type(e).__name__}: {e}"

            pred = (extract_mc(gen) if row["qtype"] == QTYPE_MC
                    else extract_tf(gen, variant))
            fout.write(json.dumps({
                "uid": int(row["uid"]),          # 全局唯一，跨 shard 不冲突
                "idx": int(i), "qtype": row["qtype"], "src": row["src"],
                "variant": variant,
                "adapter": Path(adapter).name if adapter else "",
                "gold": row["gold"], "pred": pred, "raw_output": gen,
                "num_images": int(row["num_images"]),
                "images": [Path(p).name for p in row["abs_images"]],
            }, ensure_ascii=False) + "\n")
            fout.flush()

            n = i + 1
            if n % 50 == 0 or n == len(df):
                el = time.time() - t_start
                sp = (n - done) / el if el > 0 else 0
                eta = (len(df) - n) / sp / 60 if sp > 0 else 0
                print(f"  [{n}/{len(df)}]  {sp:.1f} it/s  ETA {eta:.1f} min", flush=True)
    print(f"\n  ✓ 推理完成 -> {out_path}")


def load_predictions(paths: list[Path]) -> pd.DataFrame:
    """读所有 shard 的预测。

    注意 uid 的构造：`idx` 是**分片内部**的行号，4 个 shard 都从 0 开始，
    单靠 idx 去重会把 shard1-3 全部当成重复丢掉（曾踩过这个坑）。
    所以 uid = "shard文件名:idx"，全局唯一。
    """
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                r["uid"] = r.get("uid") or f"{p.stem}:{r['idx']}"
                rows.append(r)
    return pd.DataFrame(rows)


def merge(out_dir: Path):
    files = sorted(out_dir.glob("predictions.shard*.jsonl"))
    if not files:
        sys.exit(f"✗ {out_dir} 下没有 predictions.shard*.jsonl")
    df = load_predictions(files)
    n_raw = len(df)
    df = df.drop_duplicates(subset=["uid"], keep="first")
    if len(df) != n_raw:
        print(f"  ⚠ 去重: {n_raw} -> {len(df)}")
    m = compute_metrics(df)
    print_metrics(m, "Qwen3-VL-8B-Instruct  zero-shot  on VisFinEval")
    (out_dir / "metrics.json").write_text(
        json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  ✓ 指标已存 -> {out_dir/'metrics.json'}   (预测 {len(df)} 条)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "zeroshot"))
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条（调试用，shuffle 后取）")
    ap.add_argument("--seed", type=int, default=0, help="shuffle 与抽样种子")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--variant", default="base", help="prompt 变体名，见 PROMPT_VARIANTS")
    ap.add_argument("--qtype", default="", choices=["", "mc", "tf"], help="只跑某题型")
    ap.add_argument("--adapter", default="", help="LoRA adapter 目录（微调后评测用）")
    ap.add_argument("--uids-file", default="", help="只评测该文件列出的 uid（用于 test split）")
    ap.add_argument("--merge", action="store_true", help="只汇总，不推理")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.merge:
        merge(out_dir)
        return

    df = load_dataset()

    # 确定性 shuffle：
    #   - 让 --limit 抽到的是混合题型，而不是单个 TSV 的前 N 条
    #   - 让多卡分片负载均衡（否则 shard0 可能全是 L1、shard3 全是 L3）
    #   seed 固定 -> 每次运行顺序一致 -> 分片和续跑可复现
    # uid 已由 load_dataset() 在 load 顺序上分配好，shuffle 只改行序不改 uid。
    # 这样 prepare_data.py 切分出的 test_uids.json 能被这里正确解释。
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    print(f"  已 shuffle (seed={args.seed})  [uid 由 load_dataset 统一分配，shuffle 不改动]")

    # 只评测指定 uid（通常是 test split）。
    # ★ 必须在 shuffle 之后、qtype 过滤之前用全局 uid 过滤：
    #   zero-shot baseline 和 LoRA 评测都走这条路径 -> 同一批样本，可做配对比较。
    if args.uids_file:
        keep = set(json.loads(Path(args.uids_file).read_text(encoding="utf-8")))
        before = len(df)
        df = df[df["uid"].isin(keep)].reset_index(drop=True)
        print(f"  已按 uid 过滤: {before} -> {len(df)} 条  ({args.uids_file})")

    if args.qtype:
        df = df[df["qtype"] == args.qtype].reset_index(drop=True)
        print(f"  已过滤题型: {args.qtype} -> {len(df)} 条")
    if args.limit:
        df = df.iloc[: args.limit].reset_index(drop=True)
    if args.num_shards > 1:
        df = df.iloc[args.shard_id :: args.num_shards].reset_index(drop=True)

    print(f"  prompt 变体: {args.variant}")
    print(f"  adapter    : {args.adapter or '(无，zero-shot)'}")
    out_path = out_dir / f"predictions.shard{args.shard_id}.jsonl"
    run_inference(df, out_path, args.max_new_tokens, args.variant, args.adapter)

    if args.num_shards == 1:
        merge(out_dir)


if __name__ == "__main__":
    main()
