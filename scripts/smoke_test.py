#!/usr/bin/env python
"""
VLM-FinChart 冒烟测试 (Smoke Test)
────────────────────────────────────────────────────────────
目的:
  在启动任何完整评测/训练之前，用最小代价验证整条链路：
    数据(TSV+图) -> Processor -> Model -> 生成 -> 与 ground truth 比对

设计:
  • 只跑 N 条样本（默认 5），单卡，几分钟内出结果
  • 严格按选项字母评分，与后续正式评测口径一致
  • 打印每条的 prompt / 输出 / 是否正确，便于人眼判断格式对不对

用法:
  CUDA_VISIBLE_DEVICES=4 python scripts/smoke_test.py --n 5
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Qwen3-VL-8B-Instruct"
DATA_DIR = ROOT / "data" / "VisFinEval" / "data"

PROMPT_TMPL = """请仔细阅读图片，回答问题，只输出正确选项的字母（A/B/C/D），不要解释。

问题：{question}
A. {A}
B. {B}
C. {C}
D. {D}

答案："""


def load_samples(n: int, seed: int = 0, level: str = "L1"):
    """从 VisFinEval 的 TSV 里抽 n 条有图的样本。"""
    tsvs = sorted(DATA_DIR.glob(f"{level}_Q*.tsv"))
    if not tsvs:
        sys.exit(f"✗ 找不到 {level}_Q*.tsv，检查 {DATA_DIR}")

    frames = []
    for t in tsvs:
        try:
            df = pd.read_csv(t, sep="\t", dtype=str)
            df["__src"] = t.name
            frames.append(df)
        except Exception as e:
            print(f"  ⚠ 跳过 {t.name}: {e}")

    df = pd.concat(frames, ignore_index=True)

    # 只保留「真多选题」：
    #   VisFinEval 里混了三种题型，必须区分，否则评分口径是错的
    #     - 多选题   : A/B/C/D 列齐全，answer 是字母      <- 只用这种
    #     - 判断题   : L1_Q4，answer 是「是/否」，ABCD 全空
    #     - 多轮案例 : L3_Q4，字段结构不同
    for c in ("A", "B", "C", "D"):
        if c not in df.columns:
            sys.exit(f"✗ TSV 缺少列 {c}")
    ans = df["answer"].astype(str).str.strip().str.upper().str[:1]
    n_all = len(df)

    df = df[df["image"].notna() & ans.isin(list("ABCD"))]
    df["__imgpath"] = df["image"].apply(lambda p: DATA_DIR.parent / p)
    df = df[df["__imgpath"].apply(lambda p: p.exists())]

    print(f"  全部行数: {n_all}")
    print(f"  多选题  : {len(df)} 条 (已剔除判断题/多轮案例/空答案)")
    print(f"  来源 TSV: {len(frames)} 个")
    rng = random.Random(seed)
    idx = rng.sample(range(len(df)), min(n, len(df)))
    return df.iloc[idx].reset_index(drop=True)


def build_prompt(row) -> str:
    return PROMPT_TMPL.format(
        question=row["question"],
        A=row.get("A") or "-",
        B=row.get("B") or "-",
        C=row.get("C") or "-",
        D=row.get("D") or "-",
    )


def extract_choice(text: str) -> str:
    """从生成文本里抠出选项字母。取第一个出现的 A/B/C/D。"""
    for ch in text.strip().upper():
        if ch in "ABCD":
            return ch
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="样本数")
    ap.add_argument("--level", default="L1", help="L1/L2/L3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    print("═" * 60)
    print(" VLM-FinChart Smoke Test")
    print("═" * 60)
    print(f"  model      : {MODEL_DIR}")
    print(f"  cuda visible: {os.environ.get('CUDA_VISIBLE_DEVICES', '(all)')}")
    print(f"  torch      : {torch.__version__}  cuda={torch.version.cuda}")
    print(f"  device     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print()

    import transformers
    print(f"  transformers: {transformers.__version__}")
    print()

    print("── 1. 加载数据 ──")
    samples = load_samples(args.n, args.seed, args.level)
    print()

    print("── 2. 加载模型 ──")
    t0 = time.time()
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
    model = AutoModelForImageTextToText.from_pretrained(
        str(MODEL_DIR), dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()
    print(f"  ✓ 加载完成 {time.time()-t0:.1f}s")
    print()

    print("── 3. 推理 ──")
    correct = 0
    results = []
    for i, row in samples.iterrows():
        img = Image.open(row["__imgpath"]).convert("RGB")
        prompt = build_prompt(row)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        ).to(model.device)

        t1 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]
        dt = time.time() - t1

        pred = extract_choice(gen)
        gold = str(row["answer"]).strip().upper()[:1]
        ok = pred == gold
        correct += ok

        results.append(ok)
        print(f"  [{i+1}/{len(samples)}] {row['__src']:<12} 图={Path(row['__imgpath']).name[:38]:<38}")
        print(f"        问题: {str(row['question'])[:60]}")
        print(f"        生成: {gen.strip()[:60]!r}")
        print(f"        预测={pred}  真值={gold}  {'✓' if ok else '✗'}   {dt:.1f}s")

    print()
    print("═" * 60)
    print(f" 结果: {correct}/{len(samples)} = {correct/len(samples)*100:.1f}%")
    print("═" * 60)
    print()
    print(" 说明: 样本量极小，数字无统计意义。")
    print("       此测试只证明「链路通 + 格式对」，不是评测结果。")


if __name__ == "__main__":
    main()
