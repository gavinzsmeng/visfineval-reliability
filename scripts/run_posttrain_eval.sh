#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# 训练后评测：zero-shot vs LoRA(natural) vs LoRA(balanced)
#
# 核心设计 —— 三者必须严格可比：
#   • 同一 test split（按 uid 过滤，data/swift/test_uids.json）
#   • 同一 prompt 变体（base，且已验证与训练 prompt 逐字一致）
#   • 同一推理代码路径（merge_and_unload 后走 eval_baseline.py）
#   → 差异只来自「模型」，可以做逐样本配对检验
#
# 要回答的问题：
#   微调能否修复 baseline 发现的少数类崩溃？
#   （zero-shot 判断题「否」召回率仅 42.5%）
#
# 用法:
#   bash scripts/run_posttrain_eval.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

PY="$HOME/miniforge3/envs/vlm/bin/python"
GPUS=(4 5 6 7)
NSHARD=${#GPUS[@]}
UIDS="$ROOT/data/swift/test_uids.json"

[ -f "$UIDS" ] || { echo "✗ 缺 $UIDS，先跑 prepare_data.py"; exit 1; }

# ms-swift 的产物结构: output/lora_<cond>/v0-<时间戳>/checkpoint-<步数>
# 自动取最新的一个 checkpoint
find_ckpt() {
    local cond="$1"
    local base="$ROOT/output/lora_${cond}"
    [ -d "$base" ] || { echo ""; return; }
    # sort -V（版本排序）能正确把 checkpoint-422 排在 checkpoint-200 之后；
    # 用 -t- -k2 -n 之类按字段切会因为路径里的时间戳而错序。
    find "$base" -maxdepth 3 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -1
}

run_eval() {
    local tag="$1" adapter="$2"
    local out="$ROOT/results/${tag}"
    rm -rf "$out"; mkdir -p "$out" "$ROOT/logs"

    echo "────────────────────────────────────────"
    echo " [$tag]"
    echo "   adapter: ${adapter:-(无，zero-shot)}"
    echo "   out    : $out"

    local pids=()
    for i in $(seq 0 $((NSHARD-1))); do
        local g=${GPUS[$i]}
        local ad_arg=()
        [ -n "$adapter" ] && ad_arg=(--adapter "$adapter")
        CUDA_VISIBLE_DEVICES=$g nohup "$PY" scripts/eval_baseline.py \
            --uids-file "$UIDS" --variant base \
            "${ad_arg[@]}" \
            --shard-id "$i" --num-shards "$NSHARD" \
            --out-dir "$out" \
            > "$ROOT/logs/eval_${tag}_shard${i}.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || echo "  ✗ shard 失败"; done
    echo "  ✓ 完成"
}

echo "════════════════════════════════════════"
echo " 训练后评测    $(date '+%F %T')"
echo "════════════════════════════════════════"
echo "  GPUs  : ${GPUS[*]}"
echo "  test  : $(python3 -c "import json;print(len(json.load(open('$UIDS'))))") 条"
echo

CKPT_NAT=$(find_ckpt natural)
CKPT_BAL=$(find_ckpt balanced)
echo "  natural  checkpoint: ${CKPT_NAT:-未找到}"
echo "  balanced checkpoint: ${CKPT_BAL:-未找到}"
echo

# 1) zero-shot（基座，无 adapter）—— 作为对照
run_eval "posttrain_zeroshot" ""

# 2) LoRA natural
if [ -n "$CKPT_NAT" ]; then
    run_eval "posttrain_lora_natural" "$CKPT_NAT"
else
    echo "  ⚠ 跳过 natural（无 checkpoint）"
fi

# 3) LoRA balanced
if [ -n "$CKPT_BAL" ]; then
    run_eval "posttrain_lora_balanced" "$CKPT_BAL"
else
    echo "  ⚠ 跳过 balanced（无 checkpoint）"
fi

echo
echo "════════ 汇总 ════════"
for t in posttrain_zeroshot posttrain_lora_natural posttrain_lora_balanced; do
    [ -d "$ROOT/results/$t" ] || continue
    echo
    echo "── $t ──"
    "$PY" scripts/eval_baseline.py --merge --out-dir "$ROOT/results/$t" 2>/dev/null | \
        grep -E 'raw accuracy|常数基线|相对常数|预测分布|──' | head -14
done

echo
echo "════════ 配对比较 ════════"
"$PY" scripts/compare_conditions.py || true
