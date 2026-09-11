#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Prompt 敏感性验证
#
# 目的: baseline 上观察到两个失败模式
#         ① 判断题少数类「否」召回率仅 42.5%（低于随机）
#         ② 多图题随图数增加而崩溃
#       本脚本用多个 prompt 措辞重复实验，确认它们不是单一 prompt 的假象。
#
# 判据:
#   若所有变体下失败模式都复现  -> 是模型的性质，baseline 结论成立
#   若某个变体大幅改善          -> 之前的结论是 prompt artifact，必须修正
#
# 规模: 判断题全量 2804 x 4 变体 + 多选题抽样 3000 x 2 变体 ≈ 17k 次推理
#       4 卡约 8 分钟
# ─────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

PY="$HOME/miniforge3/envs/vlm/bin/python"
GPUS=(4 5 6 7)
NSHARD=${#GPUS[@]}

run_variant() {
    local qtype="$1" variant="$2" limit="$3"
    local out="$ROOT/results/sensitivity/${qtype}_${variant}"
    mkdir -p "$out" "$ROOT/logs"

    echo "── [$qtype/$variant] -> $out"
    local pids=()
    for i in $(seq 0 $((NSHARD-1))); do
        local g=${GPUS[$i]}
        local lim_arg=()
        [ "$limit" -gt 0 ] && lim_arg=(--limit "$limit")
        CUDA_VISIBLE_DEVICES=$g nohup "$PY" scripts/eval_baseline.py \
            --qtype "$qtype" --variant "$variant" \
            --shard-id "$i" --num-shards "$NSHARD" \
            "${lim_arg[@]}" \
            --out-dir "$out" \
            > "$ROOT/logs/sens_${qtype}_${variant}_shard${i}.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || echo "  ✗ 有 shard 失败"; done
    echo "  ✓ 完成"
}

echo "════════════════════════════════════════"
echo " Prompt 敏感性验证    $(date '+%F %T')"
echo "════════════════════════════════════════"
echo "  GPUs : ${GPUS[*]}"
echo

# 判断题：全量 2804，4 个变体
for v in base nobias correct_wrong ab_format; do
    run_variant tf "$v" 0
done

# 多选题：抽样 3000（全量 16404 x 2 变体太慢），2 个变体
for v in base nobias; do
    run_variant mc "$v" 3000
done

echo
echo "════════ 全部完成 $(date '+%F %T') ════════"
echo "  分析: python scripts/analyze_sensitivity.py"
