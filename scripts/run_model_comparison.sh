#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# 多模型横向对比
#
# 目的：把结论从「一个模型的诊断」升级为「benchmark 的系统性问题」。
#       如果只有 Qwen3-VL 在少数类上崩，那可能是这家模型的怪癖；
#       如果多个家族都崩，且常数基线都同样高，那就是 benchmark 的性质。
#
# 设计（保证可比）
#   · 同一 test split（data/swift/test_uids.json，2,889 条）
#   · 同一 prompt（base 变体，未被任何模型特调）
#   · 同一指标口径（raw / 常数基线 / macro-recall / 少数类召回）
#   · 都不设 max_pixels（各模型用自家 processor 默认）
#
# 用法:
#   bash scripts/run_model_comparison.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

PY="$HOME/miniforge3/envs/vlm/bin/python"
GPUS=(4 5 6 7)
NSHARD=${#GPUS[@]}
UIDS="$ROOT/data/swift/test_uids.json"

# tag : 模型路径 : 额外参数
MODELS=(
  "qwen3vl-8b:models/Qwen3-VL-8B-Instruct:"
  "qwen25vl-7b:models/Qwen2.5-VL-7B-Instruct:"
)

run_one() {
    local tag="$1" path="$2" extra="$3"
    local out="$ROOT/results/model_${tag}"

    if [ ! -d "$ROOT/$path" ]; then
        echo "[$tag] 跳过：$path 不存在"
        return
    fi

    echo "[$tag] 评测中 -> $out"
    rm -rf "$out"; mkdir -p "$out" "$ROOT/logs"

    local pids=()
    for i in $(seq 0 $((NSHARD-1))); do
        local g=${GPUS[$i]}
        CUDA_VISIBLE_DEVICES=$g nohup "$PY" scripts/eval_baseline.py \
            --uids-file "$UIDS" --variant base \
            --model "$ROOT/$path" $extra \
            --shard-id "$i" --num-shards "$NSHARD" \
            --out-dir "$out" \
            > "$ROOT/logs/model_${tag}_shard${i}.log" 2>&1 &
        pids+=($!)
    done
    local fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    local n
    n=$(cat "$out"/predictions.shard*.jsonl 2>/dev/null | wc -l)
    if [ "$fail" -ne 0 ] || [ "$n" -lt 2889 ]; then
        echo "[$tag] ⚠ 失败或条数不足 ($n/2889)，看 logs/model_${tag}_shard0.log"
    else
        echo "[$tag] ✓ $n 条"
    fi
}

echo "多模型对比  $(date '+%F %T')"
for entry in "${MODELS[@]}"; do
    IFS=':' read -r tag path extra <<< "$entry"
    run_one "$tag" "$path" "$extra"
done

echo
echo "分析: python scripts/compare_models.py"
