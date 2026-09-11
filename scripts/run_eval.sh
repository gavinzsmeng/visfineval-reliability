#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# 4 卡并行跑 zero-shot baseline
#
# GPU 分工：
#   本机 8 张 4090，其中 0-3 被另一个用户占用（/data/lzy）。
#   只用 4,5,6,7。
#
# 用法：
#   bash scripts/run_eval.sh              # 全量
#   LIMIT=100 bash scripts/run_eval.sh    # 小样本自测
# ─────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

PY="$HOME/miniforge3/envs/vlm/bin/python"
GPUS=(4 5 6 7)
NSHARD=${#GPUS[@]}
OUT_DIR="${OUT_DIR:-$ROOT/results/zeroshot}"
LIMIT="${LIMIT:-0}"
TAG="${TAG:-zeroshot}"

mkdir -p logs "$OUT_DIR"

echo "════════════════════════════════════════"
echo " zero-shot baseline    $(date '+%F %T')"
echo "════════════════════════════════════════"
echo "  hostname : $(hostname)"
echo "  GPUs     : ${GPUS[*]}"
echo "  shards   : $NSHARD"
echo "  out      : $OUT_DIR"
echo "  limit    : ${LIMIT:-0} (0=全量)"
echo "  model SHA: $(git -C "$ROOT/ms-swift" rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo

# 起跑前确认 GPU 没被别人占
echo "── GPU 占用检查 ──"
for g in "${GPUS[@]}"; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  printf '  GPU %s: %s MiB 已用\n' "$g" "$used"
  if [ "$used" -gt 2000 ]; then
    echo "  ⚠ GPU $g 已被占用，请改 GPUS 列表后再跑"; exit 1
  fi
done
echo

if [ "${LIMIT:-0}" -gt 0 ]; then
  LIMIT_ARG=(--limit "$LIMIT")
else
  LIMIT_ARG=()
fi

pids=()
for i in $(seq 0 $((NSHARD-1))); do
  g=${GPUS[$i]}
  log="$ROOT/logs/eval_${TAG}_shard${i}.log"
  echo "  启动 shard $i -> GPU $g  ($log)"
  CUDA_VISIBLE_DEVICES=$g nohup "$PY" scripts/eval_baseline.py \
    --shard-id "$i" --num-shards "$NSHARD" \
    "${LIMIT_ARG[@]}" \
    --out-dir "$OUT_DIR" \
    > "$log" 2>&1 &
  pids+=($!)
done

echo
echo "  所有 shard 已启动: ${pids[*]}"
echo "  实时监控:  tail -f $ROOT/logs/eval_${TAG}_shard0.log"
echo "  等待全部完成..."
echo

fail=0
for p in "${pids[@]}"; do
  wait "$p" || { echo "  ✗ PID $p 退出码非 0"; fail=1; }
done

if [ "$fail" -ne 0 ]; then
  echo
  echo "⚠ 有 shard 失败，检查 logs/eval_${TAG}_shard*.log"
  exit 1
fi

echo
echo "════════ 汇总 ════════"
"$PY" scripts/eval_baseline.py --merge --out-dir "$OUT_DIR"
