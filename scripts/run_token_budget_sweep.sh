#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# 视觉 token 预算 × 图数量  双因子实验
#
# 要回答的问题
#   发现 2 观测到「10 图题崩到 29.5%，比无脑全选 A 低 29.5pp」。
#   机制是哪个？
#
#     假设 A  视觉 token 预算耗尽
#             每图按动态分辨率编码，财报原图最大 5.8M 像素 ->
#             单图最多 4281 token，10 图 4 万+。
#             若降低每图 token 预算，高图数题应该跌得更狠。
#
#     假设 B  能力上限（与 token 数无关）
#             若曲线形状不随预算变化，说明模型就是做不了多路视觉整合。
#
# 背景事实（已实测，见 results/RESULTS.md）
#   • processor 默认 longest_edge=16777216，等于没有上限
#   • MAX_PIXELS 环境变量是 ms-swift 的约定，transformers **不读**
#   • 实测单图 token: 无上限 4281 / cap=1M 979 / cap=500K 461 / cap=250K 243
#
# 设计
#   4 档预算 × (多图题 1439 + 单图抽样 1500)
#   单图组是必要的对照：预算下降如果单图也跌，就不是"多图特有"的问题
#
# 用法
#   bash scripts/run_token_budget_sweep.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

PY="$HOME/miniforge3/envs/vlm/bin/python"
export NPROC_PER_NODE=1          # 本脚本自己按 GPU 分片，不用 torchrun
GPUS=(4 5 6 7)
NSHARD=${#GPUS[@]}

# 数据：全部多图题 + 等量单图题作对照
UIDS="$ROOT/data/token_budget_uids.json"
if [ ! -f "$UIDS" ]; then
  echo "── 生成实验样本集 ──"
  "$PY" - <<'PYEOF'
import sys, json
sys.path.insert(0, "scripts")
from pathlib import Path
from eval_baseline import load_dataset

df = load_dataset()
multi = df[df["num_images"] > 1]
single = df[df["num_images"] == 1].sample(n=min(1500, (df["num_images"]==1).sum()),
                                          random_state=0)
sel = __import__("pandas").concat([multi, single])
uids = sorted(int(x) for x in sel["uid"].tolist())
Path("data/token_budget_uids.json").write_text(json.dumps(uids), encoding="utf-8")
print(f"  多图 {len(multi)} + 单图 {len(single)} = {len(uids)} 条 -> {UIDS if False else 'data/token_budget_uids.json'}")
PYEOF
fi

echo "════════════════════════════════════════"
echo " 视觉 token 预算 sweep    $(date '+%F %T')"
echo "════════════════════════════════════════"
echo "  GPUs : ${GPUS[*]}"
echo "  样本 : $(python3 -c "import json;print(len(json.load(open('$UIDS'))))") 条"
echo

run_budget() {
    local cap="$1" tag="$2"
    local out="$ROOT/results/tokenbudget_${tag}"
    rm -rf "$out"; mkdir -p "$out" "$ROOT/logs"

    echo "────────────────────────────────────────"
    echo " [$tag]  longest_edge=${cap:-无上限}"
    local pids=()
    for i in $(seq 0 $((NSHARD-1))); do
        local g=${GPUS[$i]}
        local cap_arg=()
        [ -n "$cap" ] && cap_arg=(--max-pixels "$cap")
        CUDA_VISIBLE_DEVICES=$g nohup "$PY" scripts/eval_baseline.py \
            --uids-file "$UIDS" --variant base "${cap_arg[@]}" \
            --shard-id "$i" --num-shards "$NSHARD" \
            --out-dir "$out" \
            > "$ROOT/logs/tb_${tag}_shard${i}.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || echo "  ✗ shard 失败"; done
    echo "  ✓ 完成 ($(cat "$out"/predictions.shard*.jsonl 2>/dev/null | wc -l) 条)"
}

# 4 档预算
run_budget ""         "nocaq"      # 空字符串 = 不设上限
run_budget 1003520    "cap1m"
run_budget 501760     "cap500k"
run_budget 250880     "cap250k"

echo
echo "════════ 完成 $(date '+%F %T') ════════"
echo "  分析: python scripts/analyze_token_budget.py"
