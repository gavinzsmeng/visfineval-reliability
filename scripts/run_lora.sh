#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Qwen3-VL-8B LoRA 微调 —— VisFinEval
#
# 核心科学问题:
#   微调能否修复 baseline 上发现的失败模式？
#   特别是：判断题少数类「否」的召回率（zero-shot 仅 42.5%，低于随机）
#
# 关键设计:
#   • 按来源研报切分（prepare_data.py），无泄漏
#   • freeze_vit true + freeze_aligner true —— 只训 LM 的 LoRA
#     （与 ChartCF/ACL2026 的做法一致；也能显著省显存）
#   • 训练 prompt 与评测 prompt **完全一致**（否则口径不对齐）
#
# 显存约束:
#   Qwen3-VL-8B bf16 权重 ≈ 17.5 GB，单卡 24 GB
#   ZeRO-2 只切优化器/梯度（LoRA 部分很小），权重仍全量复制
#   真正省显存靠: freeze_vit + gradient_checkpointing + MAX_PIXELS 限制
#   若 OOM -> 改用 configs/ds_zero3.json
#
# 用法:
#   bash scripts/run_lora.sh natural     # 原始分布
#   bash scripts/run_lora.sh balanced    # 标签均衡
#   SMOKE=1 bash scripts/run_lora.sh natural   # 冒烟测试（少量 step）
# ─────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

COND="${1:-natural}"
SMOKE="${SMOKE:-0}"
GPUS="${GPUS:-4,5,6,7}"
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)

case "$COND" in
  natural)  DATA="$ROOT/data/swift" ;;
  balanced) DATA="$ROOT/data/swift_bal" ;;
  *) echo "用法: $0 [natural|balanced]"; exit 1 ;;
esac

[ -f "$DATA/train.jsonl" ] || { echo "✗ 缺 $DATA/train.jsonl，先跑 prepare_data.py"; exit 1; }

OUT="$ROOT/output/lora_${COND}"
mkdir -p "$OUT" "$ROOT/logs"
# ★ 不要写成 LOG="...$( [ x = y ] && echo z ).log"：
#   SMOKE!=1 时 [ ] 返回 1，&& 短路使命令替换退出码为 1，
#   set -e 会因此静默杀掉整个脚本（踩过，排查了半天）。
if [ "$SMOKE" = "1" ]; then
  SUFFIX="_smoke"
else
  SUFFIX=""
fi
LOG="$ROOT/logs/lora_${COND}${SUFFIX}.log"

# 冒烟模式：极小规模，只为验证管线跑得通
if [ "$SMOKE" = "1" ]; then
  EXTRA="--max_steps 20 --logging_steps 1 --save_steps 20 --dataset_num_proc 1"
  OUT="${OUT}_smoke"
else
  EXTRA="--num_train_epochs 2 --logging_steps 10 --save_steps 200 --eval_steps 200"
fi

echo "════════════════════════════════════════"
echo " LoRA 微调: $COND   $(date '+%F %T')"
echo "════════════════════════════════════════"
echo "  hostname : $(hostname)"
echo "  GPUs     : $GPUS  (nproc=$NPROC)"
echo "  data     : $DATA"
echo "  out      : $OUT"
echo "  smoke    : $SMOKE"
echo "  swift SHA: $(git -C "$ROOT/ms-swift" rev-parse --short HEAD)"
echo

# 起跑前确认 GPU 空闲
for g in $(echo "$GPUS" | tr ',' ' '); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")
  printf '  GPU %s: %s MiB\n' "$g" "$used"
  [ "$used" -gt 2000 ] && { echo "  ⚠ GPU $g 被占用，中止"; exit 1; }
done
echo

# ★ deepspeed 是 ms-swift 的**可选**依赖（不在 requirements/framework.txt 里）。
#   LoRA 的优化器状态极小，ZeRO-2 相对 DDP 收益可忽略 -> 默认用 DDP。
#   若 OOM 或需要切分权重，装 deepspeed 后加 USE_DEEPSPEED=1 重跑。
if [ "${USE_DEEPSPEED:-0}" = "1" ]; then
    DS_ARG=(--deepspeed "$ROOT/configs/ds_zero2.json")
    echo "  并行: DeepSpeed ZeRO-2"
else
    DS_ARG=()
    echo "  并行: DDP（未用 deepspeed）"
fi

# ★ 端口不能用作默认的 29500：
#   本机是共享的，另一个用户的任务（GPU 0-3）长期占着 29500。
#   用 torchrun 默认端口会直接 EADDRINUSE 启动失败。
#
# 注意：不要写成 MASTER_PORT="${MASTER_PORT:-$(python3 - <<'PY' ... PY)}"，
#       heredoc 嵌在 $() 里再嵌在 ${:-} 里，bash 解析很脆。
#       用 python3 -c 单行更稳。
if [ -z "${MASTER_PORT:-}" ]; then
  MASTER_PORT=$(python3 -c 'import socket
for p in range(29517, 29600):
    s=socket.socket()
    try:
        s.bind(("127.0.0.1",p)); s.close(); print(p); break
    except OSError: s.close()')
fi

# ★ 必须用 export 而不是行内前缀：
#   ms-swift 的 cli/main.py:use_torchrun() 靠读 NPROC_PER_NODE 决定
#   是否用 torch.distributed.run 重启自己。
#   行内前缀 + 管道(tee) 时变量没能到达 swift 进程，
#   结果 world_size=1（单卡），ETA 从 1.5h 变成 12.6h。踩过。
export CUDA_VISIBLE_DEVICES="$GPUS"
export NPROC_PER_NODE="$NPROC"
export MASTER_PORT
export MAX_PIXELS=1003520
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "  CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "  NPROC_PER_NODE       = $NPROC_PER_NODE   <- ms-swift 靠这个决定是否多卡"
echo "  MASTER_PORT          = $MASTER_PORT"
echo

# ★ 前台运行，不 background：
#   调用方(chain 脚本)需要能 wait 到真实结束，否则会误判"已完成"。
#   需要后台时由调用方 nohup ... & 包一层。
"$HOME/miniforge3/envs/vlm/bin/swift" sft \
    --model "$ROOT/models/Qwen3-VL-8B-Instruct" \
    --dataset "$DATA/train.jsonl" \
    --val_dataset "$DATA/val.jsonl" \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --freeze_vit true \
    --freeze_aligner true \
    --target_modules all-linear \
    --lora_rank 16 \
    --lora_alpha 32 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing true \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    "${DS_ARG[@]}" \
    --dataloader_num_workers 4 \
    --split_dataset_ratio 0 \
    --save_total_limit 2 \
    --seed 0 \
    --output_dir "$OUT" \
    $EXTRA \
    2>&1 | tee "$LOG"

echo
echo "  ✓ 训练结束 -> $LOG"
