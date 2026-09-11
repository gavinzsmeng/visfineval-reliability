#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# VLM 项目环境搭建
#
# 设计决定:
#   • 新建独立 env，不复用 train / s3geo —— 避免污染你其它项目
#   • torch 选 2.6.0+cu124 —— 本机 train env 已验证可用（driver 580.82.07）
#   • ms-swift 从 clone 的源码装 (-e)，Git SHA 锁死，可复现
#   • 全程记录版本，产出 requirements.lock.txt
# ─────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

ENV_NAME=vlm
PY_VER=3.11
TORCH_VER=2.6.0
CUDA_TAG=cu124

LOG="$ROOT/logs/setup_env.log"
mkdir -p logs
exec > >(tee -a "$LOG") 2>&1

echo "════════════════════════════════════════"
echo " VLM 环境搭建   $(date '+%F %T')"
echo "════════════════════════════════════════"
echo "  hostname : $(hostname)"
echo "  root     : $ROOT"
echo "  env      : $ENV_NAME (python $PY_VER)"
echo "  torch    : $TORCH_VER+$CUDA_TAG"
echo

if conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo "⚠ env '$ENV_NAME' 已存在，跳过创建"
else
  echo "── [1/4] 创建 conda env ──"
  conda create -n "$ENV_NAME" python="$PY_VER" -y
fi

PIP="$HOME/miniforge3/envs/$ENV_NAME/bin/pip"
PY="$HOME/miniforge3/envs/$ENV_NAME/bin/python"

echo
echo "── [2/4] 安装 torch $TORCH_VER+$CUDA_TAG ──"
"$PIP" install --upgrade pip -q
"$PIP" install "torch==$TORCH_VER" torchvision \
  --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

echo
echo "── [3/4] 安装 ms-swift（从 clone 源码，锁定 SHA）──"
"$PIP" install -e "$ROOT/ms-swift"

echo
echo "── [4/4] 记录版本 ──"
{
  echo "# VLM 环境版本锁定"
  echo "# 生成时间: $(date '+%F %T %z')"
  echo "# hostname: $(hostname)"
  echo "# ms-swift SHA: $(git -C "$ROOT/ms-swift" rev-parse HEAD)"
  echo "# Qwen3-VL SHA: $(git -C "$ROOT/Qwen3-VL" rev-parse HEAD)"
  echo
  "$PIP" freeze
} > "$ROOT/requirements.lock.txt"

echo
echo "════════ 验证 ════════"
"$PY" - <<'PYEOF'
import torch, transformers, importlib
print(f"  torch        : {torch.__version__}")
print(f"  torch cuda   : {torch.version.cuda}")
print(f"  cuda avail   : {torch.cuda.is_available()}")
print(f"  device count : {torch.cuda.device_count()}")
print(f"  transformers : {transformers.__version__}")
try:
    import swift
    print(f"  ms-swift     : {swift.__version__}")
except Exception as e:
    print(f"  ms-swift     : 导入失败 {e}")
try:
    import qwen_vl_utils
    print("  qwen_vl_utils: OK")
except Exception:
    print("  qwen_vl_utils: 未安装（Qwen3-VL 推理需要）")
PYEOF

echo
echo "✓ 完成 -> $LOG"
echo "  锁文件 -> $ROOT/requirements.lock.txt"
