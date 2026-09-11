#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# 把两个 LoRA adapter 上传到 HuggingFace
#
# 前置：必须先登录（交互式，只能你自己跑）
#     hf auth login
#   或者
#     export HF_TOKEN=hf_xxx
#
# 用法:
#   bash scripts/upload_to_hf.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

HF="$ROOT/scripts/hf.sh"
NS="${HF_NAMESPACE:-gavinzsmeng}"

# 先确认登录
if ! "$HF" auth whoami >/dev/null 2>&1; then
    echo "✗ 未登录 HuggingFace。先跑：  hf auth login"
    exit 1
fi
USER=$("$HF" auth whoami 2>/dev/null | head -1 | awk '{print $1}')
echo "已登录：$USER"
echo

upload_one() {
    local cond="$1" suffix="$2"
    local repo="$NS/visfineval-qwen3vl-8b-lora-${suffix}"
    local dir="$ROOT/hf_upload/$cond"

    [ -d "$dir" ] || { echo "✗ 缺 $dir，先跑准备步骤"; return 1; }
    [ -f "$dir/adapter_model.safetensors" ] || { echo "✗ $dir 缺 adapter_model.safetensors"; return 1; }

    echo "── $repo"
    if "$HF" repo view "$repo" >/dev/null 2>&1; then
        echo "   仓库已存在，直接上传"
    else
        "$HF" repos create "$repo" --type model --public --exist-ok
    fi

    "$HF" upload "$repo" "$dir" \
        --type model \
        --commit-message "VisFinEval LoRA (${cond}): Qwen3-VL-8B, report-level split, 0 leakage"

    echo "   ✓ https://huggingface.co/$repo"
    echo
}

upload_one natural  natural
upload_one balanced balanced

echo "完成。记得在 GitHub README 里加上这两个链接。"
