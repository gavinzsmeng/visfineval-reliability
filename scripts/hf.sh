#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# hf CLI wrapper —— 绕过 SOCKS 代理兼容问题
#
# 问题:
#   本机 .bashrc 里 source 了 ~/.config/mihomo/proxy.env，其中
#       ALL_PROXY=socks5h://gavin:***@127.0.0.1:17890
#   httpx 见到 socks5h:// 会去找 socksio 包，未安装则直接抛:
#       ImportError: Using SOCKS proxy, but the 'socksio' package is not installed
#
# 修复:
#   只需要 unset ALL_PROXY / all_proxy。
#   HTTPS_PROXY=http://... 是 HTTP CONNECT 代理，httpx 原生支持，保留即可。
#   —— 不要关掉所有代理，本机直连 huggingface.co 是不通的。
#
# 用法:
#   ./scripts/hf.sh download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct
#   ./scripts/hf.sh datasets info SUFE-AIFLM-Lab/VisFinEval
# ─────────────────────────────────────────────────────────────
exec env -u ALL_PROXY -u all_proxy hf "$@"
