#!/usr/bin/env bash
# 在 master 上建立到 worker 的 IPv6 双栈隧道，并把端口暴露为外部短链。
# 用法：bash scripts/tunnel_demo.sh <worker_pod_ipv6> [worker_ssh_port]
# 前置：worker 上已用 serve_demo.sh 启动 8000 端口服务。
set -euo pipefail
IP="${1:?需要 worker pod IPv6，见 mlx worker list}"
SSH_PORT="${2:-9391}"
LOCAL_PORT=9001
echo "启动隧道 master:[::]:$LOCAL_PORT -> worker:localhost:8000（Ctrl-C 结束）"
exec ssh -i ~/.ssh/id_rsa -p "$SSH_PORT" -N \
  -L "[::]:$LOCAL_PORT:localhost:8000" \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -o StrictHostKeyChecking=no "tiger@$IP"
# 隧道起来后另开终端执行：mlx export --port 9001
