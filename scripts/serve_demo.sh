#!/usr/bin/env bash
# 在 GPU worker 上启动实时情绪 Demo 服务（uvicorn + SSE）。
# 用法（在 master 上）：
#   NO_COLOR=1 TERM=dumb mlx worker login <worker_id> -- \
#     bash -lc 'cd /mlx_devbox/users/heyi.jack/playground/emotion && bash scripts/serve_demo.sh'
set -euo pipefail
VENV=/mlx_devbox/users/heyi.jack/playground/kev/.venv/bin/python
export EMOTION_MODEL_DIR="${EMOTION_MODEL_DIR:-/tmp/hf_models/Qwen3.8-27B}"
# 注意：worker 环境内置 PORT=9391（SSH 端口），不能复用该变量名
PORT="${EMOTION_PORT:-8000}"

if [ ! -d "$EMOTION_MODEL_DIR" ]; then
  echo "模型目录不存在: $EMOTION_MODEL_DIR" >&2
  echo "worker 重建后先跑 scripts/download_qwen27b.sh 恢复模型，再重跑：" >&2
  echo "  python -m emo.extract_probes --model-dir $EMOTION_MODEL_DIR" >&2
  echo "  python -m emo.calibrate    --model-dir $EMOTION_MODEL_DIR" >&2
  exit 1
fi
pgrep -af "uvicorn emo.server" || true
setsid "$VENV" -m uvicorn emo.server:app --host 0.0.0.0 --port "$PORT" \
  > /tmp/emotion_server.log 2>&1 < /dev/null &
sleep 3
pgrep -af "uvicorn emo.server"
echo "日志: tail -f /tmp/emotion_server.log（模型加载约 30~40s）"
