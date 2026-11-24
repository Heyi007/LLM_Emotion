"""实时情绪 Demo 的 FastAPI SSE 服务。

路由
----
GET  /                单文件前端（web/index.html，无外部依赖，适配内网）
GET  /api/health      健康检查（含模型加载状态）
GET  /api/emotions    28 种情绪元数据（中文名、效价、唤醒）
POST /api/chat        SSE：事件流 prepared -> token* -> done
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .emotions import AROUSAL_BY_KEY, EMOTIONS, VALENCE_BY_KEY, ZH_BY_KEY
from .engine import EmotionEngine

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"

MODEL_DIR = os.environ.get("EMOTION_MODEL_DIR", "/tmp/hf_models/Qwen3.8-27B")
PROBE_PATH = os.environ.get("EMOTION_PROBE_PATH", str(ROOT / "data/probes/probes_qwen3.8-27b.pt"))
CALIB_PATH = os.environ.get("EMOTION_CALIB_PATH", str(ROOT / "data/probes/calibration.pt"))

engine: EmotionEngine | None = None
load_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, load_error
    try:
        engine = EmotionEngine(MODEL_DIR, PROBE_PATH, calibration_path=CALIB_PATH)
    except Exception as exc:  # 服务仍可启动，/api/health 暴露错误
        load_error = repr(exc)
        print("[server] 引擎加载失败:", load_error)
    yield


app = FastAPI(title="Qwen3.8-27B 实时情绪", lifespan=lifespan)


class ChatRequest(BaseModel):
    messages: list[dict]
    max_new_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = 0.7
    top_p: float = 0.8


@app.get("/api/health")
def health():
    return {"status": "ok" if engine is not None else "degraded",
            "layer": engine.layer if engine else None,
            "error": load_error}


@app.get("/api/emotions")
def emotions_meta():
    return [{"key": e["key"], "zh": ZH_BY_KEY[e["key"]],
             "valence": VALENCE_BY_KEY[e["key"]],
             "arousal": AROUSAL_BY_KEY[e["key"]]} for e in EMOTIONS]


@app.post("/api/chat")
def chat(req: ChatRequest):
    if engine is None:
        def err_iter():
            yield "data: " + json.dumps(
                {"kind": "error", "data": {"message": load_error or "引擎未就绪"}},
                ensure_ascii=False) + "\n\n"
        return StreamingResponse(err_iter(), media_type="text/event-stream")

    def event_iter():
        try:
            for kind, data in engine.stream_chat(
                    req.messages, max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature, top_p=req.top_p):
                yield "data: " + json.dumps(
                    {"kind": kind, "data": data}, ensure_ascii=False) + "\n\n"
        except Exception as exc:
            yield "data: " + json.dumps(
                {"kind": "error", "data": {"message": repr(exc)}},
                ensure_ascii=False) + "\n\n"

    return StreamingResponse(
        event_iter(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")
