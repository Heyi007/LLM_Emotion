# LLM Emotion — Qwen3.8-27B 中文情绪探针与实时情绪 Demo

用大语言模型内部的残差流（residual stream）激活构建**中文情绪向量**，并在与模型聊天时**实时展示 LLM 的情绪**。方法复现自 Anthropic 论文 *Emotion Concepts and their Function in a LLM* (2026)：
<https://transformer-circuits.pub/2026/emotions/index.html>

模型在生成每一个 token 时，系统从指定层的残差流中读取与 28 种情绪探针的余弦相似度，经中性基线校准后输出情绪强度、效价（valence）与唤醒（arousal），通过 SSE 实时推送到前端。

## 方法与论文的对应关系

1. **间接表达故事**：为每种情绪编写短篇故事，全文不出现该情绪词及其同义词，只靠行为、肢体、对话、情境间接传达（对应论文的 emotion stories）。
2. **分层残差流激活**：注册 forward hook 抓取指定 decoder 层的残差流，跳过每条故事前 `skip_tokens` 个 token 后对位置取平均。
3. **情绪向量**：某情绪故事的平均激活 − 全部情绪的全局平均激活。
4. **中性 PCA 投影**：用中性对话激活做 SVD，投影掉累计 50% 方差的通用语义方向 `P = I − UₖUₖᵀ`。
5. **余弦读数**：探针单位化后用余弦相似度读情绪强度，默认层 42（64 层骨干中约 2/3 深度）。
6. **预备情绪**：在 chat template 的助理起始位（对应论文 “Assistant colon”）读取即将生成回复的情绪。

中文适配：中文故事仅 34–58 token（语义密度高于英文段落），`skip_tokens` 按论文同一原则由英文的 50 调小为 16。

## 目录结构

```
emotion/
├── emo/
│   ├── emotions.py        # 28 种情绪分类表（key/中文名/禁用词/效价/唤醒）
│   ├── data.py            # 故事数据加载与禁词泄漏 lint
│   ├── activations.py     # 残差流分层 hook（LayerCapture）
│   ├── extract_probes.py  # 探针抽取：位置均值→中心化→中性 PCA 投影→单位化
│   ├── calibrate.py       # 三场景中性基线 z-score 校准
│   ├── engine.py          # 实时情绪引擎（流式生成 + 逐 token 读数）
│   └── server.py          # FastAPI SSE 服务
├── data/
│   ├── stories_zh.json    # 112 条情绪故事 + 12 条中性对话 + 24 条中性用户语句
│   ├── sample.json        # 单条数据格式核对样本
│   └── probes/            # 生成物（.pt 已 gitignore，不入库）
├── web/index.html         # 零依赖单文件前端（适配内网）
└── scripts/
    ├── download_qwen27b.sh  # 内部镜像分块下载模型权重
    ├── serve_demo.sh        # worker 上启动 uvicorn
    └── tunnel_demo.sh       # master 到 worker 的 SSH 隧道
```

## 数据规模

| 内容 | 数量 |
|---|---|
| 情绪类别 | 28 种（覆盖效价 ±、唤醒 ± 的全空间） |
| 情绪故事 | 112 条（28 类 × 4 条，均通过禁词泄漏 lint） |
| 中性对话 | 12 条 |
| 中性用户语句（校准用） | 24 条 |

## 环境与依赖

- Python 3.13、CUDA GPU（实测单卡 NVIDIA-H20 96GB）
- 依赖：`torch`、`transformers`、`numpy`、`fastapi`、`uvicorn`、`pydantic`
- 本项目未单独维护虚拟环境，复用同工作区 `kev/.venv`：
  `/mlx_devbox/users/heyi.jack/playground/kev/.venv/bin/python`
- 目标模型：Qwen3.8-27B（`Qwen3_5ForCausalLM`，64 层混合骨干，hidden_size=5120，bf16），默认路径 `/tmp/hf_models/Qwen3.8-27B`

## 快速开始

以下命令中的 `python` 均指上述 venv 解释器。

### 1. 下载模型（字节内部网络）

```bash
bash scripts/download_qwen27b.sh
```

镜像非标准 HF 代理，脚本采用 200MB 分块 Range + 断点续传绕开网关长连接限制，可重复执行。模型默认落在 worker 本地易失盘 `/tmp/hf_models/`。

### 2. 抽取情绪探针

```bash
python -m emo.extract_probes --model-dir /tmp/hf_models/Qwen3.8-27B
```

关键参数（均有默认值）：`--layers 21,31,42,53`、`--default-layer 42`、`--skip-tokens 16`、`--neutral-variance 0.5`。
产物：`data/probes/probes_qwen3.8-27b.pt`（约 403MB）与 `validation_report.json`。

### 3. 中性基线校准

```bash
python -m emo.calibrate --model-dir /tmp/hf_models/Qwen3.8-27B
```

产出 `data/probes/calibration.pt`，分别为三个场景统计中性基线的均值/标准差：

- `plain`：裸文本最后 token（离线 `score_text` 用）
- `prefill`：chat template 助理起始位（预备情绪用）
- `gen`：中性回复生成过程的逐 token 激活（流式读数用）

推理输出 z-score：`z = (cos − mean) / std`，使中性输入回到 0 附近、强度可跨探针比较。

### 4. 启动服务

```bash
bash scripts/serve_demo.sh
# 等价于：python -m uvicorn emo.server:app --host 0.0.0.0 --port 8000
```

可用环境变量覆盖路径：`EMOTION_MODEL_DIR`、`EMOTION_PROBE_PATH`、`EMOTION_CALIB_PATH`；监听端口用 `EMOTION_PORT`（默认 8000，注意不要复用 worker 内置的 `PORT=9391`）。

### 5. 外部访问（mlx worker 场景）

worker 业务端口不对 master 直通，先在 master 建 IPv6 双栈隧道，再导出短链：

```bash
bash scripts/tunnel_demo.sh <worker_pod_ipv6> 9391   # master:9001 -> worker:8000
mlx export --port 9001                               # 生成 merlin-devbox 短链
```

## HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 单文件前端页面 |
| GET | `/api/health` | 健康检查与模型加载状态 |
| GET | `/api/emotions` | 28 种情绪元数据（中文名、效价、唤醒） |
| POST | `/api/chat` | SSE 流式对话，事件序列 `prepared → token* → done` |

`/api/chat` 请求体：

```json
{
  "messages": [{"role": "user", "content": "你好"}],
  "max_new_tokens": 1024,
  "temperature": 0.7,
  "top_p": 0.8
}
```

- `max_new_tokens` 限制 1–2048，默认 512；前端选择器默认 1024。
- `done` 事件含 `finish_reason`：`stop`（遇 `<|im_end|>` 自然结束）或 `length`（达到 token 上限被截断，前端会显示截断提示）。

前端面板包含：主导情绪卡、Top 8 与全部 28 情绪 z-score 条形图、效价-唤醒逐 token 轨迹、情绪强度时间线、预备情绪。

## 验证结果

探针在留出故事上的 top-1 对角区分度（`data/probes/validation_report.json`）：

| 层 | top-1 准确率 | 对角余弦 | 非对角余弦 |
|---|---|---|---|
| 21 | 1.000 | 0.599 | −0.022 |
| 31 | 1.000 | 0.619 | −0.023 |
| **42（默认）** | **1.000** | **0.628** | **−0.023** |
| 53 | 0.991 | 0.631 | −0.023 |

强对角、近零非对角说明每种情绪故事的激活都最强匹配自身探针。实测流式读数：报喜消息主导「兴奋/开心」（效价为正），投诉消息主导「委屈/愤怒/恐惧」（效价为负），中性事务咨询强度接近 0。H20 上生成速度约 15.6 token/s。

## 设计说明

- **效价/唤醒的计算**：探针空间 PCA 主成分在 28 类 × 4 条的小规模中文数据上与人工标注相关性偏弱，因此效价/唤醒改用情绪 z 分数对 `emotions.py` 中人工标注（±1/0）加权，取值落在 [−1, 1]，更稳定可解释。
- **关闭思考模式**：Qwen3.8 默认先输出思维链，引擎调用 chat template 时传 `enable_thinking=False` 以直接得到面向用户的回复。
- **强度阈值**：前端将最大正 z 分数低于 1.5 的状态显示为「中性 / 平静」。

## 注意事项

- `data/probes/*.pt` 为生成物且单文件超 GitHub 100MB 限制，已被 gitignore。新克隆仓库后需依次重跑步骤 2、3 才能启动服务。
- worker 的 `/tmp` 为易失本地盘，worker 回收后模型与产物都会丢失，需按 `download → extract_probes → calibrate → serve` 重新恢复。
- 逐 token 情绪读数存在自然抖动，整段回复的均值（`done` 事件）比单 token 更稳定，适合作为结论性判断。
