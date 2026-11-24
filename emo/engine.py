"""实时情绪引擎：聊天生成的同时逐 token 读取情绪向量。

与论文的对应关系
----------------
1. 论文在 “Assistant colon”（助理起始 token）读“预备情绪”，本引擎在 prefill
   最后一个位置得到等价信号，先于任何输出 token 产出一次快照。
2. 论文在回复 token 上测量探针激活，本引擎在每个采样步骤从 hook 取该步
   最后位置的残差流，做与离线完全相同的「中心化→中性 PCA 投影→单位化→余弦」。
3. 回复结束后给出整段回复的平均情绪、效价(valence)/唤醒(arousal)。

校准
----
真实对话激活与故事风格激活存在分布偏移，且 prefill 与生成位置分布不同。
calibration.pt 为 plain/prefill/gen 三个场景分别给出中性基线的均值/标准差，
推理输出 z=(cos-mean)/std，中性读数回到 0 附近，强度可跨探针比较。
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .activations import LayerCapture, SteeringHook
from .emotions import AROUSAL_BY_KEY, VALENCE_BY_KEY

DEFAULT_PROBE = "data/probes/probes_qwen3.8-27b.pt"
DEFAULT_CALIB = "data/probes/calibration.pt"


class EmotionEngine:
    """加载一次模型与探针，供服务长期复用。"""

    def __init__(self, model_dir: str, probe_path: str, layer: int | None = None,
                 calibration_path: str | None = DEFAULT_CALIB, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map=self.device)
        self.model.eval()
        bundle = torch.load(probe_path, map_location=self.device, weights_only=False)
        self.layer = layer or bundle["default_layer"]
        if str(self.layer) not in bundle["probes"]:
            raise KeyError(f"探针包内没有 layer {self.layer}，可选 {bundle['layers']}")
        self.keys = bundle["emotion_keys"]
        self.zh = bundle["emotion_zh"]
        # 论文探针空间 PCA 在 28 类×4 条短故事规模下相关性偏弱，效价/唤醒
        # 改用情绪 z 分数对人工标注（emotions.py）加权，稳定且可解释。
        self.valence_prior = torch.tensor(
            [VALENCE_BY_KEY[k] for k in self.keys], dtype=torch.float32, device=self.device)
        self.arousal_prior = torch.tensor(
            [AROUSAL_BY_KEY[k] for k in self.keys], dtype=torch.float32, device=self.device)
        self.probes = bundle["probes"][str(self.layer)].to(self.device).float()   # [E,H]
        self.grand_mean = bundle["grand_mean"][str(self.layer)].to(self.device).float()
        self.projector = bundle["projector"][str(self.layer)].to(self.device).float()
        self.va_comps = bundle["probe_pca"][str(self.layer)]["components"].to(self.device).float()
        # Part 3 注入向量：未做中性 PCA 投影的原始情绪方向（残差流原生坐标）。
        # 注入量 = strength × 该层残差平均范数 × 单位方向，与论文归一化口径一致。
        if "raw_probes" in bundle and str(self.layer) in bundle["raw_probes"]:
            raw = bundle["raw_probes"][str(self.layer)].to(self.device).float()
            self.steer_dirs = F.normalize(raw, dim=-1)              # [E,H] 单位方向
            self.resid_norm = float(bundle["resid_norm"][str(self.layer)])
            self.steering_ready = True
        else:
            # 旧探针包兜底：退化为投影单位探针；缺少残差范数基准，强度语义不同
            self.steer_dirs = self.probes
            self.resid_norm = 1.0
            self.steering_ready = False
        self._cap: LayerCapture | None = None
        # 中性基线校准（可选，三场景）
        self.cal = None
        if calibration_path and Path(calibration_path).exists():
            self.cal = torch.load(calibration_path, map_location=self.device,
                                  weights_only=False)
            print(f"[engine] 已加载中性基线校准（layer={self.cal['layer']}, "
                  f"gen 样本={self.cal['gen']['n']}）")
        else:
            print("[engine] 未找到校准文件，输出原始余弦（中性输入可能有噪声）")

    # —— 由情绪分数加权得到效价/唤醒（取值 [-1,1]）——
    def _va_from_scores(self, score: torch.Tensor) -> tuple[float, float]:
        w = score.clamp_min(0)
        total = float(w.sum())
        if total < 1e-6:
            return 0.0, 0.0
        valence = float((w * self.valence_prior).sum() / total)
        arousal = float((w * self.arousal_prior).sum() / total)
        return round(valence, 4), round(arousal, 4)

    # —— 单位置隐状态 -> 情绪快照；stage: plain/prefill/gen ——
    def _emotion_from_hidden(self, h: torch.Tensor, stage: str = "gen") -> dict:
        x = (h.float() - self.grand_mean) @ self.projector.t()
        x = F.normalize(x, dim=-1)
        cos = (x @ self.probes.t()).squeeze(0)                          # [E]
        if self.cal is not None and stage in self.cal:
            stats = self.cal[stage]
            score = (cos - stats["cos_mean"]) / stats["cos_std"]
        else:
            score = cos
        order = torch.argsort(score, descending=True)
        top = [{"key": self.keys[i], "zh": self.zh[self.keys[i]],
                "score": round(float(score[i]), 4)} for i in order[:6]]
        valence, arousal = self._va_from_scores(score)
        intensity = round(float(score.clamp_min(0).max()), 4)
        return {
            "top": top,
            "all": {self.keys[i]: round(float(score[i]), 4) for i in range(len(self.keys))},
            "raw": {self.keys[i]: round(float(cos[i]), 4) for i in range(len(self.keys))},
            "valence": valence,
            "arousal": arousal,
            "intensity": intensity,
        }

    def _last_hidden(self) -> torch.Tensor:
        return self._cap.get(self.layer)[:, -1, :]                    # [1,H]

    # —— 离线分析：对任意裸文本（不生成）读情绪 ——
    @torch.no_grad()
    def score_text(self, text: str) -> dict:
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        with LayerCapture(self.model, [self.layer]) as cap:
            self.model(**ids, use_cache=False)
            h = cap.get(self.layer)[:, -1, :]
        return self._emotion_from_hidden(h, stage="plain")

    # —— 流式聊天：产出 (阶段标记, 数据) ——
    @torch.no_grad()
    def stream_chat(self, messages: list[dict], max_new_tokens: int = 256,
                    temperature: float = 0.7, top_p: float = 0.8,
                    steer_key: str | None = None,
                    steer_strength: float = 0.0):
        input_ids = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            return_dict=True, enable_thinking=False).input_ids.to(self.device)
        eos = self.tok.eos_token_id
        generated: list[int] = []
        all_score: list[torch.Tensor] = []
        sent_text = ""          # 已安全发送的解码前缀（不含未闭合多字节字符）
        finish_reason = "length"   # 默认因达到 max_new_tokens 而停，遇 eos 改为 stop
        # 论文 Part 3：在探针层残差流上叠加情绪方向；负强度即沿反方向抑制该情绪。
        # 先注册 SteeringHook 再注册 LayerCapture，使探针读数反映注入后激活。
        steer_ctx = nullcontext()
        steer_meta = None
        if steer_key and abs(steer_strength) > 1e-9:
            if steer_key not in self.keys:
                raise KeyError(f"未知情绪 key: {steer_key}，可选 {self.keys}")
            idx = self.keys.index(steer_key)
            delta = self.steer_dirs[idx] * (steer_strength * self.resid_norm)
            steer_ctx = SteeringHook(self.model, self.layer, delta.contiguous())
            steer_meta = {"emotion": steer_key, "zh": self.zh[steer_key],
                          "strength": steer_strength, "layer": self.layer,
                          "resid_norm": round(self.resid_norm, 3)}
        cap = LayerCapture(self.model, [self.layer])
        self._cap = cap
        with steer_ctx, cap:
            # 1) prefill：助理起始位的“预备情绪”（对应论文 Assistant colon）
            out = self.model(input_ids=input_ids, use_cache=True)
            prepared = self._emotion_from_hidden(self._last_hidden(), stage="prefill")
            yield "prepared", prepared
            logits = out.logits[:, -1, :]
            past_kv = out.past_key_values

            for step in range(max_new_tokens):
                # 2) 采样下一个 token
                if temperature and temperature > 0:
                    logits = logits / temperature
                    if top_p and top_p < 1.0:
                        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        mask = cum > top_p
                        mask[..., 1:] = mask[..., :-1].clone()
                        mask[..., 0] = False
                        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                        logits = torch.zeros_like(logits).scatter(
                            -1, sorted_idx, sorted_logits)
                    nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                else:
                    nxt = logits.argmax(-1, keepdim=True)
                if nxt.item() == eos:
                    finish_reason = "stop"
                    break
                generated.append(nxt.item())
                # 3) 前向该 token，hook 拿到本步残差流
                out = self.model(input_ids=nxt, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                snap = self._emotion_from_hidden(self._last_hidden(), stage="gen")
                all_score.append(torch.tensor([snap["all"][k] for k in self.keys]))
                # 字节级 BPE 会把一个 emoji 拆成多个 token，逐 token 独立解码会让
                # 残缺 UTF-8 字节变成 U+FFFD（乱码）。改为对累积序列整段解码，剥掉
                # 结尾未闭合字符后按前缀差分，只发送已确认完整的部分。
                cur_text = self.tok.decode(generated, skip_special_tokens=True)
                stable = cur_text[:-1] if cur_text.endswith("�") else cur_text
                piece = stable[len(sent_text):]
                sent_text = stable
                if piece:
                    yield "token", {"text": piece, "emotion": snap, "step": step}
                logits = out.logits[:, -1, :]

        # 4) 整段回复汇总（对逐 token z 分数取均值）
        full_text = self.tok.decode(generated, skip_special_tokens=True)
        tail = full_text[len(sent_text):]   # 冲刷最后可能滞留的未闭合字符
        if tail:
            yield "token", {"text": tail,
                            "emotion": self._emotion_from_hidden(self._last_hidden(), stage="gen"),
                            "step": len(generated)}
        if all_score:
            mean_score = torch.stack(all_score).mean(0).to(self.device)
            order = torch.argsort(mean_score, descending=True)
            summary_top = [{"key": self.keys[i], "zh": self.zh[self.keys[i]],
                            "score": round(float(mean_score[i]), 4)} for i in order[:6]]
            valence, arousal = self._va_from_scores(mean_score)
            summary = {"top": summary_top, "valence": valence, "arousal": arousal,
                       "intensity": round(float(mean_score.clamp_min(0).max()), 4)}
        else:
            summary = {"top": [], "valence": 0.0, "arousal": 0.0, "intensity": 0.0}
        yield "done", {"text": full_text, "n_tokens": len(generated),
                       "finish_reason": finish_reason, "steering": steer_meta,
                       **summary}
