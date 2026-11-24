"""中性基线校准（三场景）。

真实对话激活与故事风格激活存在分布偏移，且不同读取位置的分布也不同：
  - plain  : 裸文本最后 token（离线 score_text 用）
  - prefill: chat template 助理起始位（“预备情绪”用）
  - gen    : 生成过程中逐 token 残差流（流式读数用）
对每个场景统计各探针余弦的均值/标准差，推理时输出
z=(cos-mean)/std，使中性读数回到 0 附近、强度可跨探针比较。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .activations import LayerCapture
from .data import DEFAULT_DATA, load_dataset

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "probes" / "calibration.pt"


def _cos_row(h, grand, projector, probes, va_comps):
    x = F.normalize((h.float() - grand) @ projector.t(), dim=-1)
    return (x @ probes.t()).squeeze(0).cpu()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--probe-path", default="data/probes/probes_qwen3.8-27b.pt")
    p.add_argument("--data", default=None)
    p.add_argument("--gen-tokens", type=int, default=48)
    p.add_argument("--out", default=str(OUT_PATH))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_dataset(args.data or DEFAULT_DATA)
    utters = ds.get("neutral_utters") or []
    if len(utters) < 8:
        raise SystemExit("neutral_utters 样本不足，无法稳定估计基线")

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map=device)
    model.eval()
    bundle = torch.load(args.probe_path, map_location=device, weights_only=False)
    layer = bundle["default_layer"]
    probes = bundle["probes"][str(layer)].to(device).float()
    grand = bundle["grand_mean"][str(layer)].to(device).float()
    projector = bundle["projector"][str(layer)].to(device).float()
    va_comps = bundle["probe_pca"][str(layer)]["components"].to(device).float()
    E = probes.shape[0]

    plain_rows, prefill_rows, gen_rows = [], [], []
    with torch.no_grad(), LayerCapture(model, [layer]) as cap:
        for text in utters:
            # 1) plain：裸文本
            ids = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
            model(**ids, use_cache=False)
            plain_rows.append(_cos_row(cap.get(layer)[:, -1, :],
                                       grand, projector, probes, va_comps))
            # 2) prefill：模板内助理起始位
            chat = tok.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True,
                return_tensors="pt", return_dict=True, enable_thinking=False)
            input_ids = chat.input_ids.to(device)
            out = model(input_ids=input_ids, use_cache=True)
            prefill_rows.append(_cos_row(cap.get(layer)[:, -1, :],
                                         grand, projector, probes, va_comps))
            # 3) gen：中性回复生成过程中的逐 token 激活（贪心解码）
            past_kv, logits = out.past_key_values, out.logits[:, -1, :]
            for _ in range(args.gen_tokens):
                nxt = logits.argmax(-1, keepdim=True)
                if nxt.item() == tok.eos_token_id:
                    break
                out = model(input_ids=nxt, past_key_values=past_kv, use_cache=True)
                past_kv, logits = out.past_key_values, out.logits[:, -1, :]
                gen_rows.append(_cos_row(cap.get(layer)[:, -1, :],
                                         grand, projector, probes, va_comps))

    def stats(rows):
        mat = torch.stack(rows)
        return {"cos_mean": mat.mean(0), "cos_std": mat.std(0).clamp_min(1e-6), "n": mat.shape[0]}

    cal = {"layer": layer, "texts": utters,
           "plain": stats(plain_rows), "prefill": stats(prefill_rows),
           "gen": stats(gen_rows)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cal, out)
    print(f"[save] {out} (layer {layer})")
    print(f"[n] plain={cal['plain']['n']} prefill={cal['prefill']['n']} gen={cal['gen']['n']}")


if __name__ == "__main__":
    main()
