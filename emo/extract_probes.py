"""从 Qwen3.8-27B 抽取中文情绪向量（emotion probes）。

完整复现 Sofroniew et al. (2026) 的探针构造流程：
  1. 对每条情绪故事跑前向，取指定 decoder 层的残差流激活，
     从第 skip_tokens 个 token 起对位置做平均（论文英文用 50，中文按 token 分布调小）。
  2. 情绪向量 = 该情绪所有故事的平均激活 − 全部情绪故事的全局平均激活。
  3. 用中性对话激活的主成分（累计解释 50% 方差）构造投影矩阵，把这些“通用语义方向”
     从情绪向量中投影掉。
  4. 单位化，得到最终探针；推理时对实时激活做相同的中心化 + 投影后算余弦相似度。

同时输出论文式验证：探针在自身故事上的 top-1 对角区分度，
以及探针空间 PCA 与先验效价/唤醒标签的相关性（PC1≈valence, PC2≈arousal）。

用法（H20 worker）：
  python -m emo.extract_probes --model-dir /tmp/hf_models/Qwen3.8-27B
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .activations import LayerCapture
from .data import DEFAULT_DATA, load_dataset
from .emotions import EMOTION_KEYS, ZH_BY_KEY, VALENCE_BY_KEY, AROUSAL_BY_KEY

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "probes"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default="/tmp/hf_models/Qwen3.8-27B")
    p.add_argument("--data", default=None)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--layers", default="21,31,42,53",
                   help="逗号分隔的 decoder 层索引（64 层骨干，默认覆盖 1/3~5/6）")
    p.add_argument("--default-layer", type=int, default=42, help="推理默认层（约 2/3 深度）")
    p.add_argument("--skip-tokens", type=int, default=16,
                   help="每条故事跳过的前导 token 数（论文英文为 50；中文故事仅 34~58 token，按同原则调小）")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--neutral-variance", type=float, default=0.5,
                   help="中性 PCA 投影保留到累计解释该比例方差为止的组件")
    p.add_argument("--max-tokens", type=int, default=512)
    return p.parse_args()


@torch.no_grad()
def story_means(model, tok, texts: list[str], layers: list[int],
                skip: int, batch_size: int, max_tokens: int, device: str
                ) -> dict[int, torch.Tensor]:
    """返回 {layer: [N, H]} 的每条故事位置平均激活（float32）。"""
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    accum = {l: [] for l in layers}
    length_dist: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tok(batch, add_special_tokens=False, padding=True, truncation=True,
                  max_length=max_tokens, return_tensors="pt")
        input_ids = enc.input_ids.to(device)
        attn = enc.attention_mask.to(device)
        length_dist += attn.sum(1).tolist()
        with LayerCapture(model, layers) as cap:
            model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            for l in layers:
                h = cap.get(l).float()                      # [B, L, H]
                mask = attn.unsqueeze(-1).bool()
                # 位置掩码：有效 token 且序号 >= skip
                pos = torch.arange(h.size(1), device=device).unsqueeze(0)
                keep = mask & (pos.unsqueeze(-1) >= skip)
                summed = (h * keep).sum(1)                  # [B, H]
                counts = keep.sum(1).clamp(min=1)
                accum[l].append((summed / counts).cpu())
    stats = {"min": min(length_dist), "max": max(length_dist),
             "mean": round(float(np.mean(length_dist)), 1),
             "shorter_than_skip": sum(1 for n in length_dist if n <= skip)}
    print(f"[token] 故事长度分布 {stats}")
    return {l: torch.cat(v, 0) for l, v in accum.items()}


def neutral_projector(neutral_means: torch.Tensor, variance_th: float) -> torch.Tensor:
    """中性对话激活 PCA，返回投影矩阵 P = I - U_k U_k^T（投影掉通用语义方向）。"""
    x = neutral_means - neutral_means.mean(0, keepdim=True)
    # 样本数远小于维度，用 SVD: x = U S V^T，主成分方向在 V 行中
    _, s, vh = torch.linalg.svd(x, full_matrices=False)
    explained = (s ** 2) / (s ** 2).sum()
    cum = torch.cumsum(explained, 0)
    k = int(torch.searchsorted(cum, variance_th).item()) + 1
    k = min(k, vh.shape[0])
    u_k = vh[:k]                                     # [k, H]
    h = x.shape[1]
    p = torch.eye(h) - u_k.t() @ u_k
    print(f"[pca] 中性对话投影掉 {k} 个主成分（累计方差 {float(cum[k-1]):.3f}）")
    return p


def build_probes(emotion_means: dict[str, torch.Tensor],
                 grand_mean: torch.Tensor, projector: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (原始中心化向量 [E,H], 投影并单位化的探针 [E,H])。"""
    raw = torch.stack([emotion_means[k] - grand_mean for k in EMOTION_KEYS])
    proj = raw @ projector.t()                            # P 对称：x @ P^T
    probes = proj / proj.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return raw, probes


def validate(story_vec: torch.Tensor, labels: list[str], probes: torch.Tensor) -> dict:
    """论文 Figure 2 式对角验证：每条故事在所有探针上的余弦 top-1 是否命中自身情绪。"""
    x = story_vec / story_vec.norm(dim=1, keepdim=True).clamp(min=1e-8)
    sim = x @ probes.t()                                  # [N, E]
    pred = sim.argmax(1).tolist()
    per_emotion: dict[str, dict] = {}
    for i, lab in enumerate(labels):
        d = per_emotion.setdefault(lab, {"ok": 0, "n": 0})
        d["n"] += 1
        d["ok"] += int(EMOTION_KEYS[pred[i]] == lab)
    acc = sum(p == EMOTION_KEYS.index(l) for p, l in zip(pred, labels)) / len(labels)
    diag = float(torch.tensor([sim[i, EMOTION_KEYS.index(labels[i])]
                               for i in range(len(labels))]).mean())
    off = float((sim.sum(1) - torch.tensor([sim[i, EMOTION_KEYS.index(labels[i])]
                                            for i in range(len(labels))])).mean() / (len(EMOTION_KEYS) - 1))
    return {"top1_accuracy": round(acc, 4),
            "mean_diagonal_cos": round(diag, 4),
            "mean_offdiagonal_cos": round(off, 4),
            "per_emotion": {k: round(v["ok"] / v["n"], 3) for k, v in per_emotion.items()}}


def probe_space_pca(probes: torch.Tensor) -> dict:
    """探针自身的 PCA，用先验标签对齐 PC1=效价、PC2=唤醒的符号。"""
    x = probes - probes.mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    comps = vh[:2]                                        # [2, H]
    scores = x @ comps.t()                                # [E, 2]
    valence = torch.tensor([VALENCE_BY_KEY[k] for k in EMOTION_KEYS], dtype=torch.float32)
    arousal = torch.tensor([AROUSAL_BY_KEY[k] for k in EMOTION_KEYS], dtype=torch.float32)
    signs = []
    for dim, prior in ((0, valence), (1, arousal)):
        r = torch.corrcoef(torch.stack([scores[:, dim], prior]))[0, 1]
        signs.append(1 if r.item() >= 0 else -1)
    comps = comps * torch.tensor(signs).unsqueeze(1)
    scores = x @ comps.t()
    r_val = torch.corrcoef(torch.stack([scores[:, 0], valence]))[0, 1].item()
    r_aro = torch.corrcoef(torch.stack([scores[:, 1], arousal]))[0, 1].item()
    return {"components": comps.cpu(), "signs": signs,
            "valence_corr": round(r_val, 4), "arousal_corr": round(r_aro, 4)}


def main() -> None:
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device} model={args.model_dir}")
    ds = load_dataset(args.data or DEFAULT_DATA)
    labels = [r["emotion"] for r in ds["emotional"]]
    all_stories = [r["text"] for r in ds["emotional"]]
    neutral_stories = [r["text"] for r in ds["neutral"]]

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map=device)
    model.eval()

    print("[forward] 情绪故事激活 ...")
    emo_means = story_means(model, tok, all_stories, layers,
                            args.skip_tokens, args.batch_size, args.max_tokens, device)
    print("[forward] 中性对话激活 ...")
    neu_means = story_means(model, tok, neutral_stories, layers,
                            args.skip_tokens, args.batch_size, args.max_tokens, device)

    bundle = {
        "model_dir": args.model_dir,
        "layers": layers,
        "default_layer": args.default_layer,
        "skip_tokens": args.skip_tokens,
        "emotion_keys": EMOTION_KEYS,
        "emotion_zh": ZH_BY_KEY,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "probes": {}, "grand_mean": {}, "projector": {}, "probe_pca": {},
    }
    report: dict = {}
    # 每条故事 -> 情绪索引，便于按情绪聚合
    idx_by_key = {k: [i for i, l in enumerate(labels) if l == k] for k in EMOTION_KEYS}
    for l in layers:
        means = emo_means[l]
        by_emotion = {k: means[idx_by_key[k]].mean(0) for k in EMOTION_KEYS}
        grand = means.mean(0)
        projector = neutral_projector(neu_means[l], args.neutral_variance)
        _, probes = build_probes(by_emotion, grand, projector)
        # 验证：对每条故事激活做同样的中心化+投影
        story_vec = (means - grand) @ projector.t()
        report[f"layer_{l}"] = validate(story_vec, labels, probes)
        pca = probe_space_pca(probes)
        report[f"layer_{l}"]["probe_pca"] = {
            "valence_corr": pca["valence_corr"],
            "arousal_corr": pca["arousal_corr"],
            "signs": pca["signs"],
        }
        bundle["grand_mean"][str(l)] = grand.cpu()
        bundle["projector"][str(l)] = projector.cpu()
        bundle["probes"][str(l)] = probes.cpu()
        bundle["probe_pca"][str(l)] = pca

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_path = out_dir / "probes_qwen3.8-27b.pt"
    torch.save(bundle, probe_path)
    report_path = out_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {probe_path}")
    print(f"[save] {report_path}")
    for l in layers:
        r = report[f"layer_{l}"]
        pca = r["probe_pca"]
        print(f"  layer {l:>2}: top1={r['top1_accuracy']:.3f} "
              f"diag={r['mean_diagonal_cos']:.3f} off={r['mean_offdiagonal_cos']:.3f} "
              f"| valence r={pca['valence_corr']:.3f} arousal r={pca['arousal_corr']:.3f}")


if __name__ == "__main__":
    main()
