"""故事数据加载与质量检查。

论文硬性规则：情绪故事中不得出现目标情绪词或其直接同义词。
``lint_stories`` 用 EMOTIONS 中的 ban 词表做自动检查，保证数据集合规。
"""
from __future__ import annotations

import json
from pathlib import Path

from .emotions import BAN_BY_KEY, EMOTION_KEYS

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "stories_zh.json"


def load_dataset(path: str | Path = DEFAULT_DATA) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    emotional = data["emotional"]
    neutral = data["neutral"]
    # 结构校验
    keys = [r["emotion"] for r in emotional]
    unknown = sorted(set(keys) - set(EMOTION_KEYS))
    if unknown:
        raise ValueError(f"故事中出现分类表未定义的情绪: {unknown}")
    counts = {k: keys.count(k) for k in EMOTION_KEYS}
    missing = [k for k, c in counts.items() if c == 0]
    if missing:
        raise ValueError(f"以下情绪缺少故事: {missing}")
    return {"emotional": emotional, "neutral": neutral, "counts": counts,
            "neutral_utters": data.get("neutral_utters", [])}


def lint_stories(data: dict) -> list[dict]:
    """检查每条故事是否包含其目标情绪的禁用词；返回违规记录列表（空列表=通过）。"""
    violations: list[dict] = []
    for i, rec in enumerate(data["emotional"]):
        text = rec["text"]
        for word in BAN_BY_KEY[rec["emotion"]]:
            if word in text:
                violations.append({"index": i, "emotion": rec["emotion"],
                                   "banned_word": word, "text": text[:40]})
    return violations


def stories_by_emotion(data: dict) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {k: [] for k in EMOTION_KEYS}
    for rec in data["emotional"]:
        grouped[rec["emotion"]].append(rec["text"])
    return grouped


if __name__ == "__main__":
    # 本地快速自检：python -m emo.data
    ds = load_dataset()
    v = lint_stories(ds)
    print(f"情绪故事 {len(ds['emotional'])} 条，中性对话 {len(ds['neutral'])} 条")
    print(f"每种情绪故事数: min={min(ds['counts'].values())} max={max(ds['counts'].values())}")
    if v:
        print("LINT FAILED:")
        for item in v:
            print(" ", item)
        raise SystemExit(1)
    print("LINT PASSED: 未发现情绪词泄漏")
