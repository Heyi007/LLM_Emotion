"""中文情绪分类表。

设计参考 Sofroniew et al. (2026) "Emotion Concepts and their Function in a LLM"：
选取覆盖整个效价(valence) × 唤醒(arousal) 空间的代表性情绪概念。

字段说明
--------
key   : 英文标识，与论文情绪词对齐，便于代码引用
zh    : 中文展示名（前端/报告）
ban   : 故事中禁止出现的词（情绪词本身及其直接同义词），用于数据 lint
valence : 先验效价，+1 积极 / 0 中性 / -1 消极（仅用于 PCA 符号对齐，不参与探针训练）
arousal : 先验唤醒，+1 高唤醒 / 0 中等 / -1 低唤醒（同上）
"""

from __future__ import annotations

# 顺序即前端默认展示顺序；探针矩阵的行顺序也以此为准。
EMOTIONS: list[dict] = [
    # —— 积极 · 高唤醒 ——
    {"key": "excited",    "zh": "兴奋",   "ban": ["兴奋", "激动", "亢奋"],
     "valence": 1, "arousal": 1},
    {"key": "happy",      "zh": "开心",   "ban": ["开心", "高兴", "快乐", "愉快", "欢喜"],
     "valence": 1, "arousal": 1},
    {"key": "surprised",  "zh": "惊喜",   "ban": ["惊喜", "惊讶", "意外", "吃惊"],
     "valence": 1, "arousal": 1},
    {"key": "proud",      "zh": "自豪",   "ban": ["自豪", "骄傲", "荣光"],
     "valence": 1, "arousal": 1},
    {"key": "grateful",   "zh": "感恩",   "ban": ["感恩", "感激", "感谢"],
     "valence": 1, "arousal": 1},
    {"key": "hopeful",    "zh": "充满希望", "ban": ["希望", "期待", "盼望", "憧憬"],
     "valence": 1, "arousal": 0},
    # —— 积极 · 低唤醒 ——
    {"key": "content",    "zh": "满足",   "ban": ["满足", "知足", "惬意"],
     "valence": 1, "arousal": -1},
    {"key": "calm",       "zh": "平静",   "ban": ["平静", "宁静", "安宁", "镇定"],
     "valence": 1, "arousal": -1},
    {"key": "relaxed",    "zh": "放松",   "ban": ["放松", "松弛", "轻松"],
     "valence": 1, "arousal": -1},
    {"key": "relieved",   "zh": "释然",   "ban": ["释然", "松了口气", "如释重负", "宽慰"],
     "valence": 1, "arousal": -1},
    {"key": "loving",     "zh": "关爱",   "ban": ["关爱", "疼爱", "怜爱", "温情"],
     "valence": 1, "arousal": 0},
    # —— 消极 · 高唤醒 ——
    {"key": "angry",      "zh": "愤怒",   "ban": ["愤怒", "气愤", "怒火", "生气", "恼怒"],
     "valence": -1, "arousal": 1},
    {"key": "afraid",     "zh": "恐惧",   "ban": ["恐惧", "害怕", "惧怕", "畏惧"],
     "valence": -1, "arousal": 1},
    {"key": "anxious",    "zh": "焦虑",   "ban": ["焦虑", "焦躁", "忐忑", "心慌"],
     "valence": -1, "arousal": 1},
    {"key": "panicked",   "zh": "恐慌",   "ban": ["恐慌", "惊慌", "慌乱", "惊惶"],
     "valence": -1, "arousal": 1},
    {"key": "desperate",  "zh": "绝望",   "ban": ["绝望", "无望", "走投无路", "万念俱灰"],
     "valence": -1, "arousal": 1},
    {"key": "disgusted",  "zh": "厌恶",   "ban": ["厌恶", "恶心", "反感", "憎恶"],
     "valence": -1, "arousal": 1},
    {"key": "jealous",    "zh": "嫉妒",   "ban": ["嫉妒", "妒忌", "眼红", "吃醋"],
     "valence": -1, "arousal": 1},
    {"key": "embarrassed","zh": "尴尬",   "ban": ["尴尬", "难堪", "窘迫", "难为情"],
     "valence": -1, "arousal": 1},
    {"key": "hurt",       "zh": "委屈",   "ban": ["委屈", "受委屈", "冤枉"],
     "valence": -1, "arousal": 0},
    # —— 消极 · 低唤醒 ——
    {"key": "sad",        "zh": "悲伤",   "ban": ["悲伤", "难过", "伤心", "哀伤"],
     "valence": -1, "arousal": -1},
    {"key": "depressed",  "zh": "沮丧",   "ban": ["沮丧", "消沉", "颓丧", "灰心"],
     "valence": -1, "arousal": -1},
    {"key": "lonely",     "zh": "孤独",   "ban": ["孤独", "孤单", "寂寞"],
     "valence": -1, "arousal": -1},
    {"key": "gloomy",     "zh": "忧郁",   "ban": ["忧郁", "阴郁", "愁闷", "闷闷不乐"],
     "valence": -1, "arousal": -1},
    {"key": "guilty",     "zh": "内疚",   "ban": ["内疚", "愧疚", "负罪", "自责"],
     "valence": -1, "arousal": -1},
    {"key": "weary",      "zh": "疲惫",   "ban": ["疲惫", "疲乏", "筋疲力尽", "劳累"],
     "valence": -1, "arousal": -1},
    {"key": "bored",      "zh": "无聊",   "ban": ["无聊", "乏味", "枯燥"],
     "valence": 0, "arousal": -1},
    # —— 认知性 ——
    {"key": "puzzled",    "zh": "困惑",   "ban": ["困惑", "疑惑", "迷惑", "不解"],
     "valence": 0, "arousal": 0},
]

EMOTION_KEYS: list[str] = [e["key"] for e in EMOTIONS]
ZH_BY_KEY: dict[str, str] = {e["key"]: e["zh"] for e in EMOTIONS}
BAN_BY_KEY: dict[str, list[str]] = {e["key"]: e["ban"] for e in EMOTIONS}
VALENCE_BY_KEY: dict[str, int] = {e["key"]: e["valence"] for e in EMOTIONS}
AROUSAL_BY_KEY: dict[str, int] = {e["key"]: e["arousal"] for e in EMOTIONS}
