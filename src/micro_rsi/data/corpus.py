"""固定版本语料的获取、清洗与先切分后编码。"""

import hashlib
import json
import re
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from opencc import OpenCC

from micro_rsi.tokenization.character import CharacterTokenizer

SOURCE = "https://www.gutenberg.org/ebooks/27166.txt.utf-8"
# 此电子版的顺序与常见印本不同；显式列出全部 14 篇以检查完整性。
TITLES = [
    "一件小事",
    "狂人日記",
    "鴨的喜劇",
    "端午節",
    "故鄉",
    "孔乙己",
    "藥",
    "阿Q正傳",
    "兔和貓",
    "社戲",
    "風波",
    "頭發的故事",
    "明天",
    "白光",
]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def extract_stories(raw: str) -> list[dict]:
    """保留篇目边界；移除序言、每篇尾注、行内注号和硬换行。"""
    body = raw.split("*** START OF THE PROJECT GUTENBERG EBOOK 吶喊 ***", 1)[1]
    body = body.split("*** END OF THE PROJECT GUTENBERG EBOOK 吶喊 ***", 1)[0]
    pattern = r"(?m)^\s*(" + "|".join(map(re.escape, TITLES)) + r")\s*$"
    matches = list(re.finditer(pattern, body))
    if [m.group(1) for m in matches] != TITLES:
        raise ValueError("原始文件篇目与已核实版本不符，请检查来源，禁止静默清洗")
    converter = OpenCC("t2s")
    documents = []
    for i, match in enumerate(matches):
        text = body[match.end() : matches[i + 1].start() if i + 1 < len(matches) else len(body)]
        text = re.split(r"(?m)^\s*(?:□)?注釋[︰：:]?\s*$", text)[0]
        # 原文件用空行分开印刷行，缩进标记不可靠；合并后在句末恢复段落。
        text = re.sub(r"[\s\u3000]+", "", text)
        text = re.sub(r"[\u3040-\u309f]", "", text)
        text = converter.convert(text).replace("︰", "：")
        text = re.sub(r"([。！？])([”’]?)", r"\1\2\n", text).strip()
        documents.append({"title": converter.convert(match.group(1)), "text": text})
    return documents


def prepare(raw_path: Path, output: Path, validation_fraction: float = 0.1) -> dict:
    if not 0 < validation_fraction < 1:
        raise ValueError("验证比例必须介于 0 与 1 之间")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空，防止覆盖已有实验数据：{output}")
    raw_bytes = raw_path.read_bytes()
    documents = extract_stories(raw_bytes.decode("utf-8-sig"))
    train, validation = [], []
    for doc in documents:
        cut = int(len(doc["text"]) * (1 - validation_fraction))
        train.append({"title": doc["title"], "text": doc["text"][:cut]})
        validation.append({"title": doc["title"], "text": doc["text"][cut:]})
    tokenizer = CharacterTokenizer.fit([d["text"] for d in train])
    validation_ids = [i for doc in validation for i in tokenizer.encode(doc["text"])]
    manifest = {
        "source": SOURCE,
        "catalog": "https://www.gutenberg.org/ebooks/27166",
        "credits": "Produced by Han Huei Haung",
        "source_rights": "Public domain in the USA; 原文件保留 Project Gutenberg 许可全文",
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "normalization": "OpenCC t2s; 去序言及尾注、假名注号和空白；句末恢复换行",
        "validation_fraction": validation_fraction,
        "split": "每篇前部训练、末尾连续区段验证；切分后独立构造窗口",
        "train_chars": sum(len(d["text"]) for d in train),
        "validation_chars": len(validation_ids),
        "validation_unknown_chars": validation_ids.count(0),
        "validation_unknown_rate": validation_ids.count(0) / len(validation_ids),
        "vocab_size": tokenizer.vocab_size,
        "documents": [{"title": d["title"], "chars": len(d["text"])} for d in documents],
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "train.json", {"documents": train})
    write_json(output / "validation.json", {"documents": validation})
    write_json(output / "manifest.json", manifest)
    tokenizer.save(output / "tokenizer.json")
    return manifest


def download(raw_path: Path) -> None:
    if raw_path.exists():
        raise FileExistsError(f"原始文件已存在，请复用或显式改路径：{raw_path}")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(SOURCE, headers={"User-Agent": "micro-rsi-learning/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read(2_000_001)
    if len(payload) > 2_000_000:
        raise ValueError("原始文件超出预期大小")
    extract_stories(payload.decode("utf-8-sig"))
    raw_path.write_bytes(payload)
