"""固定批量分片的说明文筛选；不逐页请求源站。"""

import argparse
import hashlib
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from opencc import OpenCC

from micro_rsi.data.corpus import write_json
from micro_rsi.tokenization.character import CharacterTokenizer

REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
FILE = "20231101.zh/train-00002-of-00006.parquet"
EXPECTED_SHA256 = "ec8f6c0dd1418b8fcd454278fb4f2bc7cd0ce8312f29c80b672533942601338f"
SOURCE = f"https://huggingface.co/datasets/wikimedia/wikipedia/resolve/{REVISION}/{FILE}"


def chinese_fraction(text):
    return sum("\u4e00" <= c <= "\u9fff" for c in text) / max(1, len(text))


def clean_article(text, converter):
    text = converter.convert(unicodedata.normalize("NFC", text))
    paragraphs = []
    for line in text.splitlines():
        # 保留英文单词边界；全部删除空白会制造模型要学习的伪长词。
        line = re.sub(r"\s+", " ", line).strip()
        if not 80 <= len(line) <= 3000 or chinese_fraction(line) < 0.65:
            continue
        if any(marker in line for marker in ["http", "www.", "{{", "}}", "{|", "�", "\\", "ISBN"]):
            continue
        if len(re.findall(r"[。！？]", line)) < 2:
            continue
        if re.search(r"(.)\1{5}", line):
            continue
        # 上游抽取缺失的内容无法在这里恢复，剔除整段，避免拼接残句。
        if re.search(r"（\s*）|\(\s*\)|“\s*”|《\s*》", line):
            continue
        paragraphs.append(line)
    return "\n".join(dict.fromkeys(paragraphs))


def split_for(article_id):
    bucket = int(hashlib.sha256(str(article_id).encode()).hexdigest()[:8], 16) % 100
    return "validation" if bucket < 3 else "test" if bucket < 6 else "train"


def prepare(raw: Path, output: Path, train_chars=5_000_000, holdout_chars=150_000):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空：{output}")
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("分片校验和与固定版本不符")
    converter = OpenCC("t2s")
    docs = {"train": [], "validation": [], "test": []}
    sizes = Counter()
    target = {"train": train_chars, "validation": holdout_chars, "test": holdout_chars}
    seen_documents, seen_paragraphs = set(), set()
    rejected = Counter()
    scanned = 0
    for batch in pq.ParquetFile(raw).iter_batches(batch_size=256):
        for row in batch.to_pylist():
            scanned += 1
            split = split_for(row["id"])
            if sizes[split] >= target[split]:
                continue
            if any(
                word in row["title"] for word in ["列表", "年表", "消歧义", "赛季", "選舉", "选举"]
            ):
                rejected["title"] += 1
                continue
            text = clean_article(row["text"], converter)
            if not 512 <= len(text) <= 16_000:
                rejected["length_or_quality"] += 1
                continue
            fingerprint = hashlib.sha256(text.encode()).hexdigest()
            paragraphs = {hashlib.sha256(p.encode()).hexdigest() for p in text.splitlines()}
            if fingerprint in seen_documents or paragraphs & seen_paragraphs:
                rejected["duplicate_document_or_paragraph"] += 1
                continue
            seen_documents.add(fingerprint)
            seen_paragraphs.update(paragraphs)
            docs[split].append(
                {
                    "title": converter.convert(row["title"]),
                    "text": text,
                    "id": row["id"],
                    "url": row["url"],
                    "sha256": fingerprint,
                }
            )
            sizes[split] += len(text)
        if all(sizes[k] >= target[k] for k in target):
            break
    if any(sizes[k] < target[k] for k in target):
        raise ValueError(f"分片中合格数据不足：{sizes}")
    counts = Counter("".join(d["text"] for d in docs["train"]))
    # 词表按频率裁剪，最多6000字符；只使用训练数据，验证/测试未知率单独报告。
    characters = sorted(c for c, count in counts.most_common(6000) if count >= 5)
    tokenizer = CharacterTokenizer(characters)
    output.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, documents in docs.items():
        write_json(output / f"{split}.json", {"documents": documents})
        unknown = sum(c not in tokenizer.stoi for d in documents for c in d["text"])
        stats[split] = {
            "documents": len(documents),
            "characters": sizes[split],
            "unknown_characters": unknown,
            "unknown_rate": unknown / sizes[split],
        }
    tokenizer.save(output / "tokenizer.json")
    manifest = {
        "source": SOURCE,
        "dataset": "wikimedia/wikipedia",
        "revision": REVISION,
        "source_sha256": digest,
        "source_shard_bytes": raw.stat().st_size,
        "rights": "数据卡：CC BY-SA 3.0 / GFDL；按条目保留url和id以供署名追溯",
        "normalization": "NFC, OpenCC t2s, 空白折叠但保留词间空格；不宣称人工逐字校对",
        "preprocessing_version": "v3-preserve-spaces-reject-empty-fields",
        "filter": (
            "中文>=65%，每段80-3000字符至少2句，每篇512-16000字符；"
            "去列表、异常标记及含空括号/空引号/空书名的段落"
        ),
        "deduplication": "规范化全文和完整段落精确去重，任何已见重复段落则剔除整篇",
        "split": "原始article id的SHA256确定94/3/3，整篇隔离；测试集训练结束后才评分",
        "vocabulary": "训练字符频率>=5、最多6000字符，另加unk",
        "scanned_rows": scanned,
        "rejected": dict(rejected),
        "splits": stats,
        "train_chars": sizes["train"],
        "validation_chars": sizes["validation"],
        "vocab_size": tokenizer.vocab_size,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="清洗已批量下载的中文维基百科固定分片")
    parser.add_argument(
        "--raw", type=Path, default=Path("data/raw/wikipedia-zh/train-00002-of-00006.parquet")
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/wiki-zh-5m-v3"))
    args = parser.parse_args()
    import json

    print(json.dumps(prepare(args.raw, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
