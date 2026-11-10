"""取得《彷徨》固定修订页面，并准备共享词表的成对数据。"""

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

from opencc import OpenCC

from micro_rsi.data.corpus import write_json
from micro_rsi.tokenization.character import CharacterTokenizer

PAGES = [
    "祝福",
    "弟兄",
    "離婚",
    "幸福的家庭",
    "傷逝 (魯迅)",
    "長明燈",
    "孤獨者",
    "高老夫子",
    "示衆",
    "肥皂",
    "在酒樓上",
]
API = "https://zh.wikisource.org/w/api.php"


class BodyParagraphs(HTMLParser):
    """只收正文段落，排除页头、许可、脚注、目录和样式。"""

    def __init__(self):
        super().__init__()
        self.stack = []
        self.paragraphs = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        tokens = set((attrs.get("class", "") + " " + attrs.get("id", "")).split())
        excluded = {
            "headerContainer",
            "licenseContainer",
            "references",
            "reflist",
            "noprint",
            "navbox",
            "toc",
            "ws-noexport",
        }
        skip = (
            bool(self.stack and self.stack[-1][1])
            or bool(tokens & excluded)
            or tag in {"style", "script", "sup", "table"}
        )
        if tag not in {"br", "hr", "img", "meta", "link", "input", "wbr"}:
            self.stack.append((tag, skip))
        if tag == "p" and not skip:
            self.current = []

    def handle_endtag(self, tag):
        if tag == "p" and self.current is not None:
            text = "".join(self.current).strip()
            if text:
                self.paragraphs.append(text)
            self.current = None
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        if self.current is not None and not (self.stack and self.stack[-1][1]):
            self.current.append(data)


def fetch_page(page: str, target: Path) -> dict:
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    query = urllib.parse.urlencode(
        {"action": "parse", "page": page, "prop": "text|revid", "format": "json"}
    )
    request = urllib.request.Request(
        API + "?" + query, headers={"User-Agent": "micro-rsi-learning/0.1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.load(response)
    if "parse" not in data:
        raise ValueError(f"获取失败：{page}：{data}")
    write_json(target, data)
    return data


def extract_page(data: dict) -> dict:
    parsed = data["parse"]
    parser = BodyParagraphs()
    parser.feed(parsed["text"]["*"])
    converter = OpenCC("t2s")
    text = re.sub(r"[\s\u3000]+", "", "".join(parser.paragraphs))
    text = converter.convert(text).replace("︰", "：")
    # 与《呐喊》保持相同的句末换行策略，不额外统一引号或修订措辞。
    text = re.sub(r"([。！？])([”’]?)", r"\1\2\n", text).strip()
    if len(text) < 500:
        raise ValueError(f"正文过短，需要检查页面结构：{parsed['title']}")
    return {"title": converter.convert(parsed["title"]), "text": text}


def prepare_pair(raw_dir: Path, nahan_dir: Path, mixed_dir: Path, control_dir: Path) -> dict:
    for path in [mixed_dir, control_dir]:
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"输出目录非空：{path}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    sources, documents = [], []
    for i, page in enumerate(PAGES):
        path = raw_dir / f"{i + 1:02d}.json"
        data = fetch_page(page, path)
        document = extract_page(data)
        documents.append(document)
        revision = data["parse"]["revid"]
        sources.append(
            {
                "page": page,
                "revision": revision,
                "url": f"https://zh.wikisource.org/w/index.php?oldid={revision}",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "chars": len(document["text"]),
            }
        )
    base = {
        split: json.loads((nahan_dir / f"{split}.json").read_text())["documents"]
        for split in ["train", "validation"]
    }
    additional = {"train": [], "validation": []}
    for doc in documents:
        cut = int(len(doc["text"]) * 0.9)
        additional["train"].append({"title": doc["title"], "text": doc["text"][:cut]})
        additional["validation"].append({"title": doc["title"], "text": doc["text"][cut:]})
    tokenizer = CharacterTokenizer.fit([d["text"] for d in base["train"] + additional["train"]])
    provenance = {
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "panghuang_sources": sources,
        "rights": "维基文库页面公版声明；原始 API 响应保留页头及许可",
        "nahan_manifest": json.loads((nahan_dir / "manifest.json").read_text()),
        "normalization": "正文 p 标签，排除页头/许可/脚注；OpenCC t2s；与呐喊一致的句末换行",
        "validation_fraction": 0.1,
        "split": "先按每篇前90%/后10%切分，再独立构造窗口",
        "shared_vocab": "只由两书训练区段建词表，两组相同；无验证文本参与",
    }
    summaries = {}
    for path, splits in [
        (mixed_dir, {s: base[s] + additional[s] for s in base}),
        (control_dir, base),
    ]:
        path.mkdir(parents=True, exist_ok=True)
        ids = [i for d in splits["validation"] for i in tokenizer.encode(d["text"])]
        manifest = {
            **provenance,
            "train_chars": sum(len(d["text"]) for d in splits["train"]),
            "validation_chars": len(ids),
            "validation_unknown_chars": ids.count(0),
            "validation_unknown_rate": ids.count(0) / len(ids),
            "vocab_size": tokenizer.vocab_size,
            "training_titles": [d["title"] for d in splits["train"]],
        }
        for split, docs in splits.items():
            write_json(path / f"{split}.json", {"documents": docs})
        tokenizer.save(path / "tokenizer.json")
        write_json(path / "manifest.json", manifest)
        summaries[str(path)] = manifest
    return summaries


def main():
    parser = argparse.ArgumentParser(description="准备《彷徨》与共享词表对照数据（会下载11个页面）")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/panghuang-wikisource"))
    parser.add_argument("--nahan-dir", type=Path, default=Path("data/processed/nahan"))
    parser.add_argument("--mixed-dir", type=Path, default=Path("data/processed/nahan-panghuang"))
    parser.add_argument("--control-dir", type=Path, default=Path("data/processed/nahan-control"))
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_pair(args.raw_dir, args.nahan_dir, args.mixed_dir, args.control_dir),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
