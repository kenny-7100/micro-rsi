"""字符与 ID 的稳定映射；0 号只用于未知字符。"""

import json
from pathlib import Path


class CharacterTokenizer:
    def __init__(self, characters: list[str]):
        if len(set(characters)) != len(characters) or any(len(c) != 1 for c in characters):
            raise ValueError("词表必须由不重复的单字符组成")
        self.characters = characters
        self.stoi = {c: i + 1 for i, c in enumerate(characters)}

    @classmethod
    def fit(cls, texts: list[str]) -> "CharacterTokenizer":
        return cls(sorted(set("".join(texts))))

    @property
    def vocab_size(self) -> int:
        return len(self.characters) + 1

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join("<unk>" if i == 0 else self.characters[i - 1] for i in ids)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    def to_dict(self) -> dict:
        return {"kind": "character", "version": 1, "characters": self.characters}

    @classmethod
    def from_dict(cls, data: dict) -> "CharacterTokenizer":
        if data.get("kind") != "character" or data.get("version") != 1:
            raise ValueError("不支持的 tokenizer 格式")
        return cls(data["characters"])

    @classmethod
    def load(cls, path: str | Path) -> "CharacterTokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
