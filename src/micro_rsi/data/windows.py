"""从单个篇目区段中取窗口，永远不跨篇目或划分边界。"""

import json
from pathlib import Path

import torch

from micro_rsi.tokenization.character import CharacterTokenizer


class TextWindows:
    def __init__(self, texts: list[str], tokenizer: CharacterTokenizer, context_length: int):
        if context_length < 1:
            raise ValueError("上下文长度必须为正")
        self.context_length = context_length
        self.documents = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in texts]
        self.counts = torch.tensor(
            [max(1, len(d) - context_length) if len(d) >= 2 else 0 for d in self.documents]
        )
        self.total = int(self.counts.sum())
        if not self.total:
            raise ValueError("没有足够长的区段来构造输入与标签，请缩短上下文或增加数据")
        self.ends = self.counts.cumsum(0)

    @classmethod
    def load(cls, path: Path, tokenizer: CharacterTokenizer, context_length: int) -> "TextWindows":
        docs = json.loads(path.read_text(encoding="utf-8"))["documents"]
        return cls([d["text"] for d in docs], tokenizer, context_length)

    def sample(
        self, batch_size: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 按可用窗口数加权，长篇不会因“先等概率选篇目”而被低估。
        indices = torch.randint(self.total, (batch_size,), generator=generator)
        xs, ys = [], []
        for index in indices.tolist():
            doc_id = int(torch.searchsorted(self.ends, index, right=True))
            previous = int(self.ends[doc_id - 1]) if doc_id else 0
            start = index - previous
            chunk = self.documents[doc_id][start : start + self.context_length + 1]
            x = torch.zeros(self.context_length, dtype=torch.long)
            y = torch.full((self.context_length,), -100, dtype=torch.long)
            # 短验证区段右侧补齐，标签 -100 被交叉熵忽略；真实位置不能读到未来补齐。
            x[: len(chunk) - 1] = chunk[:-1]
            y[: len(chunk) - 1] = chunk[1:]
            xs.append(x)
            ys.append(y)
        return torch.stack(xs), torch.stack(ys)
