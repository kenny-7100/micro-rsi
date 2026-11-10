"""可阅读的小型 GPT：显式展示因果注意力与残差连接。"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int = 256
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    ff_dim: int = 512
    dropout: float = 0.1
    init_std: float = 0.02

    def __post_init__(self):
        values = [
            self.vocab_size,
            self.context_length,
            self.embedding_dim,
            self.num_heads,
            self.num_layers,
            self.ff_dim,
        ]
        if any(type(v) is not int or v < 1 for v in values):
            raise ValueError("模型维度必须为正整数")
        if self.embedding_dim % self.num_heads:
            raise ValueError("embedding_dim 必须能被 num_heads 整除")
        if not 0 < self.init_std <= 1:
            raise ValueError("init_std 必须在 (0, 1] 内")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout 必须在 [0, 1) 内")


class CausalAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)
        self.projection = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        mask = torch.tril(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool)
        )
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        head_dim = width // self.num_heads
        # [B,T,C] -> 各自 [B,H,T,D]；Q/K 决定读取哪里，V 提供被读取的内容。
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [
            t.view(batch, length, self.num_heads, head_dim).transpose(1, 2) for t in (q, k, v)
        ]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        scores = scores.masked_fill(~self.mask[:length, :length], float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        context = (weights @ v).transpose(1, 2).contiguous().view(batch, length, width)
        return self.dropout(self.projection(context))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.embedding_dim)
        self.attention = CausalAttention(config)
        self.norm_ff = nn.LayerNorm(config.embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(config.embedding_dim, config.ff_dim),
            nn.GELU(),
            nn.Linear(config.ff_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN：先归一化再变换，残差通路保留原始信息及梯度传播路径。
        x = x + self.attention(self.norm_attention(x))
        return x + self.ff(self.norm_ff(x))


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.apply(lambda module: self._initialize(module, config))
        # 随层数缩放两条残差分支，控制深层网络初始化方差。
        residual_std = config.init_std / math.sqrt(2 * config.num_layers)
        for block in self.blocks:
            nn.init.normal_(block.attention.projection.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ff[2].weight, mean=0.0, std=residual_std)

    @staticmethod
    def _initialize(module: nn.Module, config: ModelConfig) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=config.init_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2 or not 0 < ids.shape[1] <= self.config.context_length:
            raise ValueError("输入必须为 [批大小, 长度]，长度须在上下文范围内")
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.dropout(self.token_embedding(ids) + self.position_embedding(positions))
        # 输出 logits [B,T,V]，交叉熵内部完成 log-softmax，无需预先 softmax。
        return self.head(self.norm(self.blocks(x)))
