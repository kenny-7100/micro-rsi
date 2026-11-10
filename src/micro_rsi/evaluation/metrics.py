"""固定窗口集合上的 token 平均交叉熵，隔离训练与验证随机数。"""

import torch
from torch.nn import functional as F


@torch.no_grad()
def evaluate(model, windows, batch_size, batches, device, seed=1234):
    generator = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.eval()
    total = 0.0
    tokens = 0
    try:
        for _ in range(batches):
            x, y = windows.sample(batch_size, generator)
            logits = model(x.to(device))
            total += F.cross_entropy(
                logits.flatten(0, 1), y.to(device).flatten(), reduction="sum"
            ).item()
            tokens += int((y != -100).sum())
        return total / tokens
    finally:
        model.train(was_training)
