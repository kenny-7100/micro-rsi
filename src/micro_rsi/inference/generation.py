"""从检查点独立恢复，并在 CPU 上采样以便跨设备统一随机数入口。"""

import math

import torch

from micro_rsi.models.transformer import ModelConfig, TransformerLM
from micro_rsi.tokenization.character import CharacterTokenizer


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("不支持的检查点格式")
    tokenizer = CharacterTokenizer.from_dict(checkpoint["tokenizer"])
    model = TransformerLM(ModelConfig(**checkpoint["model_config"]))
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError("检查点词表与模型不一致")
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), tokenizer, checkpoint


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, seed=42):
    if not prompt:
        raise ValueError("前缀不能为空")
    if not math.isfinite(temperature) or temperature <= 0 or max_new_tokens < 0:
        raise ValueError("temperature 须为有限正数，生成长度须非负")
    unknown = sorted(set(prompt) - set(tokenizer.characters))
    if unknown:
        raise ValueError(f"前缀含词表外字符：{unknown}；请改用已知字符")
    device = next(model.parameters()).device
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generator = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            # 第一版没有 KV cache，超长时仅保留最近的上下文并重新计算位置。
            logits = model(ids[:, -model.config.context_length :])[0, -1].float().cpu()
            logits[0] = -float("inf")  # 未知字符不是可读的续写内容。
            probabilities = torch.softmax(logits / temperature, dim=-1)
            sampled = torch.multinomial(probabilities, 1, generator=generator).item()
            ids = torch.cat([ids, torch.tensor([[sampled]], device=device)], dim=1)
        return tokenizer.decode(ids[0].tolist())
    finally:
        model.train(was_training)
