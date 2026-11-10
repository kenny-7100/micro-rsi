"""检查上下文依赖、梯度传播与 CPU/MPS 数值一致性。"""

import argparse
import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F

from micro_rsi.data.corpus import write_json
from micro_rsi.data.windows import TextWindows
from micro_rsi.device import resolve_device
from micro_rsi.inference.generation import load_model


def diagnose(checkpoint_path: Path, data: Path, output: Path, device_name="auto"):
    if output.exists():
        raise FileExistsError(f"拒绝覆盖诊断记录：{output}")
    torch.set_num_threads(4)
    device = resolve_device(device_name)
    model, tokenizer, checkpoint = load_model(checkpoint_path, device)
    expected = checkpoint["environment"]["data_sha256"]["validation.json"]
    if hashlib.sha256((data / "validation.json").read_bytes()).hexdigest() != expected:
        raise ValueError("验证集与检查点指纹不同")
    length = model.config.context_length
    if length < 2:
        raise ValueError("上下文诊断要求长度至少为 2")
    windows = TextWindows.load(data / "validation.json", tokenizer, length)
    generator = torch.Generator().manual_seed(1234)
    rows = []
    # 只保留完整窗口，保证最后一个位置有真实目标。
    for _ in range(8):
        x, y = windows.sample(8, generator)
        if (y == -100).any():
            raise ValueError("此诊断要求抽样窗口无填充；请使用长文档验证集")
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            losses = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none").view_as(y)
            shuffled = x.clone()
            permutation = torch.randperm(length - 1, generator=generator).to(device)
            shuffled[:, :-1] = x[:, permutation]
            rows.append(
                {
                    "loss": losses.mean().item(),
                    "normal_last": losses[:, -1].mean().item(),
                    "shuffled_last": F.cross_entropy(model(shuffled)[:, -1], y[:, -1]).item(),
                    "no_context_last": F.cross_entropy(model(x[:, -1:])[:, -1], y[:, -1]).item(),
                }
            )
    x, y = windows.sample(2, generator)
    model.zero_grad(set_to_none=True)
    logits = model(x.to(device))
    F.cross_entropy(logits.flatten(0, 1), y.to(device).flatten()).backward()
    gradients = {
        n: None if p.grad is None else p.grad.norm().item() for n, p in model.named_parameters()
    }
    reference_logits = logits.detach().cpu()
    reference_gradient = model.blocks[0].attention.qkv.weight.grad.detach().cpu()
    model.cpu().zero_grad(set_to_none=True)
    logits = model(x)
    F.cross_entropy(logits.flatten(0, 1), y.flatten()).backward()
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "step": checkpoint["step"],
        "device": str(device),
        "seed": 1234,
        "metrics": {k: sum(row[k] for row in rows) / len(rows) for k in rows[0]},
        "gradient_norms_eval": gradients,
        "cpu_device_max_logit_error": (logits.detach() - reference_logits).abs().max().item(),
        "cpu_device_max_qkv_gradient_error": (
            model.blocks[0].attention.qkv.weight.grad - reference_gradient
        )
        .abs()
        .max()
        .item(),
        "limitations": (
            "打乱上下文同时改变位置；单 token 对照同时缩短上下文并重置位置。"
            "差异支持使用上下文，不能单独衡量语义理解。"
        ),
    }
    write_json(output, result)
    print(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed/wiki-zh-5m-v2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    args = parser.parse_args()
    diagnose(args.checkpoint, args.data, args.output, args.device)


if __name__ == "__main__":
    main()
