"""训练结束后独立验收：全验证区段评分、固定样本与原文匹配检查。"""

import hashlib
import json
from difflib import SequenceMatcher
from pathlib import Path

import torch
from torch.nn import functional as F

from micro_rsi.data.corpus import write_json
from micro_rsi.data.windows import TextWindows
from micro_rsi.device import resolve_device
from micro_rsi.evaluation.metrics import evaluate
from micro_rsi.inference.generation import generate, load_model


@torch.no_grad()
def score_documents(model, tokenizer, documents, device):
    """无重叠地评分每个区段除首字符外的所有目标，短尾也计入。"""
    total_loss, count, known_loss, known_count = 0.0, 0, 0.0, 0
    model.eval()
    for document in documents:
        ids = tokenizer.encode(document["text"])
        for start in range(0, len(ids) - 1, model.config.context_length):
            chunk = ids[start : start + model.config.context_length + 1]
            x = torch.tensor([chunk[:-1]], device=device)
            y = torch.tensor(chunk[1:], device=device)
            logits = model(x)[0]
            losses = F.cross_entropy(logits, y, reduction="none")
            total_loss += losses.sum().item()
            count += len(y)
            known = y != 0
            known_loss += losses[known].sum().item()
            known_count += int(known.sum())
    return {
        "loss": total_loss / count,
        "targets": count,
        "known_target_loss": known_loss / known_count,
        "known_targets": known_count,
    }


def verify(run_dir: Path, checkpoint_dir: Path, requested_device="auto"):
    torch.set_num_threads(4)
    device = resolve_device(requested_device)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    data_dir = Path(config["data"]["directory"])
    for name, expected in environment["data_sha256"].items():
        if hashlib.sha256((data_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"验收数据已变化：{name}")
    validation = json.loads((data_dir / "validation.json").read_text())["documents"]
    train_docs = json.loads((data_dir / "train.json").read_text())["documents"]
    samples = json.loads((run_dir / "best_samples.json").read_text(encoding="utf-8"))
    report = {"device": str(device), "checkpoints": {}}
    for name in ["initial", "best"]:
        model, tokenizer, checkpoint = load_model(checkpoint_dir / f"{name}.pt", device)
        windows = TextWindows.load(
            data_dir / "validation.json", tokenizer, model.config.context_length
        )
        fixed_loss = evaluate(
            model,
            windows,
            config["training"]["batch_size"],
            config["training"]["eval_batches"],
            device,
        )
        generated = {}
        for category, sample in samples.items():
            text = generate(
                model, tokenizer, sample["prompt"], **config["generation"], seed=config["seed"]
            )
            continuation = text[len(sample["prompt"]) :]
            longest = max(
                SequenceMatcher(None, continuation, doc["text"], autojunk=False)
                .find_longest_match()
                .size
                for doc in train_docs
            )
            generated[category] = {
                "prompt": sample["prompt"],
                "text": text,
                "new_tokens": len(continuation),
                "longest_training_match_chars": longest,
            }
        report["checkpoints"][name] = {
            "step": checkpoint["step"],
            "fixed_validation_loss": fixed_loss,
            "full_validation": score_documents(model, tokenizer, validation, device),
            "samples": generated,
        }
        if name == "best":
            report["best_samples_match_training_process"] = all(
                generated[k]["text"] == sample["text"] for k, sample in samples.items()
            )
    initial, best = (report["checkpoints"][name] for name in ["initial", "best"])
    report["full_validation_loss_decreased"] = (
        best["full_validation"]["loss"] < initial["full_validation"]["loss"]
    )
    report["fixed_validation_loss_decreased"] = (
        best["fixed_validation_loss"] < initial["fixed_validation_loss"]
    )
    report["generation_lengths_valid"] = all(
        sample["new_tokens"] == config["generation"]["max_new_tokens"]
        for checkpoint in report["checkpoints"].values()
        for sample in checkpoint["samples"].values()
    )
    write_json(run_dir / f"acceptance-{device.type}.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not (
        report["full_validation_loss_decreased"]
        and report["fixed_validation_loss_decreased"]
        and report["generation_lengths_valid"]
    ):
        raise RuntimeError("数值或生成长度验收未通过，请检查记录")
    return report
