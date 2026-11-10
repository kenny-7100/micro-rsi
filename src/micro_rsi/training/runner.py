"""训练入口：配置、固定验证、可独立加载的检查点和完整实验记录。"""

import copy
import hashlib
import json
import math
import platform
import random
import time
import tomllib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from micro_rsi.data.corpus import write_json
from micro_rsi.data.windows import TextWindows
from micro_rsi.device import resolve_device
from micro_rsi.evaluation.metrics import evaluate
from micro_rsi.inference.generation import generate
from micro_rsi.models.transformer import ModelConfig, TransformerLM
from micro_rsi.tokenization.character import CharacterTokenizer


def load_config(path: Path) -> dict:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    required = {"seed", "device", "data", "tokenizer", "model", "training", "generation"}
    if set(config) != required:
        raise ValueError(f"配置顶层字段须为 {sorted(required)}")
    fields = {
        "data": {"directory"},
        "tokenizer": {"kind"},
        "model": {
            "kind",
            "context_length",
            "embedding_dim",
            "num_heads",
            "num_layers",
            "ff_dim",
            "init_std",
            "dropout",
        },
        "training": {
            "batch_size",
            "max_steps",
            "learning_rate",
            "weight_decay",
            "grad_clip",
            "eval_interval",
            "eval_batches",
            "checkpoint_dir",
            "run_dir",
        },
        "generation": {"max_new_tokens", "temperature"},
    }
    # 新增初始化参数保留默认值，兼容早期已保存的配置。
    config["model"].setdefault("init_std", 0.02)
    config["training"].setdefault("min_learning_rate", config["training"]["learning_rate"])
    config["training"].setdefault("schedule_start", 0)
    fields["training"].update({"min_learning_rate", "schedule_start"})
    for section, keys in fields.items():
        if set(config[section]) != keys:
            raise ValueError(f"{section} 配置字段须为 {sorted(keys)}")
    if config["tokenizer"]["kind"] != "character":
        raise ValueError("当前只支持 character tokenizer")
    if config["model"]["kind"] != "decoder_transformer":
        raise ValueError("当前只支持 decoder_transformer")
    for key in ["batch_size", "max_steps", "eval_interval", "eval_batches"]:
        if type(config["training"][key]) is not int or config["training"][key] < 1:
            raise ValueError(f"{key} 必须为正整数")
    for key in ["learning_rate", "grad_clip", "weight_decay"]:
        value = config["training"][key]
        if not math.isfinite(value) or value < 0 or (key != "weight_decay" and value == 0):
            raise ValueError(f"{key} 数值非法")
    generation = config["generation"]
    settings = config["training"]
    if not 0 < settings["min_learning_rate"] <= settings["learning_rate"]:
        raise ValueError("最低学习率须为正且不高于初始学习率")
    if type(settings["schedule_start"]) is not int or settings["schedule_start"] < 0:
        raise ValueError("schedule_start 必须为非负整数")
    if type(generation["max_new_tokens"]) is not int or generation["max_new_tokens"] < 0:
        raise ValueError("生成长度必须为非负整数")
    if not math.isfinite(generation["temperature"]) or generation["temperature"] <= 0:
        raise ValueError("temperature 必须为有限正数")
    return config


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def learning_rate_at(step: int, settings: dict) -> float:
    """按累计步数余弦衰减，恢复训练不重启学习率曲线。"""
    progress = max(
        0.0,
        min(
            1.0,
            (step - settings["schedule_start"])
            / (settings["max_steps"] - settings["schedule_start"]),
        ),
    )
    return settings["min_learning_rate"] + 0.5 * (
        settings["learning_rate"] - settings["min_learning_rate"]
    ) * (1 + math.cos(math.pi * progress))


def train(
    config_path: Path, max_steps=None, run_dir=None, checkpoint_dir=None, device=None, resume=None
) -> dict:
    config = load_config(config_path)
    settings = config["training"]
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("训练步数必须为正")
        settings["max_steps"] = max_steps
    if run_dir:
        settings["run_dir"] = str(run_dir)
    if checkpoint_dir:
        settings["checkpoint_dir"] = str(checkpoint_dir)
    if device:
        config["device"] = device
    if settings["schedule_start"] >= settings["max_steps"]:
        raise ValueError("学习率衰减起点必须小于目标累计步数")
    seed_all(config["seed"])
    # 小网络避免 CPU 上大量线程调度开销；MPS 仍负责主要训练计算。
    torch.set_num_threads(4)
    selected = resolve_device(config["device"])
    data_dir = Path(config["data"]["directory"])
    tokenizer = CharacterTokenizer.load(data_dir / "tokenizer.json")
    model_values = {k: v for k, v in config["model"].items() if k != "kind"}
    model = TransformerLM(ModelConfig(vocab_size=tokenizer.vocab_size, **model_values)).to(selected)
    datasets = {
        split: TextWindows.load(data_dir / f"{split}.json", tokenizer, model.config.context_length)
        for split in ["train", "validation"]
    }
    run_path, checkpoint_path = Path(settings["run_dir"]), Path(settings["checkpoint_dir"])
    if run_path.resolve() == checkpoint_path.resolve():
        raise ValueError("运行目录和检查点目录必须不同")
    for directory in [run_path, checkpoint_path]:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"输出目录非空，禁止覆盖：{directory}")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    fingerprints = {
        name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
        for name in ["train.json", "validation.json", "tokenizer.json"]
    }
    metadata = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": str(selected),
        "platform": platform.platform(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "data_sha256": fingerprints,
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    generator = torch.Generator().manual_seed(config["seed"])
    start_step = 0
    if resume is not None:
        source = Path(resume)
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("不支持的恢复检查点格式")
        if checkpoint["model_config"] != asdict(model.config):
            raise ValueError("恢复检查点模型配置不同")
        if checkpoint["tokenizer"] != tokenizer.to_dict():
            raise ValueError("恢复检查点 tokenizer 不同")
        if checkpoint["environment"]["data_sha256"] != fingerprints:
            raise ValueError("恢复数据指纹不同；更换数据须使用显式迁移流程")
        prior = checkpoint["config"]
        if (
            config["seed"] != prior["seed"]
            or settings["batch_size"] != prior["training"]["batch_size"]
        ):
            raise ValueError("精确恢复要求种子、批大小一致")
        if str(selected) != checkpoint["environment"]["device"]:
            raise ValueError("恢复要求相同设备类型以还原随机状态")
        start_step = checkpoint["step"]
        if start_step >= settings["max_steps"]:
            raise ValueError("max_steps 是累计步数，必须高于已有步数")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        for group in optimizer.param_groups:
            group["weight_decay"] = settings["weight_decay"]
        generator.set_state(checkpoint["sample_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if selected.type == "mps":
            torch.mps.set_rng_state(checkpoint["mps_rng_state"])
        elif selected.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        metadata["resume_from"] = str(source)
        metadata["resume_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        metadata["resume_step"] = start_step
    training_docs = json.loads((data_dir / "train.json").read_text())["documents"]
    validation_docs = json.loads((data_dir / "validation.json").read_text())["documents"]

    def known_prefix(documents):
        for document in documents:
            text = document["text"]
            for start in range(max(0, len(text) - 15)):
                prefix = text[start : start + 16]
                if all(tokenizer.encode(prefix)):
                    return prefix
        raise ValueError("训练/验证文本没有足够的连续已知字符作为验收前缀")

    prompts = {
        "training": known_prefix(training_docs),
        "validation": known_prefix(validation_docs),
        "new": (
            "地球上的水主要来自"
            if all(tokenizer.encode("地球上的水主要来自"))
            else known_prefix(training_docs[1:] or training_docs)
        ),
    }

    for directory in [run_path, checkpoint_path]:
        directory.mkdir(parents=True, exist_ok=True)
    write_json(run_path / "config.json", config)
    write_json(run_path / "data_manifest.json", manifest)
    write_json(run_path / "environment.json", metadata)

    def save_checkpoint(name, step, val_loss):
        checkpoint = {
            "format_version": 1,
            "model_config": asdict(model.config),
            "model_state": cpu_tree(model.state_dict()),
            "tokenizer": tokenizer.to_dict(),
            "optimizer_state": cpu_tree(optimizer.state_dict()),
            "step": step,
            "validation_loss": val_loss,
            "config": config,
            "environment": metadata,
            "torch_rng_state": torch.get_rng_state(),
            "sample_rng_state": generator.get_state(),
        }
        if selected.type == "mps":
            checkpoint["mps_rng_state"] = torch.mps.get_rng_state()
        if selected.type == "cuda":
            checkpoint["cuda_rng_states"] = torch.cuda.get_rng_state_all()
        target = checkpoint_path / name
        temporary = target.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(target)

    def measure(step):
        result = {"step": step}
        for split, windows in datasets.items():
            result[f"{split}_loss"] = evaluate(
                model,
                windows,
                settings["batch_size"],
                settings["eval_batches"],
                selected,
            )
        if not all(math.isfinite(result[k]) for k in ["train_loss", "validation_loss"]):
            raise RuntimeError("验证出现非有限 loss")
        result["learning_rate"] = optimizer.param_groups[0]["lr"]
        with (run_path / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result) + "\n")
        print(json.dumps(result), flush=True)
        return result

    def samples():
        return {
            name: {
                "prompt": prompt,
                "text": generate(
                    model, tokenizer, prompt, **config["generation"], seed=config["seed"]
                ),
            }
            for name, prompt in prompts.items()
        }

    started = time.perf_counter()
    initial = measure(start_step)
    save_checkpoint("initial.pt", start_step, initial["validation_loss"])
    save_checkpoint("best.pt", start_step, initial["validation_loss"])
    write_json(run_path / "initial_samples.json", samples())
    best_loss, best_step = initial["validation_loss"], start_step
    train_seconds = 0.0
    model.train()
    for step in range(start_step + 1, settings["max_steps"] + 1):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at(step, settings)
        if selected.type == "mps":
            torch.mps.synchronize()
        tick = time.perf_counter()
        x, y = datasets["train"].sample(settings["batch_size"], generator)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x.to(selected))
        loss = F.cross_entropy(logits.flatten(0, 1), y.to(selected).flatten())
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"第 {step} 步 loss 非有限")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), settings["grad_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        if selected.type == "mps":
            torch.mps.synchronize()
        train_seconds += time.perf_counter() - tick
        if step % 25 == 0:
            print(
                f"step={step} batch_loss={loss.item():.4f} "
                f"seconds_per_step={train_seconds / (step - start_step):.3f}",
                flush=True,
            )
        if step % settings["eval_interval"] == 0 or step == settings["max_steps"]:
            result = measure(step)
            if result["validation_loss"] < best_loss:
                best_loss, best_step = result["validation_loss"], step
                save_checkpoint("best.pt", step, best_loss)
    save_checkpoint("last.pt", step, result["validation_loss"])
    # 最终样本取最佳模型，而不是假定最后一步最好。
    best = torch.load(checkpoint_path / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model_state"])
    write_json(run_path / "best_samples.json", samples())
    summary = {
        **metadata,
        "steps": step,
        "start_step": start_step,
        "new_steps": step - start_step,
        "best_step": best_step,
        "initial_validation_loss": initial["validation_loss"],
        "best_validation_loss": best_loss,
        "last_validation_loss": result["validation_loss"],
        "training_seconds": train_seconds,
        "seconds_per_step": train_seconds / (step - start_step),
        "wall_seconds": time.perf_counter() - started,
        "tokens_processed": step * settings["batch_size"] * model.config.context_length,
        "new_tokens_processed": (step - start_step)
        * settings["batch_size"]
        * model.config.context_length,
        "best_checkpoint": str(checkpoint_path / "best.pt"),
    }
    if selected.type == "mps":
        summary["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
        summary["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    write_json(run_path / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary
