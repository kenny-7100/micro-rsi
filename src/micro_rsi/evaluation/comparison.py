"""相同词表下比较新增语料，并记录固定前缀的温度实验。"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from micro_rsi.data.corpus import write_json
from micro_rsi.device import resolve_device
from micro_rsi.evaluation.acceptance import score_documents
from micro_rsi.inference.generation import generate, load_model


def repetition(text: str, n: int = 4) -> float:
    """重复 n-gram 出现比例；只衡量机械重复，不代表语言质量。"""
    grams = [text[i : i + n] for i in range(max(0, len(text) - n + 1))]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def compare(output: Path, device_name="auto") -> dict:
    if output.exists():
        raise FileExistsError(f"比较记录已存在，避免覆盖：{output}")
    torch.set_num_threads(4)
    device = resolve_device(device_name)
    root = Path("data/processed")
    baseline_validation = json.loads((root / "nahan/validation.json").read_text())["documents"]
    control_validation = json.loads((root / "nahan-control/validation.json").read_text())[
        "documents"
    ]
    mixed_validation = json.loads((root / "nahan-panghuang/validation.json").read_text())[
        "documents"
    ]
    if (
        baseline_validation != control_validation
        or mixed_validation[: len(control_validation)] != control_validation
    ):
        raise ValueError("《呐喊》验证区段发生变化，不能作成对比较")
    vocab = (root / "nahan-control/tokenizer.json").read_bytes()
    if vocab != (root / "nahan-panghuang/tokenizer.json").read_bytes():
        raise ValueError("成对实验必须使用相同词表")
    panels = {
        "nahan": control_validation,
        "panghuang": mixed_validation[len(control_validation) :],
        "combined": mixed_validation,
    }
    prompts = {
        name: sample["prompt"]
        for name, sample in json.loads(Path("runs/nahan/best_samples.json").read_text()).items()
    }
    report = {
        "device": str(device),
        "seed": 42,
        "max_new_tokens": 200,
        "temperatures": [0.3, 0.5, 0.8],
        "prompts": prompts,
        "shared_tokenizer_sha256": hashlib.sha256(vocab).hexdigest(),
        "models": {},
    }
    for experiment in ["nahan", "nahan-control", "nahan-panghuang"]:
        result = {"scores": {}, "temperature_samples": {}}
        summary = json.loads(Path(f"runs/{experiment}/summary.json").read_text())
        result["training_summary"] = summary
        data_dir = root / experiment
        environment = json.loads(Path(f"runs/{experiment}/environment.json").read_text())
        for name, digest in environment["data_sha256"].items():
            if hashlib.sha256((data_dir / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"训练数据已变化：{experiment}/{name}")
        for kind in ["initial", "best", "last"]:
            if experiment == "nahan" and kind != "best":
                continue
            model, tokenizer, checkpoint = load_model(
                f"artifacts/checkpoints/{experiment}/{kind}.pt", device
            )
            if experiment != "nahan":
                if tokenizer.to_dict() != json.loads(vocab):
                    raise ValueError("检查点与共享词表不一致")
                result["scores"][kind] = {
                    "step": checkpoint["step"],
                    **{
                        panel: score_documents(model, tokenizer, docs, device)
                        for panel, docs in panels.items()
                    },
                }
            if kind == "best":
                for temperature in report["temperatures"]:
                    samples = {}
                    for category, prompt in prompts.items():
                        text = generate(model, tokenizer, prompt, 200, temperature, 42)
                        continuation = text[len(prompt) :]
                        samples[category] = {
                            "text": text,
                            "new_tokens": len(continuation),
                            "repeated_4gram_fraction": repetition(continuation),
                        }
                    result["temperature_samples"][str(temperature)] = samples
        report["models"][experiment] = result
    a = report["models"]["nahan-control"]["scores"]["initial"]
    b = report["models"]["nahan-panghuang"]["scores"]["initial"]
    report["paired_initial_scores_identical"] = all(a[p] == b[p] for p in panels)
    if not report["paired_initial_scores_identical"]:
        raise RuntimeError("成对随机初始化基线不一致")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(f"已保存温度与同验证区段对比：{output}")
    return report


def main():
    parser = argparse.ArgumentParser(description="对比《呐喊》和新增《彷徨》的实验结果")
    parser.add_argument("--output", type=Path, default=Path("runs/corpus-comparison/report.json"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()
    compare(args.output, args.device)


if __name__ == "__main__":
    main()
