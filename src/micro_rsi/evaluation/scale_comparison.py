"""共同语料上比较模型尺寸；冻结检查点后才打开独立测试集。"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from micro_rsi.data.corpus import write_json
from micro_rsi.data.windows import TextWindows
from micro_rsi.device import resolve_device
from micro_rsi.evaluation.acceptance import score_documents
from micro_rsi.evaluation.metrics import evaluate
from micro_rsi.inference.generation import load_model


def compare(output: Path, device_name="auto"):
    if output.exists():
        raise FileExistsError(f"输出已经存在：{output}")
    torch.set_num_threads(4)
    device = resolve_device(device_name)
    report = {"device": str(device), "experiments": {}}
    shared = None
    for name in ["current", "medium"]:
        run_dir = Path(f"runs/wiki-zh-{name}")
        config = json.loads((run_dir / "config.json").read_text())
        environment = json.loads((run_dir / "environment.json").read_text())
        summary = json.loads((run_dir / "summary.json").read_text())
        data_dir = Path(config["data"]["directory"])
        digests = environment["data_sha256"]
        if shared is not None and digests != shared:
            raise ValueError("两模型的训练/验证数据或词表不同")
        shared = digests
        for file, expected in digests.items():
            if hashlib.sha256((data_dir / file).read_bytes()).hexdigest() != expected:
                raise ValueError(f"验收数据改变：{file}")
        experiment = {"summary": summary, "config": config, "scores": {}}
        # 先固定训练时选出的 best.pt；不得根据 test 分数重新选择检查点。
        for kind in ["initial", "best"]:
            checkpoint_path = Path(config["training"]["checkpoint_dir"]) / f"{kind}.pt"
            model, tokenizer, checkpoint = load_model(checkpoint_path, device)
            if kind == "best" and checkpoint["step"] != summary["best_step"]:
                raise ValueError("最佳检查点与训练记录不一致")
            windows = TextWindows.load(
                data_dir / "validation.json", tokenizer, model.config.context_length
            )
            scores = {
                "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "step": checkpoint["step"],
                "fixed_validation_loss": evaluate(
                    model,
                    windows,
                    config["training"]["batch_size"],
                    config["training"]["eval_batches"],
                    device,
                ),
            }
            for split in ["validation", "test"]:
                docs = json.loads((data_dir / f"{split}.json").read_text())["documents"]
                scores[split] = score_documents(model, tokenizer, docs, device)
            experiment["scores"][kind] = scores
        report["experiments"][name] = experiment
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description="当前模型与扩大模型的共同验证和独立测试对照")
    parser.add_argument(
        "--output", type=Path, default=Path("runs/wiki-scale-comparison/report.json")
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    compare(args.output, args.device)


if __name__ == "__main__":
    main()
