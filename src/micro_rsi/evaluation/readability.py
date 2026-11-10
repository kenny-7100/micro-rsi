"""固定前缀、多个随机种子的盲化短续写及人工可读性评分。"""

import argparse
import json
import random
from pathlib import Path

import torch

from micro_rsi.data.corpus import write_json
from micro_rsi.device import resolve_device
from micro_rsi.inference.generation import generate, load_model


def collect(config_path: Path, current: Path, medium: Path, output: Path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if output.exists():
        raise FileExistsError(f"验收文件已存在，避免覆盖：{output}")
    torch.set_num_threads(4)
    models = {}
    for name, checkpoint in [("current", current), ("medium", medium)]:
        models[name] = load_model(checkpoint, resolve_device("auto"))
    tokenizers = [item[1].to_dict() for item in models.values()]
    if tokenizers[0] != tokenizers[1]:
        raise ValueError("可读性对照模型必须共享 tokenizer")
    samples = []
    for prompt_id, prompt in enumerate(config["prompts"]):
        for seed in config["seeds"]:
            for name in ["current", "medium"]:
                model, tokenizer, _ = models[name]
                full = generate(
                    model, tokenizer, prompt, config["max_new_tokens"], config["temperature"], seed
                )
                samples.append(
                    {
                        "model": name,
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "seed": seed,
                        "continuation": full[len(prompt) :],
                    }
                )
    random.Random(2026).shuffle(samples)
    for i, sample in enumerate(samples, start=1):
        sample["blind_id"] = f"S{i:02d}"
        sample["scores"] = {key: None for key in ["words", "sentences", "continuity", "repetition"]}
        sample["notes"] = ""
    rubric = {
        "words": "0=大量不成词或搭配错误；1=多数词可辨、仍有数处错误；2=词组基本自然",
        "sentences": "0=大部分句子无法理解；1=至少一句大意可理解、仍有断裂；2=多数句子大意清楚",
        "continuity": "0=与前缀无关或前后无联系；1=部分有联系；2=基本承接前缀并保持局部联系",
        "repetition": "0=明显循环或标点崩坏；1=有重复但未主导全文；2=无明显机械重复",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    blind = [
        {k: v for k, v in sample.items() if k not in {"model", "seed", "prompt_id"}}
        for sample in samples
    ]
    write_json(output.with_name("blind-review.json"), {"rubric": rubric, "samples": blind})
    write_json(
        output,
        {
            "rubric": rubric,
            "config": config,
            "model_map": {sample["blind_id"]: sample["model"] for sample in samples},
            "samples": samples,
        },
    )
    print(
        f"已生成{len(samples)}条盲化样本：{output}。"
        "请逐条填写0-2分，然后运行 micro-rsi readability score。"
    )


def score(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data["samples"]:
        raise ValueError("评分样本不能为空")
    identifiers = [sample["blind_id"] for sample in data["samples"]]
    if len(set(identifiers)) != len(identifiers) or set(identifiers) != set(data["model_map"]):
        raise ValueError("盲化样本ID重复或与映射不一致")
    groups = {model: [] for model in data["model_map"].values()}
    for sample in data["samples"]:
        if data["model_map"][sample["blind_id"]] != sample["model"]:
            raise ValueError("模型映射损坏")
        scores = sample["scores"]
        if any(
            type(scores.get(k)) is not int or scores[k] not in range(3)
            for k in ["words", "sentences", "continuity", "repetition"]
        ):
            raise ValueError(f"请先给所有指标填写0、1或2分：{sample['blind_id']}")
        groups[sample["model"]].append(scores)
    summary = {}
    for model, rows in groups.items():
        summary[model] = {
            "samples": len(rows),
            "mean": {
                key: sum(row[key] for row in rows) / len(rows)
                for key in ["words", "sentences", "continuity", "repetition"]
            },
            "pass_count": sum(
                row["words"] >= 1 and row["sentences"] >= 1 and row["repetition"] >= 1
                for row in rows
            ),
            "pass_rate": sum(
                row["words"] >= 1 and row["sentences"] >= 1 and row["repetition"] >= 1
                for row in rows
            )
            / len(rows),
        }
    for result in summary.values():
        result["model_pass"] = result["pass_rate"] >= 0.75 and result["mean"]["sentences"] >= 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    write_json(path.with_name("scores-summary.json"), summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="生成或汇总短文本盲化可读性验收")
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("collect")
    make.add_argument("--config", type=Path, default=Path("configs/evaluation/readability.json"))
    make.add_argument(
        "--current", type=Path, default=Path("artifacts/checkpoints/wiki-zh-current/best.pt")
    )
    make.add_argument(
        "--medium", type=Path, default=Path("artifacts/checkpoints/wiki-zh-medium/best.pt")
    )
    make.add_argument("--output", type=Path, default=Path("runs/readability/review.json"))
    report = sub.add_parser("score")
    report.add_argument("--file", type=Path, default=Path("runs/readability/review.json"))
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.config, args.current, args.medium, args.output)
    else:
        score(args.file)


if __name__ == "__main__":
    main()
