"""项目命令入口：环境检查、语料准备、训练、生成与验收。"""

import argparse
import json
import platform
from pathlib import Path

import torch

from micro_rsi.device import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="micro-rsi：本地语言模型学习实验")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="检查依赖、设备和自动求导")
    doctor.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    prepare = commands.add_parser("prepare", help="下载或复用《呐喊》原件，清洗并切分")
    prepare.add_argument("--raw", type=Path, default=Path("data/raw/nahan-pg27166.txt"))
    prepare.add_argument("--output", type=Path, default=Path("data/processed/nahan"))
    prepare.add_argument("--download", action="store_true", help="显式允许下载原始语料")
    prepare.add_argument("--validation-fraction", type=float, default=0.1)
    train = commands.add_parser("train", help="从随机初始化训练并记录验收样本")
    train.add_argument("--config", type=Path, default=Path("configs/pretrain/tiny_char.toml"))
    train.add_argument("--max-steps", type=int)
    train.add_argument("--resume", type=Path, help="恢复完整训练状态；max-steps 为目标累计步数")
    train.add_argument("--run-dir", type=Path)
    train.add_argument("--checkpoint-dir", type=Path)
    train.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    generate = commands.add_parser("generate", help="独立加载检查点并续写")
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=200)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    readability = commands.add_parser("readability", help="建立或汇总盲化短续写验收")
    readability_sub = readability.add_subparsers(dest="readability_command", required=True)
    collect = readability_sub.add_parser("collect", help="固定前缀、模型和种子生成盲化样本")
    collect.add_argument("--config", type=Path, default=Path("configs/evaluation/readability.json"))
    collect.add_argument(
        "--current", type=Path, default=Path("artifacts/checkpoints/wiki-zh-current/best.pt")
    )
    collect.add_argument(
        "--medium", type=Path, default=Path("artifacts/checkpoints/wiki-zh-medium/best.pt")
    )
    collect.add_argument("--output", type=Path, default=Path("runs/readability/review.json"))
    score_readability = readability_sub.add_parser("score", help="汇总逐条人工评分")
    score_readability.add_argument(
        "--file", type=Path, default=Path("runs/readability/review.json")
    )
    compare = commands.add_parser("compare", help="分析语料增量与 temperature 对比实验")
    compare.add_argument("--output", type=Path, default=Path("runs/corpus-comparison/report.json"))
    compare.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    verify = commands.add_parser("verify", help="独立验收完整验证区段和固定生成")
    verify.add_argument("--run-dir", type=Path, default=Path("runs/nahan"))
    verify.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints/nahan"))
    verify.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()

    if args.command == "doctor":
        device = resolve_device(args.device)
        # 同时验证设备上的前向计算、反向传播和结果回传。
        x = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
        loss = x.square().sum()
        loss.backward()
        expected = torch.tensor([2.0, 4.0, 6.0])
        assert x.grad is not None
        torch.testing.assert_close(x.grad.cpu(), expected)
        print(
            json.dumps(
                {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "mps_available": torch.backends.mps.is_available(),
                    "cuda_available": torch.cuda.is_available(),
                    "device": str(device),
                    "forward_backward": "ok",
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    elif args.command == "prepare":
        from micro_rsi.data.corpus import download, prepare

        if args.download:
            download(args.raw)
        print(
            json.dumps(
                prepare(args.raw, args.output, args.validation_fraction),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "train":
        from micro_rsi.training.runner import train

        train(
            args.config, args.max_steps, args.run_dir, args.checkpoint_dir, args.device, args.resume
        )
    elif args.command == "generate":
        from micro_rsi.inference.generation import generate, load_model

        torch.set_num_threads(4)
        model, tokenizer, _ = load_model(args.checkpoint, resolve_device(args.device))
        print(
            generate(
                model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, args.seed
            )
        )

    elif args.command == "verify":
        from micro_rsi.evaluation.acceptance import verify

        verify(args.run_dir, args.checkpoint_dir, args.device)

    elif args.command == "compare":
        from micro_rsi.evaluation.comparison import compare

        compare(args.output, args.device)

    elif args.command == "readability":
        from micro_rsi.evaluation.readability import collect, score

        if args.readability_command == "collect":
            collect(args.config, args.current, args.medium, args.output)
        else:
            score(args.file)
