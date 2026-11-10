"""项目命令入口；当前提供环境和设备验证。"""

import argparse
import json
import platform

import torch

from micro_rsi.device import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="micro-rsi：本地语言模型学习实验")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="检查依赖、设备和自动求导")
    doctor.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
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
