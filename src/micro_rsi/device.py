"""集中处理设备选择，让训练和推理使用相同规则。"""

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """自动选择 MPS、CUDA 或 CPU；显式请求不可用设备时给出错误。"""
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"未知设备：{requested}")

    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("当前环境无法使用 MPS，请检查 PyTorch/macOS 或选择 cpu。")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用 CUDA，请选择 auto 或 cpu。")
    return torch.device(requested)
