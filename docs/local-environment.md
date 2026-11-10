# 本地环境经验

## uv 缓存与项目位于不同文件系统

2026-11-10 初始化时，项目位于 `/Volumes/kenny/github/micro-rsi`。`uv sync` 提示无法建立硬链接，随后自动回退为文件复制，安装成功。

uv 缓存与虚拟环境位于不同文件系统时可能出现此提示。项目已在 `pyproject.toml` 中设置：

```toml
[tool.uv]
link-mode = "copy"
```

后续同步直接采用复制，避免重复尝试硬链接。这是安装文件的方式设置，不是模型运行性能优化；迁移环境后可根据磁盘布局重新判断。

## 验证 MPS 的基础计算能力

当前设备为 Apple M2、16 GB 统一内存。初始化时使用 Python 3.12.11、PyTorch 2.15.0，以下命令均通过：

```sh
uv run micro-rsi doctor
uv run micro-rsi doctor --device cpu
```

`doctor` 除查询设备可用性外，还执行张量平方求和、反向传播，并核对梯度。第一次命令实际选择 MPS，第二次选择 CPU。

这能验证基础设备计算与自动求导，无法证明未来模型的全部算子兼容性、训练速度或内存需求。增加模型后仍需进行实际训练短跑；升级 PyTorch 或迁移设备后可重新执行这些命令。
