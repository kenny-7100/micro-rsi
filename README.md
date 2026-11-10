# micro-rsi

在本地从零训练小型语言模型，学习它如何根据已有文本继续输出 token。第一阶段面向字符级自回归续写，后续逐步扩展 tokenizer、模型与训练方式。

当前状态：工程骨架和环境检查已实现；模型、训练及生成尚未实现。无需下载预训练权重。

## 环境准备

目标环境：Apple Silicon Mac，Python 3.12，uv。PyTorch 通过 MPS 使用 Apple GPU，也支持 CPU。

```sh
uv sync
uv run micro-rsi doctor
uv run micro-rsi doctor --device cpu
```

`uv sync` 会创建项目内的 `.venv`，并根据 `uv.lock` 安装运行与开发依赖。Python 版本由 `.python-version` 指定。也可使用 `uv run python -m micro_rsi doctor`。

环境检查执行设备上的张量计算和反向传播；它只能说明基础计算可用，模型的实际速度与算子兼容性仍需训练短跑验证。

## 目录结构

```text
AGENTS.md                    中文协作规范与学习目标
pyproject.toml / uv.lock      依赖声明与锁定版本
configs/pretrain/            预训练实验配置，当前为设计起点
src/micro_rsi/
  cli.py                     命令入口，当前支持 doctor
  device.py                  MPS / CUDA / CPU 设备选择
  data/                      数据读取、切分与批处理
  tokenization/              token 编解码与词表保存加载
  models/                    网络结构与前向传播
  training/                  优化、验证调度与训练状态保存
  inference/                 检查点加载与逐 token 采样
  evaluation/                指标与生成效果评估
data/raw/                    本地原始语料
data/processed/              本地处理后数据
artifacts/checkpoints/       本地模型检查点及相关元数据
runs/                        本地实验日志与配置快照
scripts/                     可重复的辅助操作
tests/                       后续功能的行为测试
docs/                        学习路线与实验说明
```

目前各业务子包只建立职责边界，不包含训练实现。数据、权重、日志和虚拟环境均被 Git 忽略。小型测试数据应放在 `tests/fixtures/`。

## 开发检查

```sh
uv run ruff check .
uv run ruff format --check .
```

已安装 pytest；后续添加功能测试后运行 `uv run pytest`。当前尚无测试用例，pytest 的“未收集到测试”退出码不作为项目验收命令。

训练起始设想见 `configs/pretrain/tiny_char.toml`，学习顺序与验收见 [学习路线](docs/roadmap.md)。配置中的步数与批大小是待测起点，不是训练速度或质量承诺。

协作中的可复用方法、实际踩坑和工程细节持续整理到 [文档导航](docs/README.md)；简明的长期协作规则保留在 [AGENTS.md](AGENTS.md)。
