# micro-rsi

在本地从随机初始化训练中文自回归语言模型，学习它如何根据已有文本继续输出 token。第一阶段使用《呐喊》、字符级 tokenizer 和小型 GPT；后续逐步扩展语料、tokenizer 与训练方式。

当前已实现语料准备、Transformer、训练验证、检查点与独立生成。无需预训练权重。实验结果见 [第一轮实验记录](docs/stage-one-results.md)。

## 环境与快速开始

目标环境：Apple Silicon Mac、Python 3.12、uv；PyTorch 优先选择 MPS，也可使用 CPU。

```sh
uv sync
uv run micro-rsi doctor
```

在项目根目录运行以下命令。数据与权重保存在本地，不随 Git 提交。

```sh
# 初次准备：下载保留原件、提取 14 篇小说、转简体、切分并建立词表
uv run micro-rsi prepare --download

# 100 步短跑，使用单独目录，防止覆盖正式结果
uv run micro-rsi train --max-steps 100 \
  --run-dir runs/my-smoke --checkpoint-dir artifacts/checkpoints/my-smoke

# 从随机初始化开始正式训练，默认 2,000 步
uv run micro-rsi train --config configs/pretrain/tiny_char.toml

# 退出训练进程后，独立加载最佳检查点续写
uv run micro-rsi generate --checkpoint artifacts/checkpoints/nahan/best.pt \
  --prompt '夜里，我一个人走在路上，' --max-new-tokens 200 --temperature 0.8

# 新进程验收：固定窗口、完整验证区段与生成对比
uv run micro-rsi verify
```

原件已存在时不要加 `--download`；若需重新清洗，请用 `--output` 指定新的空目录。训练也会拒绝覆盖非空运行/检查点目录，重跑请用新的 `--run-dir` 和 `--checkpoint-dir`。已有本地验收结果可直接执行生成命令。

`generate --device cpu` 可在 CPU 推理。词表外前缀会明确报错，不会静默删除字符。模型只有短上下文和小语料经验，输出可能重复、混乱或复述原文，不具备对话与可靠写作能力。

## 目录结构

```text
configs/pretrain/            TOML 实验配置
src/micro_rsi/
  cli.py                     doctor / prepare / train / generate / verify 命令
  device.py                  MPS / CUDA / CPU 设备选择
  data/                      下载、清洗、切分与窗口采样
  tokenization/              字符编解码与词表保存加载
  models/                    因果注意力与 Transformer
  training/                  优化、验证、检查点与实验记录
  inference/                 独立加载与逐 token 采样
  evaluation/                固定窗口交叉熵评估
data/raw/                    原件（含来源许可）
data/processed/              清洗与划分结果、词表、数据清单
artifacts/checkpoints/       initial.pt / best.pt / last.pt
runs/                        配置、数据清单、环境、loss 与生成对比
scripts/                     辅助操作入口预留
tests/                      关键行为测试
docs/                       中文方案、学习说明与实验记录
```

检查点同时保存模型配置与 tokenizer，无需依赖训练对象或原始语料即可生成；也保存优化器与 PyTorch 随机状态。`train` 默认从随机初始化开始，指定 `--resume` 可恢复完整训练状态；`--max-steps` 表示目标累计步数。恢复要求数据、词表、模型、种子、批大小和设备类型一致，并使用新的空输出目录。

实验保留 `config.json`、`environment.json`、`data_manifest.json`、`metrics.jsonl`、`initial_samples.json`、`best_samples.json` 和 `summary.json`。固定种子便于同环境比较，跨设备、版本不保证逐位复现。

## 开发检查与文档

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

[文档导航](docs/README.md)汇总学习路线、训练方案与经验；[AGENTS.md](AGENTS.md)保留长期协作规则。详细原理见 [第一阶段实现导读](docs/stage-one-guide.md)。

## 第二轮：temperature 与增加语料

已完成《呐喊》+《彷徨》的同预算、共享词表对照，预测 loss 有改善，生成仍不连贯。完整设计、命令和结果见 [第二轮实验记录](docs/corpus-expansion-results.md)，来源见 [《彷徨》说明](docs/corpus-panghuang.md)。

```sh
uv run micro-rsi generate --checkpoint artifacts/checkpoints/nahan-panghuang/best.pt \
  --prompt '夜里，我一个人走在路上，' --temperature 0.8 --max-new-tokens 200
```

## 百万字符中文与可读性对照

本轮使用 Wikimedia 开放中文百科分片，清洗得到约500万训练字符，另外隔离整篇验证与测试集。方案、来源许可和下载地址见 [语料与实验方案](docs/wiki-readability-plan.md)，结果见 [可读性实验记录](docs/wiki-readability-results.md)。

```sh
uv run micro-rsi train --config configs/pretrain/wiki_zh_current.toml
uv run micro-rsi train --config configs/pretrain/wiki_zh_medium.toml
uv run python -m micro_rsi.evaluation.scale_comparison
uv run micro-rsi readability collect
# 对盲化样本逐条评分后汇总
uv run micro-rsi readability score
```

这些命令拒绝覆盖已有非空输出目录。查看已生成的本地样本无需重新训练。

## 当前可试用的续训结果

在相同 726 万参数模型与语料上，从 6,000 步继续到 14,000 步：完整验证 loss 从 4.2218 降至 3.9749，固定短续写局部通过数从 3/24 增至 7/24，仍未达到可读性门槛。诊断证据、局限与复现见 [可读性诊断与续训](docs/training-diagnosis.md)。

```sh
uv run micro-rsi generate \
  --checkpoint artifacts/checkpoints/wiki-zh-medium-continue/best.pt \
  --prompt '这座城市位于' --max-new-tokens 80 --temperature 0.7 --seed 42
```
