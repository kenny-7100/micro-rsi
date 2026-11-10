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

检查点同时保存模型配置与 tokenizer，无需依赖训练对象或原始语料即可生成；也保存优化器与 PyTorch 随机状态，供以后扩展恢复训练。当前未提供恢复训练命令，`train` 总是从随机初始化开始。

实验保留 `config.json`、`environment.json`、`data_manifest.json`、`metrics.jsonl`、`initial_samples.json`、`best_samples.json` 和 `summary.json`。固定种子便于同环境比较，跨设备、版本不保证逐位复现。

## 开发检查与文档

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

[文档导航](docs/README.md)汇总学习路线、训练方案与经验；[AGENTS.md](AGENTS.md)保留长期协作规则。详细原理见 [第一阶段实现导读](docs/stage-one-guide.md)。
