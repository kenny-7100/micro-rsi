"""覆盖数据泄漏、因果性、优化以及独立检查点等关键行为。"""

import json
import subprocess
import sys
from dataclasses import asdict

import pytest
import torch
from torch.nn import functional as F

from micro_rsi.data.windows import TextWindows
from micro_rsi.evaluation.metrics import evaluate
from micro_rsi.inference.generation import generate, load_model
from micro_rsi.models.transformer import ModelConfig, TransformerLM
from micro_rsi.tokenization.character import CharacterTokenizer
from micro_rsi.training.runner import load_config


def tiny_model():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    return TransformerLM(
        ModelConfig(
            vocab_size=5,
            context_length=8,
            embedding_dim=16,
            num_heads=2,
            num_layers=1,
            ff_dim=32,
            dropout=0.0,
        )
    )


def test_tokenizer_roundtrip_unknown_and_persistence(tmp_path):
    tokenizer = CharacterTokenizer.fit(["中文，\n你好"])
    assert tokenizer.decode(tokenizer.encode("你好，中文\n")) == "你好，中文\n"
    assert tokenizer.encode("陌") == [0]
    tokenizer.save(tmp_path / "tokenizer.json")
    restored = CharacterTokenizer.load(tmp_path / "tokenizer.json")
    assert restored.to_dict() == tokenizer.to_dict()
    assert restored.decode([0]) == "<unk>"


def test_windows_alignment_boundaries_and_last_window():
    tokenizer = CharacterTokenizer.fit(["甲乙丙丁戊己庚辛"])
    windows = TextWindows(["甲乙丙丁", "戊己庚辛"], tokenizer, 3)
    x, y = windows.sample(50, torch.Generator().manual_seed(42))
    pairs = {
        (tokenizer.decode(a.tolist()), tokenizer.decode(b.tolist()))
        for a, b in zip(x, y, strict=True)
    }
    assert pairs == {("甲乙丙", "乙丙丁"), ("戊己庚", "己庚辛")}
    assert windows.total == 2
    with pytest.raises(ValueError):
        TextWindows(["短"], tokenizer, 3)


def test_future_tokens_cannot_change_past_logits():
    model = tiny_model().eval()
    a = torch.tensor([[1, 2, 3, 4, 1, 2]])
    b = torch.tensor([[1, 2, 3, 2, 2, 2]])
    torch.testing.assert_close(model(a)[:, :3], model(b)[:, :3], rtol=0, atol=0)
    assert not torch.allclose(model(a)[:, 3:], model(b)[:, 3:])


def test_optimizer_can_overfit_small_sequence():
    model = tiny_model()
    x, y = torch.tensor([[1, 2, 3, 4]]), torch.tensor([[2, 3, 4, 1]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    before = F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    for _ in range(40):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        loss.backward()
        optimizer.step()
    after = F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
    assert after < before * 0.1


def test_evaluation_is_repeatable_and_preserves_mode():
    model = tiny_model().train()
    tokenizer = CharacterTokenizer(list("甲乙丙丁"))
    windows = TextWindows(["甲乙丙丁" * 5], tokenizer, 4)
    before = torch.get_rng_state().clone()
    a = evaluate(model, windows, 2, 3, torch.device("cpu"))
    b = evaluate(model, windows, 2, 3, torch.device("cpu"))
    assert a == b
    assert model.training
    assert torch.equal(before, torch.get_rng_state())


def test_checkpoint_roundtrip_and_fresh_process_generation(tmp_path):
    model = tiny_model().eval()
    tokenizer = CharacterTokenizer(list("甲乙丙丁"))
    path = tmp_path / "model.pt"
    torch.save(
        {
            "format_version": 1,
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "tokenizer": tokenizer.to_dict(),
        },
        path,
    )
    restored, restored_tokenizer, _ = load_model(path, "cpu")
    ids = torch.tensor([[1, 2, 3]])
    torch.testing.assert_close(model(ids), restored(ids), rtol=0, atol=0)
    expected = generate(model, tokenizer, "甲乙", max_new_tokens=12, seed=9)
    assert len(expected) == 14  # 超过 context_length，测试滑动上下文。
    assert generate(restored, restored_tokenizer, "甲乙", 12, seed=9) == expected
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "micro_rsi",
            "generate",
            "--checkpoint",
            str(path),
            "--prompt",
            "甲乙",
            "--max-new-tokens",
            "12",
            "--seed",
            "9",
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == expected
    with pytest.raises(ValueError, match="词表外"):
        generate(model, tokenizer, "未知")
    with pytest.raises(ValueError):
        generate(model, tokenizer, "")


def test_config_rejects_unknown_fields(tmp_path):
    from pathlib import Path

    text = Path("configs/pretrain/tiny_char.toml").read_text()
    path = tmp_path / "config.toml"
    path.write_text(text.replace("seed = 42", "seed = 42\nextra = 1"))
    with pytest.raises(ValueError, match="顶层"):
        load_config(path)


def test_corpus_cleaning_and_split_without_network(tmp_path):
    from micro_rsi.data.corpus import TITLES, extract_stories, prepare

    sections = [
        f"\n{title}\n" + "這是一個測試。\n" * 100 + "\n□注釋\n不要訓練的注解" for title in TITLES
    ]
    raw = (
        "*** START OF THE PROJECT GUTENBERG EBOOK 吶喊 ***\n序言\n"
        + "".join(sections)
        + "\n*** END OF THE PROJECT GUTENBERG EBOOK 吶喊 ***"
    )
    stories = extract_stories(raw)
    assert len(stories) == 14
    assert "这是一个测试。" in stories[0]["text"]
    assert "注解" not in "".join(d["text"] for d in stories)
    raw_path = tmp_path / "raw.txt"
    raw_path.write_text(raw)
    output = tmp_path / "processed"
    manifest = prepare(raw_path, output)
    training = json.loads((output / "train.json").read_text())["documents"]
    validation = json.loads((output / "validation.json").read_text())["documents"]
    for original, tr, val in zip(stories, training, validation, strict=True):
        assert tr["text"] + val["text"] == original["text"]
    assert manifest["train_chars"] + manifest["validation_chars"] == sum(
        len(d["text"]) for d in stories
    )
    with pytest.raises(FileExistsError):
        prepare(raw_path, output)


def test_short_document_padding_does_not_create_targets():
    tokenizer = CharacterTokenizer.fit(["甲乙丙"])
    windows = TextWindows(["甲乙丙"], tokenizer, 8)
    x, y = windows.sample(1, torch.Generator().manual_seed(1))
    assert x[0, :2].tolist() == tokenizer.encode("甲乙")
    assert y[0, :2].tolist() == tokenizer.encode("乙丙")
    assert y[0, 2:].tolist() == [-100] * 6
    assert windows.total == 1


def test_training_entrypoint_checkpoint_and_artifacts(tmp_path):
    from pathlib import Path

    from micro_rsi.data.corpus import TITLES, prepare
    from micro_rsi.training.runner import train

    raw = (
        "*** START OF THE PROJECT GUTENBERG EBOOK 吶喊 ***\n"
        + "".join(f"\n{title}\n" + "夜里，我一个人走在路上，这是测试。" * 30 for title in TITLES)
        + "\n*** END OF THE PROJECT GUTENBERG EBOOK 吶喊 ***"
    )
    source = tmp_path / "raw.txt"
    source.write_text(raw)
    data = tmp_path / "data"
    prepare(source, data)
    text = Path("configs/pretrain/tiny_char.toml").read_text()
    replacements = {
        'directory = "data/processed/nahan"': f'directory = "{data}"',
        "embedding_dim = 128": "embedding_dim = 16",
        "num_layers = 4": "num_layers = 1",
        "ff_dim = 512": "ff_dim = 32",
        "context_length = 256": "context_length = 8",
        "eval_batches = 10": "eval_batches = 1",
        "max_new_tokens = 200": "max_new_tokens = 4",
    }
    for before, after in replacements.items():
        text = text.replace(before, after)
    config = tmp_path / "config.toml"
    config.write_text(text)
    summary = train(
        config,
        max_steps=2,
        run_dir=tmp_path / "run",
        checkpoint_dir=tmp_path / "checkpoints",
        device="cpu",
    )
    assert summary["steps"] == 2
    assert (tmp_path / "run" / "summary.json").exists()
    for name in ["initial.pt", "best.pt", "last.pt"]:
        model, tokenizer, checkpoint = load_model(tmp_path / "checkpoints" / name, "cpu")
        assert checkpoint["optimizer_state"] is not None
        assert len(generate(model, tokenizer, "夜里", max_new_tokens=3)) == 5
    with pytest.raises(FileExistsError):
        train(
            config,
            max_steps=1,
            run_dir=tmp_path / "run",
            checkpoint_dir=tmp_path / "checkpoints",
            device="cpu",
        )


def test_full_validation_scores_each_target_once():
    from micro_rsi.evaluation.acceptance import score_documents

    model = tiny_model()
    tokenizer = CharacterTokenizer(list("甲乙丙丁"))
    docs = [{"text": "甲乙丙丁" * 6}, {"text": "甲乙"}]
    result = score_documents(model, tokenizer, docs, "cpu")
    assert result["targets"] == 24
    assert result["known_targets"] == 24
    assert result["loss"] == pytest.approx(result["known_target_loss"])
