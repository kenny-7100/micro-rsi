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


def test_explicit_attention_matches_pytorch_causal_reference_and_gradients():
    # 用独立算子核对手写注意力的缩放、分头、遮罩及反向传播。
    model = tiny_model().eval()
    attention = model.blocks[0].attention
    x = torch.randn(2, 6, 16, requires_grad=True)
    actual = attention(x)
    q, k, v = attention.qkv(x).chunk(3, dim=-1)
    q, k, v = [t.view(2, 6, 2, 8).transpose(1, 2) for t in (q, k, v)]
    reference = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    reference = attention.projection(reference.transpose(1, 2).contiguous().view(2, 6, 16))
    torch.testing.assert_close(actual, reference, atol=1e-7, rtol=1e-5)
    probe = torch.randn_like(actual)
    parameters = (x, attention.qkv.weight, attention.projection.weight)
    actual_grad = torch.autograd.grad((actual * probe).sum(), parameters)
    reference_grad = torch.autograd.grad((reference * probe).sum(), parameters)
    for a, b in zip(actual_grad, reference_grad, strict=True):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)


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


def test_wikisource_paragraphs_exclude_navigation_license_and_notes():
    from micro_rsi.data.panghuang import BodyParagraphs, extract_page

    html = (
        '<div><div id="headerContainer"><p>导航</p></div>'
        "<p>正文<sup>注号</sup>内容<br>继续。</p>"
        '<div class="licenseContainer"><p>许可</p></div>'
        '<div class="reflist"><p>注释</p></div></div>'
    )
    parser = BodyParagraphs()
    parser.feed(html)
    assert parser.paragraphs == ["正文内容继续。"]
    data = {
        "parse": {
            "title": "測試",
            "text": {"*": html.replace("正文<sup>注号</sup>内容<br>继续。", "這是正文。" * 120)},
        }
    }
    doc = extract_page(data)
    assert doc["title"] == "测试"
    assert doc["text"].startswith("这是正文。\n")
    assert "许可" not in doc["text"]


def test_paired_data_shares_vocab_and_preserves_original_splits(tmp_path, monkeypatch):
    from micro_rsi.data import panghuang
    from micro_rsi.data.corpus import write_json

    base = tmp_path / "base"
    base.mkdir()
    for split in ["train", "validation"]:
        write_json(base / f"{split}.json", {"documents": [{"title": "原篇", "text": "甲乙"}]})
    write_json(base / "manifest.json", {"source": "测试"})
    monkeypatch.setattr(panghuang, "PAGES", ["測試"])

    def fake_fetch(page, path):
        data = {
            "parse": {
                "title": page,
                "revid": 123,
                "text": {"*": "<p>" + "丙丁正文。" * 120 + "</p>"},
            }
        }
        write_json(path, data)
        return data

    monkeypatch.setattr(panghuang, "fetch_page", fake_fetch)
    control, mixed = tmp_path / "control", tmp_path / "mixed"
    panghuang.prepare_pair(tmp_path / "raw", base, mixed, control)
    assert (control / "tokenizer.json").read_bytes() == (mixed / "tokenizer.json").read_bytes()
    for split in ["train", "validation"]:
        original = json.loads((base / f"{split}.json").read_text())["documents"]
        assert json.loads((control / f"{split}.json").read_text())["documents"] == original
        assert json.loads((mixed / f"{split}.json").read_text())["documents"][:1] == original
    with pytest.raises(FileExistsError):
        panghuang.prepare_pair(tmp_path / "raw", base, mixed, control)


def test_repetition_metric_distinguishes_loop_from_unique_text():
    from micro_rsi.evaluation.comparison import repetition

    assert repetition("甲乙丙丁戊己庚辛") == 0.0
    assert repetition("甲乙" * 30) > 0.9
    assert repetition("") == 0.0


def test_wikipedia_filter_and_article_level_split():
    from opencc import OpenCC

    from micro_rsi.data.wikipedia import clean_article, split_for

    source = (
        "这是一个中文段落，其中包含足够多的字符，并且至少有两个完整句子。"
        "继续补充一些中文内容，让它通过最低长度限制。这是第二个完整句子。" * 2 + "\n"
        "http://example.test/reference。这个网址段落不应保留下来。"
        "\n项目\n" + "A foreign-only line without enough Chinese punctuation."
    )
    result = clean_article(source, OpenCC("t2s"))
    assert result.startswith("这是一个中文段落")
    assert "http" not in result
    assert split_for("stable-article-id") == split_for("stable-article-id")
    assert split_for("stable-article-id") in {"train", "validation", "test"}


def test_wikipedia_preserves_word_spaces_and_rejects_missing_fields():
    from opencc import OpenCC

    from micro_rsi.data.wikipedia import clean_article

    converter = OpenCC("t2s")
    normal = (
        "这是关于科学研究的一段说明，英文名称为 New York，保留正常词间空格。"
        "研究开始于2020年（第一阶段），这里还保留合法的数字和括注。" * 2
    )
    cleaned = clean_article(normal, converter)
    assert "New York" in cleaned and "NewYork" not in cleaned
    assert "2020年（第一阶段）" in cleaned
    for empty in ["（）", "( )", "“”", "《》"]:
        assert clean_article(normal + empty, converter) == ""


def test_readability_score_requires_all_blind_samples_to_be_rated(tmp_path):
    from micro_rsi.data.corpus import write_json
    from micro_rsi.evaluation.readability import score

    path = tmp_path / "review.json"
    write_json(
        path,
        {
            "model_map": {"S01": "current", "S02": "medium"},
            "samples": [
                {
                    "blind_id": "S01",
                    "model": "current",
                    "scores": {"words": 1, "sentences": 1, "continuity": 0, "repetition": 2},
                },
                {
                    "blind_id": "S02",
                    "model": "medium",
                    "scores": {"words": 2, "sentences": 2, "continuity": 2, "repetition": 2},
                },
            ],
        },
    )
    result = score(path)
    assert result["medium"]["pass_count"] == 1
    assert result["current"]["pass_rate"] == 1


def test_readability_rejects_missing_scores_and_wrong_identity(tmp_path):
    from micro_rsi.data.corpus import write_json
    from micro_rsi.evaluation.readability import score

    path = tmp_path / "review.json"
    item = {
        "blind_id": "S01",
        "model": "current",
        "scores": {"words": None, "sentences": 1, "continuity": 1, "repetition": 2},
    }
    write_json(path, {"model_map": {"S01": "current"}, "samples": [item]})
    with pytest.raises(ValueError, match="填写"):
        score(path)
    write_json(path, {"model_map": {"S01": "medium"}, "samples": [item]})
    with pytest.raises(ValueError, match="映射"):
        score(path)


def test_model_checkpoint_with_old_config_still_loads(tmp_path):
    model = tiny_model()
    config = asdict(model.config)
    config.pop("init_std")
    tokenizer = CharacterTokenizer(list("甲乙丙丁"))
    path = tmp_path / "old.pt"
    torch.save(
        {
            "format_version": 1,
            "model_config": config,
            "model_state": model.state_dict(),
            "tokenizer": tokenizer.to_dict(),
        },
        path,
    )
    restored, _, _ = load_model(path, "cpu")
    model.eval()
    x = torch.tensor([[1, 2, 3]])
    torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)


def test_resume_matches_uninterrupted_training_and_rejects_changed_data(tmp_path):
    from pathlib import Path

    from micro_rsi.data.corpus import write_json
    from micro_rsi.training.runner import train

    data = tmp_path / "data"
    data.mkdir()
    text = "这是一个用于恢复训练的测试文本。模型必须得到相同的更新结果。" * 8
    tokenizer = CharacterTokenizer.fit([text])
    tokenizer.save(data / "tokenizer.json")
    for split in ["train", "validation"]:
        write_json(data / f"{split}.json", {"documents": [{"text": text}]})
    write_json(data / "manifest.json", {"purpose": "resume-test"})
    config_text = Path("configs/pretrain/tiny_char.toml").read_text()
    for before, after in {
        'directory = "data/processed/nahan"': f'directory = "{data}"',
        "embedding_dim = 128": "embedding_dim = 16",
        "num_layers = 4": "num_layers = 1",
        "ff_dim = 512": "ff_dim = 32",
        "context_length = 256": "context_length = 8",
        "batch_size = 16": "batch_size = 2",
        "eval_batches = 10": "eval_batches = 1",
        "max_new_tokens = 200": "max_new_tokens = 4",
    }.items():
        config_text = config_text.replace(before, after)
    config = tmp_path / "config.toml"
    config.write_text(config_text)

    def run(name, steps, resume=None):
        return train(
            config,
            max_steps=steps,
            run_dir=tmp_path / f"{name}-run",
            checkpoint_dir=tmp_path / f"{name}-ckpt",
            device="cpu",
            resume=resume,
        )

    run("full", 6)
    run("part", 3)
    result = run("resumed", 6, tmp_path / "part-ckpt/last.pt")
    assert result["start_step"] == 3 and result["new_steps"] == 3
    full = torch.load(tmp_path / "full-ckpt/last.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed-ckpt/last.pt", weights_only=True)
    for name in full["model_state"]:
        torch.testing.assert_close(
            full["model_state"][name], resumed["model_state"][name], rtol=0, atol=0
        )
    assert torch.equal(full["sample_rng_state"], resumed["sample_rng_state"])
    assert torch.equal(full["torch_rng_state"], resumed["torch_rng_state"])
    write_json(data / "train.json", {"documents": [{"text": text + "变化"}]})
    with pytest.raises(ValueError, match="指纹"):
        run("changed", 9, tmp_path / "part-ckpt/last.pt")
    assert not (tmp_path / "changed-run").exists()


def test_cosine_learning_rate_uses_absolute_steps():
    from micro_rsi.training.runner import learning_rate_at

    settings = {
        "learning_rate": 0.0003,
        "min_learning_rate": 0.00003,
        "schedule_start": 6000,
        "max_steps": 10000,
    }
    assert learning_rate_at(6000, settings) == pytest.approx(0.0003)
    assert learning_rate_at(8000, settings) == pytest.approx(0.000165)
    assert learning_rate_at(10000, settings) == pytest.approx(0.00003)
