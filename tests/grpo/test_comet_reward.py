"""Reference-free COMETKiwi interface and normalized composite rewards."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src.models.components.comet_reward import CometScorer
from src.models.components.translation_reward import TranslationReward


class TinyComet(torch.nn.Module):
    """Exercise actual frozen torch inference without downloading a pretrained model."""

    def __init__(self, unified: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.rows: list[dict[str, str]] = []
        self.unified = unified

    def prepare_sample(self, rows: list[dict[str, str]], stage: str) -> Any:
        assert stage == "predict" and all("ref" not in row for row in rows)
        self.rows.extend(rows)
        inputs = {
            "mt_input_ids": torch.tensor([[len(row["mt"])] for row in rows]),
            "mt_attention_mask": torch.ones(len(rows), 1, dtype=torch.long),
        }
        return (inputs,) if self.unified else inputs

    def forward(
        self, mt_input_ids: torch.Tensor, mt_attention_mask: torch.Tensor
    ) -> Any:
        assert not torch.is_grad_enabled() and not self.training
        assert not torch.is_autocast_enabled(self.weight.device.type)
        return SimpleNamespace(score=mt_input_ids[:, 0].float() * self.weight / 10)


@pytest.mark.parametrize("unified", [False, True])
def test_reference_free_comet_is_frozen_ordered_and_batched(unified: bool) -> None:
    model = TinyComet(unified)
    scorer = CometScorer(model=model, batch_size=2)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert scorer(["a", "b", "c"], ["one", "four", "x"]) == pytest.approx(
            [0.3, 0.4, 0.1]
        )
    assert [row["src"] for row in model.rows] == ["a", "b", "c"]
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in model.parameters()
    )
    assert scorer.last_input_tokens == 3
    assert set(scorer.last_timings) == {"load", "tokenization", "forward"}
    assert scorer([], []) == []
    with pytest.raises(ValueError):
        scorer(["source"], [])
    with pytest.raises(ValueError):
        CometScorer(batch_size=0)


def test_comet_bleu_reward_clips_only_reward_and_preserves_raw_metrics() -> None:
    reward = TranslationReward(
        chrf_weight=0,
        bleu_weight=0.3,
        comet_weight=0.7,
        comet_scorer=lambda s, p, r: [1.4, -0.2, 0.8],
    )
    values = reward.components(
        ["This is the correct text.", "wrong", ""],
        ["This is the correct text."] * 3,
        sources=["source"] * 3,
    )
    assert values["comet"] == [1.4, -0.2, 0.8]
    assert values["reward"][0] == pytest.approx(1)
    assert values["reward"][2] == 0
    assert all(0 <= score <= 1 for score in values["reward"])
    with pytest.raises(ValueError, match="source"):
        reward(["x"], ["ref"])
    for output in ([float("nan")], []):
        bad = TranslationReward(
            chrf_weight=0,
            comet_weight=1,
            comet_scorer=lambda s, p, r, output=output: output,
        )
        with pytest.raises(ValueError):
            bad(["x"], ["ref"], sources=["src"])


def test_huggingface_token_is_forwarded_without_becoming_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    comet = pytest.importorskip("comet")
    import huggingface_hub

    snapshot = tmp_path / "snapshot"
    (snapshot / "checkpoints").mkdir(parents=True)
    (snapshot / "checkpoints/model.ckpt").touch()
    called: dict[str, Any] = {}

    def download(repo_id: str, **kwargs: Any) -> str:
        called.update(repo_id=repo_id, **kwargs)
        return str(snapshot)

    monkeypatch.setattr(
        "src.models.components.comet_reward.environment_secret",
        lambda name: "test-secret",
    )
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(comet, "load_from_checkpoint", lambda checkpoint: TinyComet())
    scorer = CometScorer(cache_dir=str(tmp_path / "cache"))
    assert scorer(["source"], ["good"])[0] == pytest.approx(0.4)
    assert (
        called["repo_id"] == "Unbabel/wmt22-cometkiwi-da"
        and called["token"] == "test-secret"
    )
    assert "test-secret" not in repr(vars(scorer))


def test_missing_credentials_and_gated_access_fail_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("comet")
    import huggingface_hub

    monkeypatch.setattr(
        "src.models.components.comet_reward.environment_secret", lambda name: None
    )
    with pytest.raises(ValueError, match="HUGGING_FACE_API_KEY"):
        CometScorer(cache_dir=str(tmp_path))(["src"], ["translation"])
    monkeypatch.setattr(
        "src.models.components.comet_reward.environment_secret",
        lambda name: "private-token",
    )

    def denied(*args: Any, **kwargs: Any) -> str:
        raise OSError("private-token")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", denied)
    with pytest.raises(RuntimeError) as failure:
        CometScorer(cache_dir=str(tmp_path))(["src"], ["translation"])
    assert "private-token" not in str(failure.value)
