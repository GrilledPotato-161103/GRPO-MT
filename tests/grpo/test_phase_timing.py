"""Timing semantics, synchronized propagation measurements, and throughput units."""

import json
from pathlib import Path

import pytest
import torch

from src.utils.credentials import environment_secret
from src.utils.phase_timing import PhaseTiming


def test_phase_timer_records_work_rates_and_failed_phases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = iter([1.0, 3.0, 4.0, 7.0])
    monkeypatch.setattr("src.utils.phase_timing.time.perf_counter", lambda: next(clock))
    recorder = PhaseTiming(enabled=True)
    recorder.configure(tmp_path)
    with recorder.measure(
        "solver/generation", torch.device("cpu"), input_tokens=8, samples=2
    ) as work:
        work["output_tokens"] = 4
    with pytest.raises(ValueError), recorder.measure("failed", torch.device("cpu")):
        raise ValueError("synthetic failure")
    recorder.write()
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["phases"]["solver/generation"]["output_tokens_per_second"] == 2
    assert report["phases"]["solver/generation"]["mean_seconds"] == 2
    assert "failed" not in report["phases"]
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 2


def test_cuda_measurement_synchronizes_both_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(device))
    recorder = PhaseTiming(enabled=True)
    with recorder.measure("current/backward", torch.device("cuda:0")):
        pass
    assert len(calls) == 2
    disabled = PhaseTiming()
    with disabled.measure("unused", torch.device("cuda:0")):
        pass
    assert not disabled.events and len(calls) == 2


def test_named_credentials_do_not_require_a_new_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UNIT_TEST_CREDENTIAL", "local-value")
    assert environment_secret("UNIT_TEST_CREDENTIAL") == "local-value"
    monkeypatch.delenv("UNIT_TEST_CREDENTIAL")
    assert environment_secret("UNIT_TEST_CREDENTIAL") is None
