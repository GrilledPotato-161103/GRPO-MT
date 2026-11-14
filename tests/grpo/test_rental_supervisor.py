"""Training completion must not implicitly authorize deleting a rental."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.rental import supervisor as watch


class StopSupervisor(Exception):
    """End the synthetic budget-monitoring loop without real sleeping."""


@pytest.mark.parametrize("status", ["completed", "failed", "quota_stopped"])
@pytest.mark.parametrize("destroy_on_terminal", [False, True])
def test_terminal_cleanup_requires_explicit_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    destroy_on_terminal: bool,
) -> None:
    state = {
        "instance_id": 123,
        "key": str(tmp_path / "key"),
        "known_hosts": str(tmp_path / "known_hosts"),
        "host": "root@example.invalid",
        "port": 22,
        "graceful_deadline_unix": 1500,
        "hard_deadline_unix": 2000,
        "starting_credit": 100,
        "wandb_relay": False,
    }
    if destroy_on_terminal:
        state["destroy_on_terminal"] = True
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state))
    supervisor = watch.Supervisor(path)
    backup, destroy = Mock(), Mock()
    monkeypatch.setattr(supervisor, "backup", backup)
    monkeypatch.setattr(supervisor, "destroy", destroy)
    monkeypatch.setattr(supervisor, "cache_all", Mock())
    monkeypatch.setattr(
        supervisor, "remote", Mock(return_value=json.dumps({"status": status}))
    )
    monkeypatch.setattr(watch, "command", Mock(return_value='{"credit":100}'))
    monkeypatch.setattr(watch.time, "time", lambda: 1000)

    def stop(_: float) -> None:
        raise StopSupervisor

    monkeypatch.setattr(watch.time, "sleep", stop)
    if destroy_on_terminal:
        supervisor.run()
        destroy.assert_called_once_with(status)
    else:
        with pytest.raises(StopSupervisor):
            supervisor.run()
        destroy.assert_not_called()
        records = [
            json.loads(line)
            for line in (tmp_path / "supervisor.jsonl").read_text().splitlines()
        ]
        assert records[-1]["instance_preserved"]
        assert records[-1]["budget_guard_only"]
    backup.assert_called_once()


def test_instance_cleanup_preserves_user_owned_ssh_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = tmp_path / "operator_key"
    identity.write_text("existing operator identity")
    public = identity.with_suffix(".pub")
    public.write_text("existing operator public key")
    access = tmp_path / "access"
    access.mkdir()
    path = access / "state.json"
    path.write_text(
        json.dumps(
            {
                "instance_id": 123,
                "key": str(identity),
                "known_hosts": str(access / "known_hosts"),
                "host": "root@example.invalid",
                "port": 22,
                "ssh_key_id": 456,
                "ssh_key_managed": False,
            }
        )
    )
    cli = Mock(side_effect=["", '{"instances": []}'])
    monkeypatch.setattr(watch, "command", cli)

    watch.Supervisor(path).destroy("explicit test cleanup")

    assert identity.read_text() == "existing operator identity"
    assert public.read_text() == "existing operator public key"
    assert len(cli.call_args_list) == 2
    assert not any("ssh-key" in call.args[0] for call in cli.call_args_list)
