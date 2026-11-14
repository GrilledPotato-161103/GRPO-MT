"""Atomically archive run reports and checkpoints, excluding W&B internal files."""

import argparse
import tarfile
from pathlib import Path


def snapshot(root: Path, output: Path) -> None:
    temporary = output.with_suffix(".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        for path in sorted(root.rglob("*")):
            if (
                path.is_file()
                and "wandb" not in path.relative_to(root).parts
                and path.suffix != ".tmp"
            ):
                archive.add(path, arcname=path.relative_to(root).as_posix())
        for name in (
            "train_driver.log",
            "install_training.log",
            "pip_check_training.log",
        ):
            system_log = Path("/workspace") / name
            if system_log.is_file():
                archive.add(system_log, arcname=f"system_logs/{name}")
    temporary.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    snapshot(arguments.root, arguments.output)
