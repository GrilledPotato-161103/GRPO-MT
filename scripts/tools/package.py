"""Bundle the GPU test without user datasets, credentials, or Git history."""

import argparse
import tarfile
from pathlib import Path


def package(output: Path) -> None:
    """Create a portable source archive with synthetic test data only."""
    root = Path(__file__).resolve().parents[2]
    output.parent.mkdir(parents=True, exist_ok=True)
    files = [
        root / name
        for name in (
            "requirements.txt",
            "setup.py",
            "pyproject.toml",
            ".project-root",
            "README.md",
            ".env.example",
            "environment.yaml",
        )
    ]
    for directory in ("src", "configs", "tests/grpo", "scripts", "docs"):
        files.extend(
            p
            for p in (root / directory).rglob("*")
            if p.is_file()
            and not p.is_symlink()
            and "__pycache__" not in p.parts
            and p.suffix != ".pyc"
        )
    files.extend((root / "data").glob("example_*.jsonl"))
    with tarfile.open(output, "w:gz") as archive:
        for file in sorted(files):
            archive.add(file, arcname=file.relative_to(root).as_posix())
    print(output.resolve())


def main() -> None:
    """Parse the bundle destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("logs/grpo-gpu-test.tar.gz")
    )
    args = parser.parse_args()
    package(args.output)


if __name__ == "__main__":
    main()
