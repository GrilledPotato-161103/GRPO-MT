"""Preview or remove disposable caches without touching datasets or results."""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def candidates(root: Path) -> list[Path]:
    """Inspect only source directories and known root-level generated files."""
    root = root.resolve()
    found = [root / name for name in CACHE_NAMES if (root / name).exists()]
    for directory in ("src", "scripts", "tests", "configs"):
        for path in (root / directory).rglob("__pycache__"):
            if not path.is_symlink():
                found.append(path)
    found.extend(
        root / name for name in (".coverage", "coverage.xml") if (root / name).is_file()
    )
    return sorted(set(found))


def clean(root: Path, apply: bool = False) -> list[Path]:
    """Validate every deletion target stays in the project before removing it."""
    root = root.resolve()
    targets = candidates(root)
    for path in targets:
        resolved = path.resolve()
        if (
            path.is_symlink()
            or not resolved.is_relative_to(root)
            or resolved == root
            or path.name not in CACHE_NAMES | {".coverage", "coverage.xml"}
        ):
            raise ValueError(f"Unsafe cache target: {path}")
    if apply:
        for path in targets:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Delete listed caches")
    args = parser.parse_args()
    targets = clean(ROOT, args.apply)
    for path in targets:
        print(path.relative_to(ROOT))
    print(f"{'Removed' if args.apply else 'Would remove'} {len(targets)} caches")


if __name__ == "__main__":
    main()
