"""Start training in tmux and pass the calling shell environment through a FIFO."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    args = sys.argv[1:]
    if args[:1] == ["--worker"]:
        pipe, log = Path(args[1]), Path(args[2])
        with pipe.open() as stream:
            environment = json.load(stream)
        pipe.unlink(missing_ok=True)
        command = [sys.executable, "-u", "-m", "scripts.main.train", *args[3:]]
        shell = f"{shlex.join(command)} 2>&1 | tee -a {shlex.quote(str(log))}"
        os.execvpe("bash", ["bash", "-o", "pipefail", "-c", shell], environment)
    if sys.platform == "win32" or not shutil.which("tmux"):
        print("Run this command on a Linux host with tmux installed.", file=sys.stderr)
        return 2
    session = "mt-train"
    if (
        subprocess.run(
            ["tmux", "has-session", "-t", session], capture_output=True, check=False
        ).returncode
        == 0
    ):
        print(
            "Session already exists. Inspect it with: tmux attach -t mt-train",
            file=sys.stderr,
        )
        return 2
    folder = ROOT / "logs/sessions"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    pipe, log = folder / f".{stem}.pipe", folder / f"{stem}.log"
    os.mkfifo(pipe, 0o600)
    command = [
        sys.executable,
        "-u",
        "-m",
        "scripts.rental.session",
        "--worker",
        str(pipe),
        str(log),
        *args,
    ]
    try:
        subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                session,
                "-c",
                str(ROOT),
                shlex.join(command),
            ],
            check=True,
        )
        subprocess.run(
            ["tmux", "set-option", "-t", session, "remain-on-exit", "on"], check=True
        )
        deadline = time.monotonic() + 15
        while True:
            try:
                descriptor = os.open(pipe, os.O_WRONLY | os.O_NONBLOCK)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "tmux worker did not open the environment FIFO"
                    ) from None
                time.sleep(0.1)
        os.set_blocking(descriptor, True)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(dict(os.environ), stream)
        print(f"Started tmux {session}; log: {log.relative_to(ROOT)}")
        print("Monitor: tmux attach -t mt-train  |  Detach: Ctrl-b, d")
        return 0
    finally:
        pipe.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
