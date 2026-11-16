"""Prepare and launch the explicitly requested FP16 RTX 4090 experiment."""

import argparse
import hashlib
import json
import shlex
import subprocess
import tarfile
import urllib.request
from pathlib import Path

from scripts.tools.comet_reference_rental import ROOT, api, secret

OUT = ROOT / "runs/nllb-10k-fp16-20261116"
REMOTE = "/workspace/nllb-10k-fp16"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def package():
    data = ROOT / "data/nllb_10k_fp16_20261116/bounded"
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest["counts"]["sft"] != 10000:
        raise ValueError("Exactly 10,000 SFT pairs are required")
    archive = OUT / "payload.tar.gz"
    entries = []
    for directory in ("src", "scripts", "configs", "tests"):
        for path in sorted((ROOT / directory).rglob("*")):
            if directory == "configs" and "local" in path.relative_to(ROOT).parts:
                continue
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix in {".py", ".yaml", ".json", ".sh"}
            ):
                entries.append((path, path.relative_to(ROOT).as_posix()))
    for name in ("setup.py", "requirements.txt", "pyproject.toml", ".project-root"):
        entries.append((ROOT / name, name))
    for stage in ("sft", "rl", "validation_eval", "test"):
        path = data / ("test_bounded.csv" if stage == "test" else stage + ".csv")
        if digest(path) != manifest["outputs"][stage]["sha256"]:
            raise ValueError("Changed prepared data: " + stage)
        entries.append((path, "data/run/" + path.name))
    if "test_sample" in manifest["outputs"]:
        path = data / "test_sample_1000.csv"
        if digest(path) != manifest["outputs"]["test_sample"]["sha256"]:
            raise ValueError("Changed prepared test sample")
        entries.append((path, "data/run/" + path.name))
        entries.append(
            (
                data / "test_sample_1000_manifest.json",
                "data/run/test_sample_1000_manifest.json",
            )
        )
    entries.append((data / "manifest.json", "data/run/manifest.json"))
    with tarfile.open(archive, "w:gz", compresslevel=1) as handle:
        for path, name in entries:
            handle.add(path, arcname=name, recursive=False)
    record = {
        "sha256": digest(archive),
        "bytes": archive.stat().st_size,
        "entries": len(entries),
        "credentials_included": False,
        "config": "configs/rental/nllb_10k_fp16_20261116.json",
    }
    (OUT / "payload_manifest.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


def connection(instance):
    row = api(f"instances/{instance}/")["instances"]
    ports = (row.get("ports") or {}).get("22/tcp", [])
    port = str(ports[0]["HostPort"]) if ports else str(row["ssh_port"])
    host = "root@" + (row["public_ipaddr"] if ports else row["ssh_host"])
    options = [
        "-i",
        str(Path.home() / ".ssh/id_ed25519"),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={OUT / 'known_hosts'}",
    ]
    (OUT / "connection.json").write_text(
        json.dumps({"instance": instance, "host": host, "port": port}, indent=2)
    )
    return ["ssh", *options, "-p", port, host], options, port, host


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=[
            "package",
            "preflight",
            "create",
            "status",
            "check",
            "upload",
            "launch",
        ],
    )
    parser.add_argument("--offer", type=int)
    parser.add_argument("--instance", type=int)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "package":
        package()
        return
    if args.mode == "preflight":
        judge = secret("OPENROUTER_API_KEY")
        if not judge:
            raise RuntimeError(
                "The explicitly designated OPENROUTER_API_KEY is unavailable"
            )
        config = json.loads(
            (ROOT / "configs/rental/nllb_10k_fp16_20261116.json").read_text()
        )
        request = urllib.request.Request(
            config["sac"]["base_url"] + "/chat/completions",
            data=json.dumps(
                {
                    "model": config["sac"]["model"],
                    "max_tokens": 32,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Return exactly Full match if these translations have equivalent meaning; otherwise No match.",
                        },
                        {
                            "role": "user",
                            "content": "English: The book has ten pages. Vietnamese: Cuốn sách có mười trang.",
                        },
                    ],
                }
            ).encode(),
            headers={
                "Authorization": "Bearer " + judge,
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            result = json.load(response)
        label = result["choices"][0]["message"]["content"].strip()
        if label != "Full match":
            raise RuntimeError(
                "Judge smoke test did not return the expected semantic label"
            )
        (OUT / "judge_preflight.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "model": config["sac"]["model"],
                    "label": label,
                    "usage": result.get("usage", {}),
                },
                indent=2,
            )
        )
        print(
            json.dumps({"judge_preflight": "passed", "model": config["sac"]["model"]})
        )
        return
    if args.mode == "create":
        # Read only the two explicitly designated credentials. No alias discovery.
        if not secret("HF_API_KEY") or not secret("OPENROUTER_API_KEY"):
            raise RuntimeError(
                "The designated HF_API_KEY and OPENROUTER_API_KEY must be available before rental creation"
            )
        if args.offer is None or (OUT / "rental.json").exists():
            raise ValueError("A new offer and no existing rental record are required")
        if not (OUT / "payload_manifest.json").exists():
            raise ValueError("Prepare and verify the payload before renting")
        if not (OUT / "judge_preflight.json").exists():
            raise ValueError("Verify judge access before renting")
        response = api(
            f"asks/{args.offer}/",
            "PUT",
            {
                "client_id": "me",
                "image": "vastai/pytorch:cuda-13.2.1-auto",
                "disk": 64,
                "runtype": "ssh_direct",
                "label": "nllb-10k-sft-10k-rl-fp16-20261116",
                "onstart": "touch /workspace/nllb-10k-ready",
            },
        )
        response = {
            k: v
            for k, v in response.items()
            if "key" not in k.lower() and "token" not in k.lower()
        }
        (OUT / "rental.json").write_text(json.dumps(response, indent=2))
        print(json.dumps(response))
        return
    if args.instance is None:
        raise ValueError("Specify the intended instance")
    ssh, options, port, host = connection(args.instance)
    if args.mode == "status":
        record = api(f"instances/{args.instance}/")["instances"]
        print(
            json.dumps(
                {
                    k: record.get(k)
                    for k in ["id", "actual_status", "gpu_name", "dph_total"]
                }
            )
        )
        return
    if args.mode == "check":
        code = (
            "import json; from pathlib import Path; "
            f"p=Path({REMOTE!r}); r=p/'logs/nllb-1p3b-10k-fp16-20261116'; "
            "names=['worker_status.json','sft/progress.json','sft/precision.json','sft/result.json',"
            "'rl/progress.json','rl/precision.json','rl/result.json']; "
            "d={n:json.loads((r/n).read_text()) for n in names if (r/n).exists()}; "
            "d['worker_alive']=Path('/proc/'+(p/'worker.pid').read_text().strip()).exists(); "
            "d['log_tail']=(p/'worker.log').read_text(errors='replace')[-2200:]; "
            "print(json.dumps(d))"
        )
        result = subprocess.run(
            [*ssh, "/venv/main/bin/python -c " + shlex.quote(code)],
            capture_output=True,
            text=True,
            check=True,
        )
        state = json.loads(result.stdout)
        (OUT / "remote_status.json").write_text(json.dumps(state, indent=2))
        print(json.dumps(state, indent=2))
        return
    if args.mode == "upload":
        archive = OUT / "payload.tar.gz"
        expected = json.loads((OUT / "payload_manifest.json").read_text())["sha256"]
        if digest(archive) != expected:
            raise RuntimeError("Payload changed after packaging")
        subprocess.run([*ssh, "mkdir -p " + REMOTE], check=True)
        subprocess.run(
            [
                "scp",
                *options,
                "-P",
                port,
                str(archive),
                host + ":" + REMOTE + "/payload.tar.gz",
            ],
            check=True,
        )
        verifier = (
            "import hashlib,tarfile; from pathlib import Path; "
            f"p=Path({REMOTE!r}); a=p/'payload.tar.gz'; "
            f"assert hashlib.file_digest(a.open('rb'),'sha256').hexdigest()=={expected!r}; "
            "tarfile.open(a).extractall(p,filter='data'); print('Payload SHA256 verified and extracted')"
        )
        subprocess.run(
            [*ssh, "/venv/main/bin/python -c " + shlex.quote(verifier)], check=True
        )
        return
    token, judge = secret("HF_API_KEY"), secret("OPENROUTER_API_KEY")
    if not token or not judge:
        raise RuntimeError("Missing designated credential")
    launcher = (
        "import json,os,sys,subprocess; from pathlib import Path; "
        f"p=Path({REMOTE!r}); old=p/'worker.pid'; "
        "assert not old.exists(), 'Worker already launched'; "
        "values=json.load(sys.stdin); env=os.environ.copy(); env.update(values); "
        "f=(p/'worker.log').open('ab',buffering=0); "
        "c=subprocess.Popen([sys.executable,'-u','-m','scripts.rental.nllb_10k_fp16_worker'],"
        "cwd=p,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); "
        "old.write_text(str(c.pid)); print('Detached training worker PID',c.pid)"
    )
    subprocess.run(
        [*ssh, "/venv/main/bin/python -c " + shlex.quote(launcher)],
        input=json.dumps({"HF_TOKEN": token, "OPENROUTER_API_KEY": judge}),
        text=True,
        check=True,
    )


if __name__ == "__main__":
    main()
