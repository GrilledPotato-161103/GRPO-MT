"""Deploy the W&B/test-panel restart to the existing, explicitly authorized rental."""

import argparse
import json
import shlex
import subprocess
import tarfile

from scripts.tools.comet_reference_rental import secret
from scripts.tools.manage_nllb_10k_rental import OUT, REMOTE, ROOT, connection, digest

INSTANCE = 60824538
CHECKPOINT_SHA = "4311bf8a9c05847bb65af17db155c1e647e6c0a9d039488728148fe38f89c3bc"


def remote(ssh, code, **kwargs):
    return subprocess.run(
        [*ssh, "/venv/main/bin/python -c " + shlex.quote(code)],
        check=True,
        text=True,
        **kwargs,
    )


def deploy():
    backup = OUT / "restart_backup/sft_step221.ckpt"
    if digest(backup) != CHECKPOINT_SHA:
        raise RuntimeError("The local preserved checkpoint must be verified first")
    files = [
        (ROOT / name, name)
        for name in [
            "scripts/rental/nllb_10k_fp16_worker.py",
            "scripts/data/sample_test_1000.py",
            "src/data/translation_datamodule.py",
            "src/data/components/resume_sampler.py",
            "configs/rental/nllb_10k_fp16_20261116.json",
            "tests/grpo/test_wandb_restart.py",
        ]
    ]
    data = ROOT / "data/nllb_10k_fp16_20261116/bounded"
    for name in [
        "manifest.json",
        "test_sample_1000.csv",
        "test_sample_1000_manifest.json",
    ]:
        files.append((data / name, "data/run/" + name))
    panel = json.loads((data / "test_sample_1000_manifest.json").read_text())
    if (
        panel["pairs"] != 1000
        or digest(data / "test_sample_1000.csv") != panel["sha256"]
    ):
        raise RuntimeError("The local test panel has changed")
    archive = OUT / "wandb_restart_patch.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for path, target in files:
            handle.add(path, arcname=target, recursive=False)
    record = {
        "instance": INSTANCE,
        "sha256": digest(archive),
        "bytes": archive.stat().st_size,
        "files": {target: digest(path) for path, target in files},
        "preserved_checkpoint_sha256": CHECKPOINT_SHA,
        "test_pairs": 1000,
        "test_directional_records": 2000,
        "credentials_included": False,
    }
    (OUT / "wandb_restart_manifest.json").write_text(json.dumps(record, indent=2))
    ssh, options, port, host = connection(INSTANCE)
    subprocess.run(
        [
            "scp",
            *options,
            "-P",
            port,
            str(archive),
            host + ":/workspace/wandb_restart_patch.tar.gz",
        ],
        check=True,
    )
    code = f"""import hashlib, json, shutil, tarfile
from pathlib import Path
root=Path({REMOTE!r})
module=b'scripts.rental.nllb_10k_fp16_worker'
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  if module in p.read_bytes().split(b'\\0') and b'Z' not in p.with_name('stat').read_bytes().split()[2:3]:
   raise RuntimeError('Training must be stopped before patch deployment')
 except (FileNotFoundError, ProcessLookupError): pass
archive=Path('/workspace/wandb_restart_patch.tar.gz')
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={record["sha256"]!r}
saved=root/'logs/nllb-1p3b-10k-fp16-20261116/restart_original'
saved.mkdir(exist_ok=True)
expected={record["files"]!r}
with tarfile.open(archive) as tar:
 assert set(tar.getnames()) == set(expected)
 for name in expected:
  original=root/name
  backup=saved/name
  if original.is_file() and not backup.exists():
   backup.parent.mkdir(parents=True,exist_ok=True)
   shutil.copy2(original,backup)
 tar.extractall(root,filter='data')
for name,sha in expected.items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha, name
print(json.dumps({{'deployed': True,'verified_files': len(expected),'test_pairs':1000}}))
"""
    remote(ssh, code, timeout=90)


def launch():
    ssh, *_ = connection(INSTANCE)
    credentials = {
        "HF_TOKEN": secret("HF_API_KEY"),
        "OPENROUTER_API_KEY": secret("OPENROUTER_API_KEY"),
    }
    if not all(credentials.values()):
        raise RuntimeError("The explicitly designated credentials are required")
    code = f"""import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path({REMOTE!r})
output=root/'logs/nllb-1p3b-10k-fp16-20261116'
checkpoint=output/'sft/checkpoints/interrupted.ckpt'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=={CHECKPOINT_SHA!r}
module=b'scripts.rental.nllb_10k_fp16_worker'
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  if module in p.read_bytes().split(b'\\0') and p.with_name('stat').read_bytes().split()[2]!=b'Z':
   raise RuntimeError('An existing training worker is active')
 except (FileNotFoundError,ProcessLookupError): pass
config=json.loads((root/'configs/rental/nllb_10k_fp16_20261116.json').read_text())
manifest=json.loads((root/'data/run/manifest.json').read_text())
assert config['test_pairs']==1000
assert hashlib.sha256((root/'data/run'/config['test_file']).read_bytes()).hexdigest()==manifest['outputs']['test_sample']['sha256']
env=os.environ.copy()
env.update(json.load(sys.stdin))
env.update(WANDB_MODE='offline',WANDB_DISABLE_CODE='true',WANDB_DISABLE_GIT='true')
env.pop('WANDB_API_KEY',None)
pidfile=root/'worker.pid'
if pidfile.exists(): (output/'restart_original/worker.pid').write_text(pidfile.read_text())
(output/'STOP').unlink(missing_ok=True)
with (root/'worker.log').open('ab') as stream:
 process=subprocess.Popen(['/venv/main/bin/python','-u','-m','scripts.rental.nllb_10k_fp16_worker','--stage','resume_pipeline','--resume',str(checkpoint),'--resume-phase','sft'],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
pidfile.write_text(str(process.pid))
record={{'pid':process.pid,'resumed_from_sft_step':221,'test_pairs':1000,'wandb_mode':'offline_with_local_live_relay'}}
(output/'wandb_restart.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
"""
    response = remote(
        ssh, code, input=json.dumps(credentials), capture_output=True, timeout=60
    )
    record = json.loads(response.stdout)
    (OUT / "wandb_restart_launch.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["deploy", "launch"])
    args = parser.parse_args()
    deploy() if args.mode == "deploy" else launch()
