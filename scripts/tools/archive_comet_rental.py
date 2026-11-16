"""Archive task artifacts and reusable assets, excluding credentials and environments."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import tarfile

OUT = Path('/workspace/comet-reference-backup')
OUT.mkdir(exist_ok=True)
sources = [
    (Path('/workspace/comet-reference'), 'evaluation'),
    (Path('/workspace/noisy-objective-validation'), 'objective_validation'),
    (Path('/root/.cache/huggingface/hub'), 'assets/huggingface/hub'),
    (Path('/tmp/pytest-of-root'), 'tiny_model_test_artifacts'),
]
excluded = {'env', '__pycache__', '.pytest_cache', '.locks'}
manifest = {'instance_id':60753835, 'files':[], 'excluded_reproducible_directories':sorted(excluded),
            'python':platform.python_version(),
            'packages':sorted({(d.metadata['Name'], d.version) for d in importlib.metadata.distributions()})}
validation_inventory = subprocess.check_output([
    '/workspace/noisy-objective-validation/env/bin/python','-c',
    "import importlib.metadata,json; print(json.dumps(sorted({(d.metadata['Name'],d.version) for d in importlib.metadata.distributions()})))"
],text=True)
manifest['validation_packages'] = json.loads(validation_inventory)
with tarfile.open(OUT/'artifacts.tar', 'w', dereference=False) as archive:
    for directory, prefix in sources:
        if not directory.is_dir():
            raise RuntimeError(f'Missing artifact directory: {directory}')
        for path in sorted(directory.rglob('*')):
            relative = path.relative_to(directory)
            if any(part in excluded for part in relative.parts):
                continue
            name = str(Path(prefix)/relative)
            if path.is_symlink():
                link = path.readlink()
                if link.is_absolute() or not path.resolve().is_relative_to(directory.resolve()):
                    # Test-run "current" links are redundant pointers; real runs are included.
                    manifest.setdefault('excluded_external_links',[]).append(str(path))
                    continue
                archive.add(path,arcname=name,recursive=False)
                manifest['files'].append({'path':name,'kind':'symlink','target':str(link)})
            elif path.is_file():
                with path.open('rb') as handle:
                    digest = hashlib.file_digest(handle,'sha256').hexdigest()
                size = path.stat().st_size
                archive.add(path,arcname=name,recursive=False)
                manifest['files'].append({'path':name,'kind':'file','size':size,'sha256':digest})
    payload = json.dumps(manifest,indent=2).encode()
    import io
    member = tarfile.TarInfo('backup_manifest.json')
    member.size = len(payload)
    archive.addfile(member,io.BytesIO(payload))
(OUT/'backup_manifest.json').write_bytes(payload)
with (OUT/'artifacts.tar').open('rb') as handle:
    digest = hashlib.file_digest(handle,'sha256').hexdigest()
(OUT/'artifacts.sha256').write_text(digest+'  artifacts.tar\n')
print(json.dumps({'files':len(manifest['files']), 'bytes':(OUT/'artifacts.tar').stat().st_size,
                  'sha256':digest}),flush=True)
