"""Back up, verify every archived file, then destroy only the authorized rental."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from scripts.tools.comet_reference_rental import api, ROOT, OUT

INSTANCE = 60753835
backup = OUT/'complete_backup'
backup.mkdir(exist_ok=True)
connection = json.loads((OUT/'connection.json').read_text())
if connection['instance'] != INSTANCE:
    raise RuntimeError('Unexpected rental identity')
options = ['-i',connection['key'],'-o','BatchMode=yes','-o','ConnectTimeout=20',
           '-o','StrictHostKeyChecking=yes','-o',f'UserKnownHostsFile={OUT / "known_hosts"}']
ssh = ['ssh',*options,'-p',connection['port'],connection['host']]
scp = ['scp',*options,'-P',connection['port']]
mode = sys.argv[1]
if mode == 'download':
    commands = f'lcd "{backup.as_posix()}"\nreget /workspace/comet-reference-backup/artifacts.tar artifacts.tar\n'
    subprocess.run(['sftp',*options,'-P',connection['port'],'-B','262144','-R','256','-b','-',connection['host']],
                   input=commands,text=True,check=True)
elif mode == 'backup':
    subprocess.run([*scp,str(ROOT/'scripts/tools/archive_comet_rental.py'),
                    connection['host']+':/workspace/archive_comet_rental.py'],check=True)
    subprocess.run([*ssh,'/venv/main/bin/python /workspace/archive_comet_rental.py'],check=True)
    for name in ('artifacts.sha256','backup_manifest.json','artifacts.tar'):
        subprocess.run([*scp,connection['host']+':/workspace/comet-reference-backup/'+name,
                        str(backup/name)],check=True)
elif mode == 'verify':
    expected_archive = (backup/'artifacts.sha256').read_text().split()[0]
    with (backup/'artifacts.tar').open('rb') as handle:
        actual_archive = hashlib.file_digest(handle,'sha256').hexdigest()
    if actual_archive != expected_archive:
        raise RuntimeError('Archive checksum mismatch')
    manifest = json.loads((backup/'backup_manifest.json').read_text())
    if manifest['instance_id'] != INSTANCE:
        raise RuntimeError('Wrong backup identity')
    count, total = 0, 0
    with tarfile.open(backup/'artifacts.tar') as archive:
        embedded = json.load(archive.extractfile('backup_manifest.json'))
        if embedded != manifest:
            raise RuntimeError('Manifest mismatch')
        for record in manifest['files']:
            member = archive.getmember(record['path'])
            if record['kind'] == 'symlink':
                if not member.issym() or member.linkname != record['target']:
                    raise RuntimeError('Symlink mismatch: '+record['path'])
            else:
                with archive.extractfile(member) as handle:
                    digest = hashlib.file_digest(handle,'sha256').hexdigest()
                if member.size != record['size'] or digest != record['sha256']:
                    raise RuntimeError('File mismatch: '+record['path'])
                total += member.size
            count += 1
    proof = {'instance_id':INSTANCE,'complete':True,'verified_files_and_links':count,
             'file_bytes':total,'archive_sha256':actual_archive,'verified_unix':time.time(),
             'scope':'All task artifacts, raw datasets, reusable HF cache, tiny-model test artifacts; runtime environments captured as package inventories'}
    (backup/'verification.json').write_text(json.dumps(proof,indent=2))
    print(json.dumps(proof,indent=2))
elif mode == 'destroy':
    proof = json.loads((backup/'verification.json').read_text())
    if not proof.get('complete') or proof.get('instance_id') != INSTANCE:
        raise RuntimeError('Verified local backup is required before destruction')
    # Recheck archive integrity immediately before the irreversible action.
    with (backup/'artifacts.tar').open('rb') as handle:
        if hashlib.file_digest(handle,'sha256').hexdigest() != proof['archive_sha256']:
            raise RuntimeError('Backup changed since verification')
    result = api(f'instances/{INSTANCE}/','DELETE')
    sanitized = {k:v for k,v in result.items() if 'key' not in k.lower() and 'token' not in k.lower()}
    (OUT/'destroy_response.json').write_text(json.dumps(sanitized,indent=2))
    print(json.dumps(sanitized))
elif mode == 'confirm':
    listing = api('/api/v1/instances?limit=25')
    matches = [r for r in listing.get('instances',[]) if r.get('id') == INSTANCE]
    if matches:
        raise RuntimeError('Rental still appears in account inventory')
    closure = {'instance_id':INSTANCE,'destroyed':True,'absent_from_account_inventory':True,
               'confirmed_unix':time.time(),'local_backup':str(backup)}
    (OUT/'closure.json').write_text(json.dumps(closure,indent=2))
    print(json.dumps(closure,indent=2))
else:
    raise ValueError('Expected backup, verify, destroy, or confirm')
