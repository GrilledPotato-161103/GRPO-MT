"""Use the existing SSH identity; deliver the HF token through encrypted stdin."""
import json
from pathlib import Path
import subprocess
import sys
import tarfile

from scripts.tools.comet_reference_rental import api, secret, ROOT, OUT

instance = int(sys.argv[1])
record = api(f'instances/{instance}/')['instances']
key = Path.home()/'.ssh/id_ed25519'
options = ['-i',str(key),'-o','BatchMode=yes','-o','ConnectTimeout=20',
           '-o','StrictHostKeyChecking=accept-new','-o',f'UserKnownHostsFile={OUT / "known_hosts"}']
host = record['public_ipaddr']
ports = record.get('ports',{}).get('22/tcp',[])
port = str(ports[0]['HostPort']) if ports else str(record['ssh_port'])
host = 'root@'+host if ports else 'root@'+record['ssh_host']
ssh = ['ssh',*options,'-p',port,host]
mode = sys.argv[2]
if mode=='probe':
    subprocess.run([*ssh,"command -v python; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; ls /venv/main/bin/python"],check=True)
elif mode=='launch':
    archive = OUT/'source.tar.gz'
    files = ['scripts/__init__.py','scripts/tools/__init__.py','scripts/tools/score_parallel_references.py',
             'scripts/tools/comet_reference_worker.py','src/__init__.py','src/models/__init__.py',
             'src/models/components/__init__.py','src/models/components/comet_reward.py',
             'src/utils/__init__.py','src/utils/credentials.py']
    files.extend(str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'src/utils').glob('*.py')
                 if str(p.relative_to(ROOT)).replace('\\','/') not in files)
    with tarfile.open(archive,'w:gz') as handle:
        for file in files:
            handle.add(ROOT/file,arcname=file)
    subprocess.run([*ssh,'mkdir -p /workspace/comet-reference'],check=True)
    subprocess.run(['scp',*options,'-P',port,str(archive),host+':/workspace/comet-reference/source.tar.gz'],check=True)
    subprocess.run([*ssh,'cd /workspace/comet-reference && tar xzf source.tar.gz'],check=True)
    token = secret('HUGGING_FACE_API_KEY') or secret('HF_TOKEN') or secret('HF_API_KEY')
    if not token:
        raise RuntimeError('No HF token')
    # The launcher reads a single stdin payload, then detaches the worker. No token file.
    remote = "import json,sys,os,subprocess; from pathlib import Path; p=Path('/workspace/comet-reference'); old=(p/'worker.pid').read_text().strip() if (p/'worker.pid').exists() else 'none'; assert not Path('/proc/'+old).exists(), 'Worker already running'; env=os.environ.copy(); env['HF_TOKEN']=json.load(sys.stdin)['token']; f=(p/'worker.log').open('ab',buffering=0); c=subprocess.Popen([sys.executable,'-m','scripts.tools.comet_reference_worker'],cwd=p,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); (p/'worker.pid').write_text(str(c.pid)); print('Launched worker PID',c.pid)"
    command = "/venv/main/bin/python -c '"+remote.replace("'", "'\"'\"'")+"'"
    subprocess.run([*ssh,command],input=json.dumps({'token':token}),text=True,check=True)
    (OUT/'connection.json').write_text(json.dumps({'instance':instance,'host':host,'port':port,'key':str(key)},indent=2))
elif mode=='check':
    subprocess.run([*ssh,"cd /workspace/comet-reference && tail -n 18 worker.log; cat results/worker_status.json; for f in results/progress.json results/smoke.json results/test/summary.json; do test ! -f $f || cat $f; done; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"],check=True)
elif mode=='fetch':
    remote = "from pathlib import Path; import tarfile; root=Path('/workspace/comet-reference'); a=tarfile.open('/workspace/comet-reference-summary.tar.gz','w:gz'); files=sorted((root/'results').rglob('*.json'))+sorted((root/'results').rglob('*.csv')); [a.add(p,arcname=str(p.relative_to(root))) for p in files]; a.close()"
    command = "/venv/main/bin/python -c '"+remote.replace("'", "'\"'\"'")+"'"
    subprocess.run([*ssh,command],check=True)
    subprocess.run(['scp',*options,'-P',port,host+':/workspace/comet-reference-summary.tar.gz',str(OUT/'summary.tar.gz')],check=True)
    with tarfile.open(OUT/'summary.tar.gz') as archive:
        archive.extractall(OUT,filter='data')
    print('Downloaded completed scores and current summaries to', OUT/'results')
