"""Small Vast REST helper; credentials remain outside command arguments and logs."""
import json
import os
from pathlib import Path
import sys
import urllib.request
import winreg

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/comet-reference-20261115'
OUT.mkdir(parents=True, exist_ok=True)

def secret(name):
    for hive, path in ((winreg.HKEY_CURRENT_USER, 'Environment'),
                       (winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment')):
        try:
            with winreg.OpenKey(hive, path) as key:
                value, _ = winreg.QueryValueEx(key, name)
                if str(value).strip():
                    return str(value).strip()
        except OSError:
            pass
    return os.environ.get(name, '').strip()

def api(path, method='GET', body=None):
    key = (Path.home()/'.config/vastai/vast_api_key').read_text().strip()
    url = 'https://console.vast.ai'+path if path.startswith('/api/') else 'https://console.vast.ai/api/v0/'+path
    req = urllib.request.Request(url,
        data=json.dumps(body).encode() if body is not None else None,
        headers={'Authorization': 'Bearer '+key, 'Content-Type':'application/json'}, method=method)
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)

mode = sys.argv[1] if __name__ == '__main__' and len(sys.argv) > 1 else None
if mode == 'preflight':
    failures = 0
    for url in ('https://huggingface.co/Unbabel/wmt22-cometkiwi-da/resolve/main/checkpoints/model.ckpt',
                'https://huggingface.co/datasets/vinai/PhoMT/resolve/main/PhoMT.zip'):
        try:
            token = secret('HUGGING_FACE_API_KEY') or secret('HF_TOKEN') or secret('HF_API_KEY')
            headers = {'Range':'bytes=0-0', 'Authorization':'Bearer '+token}
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
                print(json.dumps({'url':url,'status':response.status,'range':response.headers.get('Content-Range')}))
        except Exception as error:
            failures += 1
            print(json.dumps({'url':url,'error':str(error),
                              'detail':getattr(error,'headers',{}).get('X-Error-Message')}))
    sys.exit(bool(failures))
elif mode == 'inventory':
    listing = api('/api/v1/instances?limit=25')
    (OUT/'inventory.json').write_text(json.dumps(listing, indent=2))
    print(json.dumps([{k:r.get(k) for k in ('id','gpu_name','actual_status','label','ssh_host','ssh_port')}
        for r in listing.get('instances', [])], indent=2))
    print(json.dumps({n: bool(secret(n)) for n in ('HUGGING_FACE_API_KEY','HF_TOKEN','HF_API_KEY')}))
elif mode == 'search':
    query = {'verified':{'eq':True},'rentable':{'eq':True},'rented':{'eq':False},
        'gpu_name':{'eq':'RTX 4090'},'num_gpus':{'eq':1},'reliability':{'gte':.99},
        'cpu_ram':{'gte':24000},'disk_space':{'gte':32},'inet_down':{'gte':200},
        'type':'ondemand','limit':50,'order':[['dph_total','asc']], 'allocated_storage':32}
    offers = api('bundles/', 'POST', query)
    (OUT/'offers.json').write_text(json.dumps(offers, indent=2))
    print(json.dumps([{k:r.get(k) for k in ('id','gpu_name','dph_total','storage_total_cost','geolocation','inet_down','cpu_ram','reliability')}
        for r in offers.get('offers', [])[:12]], indent=2))
elif mode == 'create':
    offer = int(sys.argv[2])
    result = api(f'asks/{offer}/', 'PUT', {'client_id':'me','image':'vastai/pytorch:cuda-13.2.1-auto',
        'disk':32,'label':'comet-reference-quality-20261115','runtype':'ssh_direct','onstart':'touch /workspace/comet-reference-ready'})
    result = {k:v for k,v in result.items() if 'key' not in k.lower() and 'token' not in k.lower()}
    (OUT/'rental.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
elif mode == 'status':
    result = api(f'instances/{int(sys.argv[2])}/')
    (OUT/'instance.json').write_text(json.dumps(result, indent=2))
    r = result['instances']
    print(json.dumps({k:r.get(k) for k in ('id','actual_status','status_msg','ssh_host','ssh_port','public_ipaddr','gpu_name','dph_total')}, indent=2))
