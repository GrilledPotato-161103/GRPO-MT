"""Rental worker: bootstrap COMET, retrieve official assets, score full references."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path('/workspace/comet-reference')
OUT = ROOT/'results'
OUT.mkdir(parents=True, exist_ok=True)

def state(phase, **extra):
    temporary = OUT/'worker_status.tmp'
    temporary.write_text(json.dumps({'phase':phase,'unix_time':time.time(),**extra},indent=2))
    temporary.replace(OUT/'worker_status.json')
    print(json.dumps({'phase':phase,**extra}),flush=True)

def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()

def prepare_csv(archive, split):
    names = [n for n in archive.namelist() if '__MACOSX' not in n and not n.endswith('/')]
    aliases = ('train','training') if split=='train' else ('test','testing')
    relevant = [n for n in names if any(a in Path(n).name.lower() for a in aliases)]
    csvs = [n for n in relevant if n.lower().endswith('.csv')]
    destination = ROOT/'data'/f'{split}.csv'
    temporary = destination.with_suffix('.tmp')
    if len(csvs)==1:
        member = csvs[0]
        count = 0
        with io.TextIOWrapper(archive.open(member),encoding='utf-8-sig',newline='') as source, temporary.open('w',encoding='utf-8',newline='') as target:
            reader = csv.DictReader(source)
            if not {'en','vi'}.issubset(reader.fieldnames or []):
                raise ValueError(f'Unexpected CSV columns: {reader.fieldnames}')
            writer = csv.writer(target)
            writer.writerow(['en','vi'])
            for row in reader:
                if any(not row.get(k,'').strip() for k in ('en','vi')):
                    raise ValueError('Empty parallel row')
                writer.writerow([row['en'],row['vi']])
                count += 1
        members = [member]
    else:
        # Prefer the official detokenized text over already segmented variants.
        raw = [n for n in relevant if 'detoken' in n.lower()]
        if raw:
            relevant = raw
        languages = {lang:[n for n in relevant if n.lower().endswith('.'+lang)] for lang in ('en','vi')}
        if any(len(v)!=1 for v in languages.values()):
            raise ValueError(f'Ambiguous parallel members: {languages}; inspect archive_members.json')
        members = [languages['en'][0],languages['vi'][0]]
        count = 0
        with io.TextIOWrapper(archive.open(members[0]),encoding='utf-8-sig') as english, io.TextIOWrapper(archive.open(members[1]),encoding='utf-8-sig') as vietnamese, temporary.open('w',encoding='utf-8',newline='') as target:
            writer = csv.writer(target)
            writer.writerow(['en','vi'])
            for en,vi in zip(english,vietnamese,strict=True):
                en,vi = en.rstrip('\r\n'),vi.rstrip('\r\n')
                if not en.strip() or not vi.strip():
                    raise ValueError('Empty parallel row')
                writer.writerow([en,vi])
                count += 1
    if not count:
        raise ValueError('Empty split')
    temporary.replace(destination)
    return {'pairs':count,'archive_members':members,'sha256':digest(destination)}

def main():
    os.environ.update(HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',
                      TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
    state('installing')
    subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check',
                    'unbabel-comet==2.2.7','transformers==4.57.6','setuptools<81',
                    'lightning>=2.2,<3','hydra-core==1.3.2','rich'],check=True)
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    state('downloading',gpu=torch.cuda.get_device_name(),torch=torch.__version__)
    token = os.environ['HF_TOKEN']
    api = HfApi(token=token)
    model_revision = api.model_info('Unbabel/wmt22-cometkiwi-da').sha
    data_revision = api.dataset_info('vinai/PhoMT').sha
    checkpoint = Path(hf_hub_download('Unbabel/wmt22-cometkiwi-da','checkpoints/model.ckpt',revision=model_revision,token=token))
    hf_hub_download('Unbabel/wmt22-cometkiwi-da','hparams.yaml',revision=model_revision,token=token)
    dataset = Path(hf_hub_download('vinai/PhoMT','PhoMT.zip',repo_type='dataset',revision=data_revision,token=token))
    state('preparing_data')
    (ROOT/'data').mkdir(exist_ok=True)
    with zipfile.ZipFile(dataset) as archive:
        (OUT/'archive_members.json').write_text(json.dumps(archive.namelist(),indent=2))
        splits = {s:prepare_csv(archive,s) for s in ('test','train')}
    provenance = {'dataset':'vinai/PhoMT','dataset_revision':data_revision,'archive_sha256':digest(dataset),
                  'model':'Unbabel/wmt22-cometkiwi-da','model_revision':model_revision,
                  'checkpoint_sha256':digest(checkpoint),'splits':splits,
                  'scope':'Full official train/test, unaugmented, no NLLB length filtering or model generation'}
    (OUT/'asset_provenance.json').write_text(json.dumps(provenance,indent=2))
    state('scoring',splits=splits)
    subprocess.run([sys.executable,'-m','scripts.tools.score_parallel_references',
                    '--train-csv',str(ROOT/'data/train.csv'),'--test-csv',str(ROOT/'data/test.csv'),
                    '--checkpoint',str(checkpoint),'--output',str(OUT),'--batch-size','32'],check=True,cwd=ROOT)
    state('complete')

if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        state('failed',error_type=type(error).__name__)
        raise
