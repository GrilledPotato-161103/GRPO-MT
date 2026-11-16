"""Reproducible local audit of official PhoMT text and bidirectional QE scores."""
import csv
import hashlib
import io
import json
import posixpath
import random
import re
import shutil
import tarfile
import time
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/comet-reference-20261115'
OUT = RUN/'dataset_analysis'
RAW = ROOT/'data/reference_audit'
OUT.mkdir(exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)
VI = set('ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ')
DTYPE = np.dtype([('en_chars','u4'),('vi_chars','u4'),('en_units','u2'),('vi_units','u2'),
                  ('en_lower','?'),('vi_lower','?'),('vi_ascii','?'),('copy_like','?'),
                  ('same_text','?'),('digits_en','?'),('digits_vi','?'),
                  ('en_fragment','?'),('vi_fragment','?'),('en_hash','S16'),('vi_hash','S16'),('pair_hash','S16')])

def digest(text):
    return hashlib.blake2b(text.encode(),digest_size=16).digest()

def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())

def hashed_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def save(name, value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def restore():
    provenance = json.loads((RUN/'results/asset_provenance.json').read_text())
    targets = [(f'evaluation/data/{s}.csv', RAW/f'{s}.csv',provenance['splits'][s]['sha256']) for s in ('train','test')]
    revision = provenance['dataset_revision']
    targets.append((f'assets/huggingface/hub/datasets--vinai--PhoMT/snapshots/{revision}/PhoMT.zip', RAW/'PhoMT.zip',provenance['archive_sha256']))
    with tarfile.open(RUN/'complete_backup/artifacts.tar') as archive:
        for name,destination,expected in targets:
            if not destination.exists():
                member = archive.getmember(name)
                while member.issym():
                    member = archive.getmember(posixpath.normpath(posixpath.join(posixpath.dirname(member.name),member.linkname)))
                with archive.extractfile(member) as source, destination.with_suffix('.tmp').open('wb') as target:
                    shutil.copyfileobj(source,target)
                destination.with_suffix('.tmp').replace(destination)
            assert hashed_file(destination) == expected, 'Input checksum mismatch'
    with zipfile.ZipFile(RAW/'PhoMT.zip') as archive:
        (OUT/'official_README.txt').write_bytes(archive.read('PhoMT/README.md'))

def profile(split):
    scores = pd.read_csv(RUN/f'results/{split}/scores.csv')
    assert np.array_equal(scores.pair_index.to_numpy(),np.arange(len(scores)))
    q = scores[['en_vi_cometkiwi','vi_en_cometkiwi']].to_numpy()
    cache = OUT/f'{split}_features.npy'
    if cache.exists():
        return np.load(cache,allow_pickle=False),q
    features = np.zeros(len(q),dtype=DTYPE)
    reservoirs, seen = {}, {}
    rng = random.Random(20261115)
    boundaries = [.5,.65,.75,.85]
    with (RAW/f'{split}.csv').open(encoding='utf-8',newline='') as handle:
        for i,row in enumerate(csv.DictReader(handle)):
            en,vi = row['en'],row['vi']
            a,b = normalized(en),normalized(vi)
            eu,vu = len(en.split()),len(vi.split())
            initial_en = next((c for c in en if c.isalpha()),'A')
            initial_vi = next((c for c in vi if c.isalpha()),'A')
            vi_ascii = not any(c in VI for c in b)
            copy_like = False
            if vi_ascii and min(eu,vu)>=4:
                aw,bw = set(re.findall(r'\w+',a)),set(re.findall(r'\w+',b))
                copy_like = len(aw & bw)/max(1,len(aw | bw)) >= .8
            features[i] = (len(en),len(vi),eu,vu,initial_en.islower(),initial_vi.islower(),vi_ascii,copy_like,
                           a==b,any(c.isdigit() for c in en),any(c.isdigit() for c in vi),
                           not en.rstrip().endswith(('.', '!', '?', '…')),not vi.rstrip().endswith(('.', '!', '?', '…')),
                           digest(a),digest(b),digest(a+'\0'+b))
            for column,direction in enumerate(('en_vi','vi_en')):
                cell = direction+'_'+str(int(np.searchsorted(boundaries,q[i,column])))
                seen[cell] = seen.get(cell,0)+1
                sample = reservoirs.setdefault(cell,[])
                record = dict(pair_index=i,en=en,vi=vi,en_vi=float(q[i,0]),vi_en=float(q[i,1]),
                              copy_like=bool(copy_like),vi_without_diacritics=bool(vi_ascii))
                if len(sample)<24:
                    sample.append(record)
                else:
                    position = rng.randrange(seen[cell])
                    if position<24:
                        sample[position] = record
            if (i+1)%250000==0:
                print(json.dumps({'split':split,'profiled_pairs':i+1}),flush=True)
    assert i+1 == len(q)
    np.save(cache,features,allow_pickle=False)
    save(f'{split}_stratified_review.json',{'cells':reservoirs,'population_counts':seen,'seed':20261115,
         'note':'Review candidates only; no human labels or inferred error rates'})
    return features,q

def describe(values):
    return {'mean':float(np.mean(values)),'median':float(np.median(values)),
            'p05':float(np.quantile(values,.05)),'p95':float(np.quantile(values,.95)),'max':float(np.max(values))}

def group_stats(mask,q):
    return {'pairs':int(mask.sum()),'share':float(mask.mean()),
            'means':q[mask].mean(axis=0).tolist() if mask.any() else None,
            'fraction_either_direction_below_05':float((q[mask].min(axis=1)<.5).mean()) if mask.any() else None}

def summarize(features,q):
    result = {'pairs':len(q),'directions':{},'lengths':{},'flags':{},'length_bins':{},'order_blocks':[]}
    for column,direction in enumerate(('en_vi','vi_en')):
        v = q[:,column]
        result['directions'][direction] = {**describe(v),'std':float(v.std()),
            'fractions_below':{str(t):float((v<t).mean()) for t in (.5,.6,.65,.7,.75,.8,.85)},
            'fraction_above_085':float((v>.85).mean())}
    delta = q[:,0]-q[:,1]
    result['direction_comparison'] = {'pearson':float(np.corrcoef(q.T)[0,1]),
        'en_vi_minus_vi_en':describe(delta),'fraction_abs_delta_above_005':float((abs(delta)>.05).mean()),
        'fraction_abs_delta_above_01':float((abs(delta)>.1).mean()),
        'both_below_05':float((q.max(axis=1)<.5).mean()),'either_below_05':float((q.min(axis=1)<.5).mean())}
    for name in ('en_chars','vi_chars','en_units','vi_units'):
        result['lengths'][name] = describe(features[name])
    result['lengths']['vi_over_en_whitespace_units'] = describe(features['vi_units']/np.maximum(1,features['en_units']))
    for name in ('en_lower','vi_lower','vi_ascii','copy_like','same_text','digits_en','digits_vi','en_fragment','vi_fragment'):
        result['flags'][name] = group_stats(features[name],q)
    for direction,column,name in (('en_vi',0,'en_units'),('vi_en',1,'vi_units')):
        bins = [(0,5),(5,10),(10,20),(20,40),(40,100000)]
        result['length_bins'][direction] = [{'source_whitespace_units':f'{lo+1}-{hi}' if hi<100000 else '>40',
            **group_stats((features[name]>lo)&(features[name]<=hi),q)} for lo,hi in bins]
    ratio = features['vi_chars']/np.maximum(1,features['en_chars'])
    result['flags']['extreme_character_ratio'] = group_stats((ratio<.5)|(ratio>2.5),q)
    for indices in np.array_split(np.arange(len(q)),20):
        result['order_blocks'].append({'start':int(indices[0]),'end':int(indices[-1]),'pairs':len(indices),
            'means':q[indices].mean(axis=0).tolist(),'mean_en_units':float(features['en_units'][indices].mean())})
    for language in ('en','vi','pair'):
        unique,counts = np.unique(features[language+'_hash'],return_counts=True)
        result.setdefault('duplicates',{})[language] = {'unique':len(unique),'excess_rows':int((counts-1).sum()),
                'rows_in_repeated_groups':int(counts[counts>1].sum()),'max_multiplicity':int(counts.max())}
    _,first = np.unique(features['pair_hash'],return_index=True)
    for language in ('en','vi'):
        _,counts = np.unique(features[language+'_hash'][first],return_counts=True)
        result['duplicates'][language]['sources_with_multiple_distinct_targets'] = int((counts>1).sum())
    return result

def overlap(train,test):
    result={}
    for name in ('en','vi','pair'):
        mask = np.isin(test[name+'_hash'],train[name+'_hash'])
        result[name] = {'test_pairs_overlapping_raw_train':int(mask.sum()),'test_share':float(mask.mean())}
    return result

def matched_predictions(test_q):
    with (RAW/'test.csv').open(encoding='utf-8',newline='') as handle:
        pairs = list(csv.DictReader(handle))
    index = {(r['en'],r['vi']):i for i,r in enumerate(pairs)}
    result = {}
    for policy in ('sft','rl'):
        path = ROOT/f'runs/nllb-1p3b-60596557-20261115/run/evaluation/test_comparison/{policy}/translation_metrics/test_predictions.jsonl'
        records = [json.loads(line) for line in path.open(encoding='utf-8')]
        for column,direction in enumerate(('en_vi','vi_en')):
            selected = [r for r in records if r['direction']==direction]
            ids = [index[(r['source'],r['reference']) if column==0 else (r['reference'],r['source'])] for r in selected]
            baseline = test_q[ids,column]
            prediction = np.array([r['comet'] for r in selected])
            result[policy+'_'+direction] = {'exact_text_matched_pairs':len(ids),
                'reference_mean':float(baseline.mean()),'model_mean':float(prediction.mean()),
                'mean_model_minus_reference':float((prediction-baseline).mean()),
                'fraction_model_score_above_reference':float((prediction>baseline).mean())}
    return result

def figures(stats,arrays):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#007f86','#d17424']
    fig,axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for col,direction in enumerate(('en_vi','vi_en')):
        for split,color in zip(('train','test'),colors):
            q = arrays[split][1][:,col]
            axes[0,col].hist(q,bins=np.linspace(.1,.95,85),density=True,histtype='step',linewidth=1.8,label=split,color=color)
            axes[1,col].plot(np.arange(1,21)*5,[r['means'][col] for r in stats[split]['order_blocks']],label=split,color=color,marker='.')
        axes[0,col].set(title=direction.replace('_',' → ').upper(),xlabel='Raw COMETKiwi score',ylabel='Density')
        axes[1,col].set(xlabel='Corpus position percentile (not time)',ylabel='Mean score in block',ylim=(.55,.9))
        for ax in axes[:,col]:
            ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('PhoMT references: directional distributions and corpus order')
    fig.savefig(OUT/'score_distributions.png',dpi=180)
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for col,direction in enumerate(('en_vi','vi_en')):
        for split,color in zip(('train','test'),colors):
            data = stats[split]['length_bins'][direction]
            axes[col].plot(range(len(data)),[r['means'][col] for r in data],marker='o',label=split,color=color)
        axes[col].set(title=direction.replace('_',' → ').upper(),xticks=range(5),
                      xticklabels=['1–5','6–10','11–20','21–40','>40'],xlabel='Source whitespace units',ylabel='Mean COMETKiwi',ylim=(.5,.9))
        axes[col].legend(); axes[col].grid(alpha=.2)
    fig.suptitle('Length-conditioned scores; VI whitespace units are mostly syllables, not words')
    fig.savefig(OUT/'length_conditioning.png',dpi=180)
    plt.close(fig)

def main():
    started=time.time()
    restore()
    arrays = {split:profile(split) for split in ('test','train')}
    stats = {split:summarize(*arrays[split]) for split in arrays}
    stats['raw_train_test_overlap'] = overlap(arrays['train'][0],arrays['test'][0])
    stats['matched_model_comparison'] = matched_predictions(arrays['test'][1])
    stats['limitations'] = ['COMET is a proxy, not human correctness','Flags are triage signals, not error labels',
        'Raw official training differs from leakage-filtered prepared training','Whitespace units do not measure comparable EN/VI words',
        'Corpus-order blocks are not domain labels or temporal trends','Training/test scored; official development split not scored']
    save('statistics.json',stats)
    figures(stats,arrays)
    print(json.dumps({'status':'complete','seconds':time.time()-started,'output':str(OUT)},indent=2),flush=True)

if __name__=='__main__':
    main()
