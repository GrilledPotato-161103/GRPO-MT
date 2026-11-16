"""Verify population accounting, weighted aggregates, joins and report artifacts."""
import hashlib
import json
from pathlib import Path

import numpy as np

root=Path(__file__).resolve().parents[2]
out=root/'runs/comet-reference-20261115/dataset_analysis'
stats=json.loads((out/'statistics.json').read_text(encoding='utf-8'))
checks=[]
for split,expected in (('train',2977999),('test',19151)):
    s=stats[split]
    assert s['pairs']==expected
    means=np.array([s['directions'][d]['mean'] for d in ('en_vi','vi_en')])
    for name,flag in s['flags'].items():
        assert 0<=flag['pairs']<=expected
        assert abs(flag['share']-flag['pairs']/expected)<1e-14
    for direction,groups in s['length_bins'].items():
        assert sum(g['pairs'] for g in groups)==expected
        weighted=sum(np.array(g['means'])*g['pairs'] for g in groups)/expected
        np.testing.assert_allclose(weighted,means,atol=1e-12,rtol=0)
    blocks=s['order_blocks']
    assert blocks[0]['start']==0 and blocks[-1]['end']==expected-1
    assert all(a['end']+1==b['start'] for a,b in zip(blocks,blocks[1:]))
    np.testing.assert_allclose(sum(np.array(b['means'])*b['pairs'] for b in blocks)/expected,means,atol=1e-12,rtol=0)
    for direction in ('en_vi','vi_en'):
        below=list(s['directions'][direction]['fractions_below'].values())
        assert all(0<=a<=b<=1 for a,b in zip(below,below[1:]))
    for name,d in s['duplicates'].items():
        assert d['excess_rows']==expected-d['unique']
    checks.append(split+': populations, weighted means, thresholds and duplicates')
review=json.loads((out/'training_calibration_review.json').read_text(encoding='utf-8'))
assert len(review['items'])==240
assert all(0<=r['pair_index']<2977999 for r in review['items'])
assert all(all(v is None or v=='' for v in r['annotation'].values()) for r in review['items'])
assert all(v['exact_text_matched_pairs']==1600 for v in stats['matched_model_comparison'].values())
checks.extend(['240 training-only review items, annotation fields blank','1600 exact text matches per policy/direction'])
report=(out/'report.html').read_text(encoding='utf-8')
assert report.count('data:image/png;base64,')==2 and '<table>' in report
checks.append('Self-contained report contains both rendered figures')
files=['statistics.json','supplement.json','optimization_alignment.json','training_rollout_sensitivity.json',
       'training_calibration_review.json','report.html','score_distributions.png','length_conditioning.png']
proof={'passed':True,'checks':checks,'artifacts_sha256':{n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in files},
       'limitations':'Descriptive-accounting checks, not validation of QE correctness or proposed hyperparameters'}
(out/'verification.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
print(json.dumps({'passed':True,'checks':checks},indent=2))
