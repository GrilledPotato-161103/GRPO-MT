"""Present corpus findings and audit prior optimization choices without new inference."""
import base64
import csv
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/comet-reference-20261115'
OUT = RUN/'dataset_analysis'
stats = json.loads((OUT/'statistics.json').read_text(encoding='utf-8'))

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def table(headers, rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'

def image(name):
    return '<img src="data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()+'" alt="'+name+'">'

counts,scalars=[],[]
source=ROOT/'runs/nllb-1p3b-60596557-20261115/run/rl/diagnostics/rollouts.jsonl'
with source.open(encoding='utf-8') as handle:
    for line in handle:
        r=json.loads(line)
        size=r['group_size']
        sac=np.array(r['components']['sac']).reshape(-1,size)
        comet=np.array(r['components']['comet']).reshape(-1,size)
        cap=np.array(r['capped']).reshape(-1,size)
        eligible=(sac==1).all(-1)&(comet>=.75).all(-1)&~cap.any(-1)
        scalars.append(np.array(r['rewards']).reshape(-1,size)[eligible])
q=np.concatenate(scalars)
delta=q[:,:,None]-q[:,None,:]
sensitivity=[]
for floor in (0,.002,.005,.01,.02):
    contrast=np.sign(delta)*np.maximum(abs(delta)-floor,0)
    for scale in (.02,.05,.1):
        advantage=np.clip(contrast/scale,-2,2).mean(-1)
        sensitivity.append({'floor':floor,'scale':scale,'eligible_groups':len(q),
                            'suppressed_fraction':float((np.ptp(q,axis=1)<=floor).mean()),
                            'mean_absolute_coefficient':float(abs(advantage).mean())})
save('training_rollout_sensitivity.json',{'scope':'Training-rollout scalar replay; not evidence of judge noise or quality gains','results':sensitivity})

extras={}
for split in ('train','test'):
    f=np.load(OUT/f'{split}_features.npy')
    scores=pd.read_csv(RUN/f'results/{split}/scores.csv').iloc[:,1:].to_numpy()
    examples=[]
    wanted=set(np.flatnonzero(f['copy_like'])[:12].tolist())
    with (ROOT/f'data/reference_audit/{split}.csv').open(encoding='utf-8',newline='') as handle:
        for i,r in enumerate(csv.DictReader(handle)):
            if i in wanted:
                examples.append(dict(pair_index=i,**r,en_vi=float(scores[i,0]),vi_en=float(scores[i,1])))
            if len(examples)==len(wanted): break
    extras[split]={'copy_like_examples':examples,
        'prefix_means':{str(p):scores[:int(len(scores)*p)].mean(axis=0).tolist() for p in (.1,.25,.5,.75,1)},
        'short_en_share':float((f['en_units']<=10).mean()),
        'short_vi_share':float((f['vi_units']<=10).mean()),
        'lowercase_vi_with_lowercase_en':int((f['vi_lower'] & f['en_lower']).sum()),
        'long_inputs_whitespace_units':{lang:{str(t):int((f[lang+'_units']>t).sum()) for t in (128,256,512)} for lang in ('en','vi')}}
save('supplement.json',extras)

# Review candidates come from training only. Blank annotation fields are intentional.
review=[]
sample=json.loads((OUT/'train_stratified_review.json').read_text(encoding='utf-8'))
for cell,records in sample['cells'].items():
    direction=cell.rsplit('_',1)[0]
    for r in records:
        review.append({'pair_index':r['pair_index'],'direction':direction,
                       'source':r['en'] if direction=='en_vi' else r['vi'],
                       'target':r['vi'] if direction=='en_vi' else r['en'],
                       'comet':r[direction],'stratum':cell,
                       'annotation':{'adequacy':None,'fluency':None,'orthography':None,
                                     'wrong_language':None,'context_required':None,'reviewer_notes':''}})
save('training_calibration_review.json',{'note':'No test items. This is a stratified review worksheet, not a labeled calibration set. Apply population weights for aggregate estimates.','items':review})

alignment=[
 {'measure':'Noise floor 0.02 and scale 0.1','decision':'Unvalidated conservative control, not a measured noise estimate',
  'finding':'Suppresses 77.62% of eligible archived groups; scalar spread across sentences cannot calibrate candidate-ranking noise',
  'next':'Review training candidate pairs and sweep floor/scale separately; monitor effective gradient scale and source coverage'},
 {'measure':'QE >= 0.75 as anchor screen','decision':'Retain only as a heuristic ablation screen',
  'finding':'Would screen out 20.67% EN→VI and 19.36% VI→EN raw training references; none of these fractions is an error rate',
  'next':'Require source-conditioned adequacy and naturalness evidence; calibrate by source length/style using training data'},
 {'measure':'Correct SAC 0.7 candidates','decision':'Retain as a candidate repair route, not automatic ground truth',
  'finding':'Idiomatic translations and missing context can be marked down alongside actual omissions',
  'next':'Keep scalar-only mode as a control; require calibrated criterion evidence or adjudication for recommended retraining'},
 {'measure':'Direct directional scoring','decision':'Supported; retain separate directional decisions',
  'finding':'Raw directional means are close but 18.68% of training pairs differ by >0.05; correlation 0.899 does not make views interchangeable',
  'next':'Do not punish a forward translation for an independently weak reverse candidate'},
 {'measure':'Soft preference labels and weaker self-imitation','decision':'Mechanistically justified; numerical weights remain uncalibrated',
  'finding':'References and candidates both have failure modes, and normalized sources can have multiple valid targets',
  'next':'Retain finite-margin safeguards; use reviewed adequacy/fluency and monitor rejected/chosen likelihoods'},
 {'measure':'Capitalization and sentence completeness checks','decision':'Use target conventions and context; reject blanket penalties',
  'finding':'55,014 training VI references begin lowercase; 204,606 lack terminal .!?; these features include valid fragments',
  'next':'Distinguish complete sentences, fragments, titles, quotations and names; mark unavailable context unknown'},
 {'measure':'Deduplication and leakage controls','decision':'Retain grouped preparation and verify its outputs separately',
  'finding':'Raw official splits have 18 repeated normalized bilingual pairs and 891 test EN sources appearing in training',
  'next':'Do not treat raw overlaps as proof of leakage in the project’s already filtered partitions'},
 {'measure':'Evaluation protocol','decision':'Strengthen representativeness and separate evaluator from training objective',
  'finding':'The model comparison covers 1,600 exact matched pairs; order blocks vary substantially',
  'next':'Use a fixed representative development set and a separate adjudicator; report the untouched full test only after selection'}]
save('optimization_alignment.json',alignment)

metric_rows=[]
for split in ('train','test'):
    for direction in ('en_vi','vi_en'):
        d=stats[split]['directions'][direction]
        metric_rows.append([split,direction.replace('_','→'),f"{stats[split]['pairs']:,}",f"{d['mean']:.5f}",f"{d['median']:.5f}",f"{d['p05']:.5f}",f"{100*d['fractions_below']['0.5']:.2f}%",f"{100*d['fractions_below']['0.75']:.2f}%"])
matched=[]
for direction in ('en_vi','vi_en'):
    a=stats['matched_model_comparison']['sft_'+direction]
    b=stats['matched_model_comparison']['rl_'+direction]
    matched.append([direction.replace('_','→'),a['exact_text_matched_pairs'],f"{a['reference_mean']:.5f}",f"{a['model_mean']:.5f}",f"{b['model_mean']:.5f}",f"{b['mean_model_minus_reference']:+.5f}"])
lengths=[]
for split in ('train','test'):
    d=stats[split]['lengths']
    lengths.append([split,f"{d['en_units']['mean']:.2f}",f"{d['vi_units']['mean']:.2f}",f"{100*extras[split]['short_en_share']:.2f}%",f"{100*stats[split]['flags']['vi_lower']['share']:.2f}%",f"{100*stats[split]['flags']['vi_fragment']['share']:.2f}%"])
duplicates=[]
for split in ('train','test'):
    d=stats[split]['duplicates']
    duplicates.append([split,d['pair']['excess_rows'],d['en']['excess_rows'],d['vi']['excess_rows'],d['en']['sources_with_multiple_distinct_targets'],d['vi']['sources_with_multiple_distinct_targets']])
composition=[('News',40990,257),('Blogspot',92545,814),('TED-Talks',316808,2000),('MediaWiki',490505,3270),('WikiHow',507379,3246),('OpenSubtitles',1529772,9564)]
body='''<!doctype html><html lang="en"><meta charset="utf-8"><title>PhoMT corpus audit</title><style>
body{font:16px/1.6 Segoe UI,Arial,sans-serif;background:#f5f7f8;color:#16303b;margin:0}main{max-width:1180px;margin:auto;background:white;padding:36px}h1,h2{line-height:1.25}h1{font-size:34px}h2{margin-top:36px;font-size:24px}table{width:100%;border-collapse:collapse;font-size:14px;margin:18px 0}td,th{text-align:left;padding:10px;border-bottom:1px solid #d9e2e7;vertical-align:top}th{background:#e8f3f3}img{width:100%;height:auto}aside{padding:16px;background:#fff2de;border-left:4px solid #b86a1a}small{color:#516772}a{color:#006d76}p{max-width:1000px}code{background:#eef2f5;padding:2px 5px}</style><main>
<h1>PhoMT: dataset characteristics and optimization decisions</h1>
<p>Full official training and test pairs, scored in both directions with <code>Unbabel/wmt22-cometkiwi-da</code>. Analysis date: 15 November 2026. All computations are local; the rental was destroyed.</p>
<aside>The corpus has a strong high-score center and a heavier low-score training tail. COMET is a proxy, not a human error rate. Low scores, duplicate sources, fragments and copying flags require interpretation.</aside>
<h2>1. Full reference score distributions</h2>'''
body+=table(['Split','Direction','Pairs','Mean','Median','P05','Below 0.50','Below 0.75'],metric_rows)
body+='<p>The means are depressed by a low-score tail. Training/test P95 values are nearly identical, while training P05 values are lower. A score below 0.75 does not establish a defective reference.</p>'+image('score_distributions.png')
body+='<h2>2. Corpus composition and text shape</h2>'+table(['Source domain (paper)','Train pairs','Test pairs'],composition)
body+='<p>Published composition: OpenSubtitles contributes 51.37% of training and 49.94% of test. Row-level domain/document labels are not supplied in this downloaded text archive; corpus-order blocks must not be relabeled as domains without provenance. English was originally the source language, so reversing pairs is not guaranteed to restore omitted facts.</p>'
body+=table(['Split','EN mean units','VI mean units','EN ≤10 units','VI lowercase initial','VI no terminal .!?'],lengths)
body+='<p>Whitespace units are not comparable linguistic words: Vietnamese spacing often separates syllables. Median VI/EN unit ratio is 1.25 in both splits. Fragments, titles and quotations can legitimately differ from full-sentence capitalization/punctuation. The terminal-mark flag also includes sentences ending with a quotation mark after punctuation. All pairs were scored, but exact internal COMET tokenizer truncation was not audited; interpret very long inputs with care.</p>'+image('length_conditioning.png')
body+='<h2>3. Directional asymmetry and data hygiene</h2><p>Training forward/reverse correlation is 0.8993; 18.68% of pairs differ by more than 0.05. Test correlation is 0.8866, with 16.88% above that gap. Similar global means conceal different conditional behavior.</p>'
body+=table(['Split','Repeated pair excess','Repeated EN excess','Repeated VI excess','EN with multiple targets','VI with multiple targets'],duplicates)
body+='<p>Normalization: NFKC + casefold + whitespace collapse. Raw training overlaps 18 test pair rows, 891 test EN source rows and 225 test VI source rows. These counts do not establish leakage in the project’s prepared partitions, which use connected-group removal.</p><p>A conservative copy-like flag identifies 477 training and 3 test pairs (not an error count). The mean training flagged score is 0.55110 EN→VI / 0.55690 VI→EN. Names, codes and legitimate quotations can also be flagged.</p>'
body+='<h2>4. Exact-source comparison with prior model predictions</h2>'+table(['Direction','Matched pairs','Reference','SFT','RL','RL − reference'],matched)
body+='<p>All 1,600 pairs match exact EN/VI text in the official test data. These are subset comparisons, not full-test model results. Higher scores from an evaluator used in training do not independently verify better adequacy or naturalness.</p>'
body+='<h2>5. Reconciliation of the proposed safeguards</h2>'+table(['Measure','Decision','Evidence','Aligned next step'],[[r['measure'],r['decision'],r['finding'],r['next']] for r in alignment])
body+='<h2>6. Training-rollout sensitivity audit</h2>'+table(['Dead band','Fixed scale','Groups suppressed','Mean |coefficient|'],[[r['floor'],r['scale'],f"{100*r['suppressed_fraction']:.2f}%",f"{r['mean_absolute_coefficient']:.5f}"] for r in sensitivity])
body+='<p>Same 20,663 eligible Full-match groups in every row. Historical standardized coefficients average |A|≈0.7474. Neither floor nor scale is calibrated by the corpus statistics; they have separate effects on update frequency and magnitude. Zero floor is an offline control, not currently an accepted training option.</p>'
body+='<h2>7. Review artifacts and boundaries</h2><p>A 240-item training-only directional review worksheet is saved as <code>training_calibration_review.json</code>, stratified across five score bands per direction. Annotation fields are blank; this is not a labeled calibration set. Review low-score idioms, missing context, language-copying cases, good paraphrases, capitalization and contradictory source/target information. Use population weights when estimating aggregate errors from the stratified sample.</p><p>The full test is descriptive evidence only: do not fit reward thresholds or select model hyperparameters to its values. No new training, judge calls or GPU inference was run.</p>'
body+='<p>Sources: <a href="https://aclanthology.org/2021.emnlp-main.369/">PhoMT paper, Table 1 and construction</a>; <a href="https://huggingface.co/Unbabel/wmt22-cometkiwi-da">COMETKiwi model card</a>. Corpus measurements are computed from verified local data and score files.</p></main></html>'
(OUT/'report.html').write_text(body,encoding='utf-8')
print(json.dumps({'report':str(OUT/'report.html'),'training_review_items':len(review),'sensitivity':sensitivity},indent=2))
