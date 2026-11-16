"""Export and structurally validate the ten documented research prompt proposals."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
document = ROOT/'docs/research/sac_grading_engines_20261115.md'
destination = document.with_name('sac_grading_prompts_20261115.json')
text = document.read_text(encoding='utf-8')
common = re.search(r'### Common system policy\s+```text\n(.*?)\n```',text,re.S).group(1)
payload_region = text.split('Default source-conditioned payload',1)[1].split('Use a research envelope',1)[0]
example = json.loads(re.search(r'```json\n(.*?)\n```',payload_region,re.S).group(1))
matches = list(re.finditer(r'^## (\d+)\. (.+)$',text,re.M))
assert [int(m.group(1)) for m in matches] == list(range(1,11))
proposals = []
for i,match in enumerate(matches):
    section = text[match.end():matches[i+1].start() if i+1<len(matches) else len(text)]
    prompts = [p for p in re.findall(r'```text\n(.*?)\n```',section,re.S) if p.startswith('ROLE: ')]
    assert prompts, 'Each proposal must include at least one concrete stage prompt'
    for label in ('**Engine.**','**Scoring/use.**','**Fit and risk.**','**Discriminating test.**'):
        assert label in section, f'Proposal {i+1} lacks {label}'
    proposals.append({'id':int(match.group(1)),'title':match.group(2),
        'stages':[{'role':p.splitlines()[0].removeprefix('ROLE: '),
                   'system_prompt':common+'\n\n'+p} for p in prompts]})
roles = [s['role'] for p in proposals for s in p['stages']]
assert len(roles)==len(set(roles))==16
pack={'format_version':'sac-research-prompts-v1','source_document':document.relative_to(ROOT).as_posix(),
      'runtime_compatible':False,
      'notice':'Research templates, not calibrated or deployed graders. New schemas, validators, routing and per-stage payload filters are required.',
      'payload_policy':'Filter JSON user inputs to the stage-specific fields declared in each prompt. The shared example is not valid for source-blind or pairwise stages. Do not send references, policy identities or prior quality scores unless a role explicitly requires anonymized assessment records.',
      'shared_example_input':example,'common_system_policy':common,'proposals':proposals}
destination.write_text(json.dumps(pack,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
# Verify the serialized artifact and local document links without network calls.
assert len(json.loads(destination.read_text(encoding='utf-8'))['proposals'])==10
assert text.count('```')%2==0
for target in re.findall(r'\]\(([^)]+)\)',text):
    if not target.startswith(('http://','https://','#')):
        assert (document.parent/target.split('#',1)[0]).resolve().exists(), target
print(json.dumps({'proposals':10,'stage_prompts':len(roles),'json_valid':True,
                  'local_links_valid':True,'destination':str(destination)},indent=2))
