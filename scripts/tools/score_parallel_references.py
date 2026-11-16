"""Score existing EN/VI pairs with reference-free COMETKiwi, in both directions.

The parallel target is the hypothesis (`mt`), never COMET's `ref` input.
Chunk files permit exact resume with immutable input/model fingerprints.
"""
import argparse
import csv
import hashlib
import heapq
import json
from pathlib import Path
import time


def fingerprint(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def csv_pairs(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'en', 'vi'}.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: CSV requires en and vi columns')
        for index, row in enumerate(reader):
            if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ('en', 'vi')):
                raise ValueError(f'{path}: invalid parallel pair {index}')
            yield index, row['en'], row['vi']


def chunks(iterator, size):
    chunk = []
    for item in iterator:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-csv', type=Path, required=True)
    parser.add_argument('--test-csv', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--chunk-size', type=int, default=2048)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.batch_size < 1 or args.chunk_size < 1:
        parser.error('batch and chunk sizes must be positive')
    import numpy as np
    import torch
    from comet import load_from_checkpoint
    from src.models.components.comet_reward import CometScorer

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'purpose':'parallel_reference_quality_not_model_generated_translation',
        'model':'Unbabel/wmt22-cometkiwi-da', 'checkpoint_sha256':fingerprint(args.checkpoint),
        'inputs':{s:{'path':str(p.resolve()),'sha256':fingerprint(p)} for s,p in
                  (('train',args.train_csv),('test',args.test_csv))},
        'chunk_size':args.chunk_size, 'dtype':'float32', 'torch_version':torch.__version__,
        'device':args.device, 'batch_size':args.batch_size,
        'directions':{'en_vi':{'src':'en','mt':'vi'},'vi_en':{'src':'vi','mt':'en'}},
        'score_scale':'raw COMETKiwi; not accuracy or a calibrated probability'}
    manifest_path = args.output/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Existing evaluation has incompatible data/model/settings')
    atomic_json(manifest_path, manifest)
    model = load_from_checkpoint(str(args.checkpoint))
    scorer = CometScorer(model=model, device=args.device, batch_size=args.batch_size)
    smoke_sources = ['This is a book.', 'This is a book.',
                     'Đây là một cuốn sách.', 'Đây là một cuốn sách.']
    smoke_targets = ['Đây là một cuốn sách.', 'Trời đang mưa.',
                     'This is a book.', 'It is raining.']
    smoke_scores = scorer(smoke_sources, smoke_targets)
    if not (smoke_scores[0] > smoke_scores[1] and smoke_scores[2] > smoke_scores[3]):
        raise RuntimeError('Bilingual COMET quality-ordering check failed')
    atomic_json(args.output/'smoke.json', {'scores':smoke_scores,
                'correct_beats_unrelated_both_directions':True})
    started = time.time()
    report = {'manifest':manifest, 'splits':{}}
    # Finish the small official test split first to provide an early baseline.
    for split, path in (('test',args.test_csv),('train',args.train_csv)):
        directory = args.output/split
        directory.mkdir(exist_ok=True)
        all_scores = []
        lowest = {'en_vi':[], 'vi_en':[]}
        count = 0
        running_sum = np.zeros(2, dtype=np.float64)
        export_tmp = directory/'scores.csv.tmp'
        with export_tmp.open('w',encoding='utf-8',newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['pair_index','en_vi_cometkiwi','vi_en_cometkiwi'])
            for part, pairs in enumerate(chunks(csv_pairs(path),args.chunk_size)):
                cache = directory/f'chunk_{part:06d}.npz'
                if cache.exists():
                    with np.load(cache,allow_pickle=False) as saved:
                        indices, values = saved['indices'], saved['scores']
                    if indices.tolist() != [r[0] for r in pairs] or values.shape != (len(pairs),2):
                        raise ValueError('Invalid cached chunk')
                else:
                    sources, targets = [], []
                    for _,en,vi in pairs:
                        sources.extend((en,vi))
                        targets.extend((vi,en))
                    values = np.array(scorer(sources,targets),dtype=np.float64).reshape(-1,2)
                    indices = np.array([r[0] for r in pairs], dtype=np.int64)
                    with cache.with_suffix('.tmp').open('wb') as stream:
                        np.savez_compressed(stream,indices=indices,scores=values)
                    cache.with_suffix('.tmp').replace(cache)
                if not np.isfinite(values).all():
                    raise ValueError('Non-finite COMET scores')
                for (index,en,vi), (forward,reverse) in zip(pairs,values,strict=True):
                    writer.writerow([index,float(forward),float(reverse)])
                    for direction,score in (('en_vi',forward),('vi_en',reverse)):
                        heap = lowest[direction]
                        entry = (-float(score),index,en,vi)
                        if len(heap)<100:
                            heapq.heappush(heap,entry)
                        elif entry>heap[0]:
                            heapq.heapreplace(heap,entry)
                all_scores.append(values)
                running_sum += values.sum(axis=0)
                count += len(pairs)
                handle.flush()
                progress = {'split':split,'pairs_complete':count,'elapsed_seconds':time.time()-started,
                            'status':'scoring','running_mean':(running_sum/count).tolist()}
                atomic_json(args.output/'progress.json',progress)
                print(json.dumps(progress),flush=True)
        export_tmp.replace(directory/'scores.csv')
        values = np.concatenate(all_scores)
        summary = {'parallel_pairs':count,'directions':{}}
        for column,direction in enumerate(('en_vi','vi_en')):
            scores = values[:,column]
            summary['directions'][direction] = {'mean':float(scores.mean()),'std':float(scores.std()),
                'min':float(scores.min()),'max':float(scores.max()),
                'percentiles':dict(zip(('p01','p05','p10','p25','p50','p75','p90','p95','p99'),
                                      np.quantile(scores,[.01,.05,.10,.25,.50,.75,.90,.95,.99]).tolist()))}
            atomic_json(directory/f'{direction}_lowest100.json',[
                {'pair_index':i,'score':-neg,'en':en,'vi':vi}
                for neg,i,en,vi in sorted(lowest[direction],reverse=True)])
        atomic_json(directory/'summary.json',summary)
        report['splits'][split] = summary
        atomic_json(args.output/'summary.json',report)
    atomic_json(args.output/'progress.json',{'status':'complete','elapsed_seconds':time.time()-started})


if __name__ == '__main__':
    main()
