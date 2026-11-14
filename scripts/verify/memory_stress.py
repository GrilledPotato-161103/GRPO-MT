"""Bounded real-model SFT/RL stress with full token caps and resident COMET."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import lightning as L
import torch


class MemoryStress(L.Callback):
    """Force maximum rollout length and retain the production COMET evaluator."""

    def on_train_start(self, trainer, pl_module):
        scores = pl_module.reward_fn.comet(
            ["The library has many books."], ["Thư viện có nhiều sách."]
        )
        assert all(math.isfinite(x) for x in scores)
        assert pl_module.reward_fn.comet._ready
        assert next(pl_module.reward_fn.comet.model.parameters()).is_cuda
        pl_module.generation_config.min_new_tokens = pl_module.hparams.max_new_tokens
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.shapes = []
        self.start_step = trainer.global_step

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        inputs = pl_module._encode(
            batch['source'], batch['source_lang'], batch['target_lang']
        )
        lengths = inputs['attention_mask'].sum(-1)
        assert bool((lengths == pl_module.hparams.max_source_length).all()), lengths
        self.shapes.append({'samples': len(batch['source']), 'source_tokens': lengths.tolist()})
        if hasattr(pl_module, 'supervised_inputs'):
            target = pl_module.supervised_inputs(batch)['labels']
            assert bool(((target != -100).sum(-1) == pl_module.hparams.max_target_length).all())

    def on_train_end(self, trainer, pl_module):
        report = {
            'start_step': self.start_step,
            'end_step': trainer.global_step,
            'comet_resident_before_backward': True,
            'forced_generation_tokens': pl_module.hparams.max_new_tokens,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'batches': self.shapes,
        }
        Path(trainer.default_root_dir, 'memory_stress.json').write_text(json.dumps(report, indent=2))


def main():
    from omegaconf import OmegaConf
    from scripts.data.prepare_stages import prepare
    from scripts.main.common import load_config, environment
    from scripts.main import pipeline
    from scripts.verify.pipeline import synthetic_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=5)
    args = parser.parse_args()
    config = synthetic_config(args.config, args.output, args.steps)
    cfg = load_config(config, ['wandb.enabled=false', 'data_loader.num_workers=0'])
    data = args.output.resolve() / 'data'
    for name, count in [('train', 80), ('validation', 2), ('test', 2)]:
        with (data / f'{name}.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['en', 'vi'])
            for index in range(count):
                writer.writerow([
                    f'{name} item {index}. ' + 'The library contains books about science and history. ' * 100,
                    f'{name} mục {index}. ' + 'Thư viện có nhiều sách về khoa học và lịch sử. ' * 100,
                ])
    # The short synthetic preparation must be replaced with a matching manifest.
    import shutil
    shutil.rmtree(data / 'stages')
    prepare(data/'train.csv', data/'validation.csv', data/'test.csv', data/'stages')
    cfg.update(start_stage='sft', end_stage='rl', fresh_start=True)
    root = Path(cfg['output_dir'])
    root.mkdir(parents=True)
    (root/'rental_config.json').write_text(json.dumps(cfg, indent=2))
    OmegaConf.save(OmegaConf.create(cfg), config)
    environment(cfg)
    os.environ.update(TRAIN_RUN_GROUP=cfg['run_group'], TRAIN_CACHE_READY_REVISION=cfg['policy_revision'], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    original = pipeline.stage_overrides
    def options(*positional, **keywords):
        return original(*positional, **keywords) + ['+callbacks.memory_stress._target_=scripts.verify.memory_stress.MemoryStress']
    pipeline.stage_overrides = options
    pipeline.run(root, None)
    report = {'status': 'passed', 'steps_per_stage': args.steps, 'stages': {}}
    for stage, phase in [('sft', 'sft/optimizer'), ('rl', 'translation/optimizer')]:
        folder = root / stage
        result = json.loads((folder/'result.json').read_text())
        events = [json.loads(x) for x in (folder/'phase_timing/events.jsonl').read_text().splitlines()]
        updates = [e for e in events if e['phase'] == phase and e.get('succeeded')]
        audit = json.loads((folder/'diagnostics/trainability.json').read_text())
        validation = json.loads((folder/'translation_metrics/val.json').read_text())
        assert result['global_step'] == len(updates) == args.steps
        assert audit['backbone_frozen']
        assert all(math.isfinite(float(x)) for x in result['metrics'].values())
        assert set(validation['directions']) == {'en_vi', 'vi_en'}
        for metrics in validation['directions'].values():
            assert {'bleu', 'chrf', 'comet'} <= metrics.keys()
            assert all(math.isfinite(float(x)) for x in metrics.values())
        checkpoint = torch.load(result['checkpoint'], map_location='cpu', weights_only=False)
        assert checkpoint['global_step'] == args.steps and checkpoint['optimizer_states']
        memory = json.loads((folder/'memory_stress.json').read_text())
        if stage == 'rl':
            generation = [e for e in events if e['phase'] == 'translation/generation']
            assert generation and all(e['output_tokens'] == e['samples'] * cfg['target_cap'] for e in generation)
            diagnostics = [json.loads(x) for x in (folder/'diagnostics/updates.jsonl').read_text().splitlines()]
            assert [x['global_step'] for x in diagnostics] == list(range(1, args.steps+1))
            sac = list((args.output.resolve()/'sac_cache').rglob('*.json'))
            assert sac, 'Real SAC API cache evidence required'
        report['stages'][stage] = {'global_step': result['global_step'], 'optimizer_calls': len(updates), 'backbone_frozen': True, 'validation': validation, 'memory': memory}
    (root/'memory_stress_verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': 'passed', 'sft_updates': args.steps, 'rl_updates': args.steps, 'report': str(root/'memory_stress_verification.json')}), flush=True)


if __name__ == '__main__':
    main()
