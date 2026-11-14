"""Rebatch remaining bilingual views without replaying consumed examples."""

import csv
import json

import pytest

from src.data.translation_datamodule import TranslationDataModule


def test_resume_batch_plan_preserves_remaining_pairs(tmp_path):
    source = tmp_path / 'pairs.csv'
    with source.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['en', 'vi'])
        writer.writerows([(f'English {i}', f'Vietnamese {i}') for i in range(8)])
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'batches': [[8, 9, 2, 3], [12, 13]]}))
    data = TranslationDataModule(train_file=str(source), val_file=str(source),
        source_column='en', target_column='vi', bidirectional=True,
        paired_batches=True, batch_size=4, num_workers=0,
        resume_batch_plan=str(plan))
    data.setup('fit')
    batches = list(data.train_dataloader())
    assert [b['source'] for b in batches] == [
        ['English 4', 'Vietnamese 4', 'English 1', 'Vietnamese 1'],
        ['English 6', 'Vietnamese 6'],
    ]
    plan.write_text(json.dumps({'batches': [[8, 11]]}))
    with pytest.raises(ValueError, match='bilingual pairs'):
        data.train_dataloader()
