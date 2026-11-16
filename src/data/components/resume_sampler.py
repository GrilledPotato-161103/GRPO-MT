"""Skip completed batches while preserving Lightning's nominal epoch length."""

from itertools import islice


class ResumeBatchSampler:
    """Keep restored loop counters aligned with a partially consumed epoch.

    The wrapper deliberately reports the full epoch length: restored Lightning
    batch counters already account for the skipped prefix. Subsequent epochs
    delegate normally. It must not be combined with reset/rebased loop counters.
    """

    def __init__(self, sampler, completed: int, resume_epoch: int = 0):
        if type(completed) is not int or not 0 <= completed < len(sampler):
            raise ValueError("Invalid completed-batch offset")
        self.base, self.completed, self.resume_epoch = sampler, completed, resume_epoch
        self.epoch = resume_epoch
        self.sampler = self

    def __len__(self):
        return len(self.base)

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

    def __iter__(self):
        offset = self.completed if self.epoch == self.resume_epoch else 0
        return islice(iter(self.base), offset, None)
