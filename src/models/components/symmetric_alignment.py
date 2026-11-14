"""Align the two directional views of each parallel pair and candidate index."""


def paired_candidate_indices(batch: dict[str, list[str]], group_size: int) -> list[int]:
    """Reject incomplete or mismatched pairs before generation or paid scoring."""
    count = len(batch["source"])
    if group_size < 1 or any(
        name not in batch or len(batch[name]) != count
        for name in ("reference", "source_lang", "target_lang", "pair_id")
    ):
        raise ValueError("Symmetric SAC requires complete paired bilingual batches")
    pairs: dict[str, list[int]] = {}
    for index, pair in enumerate(batch["pair_id"]):
        pairs.setdefault(pair, []).append(index)
    partners = [-1] * count
    sources = batch.get("clean_source", batch["source"])
    for indices in pairs.values():
        if len(indices) != 2:
            raise ValueError(
                "Every symmetric pair must have exactly two directional views"
            )
        first, second = indices
        if (
            batch["source_lang"][first] != batch["target_lang"][second]
            or batch["target_lang"][first] != batch["source_lang"][second]
            or batch["source_lang"][first] == batch["target_lang"][first]
            or sources[first] != batch["reference"][second]
            or sources[second] != batch["reference"][first]
        ):
            raise ValueError(
                "Symmetric pair texts/languages must be exact reversed clean views"
            )
        partners[first], partners[second] = second, first
    return [
        partners[row] * group_size + candidate
        for row in range(count)
        for candidate in range(group_size)
    ]
