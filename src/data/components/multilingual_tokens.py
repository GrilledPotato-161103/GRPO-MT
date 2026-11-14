"""Direction-aware NLLB language tokens, shared by filtering, SFT and generation."""

from typing import Any


def language_token_ids(
    tokenizer: Any, names: list[str], codes: dict[str, str]
) -> list[int]:
    result = []
    for name in names:
        if name not in codes:
            raise ValueError(f"No translation language code configured for {name}")
        value = tokenizer.convert_tokens_to_ids(codes[name])
        if value is None or value == tokenizer.unk_token_id:
            raise ValueError(f"Tokenizer does not contain language token {codes[name]}")
        result.append(int(value))
    return result


def multilingual_ids(
    tokenizer: Any,
    texts: list[str],
    source_langs: list[str],
    target_langs: list[str],
    codes: dict[str, str],
    *,
    targets: bool = False,
    truncation: bool = False,
    max_length: int | None = None,
) -> list[list[int]]:
    """Tokenize language-homogeneous chunks and restore the mixed batch order.

    Each sequence retains its own language prefix and EOS. The mutable tokenizer
    language settings are restored, including on an exception.
    """
    if not len(texts) == len(source_langs) == len(target_langs):
        raise ValueError("Every multilingual text needs both language labels")
    desired = language_token_ids(
        tokenizer, target_langs if targets else source_langs, codes
    )
    language_token_ids(tokenizer, target_langs if not targets else source_langs, codes)
    groups: dict[tuple[str, str], list[int]] = {}
    for index, pair in enumerate(zip(source_langs, target_langs, strict=True)):
        groups.setdefault(pair, []).append(index)
    result: list[list[int]] = [[] for _ in texts]
    old_source, old_target = tokenizer.src_lang, tokenizer.tgt_lang
    try:
        for (source, target), indices in groups.items():
            tokenizer.src_lang, tokenizer.tgt_lang = codes[source], codes[target]
            options: dict[str, Any] = {
                "padding": False,
                "truncation": truncation,
                "return_attention_mask": False,
                "return_token_type_ids": False,
            }
            if truncation and max_length is not None:
                options["max_length"] = max_length
            batch = [texts[index] for index in indices]
            encoded = (
                tokenizer(text_target=batch, **options)
                if targets
                else tokenizer(batch, **options)
            )
            for index, ids in zip(indices, encoded["input_ids"], strict=True):
                if not ids or ids[0] != desired[index]:
                    raise ValueError(
                        "NLLB requires prefix language tokens; use legacy_behaviour=False"
                    )
                result[index] = list(ids)
    finally:
        tokenizer.src_lang, tokenizer.tgt_lang = old_source, old_target
    return result
