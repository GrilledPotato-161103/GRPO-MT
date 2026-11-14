"""Shared translation instructions and native chat rendering, without truncation."""

from typing import Any

PRECISE_TRANSLATION_PROMPT = (
    "Translate {source_lang} to {target_lang} precisely. "
    "Preserve the meaning, names, numbers, and negation. "
    "Return only the translation.\nSource: {source}"
)


def translation_prompts(
    tokenizer: Any,
    sources: list[str],
    source_langs: list[str],
    target_langs: list[str],
    prompt_template: str,
    use_chat_template: bool = False,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[str]:
    """Let each instruction model supply its own role/thinking delimiters."""
    if not len(sources) == len(source_langs) == len(target_langs):
        raise ValueError("Every translation source needs both language labels")
    content = [
        prompt_template.format(source=text, source_lang=src, target_lang=tgt)
        for text, src, tgt in zip(sources, source_langs, target_langs, strict=True)
    ]
    if not use_chat_template:
        return content
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Native chat prompting requires a tokenizer chat_template")
    options = dict(chat_template_kwargs or {})
    if {"tokenize", "add_generation_prompt", "continue_final_message"} & options.keys():
        raise ValueError("Chat options cannot override generation-prompt boundaries")
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            **options,
        )
        for text in content
    ]


def prompt_overhead(
    tokenizer: Any,
    prompt_template: str,
    use_chat_template: bool,
    chat_template_kwargs: dict[str, Any] | None = None,
    languages: tuple[str, str] = ("English", "Vietnamese"),
) -> int:
    """Reserve the larger complete empty-source prompt across both directions."""
    prompts = translation_prompts(
        tokenizer,
        ["", ""],
        list(languages),
        list(reversed(languages)),
        prompt_template,
        use_chat_template,
        chat_template_kwargs,
    )
    tokens = tokenizer(
        prompts, add_special_tokens=not use_chat_template, truncation=False
    )["input_ids"]
    return max(map(len, tokens))
