"""Create tiny, random Transformers models locally, without Hub downloads."""

import argparse
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    T5Config,
    T5ForConditionalGeneration,
)


def create_assets(output: Path, architecture: str = "seq2seq") -> Path:
    """Write a tiny HF policy and tokenizer to a loadable directory."""
    output.mkdir(parents=True, exist_ok=True)
    words = [
        "[PAD]",
        "[EOS]",
        "[UNK]",
        "Translate",
        "English",
        "to",
        "Vietnamese",
        ":",
        "Hello",
        "Thank",
        "you",
        "Good",
        "morning",
        "Xin",
        "chào",
        "Cảm",
        "ơn",
        ".",
    ]
    backend = Tokenizer(
        WordLevel(dict(zip(words, range(len(words)), strict=True)), unk_token="[UNK]")
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )  # type: ignore[no-untyped-call]
    model: Any
    if architecture == "seq2seq":
        model = T5ForConditionalGeneration(
            T5Config(  # type: ignore[no-untyped-call]
                vocab_size=len(words),
                d_model=16,
                d_ff=32,
                num_layers=1,
                num_decoder_layers=1,
                num_heads=2,
                decoder_start_token_id=0,
                pad_token_id=0,
                eos_token_id=1,
            )
        )
    elif architecture == "causal":
        model = GPT2LMHeadModel(
            GPT2Config(  # type: ignore[no-untyped-call]
                vocab_size=len(words),
                n_embd=16,
                n_layer=1,
                n_head=2,
                n_positions=128,
                bos_token_id=1,
                eos_token_id=1,
                pad_token_id=0,
            )
        )
    else:
        raise ValueError("architecture must be seq2seq or causal")
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    return output


def main() -> None:
    """Create assets from command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--architecture", choices=["seq2seq", "causal"], default="seq2seq"
    )
    args = parser.parse_args()
    create_assets(args.output, args.architecture)


if __name__ == "__main__":
    main()
