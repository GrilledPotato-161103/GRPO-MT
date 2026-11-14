#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="grpo-mt",
    version="0.1.0",
    description="Hydra and Lightning GRPO for machine translation",
    python_requires=">=3.11",
    install_requires=["torch>=2.2", "lightning>=2.2,<3", "hydra-core==1.3.2",
                      "transformers>=4.45,<5", "datasets>=3,<5", "sacrebleu>=2.4,<3",
                      "sentencepiece>=0.2", "httpx>=0.27,<1", "rootutils", "rich", "hydra-colorlog==1.2.0"],
    packages=find_packages(exclude=("tests*", "archive*")),
    include_package_data=True,
    package_data={"configs": ["**/*.yaml", "**/*.json"]},
    extras_require={
        "lora": ["peft>=0.14,<0.19", "accelerate>=1,<2"],
        "qlora": ["peft>=0.14,<0.19", "accelerate>=1,<2", "bitsandbytes>=0.45,<1"],
        "comet": ["unbabel-comet>=2.2.7,<2.3", "setuptools>=65,<81"],
        "logging": ["wandb>=0.24,<1"],
        "dev": ["pytest>=7", "pytest-cov>=4", "ruff", "mypy"],
    },
    entry_points={
        "console_scripts": [
            "grpo-train = scripts.main.train:main",
            "grpo-eval = scripts.main.eval:main",
        ]
    },
)
