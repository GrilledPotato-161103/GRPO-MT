.PHONY: help preview train test check clean package
help:
	@echo "preview train test check clean package"
preview:
	python -m scripts.main.train --dry-run
train:
	python -m scripts.main.train
test:
	python -m pytest tests/grpo -q
check:
	python -m ruff check src scripts tests/grpo
	python -m mypy
clean:
	python -m scripts.tools.clean --apply
package:
	python -m scripts.tools.package --output logs/grpo-mt-source.tar.gz
