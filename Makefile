.PHONY: setup bootstrap check test all llm-live deps-audit

setup bootstrap:
	uv sync --extra dev

check:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy src
	uv run bandit -q -r src
	uv run detect-secrets scan --exclude-files '(^\\.git/|^\\.beads/|^\\.venv/|^\\.mypy_cache/|^\\.ruff_cache/|^\\.pytest_cache/)' > /tmp/structured_jepa_detect_secrets.json

test:
	uv run pytest

all:
	$(MAKE) check
	$(MAKE) test

llm-live:
	@echo "No LLM live tests in this project."

deps-audit:
	uv run pip-audit
