.PHONY: verify-types test
verify-types:
	@echo "Running type checks..."
	@uvx pyrefly check

test:
	@echo "Running tests..."
	@uv sync --extra dev --extra plotting
	@uv add --dev pytest pytest-cov
	@uv run python -m pytest --ignore=tests/test_analyzer-timereturn.py -v