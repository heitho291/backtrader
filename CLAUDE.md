# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **backtrader**, a Python-based backtesting and live trading platform for algorithmic trading strategies. It's a mature open-source project (GPLv3+) that supports Python 3.13+ and provides comprehensive trading simulation capabilities with multiple data feeds and broker integrations.

## Core Architecture

### Central Components
- **Cerebro** (`backtrader/cerebro.py`): The main engine that orchestrates strategies, data feeds, brokers, and analysis
- **Strategy** (`backtrader/strategy.py`): Base class for trading strategies with lifecycle methods (`next()`, `start()`, `stop()`)
- **Data Feeds** (`backtrader/feeds/`): Connectors for various data sources (CSV, Yahoo, Interactive Brokers, Oanda, etc.)
- **Brokers** (`backtrader/brokers/`): Trading execution simulation and live broker integration
- **Indicators** (`backtrader/indicators/`): 122+ built-in technical indicators with extensible framework

### Key Modules Structure
- `backtrader/` - Core library code
- `samples/` - Example strategies and usage patterns
- `tests/` - Unit tests using Python's unittest framework
- `tools/` - Command-line utilities including `bt-run.py`
- `datas/` - Sample CSV data files for testing

### Data Flow Architecture
1. **Data Feeds** → **Cerebro** → **Strategy**
2. **Strategy** generates **Orders** → **Broker** executes
3. **Analyzers** and **Observers** track performance metrics
4. **Writers** output results, **Plotting** visualizes data

## Development Commands

### Installation & Setup (using uv - recommended)
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup development environment
git clone <repository-url>
cd backtrader

# Install in development mode with all dependencies
uv sync --extra dev --extra all

# Install specific extras only
uv sync --extra dev --extra plotting
```

### Alternative Installation (using pip)
```bash
# Install from source (development mode)
pip install -e .

# Install with plotting support
pip install -e .[plotting]

# Install development dependencies
pip install -r requirements-dev.txt
```

### Testing
```bash
# Using uv (recommended)
uv run pytest

# Run with coverage
uv run pytest --cov=backtrader --cov-report=html

# Run specific test file
uv run pytest tests/test_strategy_unoptimized.py -v

# Using traditional unittest
cd tests && python -m unittest discover -v
```

### Code Quality
```bash
# Format code with black
uv run black .

# Sort imports with isort
uv run isort .

# Run linter
uv run flake8 .

# Type checking
uv run mypy backtrader/
```

### Running Examples
```bash
# Use the btrun command-line tool
uv run python tools/bt-run.py --help

# Or use the installed command (after uv sync)
btrun --help

# Run sample strategies
uv run python samples/sma-crossover/sma-crossover.py
uv run python samples/optimization/optimization.py
```

### Package Building
```bash
# Build distribution packages with uv
uv build

# Alternative with build tool
uv run python -m build

# Upload to PyPI (maintainers only)
uv run twine upload dist/*
```

## Code Patterns & Conventions

### Strategy Development
- Inherit from `bt.Strategy` or `bt.SignalStrategy`
- Implement `__init__()` for indicator setup, `next()` for trading logic
- Use `self.buy()`, `self.sell()`, `self.close()` for order management
- Access data via `self.data[0]` (close), `self.data.high[0]`, etc.

### Indicator Creation
- Inherit from `bt.Indicator`
- Define `lines` tuple for output lines
- Implement `next()` method for calculations
- Use `period` parameter for lookback requirements

### Data Feed Integration
- Inherit from appropriate feed class in `backtrader/feeds/`
- Implement required methods for data parsing/fetching
- Handle timeframe conversions and resampling

### Testing Patterns
- Tests use `testcommon.py` for shared utilities
- Each test file targets specific functionality (indicators, strategies, etc.)
- Tests verify mathematical accuracy against known reference values

## Live Trading Integration

The platform supports live trading through:
- **Interactive Brokers** (via IbPy)
- **Oanda** (REST API)
- **Visual Chart** (Windows-specific)

Broker-specific stores (`backtrader/stores/`) handle connection management and data streaming.

## Performance Considerations

- Use `cerebro.optreturn=False` to return full objects instead of just parameters
- Enable `runonce=True` for vectorized indicator calculations when possible
- Consider memory usage with `cerebro.adddata()` vs `cerebro.resampledata()`
- Plotting requires matplotlib and can be memory-intensive for large datasets