# Backtrader Options Trading Module

This directory contains comprehensive options trading functionality for Backtrader, including examples, tests, and strategies.

## Features

### Core Options Components

1. **Option Contract (`backtrader.option`)**
   - Option contract representation with Greeks calculation
   - Support for calls, puts, expiration dates, strikes
   - Intrinsic value and moneyness calculations

2. **Options Pricing Models (`backtrader.optionpricing`)**
   - Black-Scholes model with Greeks
   - Binomial tree model
   - Implied volatility calculation

3. **Options Data Feeds (`backtrader.feeds.optiondata`)**
   - Synthetic option data generation
   - Option chain management
   - Historical option data loading

4. **Options Broker (`backtrader.brokers.optionbroker`)**
   - Options-aware broker with margin handling
   - Automatic expiration and assignment
   - Portfolio Greeks tracking

5. **Options Strategy Base (`backtrader.optionstrategy`)**
   - Base class for options strategies
   - Common options trading methods
   - Built-in strategies (spreads, straddles, etc.)

6. **Options Commission (`backtrader.optioncommission`)**
   - Specialized commission schemes for options
   - Different structures for equity/index options

## Examples and Tests

### Basic Testing
```bash
cd c:\src\backtrader
python [options-test.py](http://_vscodecontentref_/0)