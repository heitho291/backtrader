#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# Sample Options Strategy - Covered Call Example
#
# This sample demonstrates how to use the new options functionality in
# Backtrader to implement a covered call strategy.
#
###############################################################################
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime
import argparse

import backtrader as bt
from backtrader.feeds.optiondata import SyntheticOptionData, OptionChain
from backtrader.optionstrategy import OptionStrategy
from backtrader.option import OptionContract
from backtrader.brokers.optionbroker import OptionBroker


class CoveredCallStrategy(OptionStrategy):
    '''
    A simple covered call strategy that:
    1. Buys the underlying stock
    2. Sells call options against the position
    3. Manages the positions based on time decay and profit targets
    '''
    
    params = (
        ('strike_pct', 1.05),      # Sell calls 5% OTM
        ('dte_entry', 30),         # Enter calls with 30 DTE
        ('dte_exit', 7),           # Exit calls with 7 DTE remaining
        ('profit_target', 0.5),    # Close at 50% profit
        ('stock_quantity', 100),   # Number of shares per covered call
    )
    
    def __init__(self):
        super(CoveredCallStrategy, self).__init__()
        
        # Assume first data feed is the underlying stock
        self.stock_data = self.datas[0]
        
        # Track our options positions
        self.call_data = None
        self.call_entry_price = 0.0
        self.call_entry_date = None
        self.stock_position_size = 0
        
        # We'll need to find option data feeds for our strikes
        self.available_calls = []
        
        # Identify option data feeds
        for data in self.datas[1:]:  # Skip first (stock) data
            if (hasattr(data, 'contract') and 
                data.contract.is_call() and
                data.contract.p.symbol == self.stock_data._name):
                self.available_calls.append(data)
        
        print(f"Found {len(self.available_calls)} call option data feeds")
    
    def next(self):
        current_date = self.datetime.date(0)
        stock_price = self.stock_data.close[0]
        
        # Step 1: Ensure we own the underlying stock
        current_stock_pos = self.getposition(self.stock_data)
        if current_stock_pos.size < self.p.stock_quantity:
            shares_to_buy = self.p.stock_quantity - current_stock_pos.size
            print(f'{current_date}: Buying {shares_to_buy} shares at ${stock_price:.2f}')
            self.buy(data=self.stock_data, size=shares_to_buy)
            self.stock_position_size = self.p.stock_quantity
        
        # Step 2: Manage covered call position
        if self.call_data is None:
            # Look for a new call to sell
            self._enter_covered_call(stock_price, current_date)
        else:
            # Check if we should close existing call
            self._manage_covered_call(current_date)
    
    def _enter_covered_call(self, stock_price, current_date):
        '''Enter a new covered call position'''
        target_strike = stock_price * self.p.strike_pct
        
        # Find the best call option to sell
        best_call = self._find_best_call(target_strike, current_date)
        
        if best_call:
            self.call_data = best_call
            call_price = best_call.close[0]
            
            print(f'{current_date}: Selling call {best_call.contract.p.strike} '
                  f'strike for ${call_price:.2f}')
            
            # Sell the call option
            order = self.sell_call(data=best_call, size=1)
            self.call_entry_price = call_price
            self.call_entry_date = current_date
    
    def _manage_covered_call(self, current_date):
        '''Manage existing covered call position'''
        if not self.call_data:
            return
        
        call_position = self.getposition(self.call_data)
        if call_position.size >= 0:  # No short position
            self.call_data = None
            return
        
        # Check time-based exit
        days_to_expiry = self.call_data.contract.days_to_expiry(current_date)
        if days_to_expiry <= self.p.dte_exit:
            print(f'{current_date}: Closing call due to {days_to_expiry} DTE remaining')
            self.buy_call(data=self.call_data, size=1)  # Buy to close
            self.call_data = None
            return
        
        # Check profit target
        current_call_price = self.call_data.close[0]
        if current_call_price <= self.call_entry_price * (1 - self.p.profit_target):
            profit = (self.call_entry_price - current_call_price) / self.call_entry_price
            print(f'{current_date}: Closing call at {profit:.1%} profit')
            self.buy_call(data=self.call_data, size=1)  # Buy to close
            self.call_data = None
            return
    
    def _find_best_call(self, target_strike, current_date):
        '''Find the best call option to sell'''
        best_call = None
        best_score = 0
        
        for call_data in self.available_calls:
            # Check if this call is suitable
            contract = call_data.contract
            days_to_expiry = contract.days_to_expiry(current_date)
            
            # Skip if too close to expiry or too far out
            if days_to_expiry < 20 or days_to_expiry > 60:
                continue
            
            # Skip if strike is too far from target
            strike_diff = abs(contract.p.strike - target_strike)
            if strike_diff > target_strike * 0.1:  # Within 10%
                continue
            
            # Calculate a score based on DTE and strike proximity
            dte_score = max(0, 1 - abs(days_to_expiry - self.p.dte_entry) / 30)
            strike_score = max(0, 1 - strike_diff / (target_strike * 0.05))
            total_score = dte_score * strike_score
            
            if total_score > best_score:
                best_score = total_score
                best_call = call_data
        
        return best_call
    
    def notify_order(self, order):
        '''Order notification handler'''
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                action = 'BUY'
            else:
                action = 'SELL'
            
            print(f'ORDER COMPLETED: {action} {order.executed.size} '
                  f'@ ${order.executed.price:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f'ORDER FAILED: {order.status}')
    
    def notify_trade(self, trade):
        '''Trade notification handler'''
        if not trade.isclosed:
            return
        
        print(f'TRADE CLOSED: PnL ${trade.pnl:.2f}, Commission ${trade.commission:.2f}')
    
    def stop(self):
        '''Called at the end of the strategy'''
        print('Strategy completed')
        print(f'Final portfolio value: ${self.broker.getvalue():.2f}')
        
        # Print portfolio Greeks if available
        if hasattr(self.broker, 'get_portfolio_greeks'):
            greeks = self.broker.get_portfolio_greeks()
            print(f'Portfolio Greeks: {greeks}')


def runstrat(args=None):
    args = parse_args(args)
    
    cerebro = bt.Cerebro()
    
    # Add the options-aware broker
    cerebro.broker = OptionBroker()
    cerebro.broker.setcash(args.cash)
    cerebro.broker.setcommission(commission=args.commission)
    
    # Load stock data
    print(f'Loading stock data from: {args.data}')
    stock_data = bt.feeds.YahooFinanceCSVData(
        dataname=args.data,
        fromdate=datetime.datetime.strptime(args.fromdate, '%Y-%m-%d'),
        todate=datetime.datetime.strptime(args.todate, '%Y-%m-%d')
    )
    cerebro.adddata(stock_data, name='STOCK')
    
    # Create synthetic option data feeds
    print('Creating synthetic option data feeds...')
    
    # Create options for different strikes and expirations
    base_date = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')
    
    # Generate monthly expirations for the next year
    expiry_dates = []
    for i in range(12):
        expiry_month = base_date.month + i
        expiry_year = base_date.year + (expiry_month - 1) // 12
        expiry_month = ((expiry_month - 1) % 12) + 1
        
        # Third Friday of the month (simplified)
        expiry_date = datetime.date(expiry_year, expiry_month, 15)
        expiry_dates.append(expiry_date)
    
    # Generate strikes around current price (rough estimate)
    base_price = 100  # We'll adjust this dynamically
    strikes = []
    for i in range(-10, 11):  # 21 strikes
        strike = base_price + (i * 5)  # $5 increments
        if strike > 0:
            strikes.append(strike)
    
    # Create call option data feeds
    option_count = 0
    for expiry in expiry_dates[:6]:  # Only first 6 months
        for strike in strikes:
            # Create synthetic call option data
            call_data = SyntheticOptionData(
                symbol='STOCK',
                expiry=expiry,
                strike=strike,
                option_type='call',
                underlying_data=stock_data,
                volatility=args.volatility,
                risk_free_rate=args.risk_free_rate
            )
            
            option_name = f'CALL_{expiry.strftime("%y%m%d")}_{strike}'
            cerebro.adddata(call_data, name=option_name)
            option_count += 1
    
    print(f'Created {option_count} synthetic call options')
    
    # Add the strategy
    cerebro.addstrategy(CoveredCallStrategy)
    
    # Run the backtest
    print('Starting backtest...')
    result = cerebro.run()
    
    # Plot if requested
    if args.plot:
        cerebro.plot(style='candlestick', volume=False)


def parse_args(pargs=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Options Covered Call Strategy')
    
    parser.add_argument('--data', required=False,
                        default='../../datas/orcl-1995-2014.txt',
                        help='Data file to read from')
    
    parser.add_argument('--fromdate', required=False, default='2005-01-01',
                        help='Starting date in YYYY-MM-DD format')
    
    parser.add_argument('--todate', required=False, default='2006-12-31',
                        help='Ending date in YYYY-MM-DD format')
    
    parser.add_argument('--cash', default=10000.0, type=float,
                        help='Starting cash')
    
    parser.add_argument('--commission', default=0.001, type=float,
                        help='Commission factor')
    
    parser.add_argument('--volatility', default=0.25, type=float,
                        help='Volatility for option pricing')
    
    parser.add_argument('--risk-free-rate', default=0.02, type=float,
                        help='Risk-free rate for option pricing')
    
    parser.add_argument('--plot', action='store_true',
                        help='Plot the results')
    
    return parser.parse_args(pargs)


if __name__ == '__main__':
    runstrat()
