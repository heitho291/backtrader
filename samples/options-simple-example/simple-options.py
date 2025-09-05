#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# Simple Options Trading Example
#
# Demonstrates basic options functionality in Backtrader
#
###############################################################################
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime
import sys
import os

# Add the backtrader directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import backtrader as bt
from backtrader.option import OptionContract
from backtrader.optionpricing import BlackScholesModel
from backtrader.feeds.optiondata import SyntheticOptionData
from backtrader.brokers.optionbroker import OptionBroker
from backtrader.optionstrategy import OptionStrategy
from backtrader.optioncommission import EquityOptionCommissionInfo


class SimpleOptionsStrategy(OptionStrategy):
    '''Simple options strategy - buy calls when RSI < 30'''
    
    params = (
        ('rsi_period', 14),
        ('rsi_low', 30),
        ('rsi_high', 70),
        ('debug', True),
    )
    
    def __init__(self):
        # Add RSI indicator
        self.rsi = bt.indicators.RSI(self.datas[0], period=self.p.rsi_period)
        self.option_position = None
        
        if self.p.debug:
            print("Simple Options Strategy initialized")
    
    def log(self, txt, dt=None):
        '''Logging function'''
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')
    
    def next(self):
        # Wait for RSI to be calculated
        if len(self.rsi) == 0:
            return
        
        current_rsi = self.rsi[0]
        stock_price = self.datas[0].close[0]
        
        # Buy call when RSI is oversold and we don't have a position
        if (current_rsi < self.p.rsi_low and 
            self.option_position is None and 
            len(self.datas) > 1):
            
            self.option_position = self.buy_call(data=self.datas[1], size=1)
            
            if self.p.debug:
                self.log(f"BUYING Call: RSI={current_rsi:.2f}, Stock=${stock_price:.2f}")
        
        # Sell call when RSI is overbought and we have a position
        elif (current_rsi > self.p.rsi_high and 
              self.option_position is not None):
            
            option_pos = self.getposition(self.datas[1])
            if option_pos.size > 0:
                self.sell_call(data=self.datas[1], size=option_pos.size)
                
                if self.p.debug:
                    self.log(f"SELLING Call: RSI={current_rsi:.2f}, Stock=${stock_price:.2f}")
                
                self.option_position = None
    
    def notify_order(self, order):
        if order.status == order.Completed:
            action = 'BUY' if order.isbuy() else 'SELL'
            if self.p.debug:
                self.log(f'ORDER {action}: {order.executed.size} @ ${order.executed.price:.4f}')
    
    def notify_trade(self, trade):
        if trade.isclosed and self.p.debug:
            self.log(f'TRADE PnL: ${trade.pnl:.2f}')


def run_simple_options_example():
    '''Run the simple options example'''
    print("Simple Options Trading Example")
    print("=" * 40)
    
    # Create Cerebro
    cerebro = bt.Cerebro()
    
    # Load stock data
    data_path = os.path.join(os.path.dirname(__file__), 
                            '..', '..', 'datas', '2006-day-001.txt')
    
    if os.path.exists(data_path):
        stock_data = bt.feeds.BacktraderCSVData(dataname=data_path)
        cerebro.adddata(stock_data, name='STOCK')
        
        # Create call option data
        call_option = SyntheticOptionData(
            symbol='STOCK',
            expiry=datetime.date(2006, 3, 17),
            strike=105.0,
            option_type='call',
            underlying_data=stock_data,
            volatility=0.30
        )
        
        cerebro.adddata(call_option, name='CALL_105')
        
        # Set up options broker
        cerebro.broker = OptionBroker()
        cerebro.broker.setcash(25000)
        
        # Add commission
        option_comm = EquityOptionCommissionInfo(commission=1.0)
        cerebro.broker.addcommissioninfo(option_comm, name='CALL_105')
        
        # Add strategy
        cerebro.addstrategy(SimpleOptionsStrategy, debug=True)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        print(f"Starting value: ${cerebro.broker.getvalue():.2f}")
        
        # Run backtest
        results = cerebro.run()
        
        print(f"Final value: ${cerebro.broker.getvalue():.2f}")
        
        # Print results
        strategy = results[0]
        
        trade_analysis = strategy.analyzers.trades.get_analysis()
        if trade_analysis.get('total', {}).get('total', 0) > 0:
            print(f"Total trades: {trade_analysis['total']['total']}")
            print(f"Winning trades: {trade_analysis.get('won', {}).get('total', 0)}")
            
        returns = strategy.analyzers.returns.get_analysis()
        if returns:
            print(f"Total return: {returns.get('rtot', 0) * 100:.2f}%")
    
    else:
        print("Error: Sample data file not found")
        print(f"Looking for: {data_path}")


if __name__ == '__main__':
    run_simple_options_example()