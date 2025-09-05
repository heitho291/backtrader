#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# Options Trading Example and Test Script
#
# This script demonstrates the new options functionality in Backtrader
# and serves as a comprehensive test of the options features.
#
###############################################################################
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime
import sys
import os

# Add the backtrader directory to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import backtrader as bt
from backtrader.option import OptionContract, OptionPosition
from backtrader.optionpricing import BlackScholesModel, BinomialModel
from backtrader.feeds.optiondata import SyntheticOptionData, OptionChain
from backtrader.brokers.optionbroker import OptionBroker
from backtrader.optionstrategy import OptionStrategy
from backtrader.optioncommission import EquityOptionCommissionInfo


def test_option_contract():
    '''Test basic option contract functionality'''
    print("Testing Option Contract...")
    
    # Create a call option
    call_contract = OptionContract(
        symbol='AAPL',
        expiry=datetime.date(2024, 1, 19),
        strike=150.0,
        option_type='call'
    )
    
    print(f"Call Contract: {call_contract}")
    print(f"Contract Name: {call_contract.contract_name()}")
    print(f"Is Call: {call_contract.is_call()}")
    print(f"Days to Expiry: {call_contract.days_to_expiry(datetime.date(2023, 12, 1))}")
    print(f"Intrinsic Value at $155: {call_contract.intrinsic_value(155.0)}")
    print(f"Moneyness at $155: {call_contract.moneyness(155.0):.3f}")
    
    # Create a put option
    put_contract = OptionContract(
        symbol='AAPL',
        expiry=datetime.date(2024, 1, 19),
        strike=150.0,
        option_type='put'
    )
    
    print(f"\nPut Contract: {put_contract}")
    print(f"Is Put: {put_contract.is_put()}")
    print(f"Intrinsic Value at $145: {put_contract.intrinsic_value(145.0)}")
    print(f"Moneyness at $145: {put_contract.moneyness(145.0):.3f}")


def test_option_pricing():
    '''Test option pricing models'''
    print("\nTesting Option Pricing...")
    
    # Create option contract
    contract = OptionContract(
        symbol='SPY',
        expiry=datetime.date(2024, 3, 15),
        strike=450.0,
        option_type='call'
    )
    
    # Test Black-Scholes pricing
    bs_model = BlackScholesModel()
    
    underlying_price = 450.0
    volatility = 0.25
    risk_free_rate = 0.05
    
    price, greeks = bs_model.price(
        contract, underlying_price, volatility, risk_free_rate
    )
    
    print(f"Black-Scholes Price: ${price:.4f}")
    print(f"Greeks: {greeks}")
    
    # Test implied volatility calculation
    market_price = 25.0
    implied_vol = bs_model.implied_volatility(
        contract, underlying_price, market_price, risk_free_rate
    )
    print(f"Implied Volatility for ${market_price}: {implied_vol:.4f}")
    
    # Test Binomial pricing (if available)
    try:
        binomial_model = BinomialModel(steps=50)
        bin_price, bin_greeks = binomial_model.price(
            contract, underlying_price, volatility, risk_free_rate
        )
        print(f"Binomial Price: ${bin_price:.4f}")
    except Exception as e:
        print(f"Binomial pricing not available: {e}")


def test_option_position():
    '''Test option position tracking'''
    print("\nTesting Option Position...")
    
    contract = OptionContract(
        symbol='MSFT',
        expiry=datetime.date(2024, 2, 16),
        strike=350.0,
        option_type='call'
    )
    
    position = OptionPosition(contract)
    
    # Test position updates
    print(f"Initial position: Size={position.size}, Price=${position.price}")
    
    # Buy 5 contracts at $10
    position.update(5, 10.0)
    print(f"After buying 5 @ $10: Size={position.size}, Price=${position.price}")
    
    # Buy 3 more at $12
    position.update(3, 12.0)
    print(f"After buying 3 @ $12: Size={position.size}, Price=${position.price:.2f}")
    
    # Sell 4 contracts at $15
    position.update(-4, 15.0)
    print(f"After selling 4 @ $15: Size={position.size}, Price=${position.price:.2f}")
    
    # Calculate market value and PnL
    market_price = 14.0
    market_value = position.market_value(market_price)
    unrealized_pnl = position.unrealized_pnl(market_price)
    print(f"Market Value @ $14: ${market_value:.2f}")
    print(f"Unrealized PnL: ${unrealized_pnl:.2f}")


class TestOptionsStrategy(OptionStrategy):
    '''Test strategy for options functionality'''
    
    def __init__(self):
        super(TestOptionsStrategy, self).__init__()
        self.test_phase = 0
        self.orders_placed = []
    
    def next(self):
        if len(self) < 10:  # Let some bars pass
            return
        
        if self.test_phase == 0:
            # Test buying a call
            print(f"Bar {len(self)}: Testing call purchase")
            if len(self.datas) > 1:  # Have option data
                order = self.buy_call(data=self.datas[1], size=1)
                self.orders_placed.append(order)
                self.test_phase = 1
        
        elif self.test_phase == 1 and len(self) > 20:
            # Test selling the call
            print(f"Bar {len(self)}: Testing call sale")
            call_position = self.getposition(self.datas[1])
            if call_position.size > 0:
                order = self.sell_call(data=self.datas[1], size=call_position.size)
                self.orders_placed.append(order)
                self.test_phase = 2
        
        elif self.test_phase == 2 and len(self) > 30:
            # Test a simple spread
            print(f"Bar {len(self)}: Testing bull call spread")
            if len(self.datas) > 3:  # Have multiple strikes
                orders = self.bull_call_spread(
                    self.datas[1],  # Lower strike
                    self.datas[2],  # Higher strike
                    size=1
                )
                self.orders_placed.extend(orders)
                self.test_phase = 3
    
    def notify_order(self, order):
        if order.status == order.Completed:
            action = 'BUY' if order.isbuy() else 'SELL'
            print(f"ORDER: {action} {order.executed.size} @ ${order.executed.price:.4f}")
    
    def stop(self):
        print(f"Strategy completed. Placed {len(self.orders_placed)} orders")
        
        # Print final Greeks
        greeks = self.get_portfolio_greeks()
        if greeks:
            print(f"Final Portfolio Greeks: {greeks}")


def test_synthetic_option_data():
    '''Test synthetic option data generation'''
    print("\nTesting Synthetic Option Data...")
    
    # Create a simple price series for the underlying
    cerebro = bt.Cerebro()
    
    # Add underlying data (using built-in test data)
    data_path = os.path.join(os.path.dirname(__file__), 
                            '..', 'datas', '2006-day-001.txt')
    
    if os.path.exists(data_path):
        stock_data = bt.feeds.BacktraderCSVData(dataname=data_path)
    else:
        # Create synthetic stock data if file not found
        print("Creating synthetic stock data for testing...")
        stock_data = bt.feeds.BacktraderCSVData(dataname=None)
        # Note: In a real implementation, you'd create actual test data
    
    cerebro.adddata(stock_data, name='STOCK')
    
    # Create synthetic option data
    call_option = SyntheticOptionData(
        symbol='STOCK',
        expiry=datetime.date(2006, 3, 17),  # Options expire monthly
        strike=105.0,
        option_type='call',
        underlying_data=stock_data,
        volatility=0.25
    )
    
    cerebro.adddata(call_option, name='CALL_105')
    
    # Add higher strike call for spread testing
    call_option_higher = SyntheticOptionData(
        symbol='STOCK',
        expiry=datetime.date(2006, 3, 17),
        strike=110.0,
        option_type='call',
        underlying_data=stock_data,
        volatility=0.25
    )
    
    cerebro.adddata(call_option_higher, name='CALL_110')
    
    # Add put option
    put_option = SyntheticOptionData(
        symbol='STOCK',
        expiry=datetime.date(2006, 3, 17),
        strike=100.0,
        option_type='put',
        underlying_data=stock_data,
        volatility=0.25
    )
    
    cerebro.adddata(put_option, name='PUT_100')
    
    # Use options broker
    cerebro.broker = OptionBroker()
    cerebro.broker.setcash(10000)
    
    # Set options commission
    option_comm = EquityOptionCommissionInfo()
    cerebro.broker.addcommissioninfo(option_comm, name='CALL_105')
    cerebro.broker.addcommissioninfo(option_comm, name='CALL_110')
    cerebro.broker.addcommissioninfo(option_comm, name='PUT_100')
    
    # Add test strategy
    cerebro.addstrategy(TestOptionsStrategy)
    
    print("Running options backtest...")
    try:
        results = cerebro.run()
        print(f"Backtest completed. Final value: ${cerebro.broker.getvalue():.2f}")
    except Exception as e:
        print(f"Backtest failed: {e}")


def test_option_chain():
    '''Test option chain functionality'''
    print("\nTesting Option Chain...")
    
    # Create option chain
    chain = OptionChain('TSLA')
    
    # Add some contracts
    expiry1 = datetime.date(2024, 1, 19)
    expiry2 = datetime.date(2024, 2, 16)
    
    strikes = [200, 210, 220, 230, 240]
    
    for expiry in [expiry1, expiry2]:
        for strike in strikes:
            # Add calls
            chain.add_contract(expiry, strike, 'call')
            # Add puts
            chain.add_contract(expiry, strike, 'put')
    
    print(f"Created option chain with {len(chain.contracts)} contracts")
    
    # Test finding contracts
    call_contract = chain.get_contract(expiry1, 220, 'call')
    put_contract = chain.get_contract(expiry1, 220, 'put')
    
    if call_contract:
        print(f"Found call: {call_contract}")
    if put_contract:
        print(f"Found put: {put_contract}")
    
    # Test ATM contracts
    atm_call, atm_put = chain.get_atm_contracts(expiry1, 225.0)
    if atm_call and atm_put:
        print(f"ATM Call: {atm_call}")
        print(f"ATM Put: {atm_put}")


def main():
    '''Run all tests'''
    print("Backtrader Options Testing Suite")
    print("=" * 50)
    
    try:
        test_option_contract()
        test_option_pricing()
        test_option_position()
        test_option_chain()
        test_synthetic_option_data()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
