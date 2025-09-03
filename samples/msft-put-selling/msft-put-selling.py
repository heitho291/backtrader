#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# MSFT Put Selling Strategy - Sell on Dips
#
# Strategy that sells up to 3 MSFT put option contracts, selling 1 contract 
# every time the stock price drops 5% from the previous sell time.
# Uses 30 delta puts with 6 weeks expiration.
# Closes positions at 60% profit or holds to expiration.
#
###############################################################################
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime
import sys
import os

# Add the backtrader directory to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import backtrader as bt
from backtrader.option import OptionContract
from backtrader.feeds.optiondata import SyntheticOptionData
from backtrader.brokers.optionbroker import OptionBroker
from backtrader.optionstrategy import OptionStrategy
from backtrader.optioncommission import EquityOptionCommissionInfo


class MSFTPutSellingStrategy(OptionStrategy):
    '''
    MSFT Put Selling Strategy that sells put options on price dips
    
    Rules:
    - Sell up to 3 put option contracts maximum
    - Sell 1 contract each time MSFT drops 5% from previous sell price
    - Use 30 delta puts with 6 weeks (42 days) to expiration
    - Close positions at 60% profit or hold to expiration
    - If assigned at expiration, accept the stock
    '''
    
    params = (
        ('max_contracts', 3),           # Maximum number of contracts to sell
        ('drop_threshold', 0.05),       # 5% drop threshold
        ('target_dte', 42),            # Target 6 weeks (42 days) to expiration
        ('dte_tolerance', 7),          # Allow +/- 7 days from target
        ('target_delta', 0.30),        # Target 30 delta
        ('delta_tolerance', 0.05),     # Allow +/- 5 delta points
        ('profit_target', 0.60),       # Close at 60% profit
        ('option_type', 'put'),        # Option type
        ('debug', True),               # Print debug information
    )
    
    def __init__(self):
        super(MSFTPutSellingStrategy, self).__init__()
        
        # Strategy state
        self.last_sell_price = None     # Price at last option sale
        self.contracts_sold = 0         # Number of contracts currently sold
        self.sell_history = []          # History of sales
        self.open_positions = {}        # Track individual option positions with entry prices
        
        # Data references
        self.msft_data = self.datas[0]  # Underlying MSFT stock data
        self.option_feeds = self.datas[1:]  # Option data feeds
        
        # Track stock price for monitoring
        self.current_price = None
        
        if self.p.debug:
            print("MSFT Put Selling Strategy initialized")
            print(f"Max contracts: {self.p.max_contracts}")
            print(f"Drop threshold: {self.p.drop_threshold * 100}%")
            print(f"Target DTE: {self.p.target_dte} days")
            print(f"Target Delta: {self.p.target_delta}")
            print(f"Profit target: {self.p.profit_target * 100}%")
    
    def log(self, txt, dt=None):
        '''Logging function for strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')
    
    def next(self):
        '''Main strategy logic called on each bar'''
        self.current_price = self.msft_data.close[0]
        
        # Check for profit taking opportunities
        self.check_profit_targets()
        
        # Check for option expirations
        self.check_option_expirations()
        
        # Check if we should sell more options
        if self.should_sell_option():
            self.sell_option()
    
    def should_sell_option(self):
        '''Determine if we should sell an option'''
        # Don't sell if we already have max contracts
        if self.contracts_sold >= self.p.max_contracts:
            return False
        
        # Don't sell if we don't have a previous sell price (first sale)
        if self.last_sell_price is None:
            if self.p.debug:
                self.log(f"First sale opportunity at ${self.current_price:.2f}")
            return True
        
        # Calculate price drop from last sell
        price_drop = (self.last_sell_price - self.current_price) / self.last_sell_price
        
        if price_drop >= self.p.drop_threshold:
            if self.p.debug:
                self.log(f"5% drop detected: {price_drop*100:.2f}% "
                        f"(from ${self.last_sell_price:.2f} to ${self.current_price:.2f})")
            return True
        
        return False
    
    def sell_option(self):
        '''Sell a put option contract'''
        try:
            # Find suitable option contract
            option_data = self.find_suitable_put_option()
            
            if option_data is None:
                if self.p.debug:
                    self.log("No suitable put option contract found")
                return
            
            # Place sell order (short the put)
            order = self.sell(data=option_data, size=1)
            
            if order:
                # Update strategy state
                self.last_sell_price = self.current_price
                self.contracts_sold += 1
                
                # Record sale
                sale_record = {
                    'date': self.datas[0].datetime.date(0),
                    'stock_price': self.current_price,
                    'option_data': option_data,
                    'order': order,
                    'contract_number': self.contracts_sold,
                    'entry_price': None  # Will be filled in notify_order
                }
                self.sell_history.append(sale_record)
                
                if self.p.debug:
                    strike = getattr(option_data, 'strike', 'Unknown')
                    expiry = getattr(option_data, 'expiry', 'Unknown')
                    delta = self.calculate_option_delta(option_data)
                    self.log(f"SOLD Put #{self.contracts_sold}: "
                            f"Strike ${strike}, Expiry {expiry}, "
                            f"Delta {delta:.3f}, Stock @ ${self.current_price:.2f}")
        
        except Exception as e:
            if self.p.debug:
                self.log(f"Error selling option: {e}")
    
    def find_suitable_put_option(self):
        '''Find a suitable put option contract to sell (30 delta, 6 weeks)'''
        current_date = self.datas[0].datetime.date(0)
        best_option = None
        best_score = float('inf')
        
        # Look through available option feeds
        for option_data in self.option_feeds:
            if (hasattr(option_data, 'expiry') and 
                hasattr(option_data, 'strike') and 
                hasattr(option_data, 'option_type') and
                option_data.option_type == 'put'):
                
                # Check days to expiration
                days_to_expiry = (option_data.expiry - current_date).days
                dte_diff = abs(days_to_expiry - self.p.target_dte)
                
                if dte_diff <= self.p.dte_tolerance:
                    # Calculate delta
                    delta = self.calculate_option_delta(option_data)
                    
                    if delta is not None:
                        # For puts, delta is negative, so we want around -0.30
                        target_delta = -self.p.target_delta
                        delta_diff = abs(delta - target_delta)
                        
                        if delta_diff <= self.p.delta_tolerance:
                            # Score based on how close to target delta and DTE
                            score = delta_diff * 10 + dte_diff * 0.1
                            
                            if score < best_score:
                                best_score = score
                                best_option = option_data
        
        # If no perfect match, find the closest put option
        if best_option is None and self.option_feeds:
            for option_data in self.option_feeds:
                if (hasattr(option_data, 'option_type') and 
                    option_data.option_type == 'put'):
                    
                    # Check if it's reasonably close to target
                    days_to_expiry = (option_data.expiry - current_date).days
                    if 30 <= days_to_expiry <= 60:  # Reasonable DTE range
                        strike = getattr(option_data, 'strike', 0)
                        # Look for strikes below current price (out-of-the-money puts)
                        if strike < self.current_price * 0.95:  # At least 5% OTM
                            best_option = option_data
                            break
        
        return best_option
    
    def calculate_option_delta(self, option_data):
        '''Calculate option delta using Black-Scholes'''
        try:
            from backtrader.optionpricing import BlackScholesModel
            
            if not hasattr(option_data, 'strike') or not hasattr(option_data, 'expiry'):
                return None
            
            current_date = self.datas[0].datetime.date(0)
            days_to_expiry = (option_data.expiry - current_date).days
            
            if days_to_expiry <= 0:
                return None
            
            # Create a temporary contract for delta calculation
            contract = OptionContract(
                symbol='MSFT',
                expiry=option_data.expiry,
                strike=option_data.strike,
                option_type=getattr(option_data, 'option_type', 'put')
            )
            
            bs_model = BlackScholesModel()
            
            # Use reasonable defaults for pricing
            volatility = getattr(option_data, 'volatility', 0.25)
            risk_free_rate = 0.05
            
            try:
                price, greeks = bs_model.price(
                    contract, self.current_price, volatility, risk_free_rate
                )
                return greeks.get('delta', None)
            except:
                return None
        
        except ImportError:
            # Fallback approximation for delta
            if hasattr(option_data, 'strike'):
                moneyness = self.current_price / option_data.strike
                # Rough approximation: OTM puts have delta around -0.3 when 5% OTM
                if option_data.option_type == 'put':
                    if moneyness > 1.05:  # 5% OTM
                        return -0.30
                    elif moneyness > 1.0:  # ATM to 5% OTM
                        return -0.50
                    else:  # ITM
                        return -0.70
            return None
    
    def check_profit_targets(self):
        '''Check if any positions have reached 60% profit target'''
        for option_data, entry_info in self.open_positions.items():
            position = self.getposition(option_data)
            
            if position.size < 0:  # We're short (sold puts)
                current_option_price = option_data.close[0] if len(option_data) > 0 else 0
                entry_price = entry_info['entry_price']
                
                if entry_price > 0:
                    # For short positions, profit = entry_price - current_price
                    profit_per_contract = entry_price - current_option_price
                    profit_percentage = profit_per_contract / entry_price
                    
                    if profit_percentage >= self.p.profit_target:
                        self.close_profitable_position(option_data, profit_percentage)
    
    def close_profitable_position(self, option_data, profit_pct):
        '''Close a position that has reached profit target'''
        position = self.getposition(option_data)
        
        if position.size < 0:  # Confirm we're short
            # Buy to close the position
            order = self.buy(data=option_data, size=abs(position.size))
            
            if self.p.debug:
                strike = getattr(option_data, 'strike', 'Unknown')
                self.log(f"CLOSING PROFITABLE Put: Strike ${strike}, "
                        f"Profit {profit_pct*100:.1f}%")
            
            # Remove from open positions tracking
            if option_data in self.open_positions:
                del self.open_positions[option_data]
            
            self.contracts_sold = max(0, self.contracts_sold - abs(position.size))
    
    def check_option_expirations(self):
        '''Check for option expirations and handle assignment'''
        current_date = self.datas[0].datetime.date(0)
        
        for option_data in list(self.open_positions.keys()):
            position = self.getposition(option_data)
            
            if position.size < 0 and hasattr(option_data, 'expiry'):
                # Check if option expires today
                if option_data.expiry == current_date:
                    self.handle_put_expiration(option_data, position)
    
    def handle_put_expiration(self, option_data, position):
        '''Handle put option expiration (assignment if ITM)'''
        if hasattr(option_data, 'strike'):
            strike = option_data.strike
            current_price = self.current_price
            
            # For put options, we get assigned if stock price < strike (ITM)
            if current_price < strike:
                # We're assigned - must buy stock at strike price
                intrinsic_value = strike - current_price
                
                if self.p.debug:
                    self.log(f"PUT ASSIGNED: Strike ${strike:.2f}, "
                            f"Stock ${current_price:.2f}, "
                            f"Loss ${intrinsic_value:.2f} per share")
                
                # In a real implementation, we'd receive the stock
                # For this simulation, we just close the position
                self.close(data=option_data)
            
            else:
                # Option expires worthless (good for us as sellers)
                if self.p.debug:
                    self.log(f"PUT EXPIRED WORTHLESS: Strike ${strike:.2f}, "
                            f"Stock ${current_price:.2f} - Full profit!")
            
            # Remove from tracking
            if option_data in self.open_positions:
                del self.open_positions[option_data]
            
            self.contracts_sold = max(0, self.contracts_sold - abs(position.size))
    
    def notify_order(self, order):
        '''Handle order notifications'''
        if order.status in [order.Completed]:
            action = 'SELL' if order.issell() else 'BUY'
            
            # Track entry prices for profit calculation
            if order.issell():  # Opening position
                for sale_record in self.sell_history:
                    if (sale_record['order'] == order and 
                        sale_record['entry_price'] is None):
                        sale_record['entry_price'] = order.executed.price
                        
                        # Add to open positions tracking
                        self.open_positions[sale_record['option_data']] = {
                            'entry_price': order.executed.price,
                            'entry_date': self.datas[0].datetime.date(0),
                            'sale_record': sale_record
                        }
                        break
            
            if self.p.debug:
                self.log(f'ORDER {action}: {order.executed.size} contracts @ '
                        f'${order.executed.price:.4f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if self.p.debug:
                self.log(f'ORDER FAILED: {order.status}')
    
    def notify_trade(self, trade):
        '''Handle trade notifications'''
        if not trade.isclosed:
            return
        
        if self.p.debug:
            self.log(f'TRADE CLOSED: PnL ${trade.pnl:.2f}')
    
    def stop(self):
        '''Called when strategy ends'''
        if self.p.debug:
            self.log('Strategy completed')
            self.log(f'Total puts sold: {len(self.sell_history)}')
            self.log(f'Final contracts outstanding: {self.contracts_sold}')
            
            # Calculate total PnL
            total_pnl = 0
            for trade in self.trades:
                if trade.isclosed:
                    total_pnl += trade.pnl
            
            self.log(f'Total PnL: ${total_pnl:.2f}')
            self.log(f'Final portfolio value: ${self.broker.getvalue():.2f}')


def create_msft_put_option_data(stock_data, strike_prices, expiry_date):
    '''Create synthetic put option data for different strikes'''
    option_feeds = []
    
    for strike in strike_prices:
        option_data = SyntheticOptionData(
            symbol='MSFT',
            expiry=expiry_date,
            strike=strike,
            option_type='put',  # Changed to put options
            underlying_data=stock_data,
            volatility=0.25,  # 25% implied volatility
            risk_free_rate=0.05
        )
        option_feeds.append(option_data)
    
    return option_feeds


def run_msft_put_selling_strategy():
    '''Run the MSFT put selling strategy'''
    print("MSFT Put Selling Strategy")
    print("=" * 50)
    
    # Create Cerebro engine
    cerebro = bt.Cerebro()
    
    # Add MSFT stock data (using sample data as proxy)
    data_path = os.path.join(os.path.dirname(__file__), 
                            '..', '..', 'datas', '2006-day-001.txt')
    
    if os.path.exists(data_path):
        msft_data = bt.feeds.BacktraderCSVData(dataname=data_path)
    else:
        print("Warning: Sample data file not found, creating minimal data")
        return
    
    cerebro.adddata(msft_data, name='MSFT')
    
    # Create put option contracts with different strikes
    # Use strikes below current price for OTM puts (typical for selling)
    expiry_date = datetime.date(2006, 3, 17)  # 6 weeks out
    base_price = 100  # Assuming stock around $100
    strike_prices = [85, 90, 95, 100, 105]  # Range including OTM puts
    
    option_feeds = create_msft_put_option_data(msft_data, strike_prices, expiry_date)
    
    for i, option_feed in enumerate(option_feeds):
        cerebro.adddata(option_feed, name=f'MSFT_Put_{strike_prices[i]}')
    
    # Set up options broker with margin requirements for short options
    cerebro.broker = OptionBroker()
    cerebro.broker.setcash(100000)  # $100,000 starting capital (need more for selling)
    
    # Add options commission (higher for selling due to margin requirements)
    option_comm = EquityOptionCommissionInfo(
        commission=2.0,      # $2 per contract (higher for selling)
        margin=None,         # Margin handled by broker
        mult=100             # Standard option multiplier
    )
    
    for i, strike in enumerate(strike_prices):
        cerebro.broker.addcommissioninfo(option_comm, name=f'MSFT_Put_{strike}')
    
    # Add the strategy
    cerebro.addstrategy(
        MSFTPutSellingStrategy,
        max_contracts=3,
        drop_threshold=0.05,  # 5%
        target_dte=42,        # 6 weeks
        target_delta=0.30,    # 30 delta
        profit_target=0.60,   # 60% profit
        debug=True
    )
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade_analyzer")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    
    # Run the strategy
    print(f"Starting portfolio value: ${cerebro.broker.getvalue():.2f}")
    
    try:
        results = cerebro.run()
        strategy = results[0]
        
        print(f"\nFinal portfolio value: ${cerebro.broker.getvalue():.2f}")
        
        # Print analysis
        trade_analysis = strategy.analyzers.trade_analyzer.get_analysis()
        if trade_analysis:
            print(f"\nTrade Analysis:")
            print(f"Total trades: {trade_analysis.get('total', {}).get('total', 0)}")
            print(f"Winning trades: {trade_analysis.get('won', {}).get('total', 0)}")
            print(f"Losing trades: {trade_analysis.get('lost', {}).get('total', 0)}")
        
        # Print drawdown
        drawdown = strategy.analyzers.drawdown.get_analysis()
        if drawdown:
            print(f"Max drawdown: {drawdown.get('max', {}).get('drawdown', 0):.2f}%")
        
        # Print returns
        returns = strategy.analyzers.returns.get_analysis()
        if returns:
            print(f"Total return: {returns.get('rtot', 0) * 100:.2f}%")
    
    except Exception as e:
        print(f"Strategy execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_msft_put_selling_strategy()