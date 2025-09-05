#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# Put Selling Strategy - Sell on Dips
#
# Strategy that sells put option contracts using 1/3 of capital each time,
# selling when the stock price drops 5% from the previous sell time.
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


class PutSellingStrategy(OptionStrategy):
    '''
    Put Selling Strategy that sells put options on price dips using capital allocation
    
    Rules:
    - Use 1/3 of total capital for each option sale
    - Sell when stock drops 5% from previous sell price
    - Use 30 delta puts with 6 weeks (42 days) to expiration
    - Close positions at 60% profit or hold to expiration
    - If assigned at expiration, accept the stock
    '''
    
    params = (
        ('total_capital', 100000),      # Total capital to allocate
        ('capital_fraction', 0.333),    # Fraction of capital to use per trade (1/3)
        ('drop_threshold', 0.05),       # 5% drop threshold
        ('target_dte', 42),            # Target 6 weeks (42 days) to expiration
        ('dte_tolerance', 7),          # Allow +/- 7 days from target
        ('target_delta', 0.30),        # Target 30 delta
        ('delta_tolerance', 0.05),     # Allow +/- 5 delta points
        ('profit_target', 0.60),       # Close at 60% profit
        ('option_type', 'put'),        # Option type
        ('symbol', 'STOCK'),           # Symbol for underlying (configurable)
        ('debug', True),               # Print debug information
    )
    
    def __init__(self):
        super(PutSellingStrategy, self).__init__()
        
        # Strategy state
        self.last_sell_price = None     # Price at last option sale
        self.active_trades = []         # List of active trades with capital allocation
        self.sell_history = []          # History of sales
        self.open_positions = {}        # Track individual option positions with entry prices
        
        # Capital management - now dynamic
        self.initial_capital = self.p.total_capital  # Store initial capital for reference
        self.allocated_capital = 0      # Total capital currently allocated to margin
        self.available_capital = self.p.total_capital  # Available capital for new trades
        
        # Data references
        self.stock_data = self.datas[0]  # Underlying stock data
        self.option_feeds = self.datas[1:]  # Option data feeds
        
        # Track stock price for monitoring
        self.current_price = None
        
        if self.p.debug:
            print(f"Put Selling Strategy initialized for {self.p.symbol}")
            print(f"Initial capital: ${self.initial_capital:,.2f}")
            print(f"Capital per trade: {self.initial_capital * self.p.capital_fraction:,.2f}")
            print(f"Drop threshold: {self.p.drop_threshold * 100}%")
            print(f"Target DTE: {self.p.target_dte} days")
            print(f"Target Delta: {self.p.target_delta}")
            print(f"Profit target: {self.p.profit_target * 100}%")

    def get_current_total_capital(self):
        '''Get current total capital (portfolio value)'''
        return self.broker.getvalue()
    
    def log(self, txt, dt=None):
        '''Logging function for strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')
    
    def next(self):
        '''Main strategy logic called on each bar'''
        self.current_price = self.stock_data.close[0]
        
        # Update capital allocation status
        self.update_capital_status()
        
        # Check for profit taking opportunities
        self.check_profit_targets()
        
        # Check for option expirations
        self.check_option_expirations()
        
        # Check if we should sell more options
        if self.should_sell_option():
            self.sell_option()
    
    def update_capital_status(self):
        '''Update available and allocated capital based on current portfolio value'''
        # Get current total capital from broker (includes all cash and positions)
        current_total_capital = self.get_current_total_capital()
        
        # Recalculate allocated capital based on active positions (margin requirements)
        self.allocated_capital = 0
        active_trades_temp = []
        
        for trade_info in self.active_trades:
            option_data = trade_info['option_data']
            position = self.getposition(option_data)
            
            if position.size != 0:  # Position still active
                # Calculate current margin requirement (strike price * contracts * 100)
                if hasattr(option_data, 'strike'):
                    margin_required = abs(position.size) * option_data.strike * 100
                    trade_info['current_margin'] = margin_required
                    self.allocated_capital += margin_required
                    active_trades_temp.append(trade_info)
        
        # Update active trades list
        self.active_trades = active_trades_temp
        
        # Calculate available capital (total portfolio value minus margin requirements)
        self.available_capital = max(0, current_total_capital - self.allocated_capital)
        
        if self.p.debug and len(self) % 20 == 0:  # Log every 20 bars
            # Use current total capital for calculating target per trade
            target_per_trade = current_total_capital * self.p.capital_fraction
            usable_capital = min(target_per_trade, self.available_capital)
            utilization = (self.allocated_capital / current_total_capital) * 100 if current_total_capital > 0 else 0
            
            # Calculate total return since start
            total_return = ((current_total_capital - self.initial_capital) / self.initial_capital) * 100 if self.initial_capital > 0 else 0
            
            self.log(f"Capital Status - Current Total: ${current_total_capital:,.2f} "
                    f"(+{total_return:.1f}% from ${self.initial_capital:,.2f})")
            self.log(f"Available: ${self.available_capital:,.2f}, "
                    f"Allocated to Margin: ${self.allocated_capital:,.2f} ({utilization:.1f}%)")
            self.log(f"Next Trade - Target: ${target_per_trade:,.2f}, "
                    f"Usable: ${usable_capital:,.2f}, "
                    f"Active Trades: {len(self.active_trades)}")

    def should_sell_option(self):
        '''Determine if we should sell an option'''
        # Use current total capital for calculations
        current_total_capital = self.get_current_total_capital()
        target_capital = current_total_capital * self.p.capital_fraction
        trade_capital = min(target_capital, self.available_capital)
        
        # Need at least enough for one contract (estimate $5000 minimum)
        min_capital_needed = 5000  # Conservative estimate
        
        if trade_capital < min_capital_needed:
            if self.p.debug and len(self) % 50 == 0:  # Log less frequently to avoid spam
                self.log(f"Insufficient capital for new trade: "
                        f"Target ${target_capital:,.2f}, "
                        f"Available ${self.available_capital:,.2f}, "
                        f"Usable ${trade_capital:,.2f}")
            return False
        
        # Don't sell if we don't have a previous sell price (first sale)
        if self.last_sell_price is None:
            if self.p.debug:
                self.log(f"First sale opportunity at ${self.current_price:.2f} "
                        f"with ${trade_capital:,.2f} capital (total: ${current_total_capital:,.2f})")
            return True
        
        # Calculate price drop from last sell
        price_drop = (self.last_sell_price - self.current_price) / self.last_sell_price
        
        if price_drop >= self.p.drop_threshold:
            if self.p.debug:
                self.log(f"5% drop detected: {price_drop*100:.2f}% "
                        f"(from ${self.last_sell_price:.2f} to ${self.current_price:.2f}) "
                        f"with ${trade_capital:,.2f} available capital")
            return True
        
        return False

    def sell_option(self):
        '''Sell a put option contract using capital allocation'''
        try:
            # Find suitable option contract
            option_data = self.find_suitable_put_option()
            
            if option_data is None:
                if self.p.debug:
                    self.log("No suitable put option contract found")
                return
            
            # Use current total capital for position sizing
            current_total_capital = self.get_current_total_capital()
            target_capital = current_total_capital * self.p.capital_fraction
            trade_capital = min(target_capital, self.available_capital)
            
            # Ensure we have minimum capital for at least one contract
            strike_price = getattr(option_data, 'strike', 100)
            min_capital_needed = strike_price * 100  # One contract margin requirement
            
            if trade_capital < min_capital_needed:
                if self.p.debug:
                    self.log(f"Insufficient capital for trade. "
                            f"Need: ${min_capital_needed:,.2f}, "
                            f"Available: ${trade_capital:,.2f}, "
                            f"Target: ${target_capital:,.2f}")
                return
            
            # For put selling, margin requirement is approximately strike * contracts * 100
            # Calculate how many contracts we can sell with allocated capital
            max_contracts = int(trade_capital / (strike_price * 100))
            
            if max_contracts < 1:
                if self.p.debug:
                    self.log(f"Cannot afford even 1 contract with available capital: ${trade_capital:,.2f}")
                return
            
            # Limit to reasonable number of contracts (e.g., max 10)
            contracts_to_sell = min(max_contracts, 10)
            
            # Recalculate actual capital used based on contracts we can afford
            actual_capital_used = contracts_to_sell * strike_price * 100
            
            # Place sell order (short the put)
            order = self.sell(data=option_data, size=contracts_to_sell)
            
            if order:
                # Update strategy state
                self.last_sell_price = self.current_price
                
                # Create trade tracking record
                trade_info = {
                    'date': self.datas[0].datetime.date(0),
                    'stock_price': self.current_price,
                    'option_data': option_data,
                    'order': order,
                    'contracts': contracts_to_sell,
                    'allocated_capital': actual_capital_used,  # Use actual capital used
                    'target_capital': target_capital,         # Track what we wanted to use
                    'available_capital': self.available_capital,  # Track what was available
                    'current_total_capital': current_total_capital,  # Track total capital at time of trade
                    'strike_price': strike_price,
                    'entry_price': None,  # Will be filled in notify_order
                    'current_margin': 0   # Will be updated
                }
                
                # Record sale
                self.sell_history.append(trade_info.copy())
                
                if self.p.debug:
                    expiry = getattr(option_data, 'expiry', 'Unknown')
                    delta = self.calculate_option_delta(option_data)
                    capital_efficiency = (actual_capital_used / target_capital) * 100 if target_capital > 0 else 0
                    self.log(f"SOLD {contracts_to_sell} Put contracts: "
                            f"Strike ${strike_price}, Expiry {expiry}, "
                            f"Delta {delta:.3f}, Stock @ ${self.current_price:.2f}")
                    self.log(f"Capital: Total ${current_total_capital:,.2f}, Target ${target_capital:,.2f}, "
                            f"Available ${self.available_capital:,.2f}, "
                            f"Used ${actual_capital_used:,.2f} ({capital_efficiency:.1f}% of target)")
        
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
                symbol=self.p.symbol,
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
                        f"Contracts: {abs(position.size)}, "
                        f"Profit {profit_pct*100:.1f}%")
            
            # Remove from open positions tracking
            if option_data in self.open_positions:
                del self.open_positions[option_data]
    
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
            contracts = abs(position.size)
            
            # For put options, we get assigned if stock price < strike (ITM)
            if current_price < strike:
                # We're assigned - must buy stock at strike price
                intrinsic_value = strike - current_price
                total_loss = intrinsic_value * contracts * 100
                
                if self.p.debug:
                    self.log(f"PUT ASSIGNED: {contracts} contracts @ Strike ${strike:.2f}, "
                            f"Stock ${current_price:.2f}, "
                            f"Total Loss: ${total_loss:,.2f}")
                
                # In a real implementation, we'd receive the stock
                # For this simulation, we just close the position
                self.close(data=option_data)
            
            else:
                # Option expires worthless (good for us as sellers)
                if self.p.debug:
                    self.log(f"PUT EXPIRED WORTHLESS: {contracts} contracts @ Strike ${strike:.2f}, "
                            f"Stock ${current_price:.2f} - Full profit!")
            
            # Remove from tracking
            if option_data in self.open_positions:
                del self.open_positions[option_data]
    
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
                            'sale_record': sale_record,
                            'contracts': sale_record['contracts']
                        }
                        
                        # Add to active trades for capital tracking
                        self.active_trades.append(sale_record)
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
            current_total_capital = self.get_current_total_capital()
            
            self.log('Strategy completed')
            self.log(f'Total puts sold: {len(self.sell_history)}')
            self.log(f'Final active trades: {len(self.active_trades)}')
            self.log(f'Final allocated capital: ${self.allocated_capital:,.2f}')
            self.log(f'Final available capital: ${self.available_capital:,.2f}')
            
            # Calculate total PnL
            total_pnl = 0
            for trade in self.trades:
                if trade.isclosed:
                    total_pnl += trade.pnl
            
            self.log(f'Initial capital: ${self.initial_capital:,.2f}')
            self.log(f'Final portfolio value: ${current_total_capital:,.2f}')
            self.log(f'Total PnL: ${total_pnl:.2f}')
            
            # Calculate return on initial capital
            if self.initial_capital > 0:
                total_return = ((current_total_capital - self.initial_capital) / self.initial_capital) * 100
                self.log(f'Total Return: {total_return:.2f}%')
                
                # Show credits received from option sales
                total_credits = 0
                for sale in self.sell_history:
                    if sale.get('entry_price'):
                        total_credits += sale['contracts'] * sale['entry_price'] * 100
                
                self.log(f'Total option credits received: ${total_credits:,.2f}')


def create_put_option_data(stock_data, strike_prices, expiry_date, symbol='STOCK'):
    '''Create synthetic put option data for different strikes'''
    option_feeds = []
    
    for strike in strike_prices:
        option_data = SyntheticOptionData(
            symbol=symbol,
            expiry=expiry_date,
            strike=strike,
            option_type='put',
            underlying_data=stock_data,
            volatility=0.25,  # 25% implied volatility
            risk_free_rate=0.05
        )
        option_feeds.append(option_data)
    
    return option_feeds


def run_put_selling_strategy(symbol='STOCK', data_file=None, total_capital=100000):
    '''Run the put selling strategy'''
    print(f"Put Selling Strategy for {symbol}")
    print(f"Total Capital: ${total_capital:,.2f}")
    print("=" * 50)
    
    # Create Cerebro engine
    cerebro = bt.Cerebro()
    
    # Add stock data
    if data_file:
        stock_data = bt.feeds.BacktraderCSVData(dataname=data_file)
    else:
        # Use default sample data
        data_path = os.path.join(os.path.dirname(__file__), 
                                '..', '..', 'datas', '2006-day-001.txt')
        
        if os.path.exists(data_path):
            stock_data = bt.feeds.BacktraderCSVData(dataname=data_path)
        else:
            print("Warning: Sample data file not found, specify data_file parameter")
            return
    
    cerebro.adddata(stock_data, name=symbol)
    
    # Create put option contracts with different strikes
    # Use strikes below current price for OTM puts (typical for selling)
    expiry_date = datetime.date(2006, 3, 17)  # 6 weeks out
    base_price = 100  # Assuming stock around $100
    strike_prices = [85, 90, 95, 100, 105]  # Range including OTM puts
    
    option_feeds = create_put_option_data(stock_data, strike_prices, expiry_date, symbol)
    
    for i, option_feed in enumerate(option_feeds):
        cerebro.adddata(option_feed, name=f'{symbol}_Put_{strike_prices[i]}')
    
    # Set up options broker with margin requirements for short options
    cerebro.broker = OptionBroker()
    cerebro.broker.setcash(total_capital)  # Set the capital amount
    
    # Add options commission (higher for selling due to margin requirements)
    option_comm = EquityOptionCommissionInfo(
        commission=2.0,      # $2 per contract (higher for selling)
        margin=None,         # Margin handled by broker
        mult=100             # Standard option multiplier
    )
    
    for i, strike in enumerate(strike_prices):
        cerebro.broker.addcommissioninfo(option_comm, name=f'{symbol}_Put_{strike}')
    
    # Add the strategy
    cerebro.addstrategy(
        PutSellingStrategy,
        total_capital=total_capital,  # Pass total capital to strategy
        capital_fraction=0.333,       # Use 1/3 of capital per trade
        drop_threshold=0.05,          # 5%
        target_dte=42,               # 6 weeks
        target_delta=0.30,           # 30 delta
        profit_target=0.60,          # 60% profit
        symbol=symbol,               # Pass symbol to strategy
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
    # Example usage:
    # For MSFT with $50k: run_put_selling_strategy('MSFT', 'path/to/msft_data.csv', 50000)
    # For SPY with $100k: run_put_selling_strategy('SPY', 'path/to/spy_data.csv', 100000)
    # For default sample data: run_put_selling_strategy()
    
    run_put_selling_strategy()