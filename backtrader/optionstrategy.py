#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# Copyright (C) 2015-2023 Daniel Rodriguez
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime
from .strategy import Strategy
from .utils.py3 import with_metaclass


class OptionStrategy(Strategy):
    '''
    Base strategy class with option-specific functionality.
    Provides helper methods for common option strategies.
    '''
    
    def __init__(self):
        super(OptionStrategy, self).__init__()
        self.option_orders = {}  # Track option orders by strategy
        self.option_positions = {}  # Track option positions
    
    # Helper methods for option trading
    
    def buy_call(self, data=None, size=None, **kwargs):
        '''Buy call option(s)'''
        if data is None:
            data = self.datas[0]
        
        if size is None:
            size = 1  # Default to 1 contract
        
        order = self.buy(data=data, size=size, **kwargs)
        self._track_option_order(order, 'buy_call')
        return order
    
    def sell_call(self, data=None, size=None, **kwargs):
        '''Sell call option(s)'''
        if data is None:
            data = self.datas[0]
        
        if size is None:
            size = 1
        
        order = self.sell(data=data, size=size, **kwargs)
        self._track_option_order(order, 'sell_call')
        return order
    
    def buy_put(self, data=None, size=None, **kwargs):
        '''Buy put option(s)'''
        if data is None:
            data = self.datas[0]
        
        if size is None:
            size = 1
        
        order = self.buy(data=data, size=size, **kwargs)
        self._track_option_order(order, 'buy_put')
        return order
    
    def sell_put(self, data=None, size=None, **kwargs):
        '''Sell put option(s)'''
        if data is None:
            data = self.datas[0]
        
        if size is None:
            size = 1
        
        order = self.sell(data=data, size=size, **kwargs)
        self._track_option_order(order, 'sell_put')
        return order
    
    # Option spread strategies
    
    def bull_call_spread(self, lower_strike_data, higher_strike_data, size=1, **kwargs):
        '''
        Execute bull call spread:
        - Buy call at lower strike
        - Sell call at higher strike
        '''
        orders = []
        
        # Buy lower strike call
        buy_order = self.buy_call(data=lower_strike_data, size=size, **kwargs)
        orders.append(buy_order)
        
        # Sell higher strike call
        sell_order = self.sell_call(data=higher_strike_data, size=size, **kwargs)
        orders.append(sell_order)
        
        self._track_spread('bull_call_spread', orders)
        return orders
    
    def bear_put_spread(self, lower_strike_data, higher_strike_data, size=1, **kwargs):
        '''
        Execute bear put spread:
        - Buy put at higher strike
        - Sell put at lower strike
        '''
        orders = []
        
        # Buy higher strike put
        buy_order = self.buy_put(data=higher_strike_data, size=size, **kwargs)
        orders.append(buy_order)
        
        # Sell lower strike put
        sell_order = self.sell_put(data=lower_strike_data, size=size, **kwargs)
        orders.append(sell_order)
        
        self._track_spread('bear_put_spread', orders)
        return orders
    
    def iron_condor(self, put_lower_data, put_higher_data, 
                    call_lower_data, call_higher_data, size=1, **kwargs):
        '''
        Execute iron condor:
        - Sell put at higher strike
        - Buy put at lower strike  
        - Sell call at lower strike
        - Buy call at higher strike
        '''
        orders = []
        
        # Put spread (sell higher, buy lower)
        sell_put = self.sell_put(data=put_higher_data, size=size, **kwargs)
        buy_put = self.buy_put(data=put_lower_data, size=size, **kwargs)
        orders.extend([sell_put, buy_put])
        
        # Call spread (sell lower, buy higher)  
        sell_call = self.sell_call(data=call_lower_data, size=size, **kwargs)
        buy_call = self.buy_call(data=call_higher_data, size=size, **kwargs)
        orders.extend([sell_call, buy_call])
        
        self._track_spread('iron_condor', orders)
        return orders
    
    def straddle(self, call_data, put_data, size=1, direction='long', **kwargs):
        '''
        Execute straddle (same strike call and put):
        - Long straddle: buy call and put
        - Short straddle: sell call and put
        '''
        orders = []
        
        if direction.lower() == 'long':
            call_order = self.buy_call(data=call_data, size=size, **kwargs)
            put_order = self.buy_put(data=put_data, size=size, **kwargs)
        else:  # short
            call_order = self.sell_call(data=call_data, size=size, **kwargs)
            put_order = self.sell_put(data=put_data, size=size, **kwargs)
        
        orders.extend([call_order, put_order])
        self._track_spread(f'{direction}_straddle', orders)
        return orders
    
    def strangle(self, call_data, put_data, size=1, direction='long', **kwargs):
        '''
        Execute strangle (different strike call and put):
        - Long strangle: buy OTM call and put
        - Short strangle: sell OTM call and put
        '''
        orders = []
        
        if direction.lower() == 'long':
            call_order = self.buy_call(data=call_data, size=size, **kwargs)
            put_order = self.buy_put(data=put_data, size=size, **kwargs)
        else:  # short
            call_order = self.sell_call(data=call_data, size=size, **kwargs)
            put_order = self.sell_put(data=put_data, size=size, **kwargs)
        
        orders.extend([call_order, put_order])
        self._track_spread(f'{direction}_strangle', orders)
        return orders
    
    def covered_call(self, underlying_data, call_data, size=1, **kwargs):
        '''
        Execute covered call:
        - Own underlying stock
        - Sell call option
        '''
        orders = []
        
        # Buy underlying if not already owned
        underlying_pos = self.getposition(underlying_data)
        if underlying_pos.size < size * 100:  # Need 100 shares per contract
            shares_needed = (size * 100) - underlying_pos.size
            stock_order = self.buy(data=underlying_data, size=shares_needed, **kwargs)
            orders.append(stock_order)
        
        # Sell call
        call_order = self.sell_call(data=call_data, size=size, **kwargs)
        orders.append(call_order)
        
        self._track_spread('covered_call', orders)
        return orders
    
    def protective_put(self, underlying_data, put_data, size=1, **kwargs):
        '''
        Execute protective put:
        - Own underlying stock
        - Buy put option for protection
        '''
        orders = []
        
        # Buy underlying if not already owned
        underlying_pos = self.getposition(underlying_data)
        if underlying_pos.size < size * 100:
            shares_needed = (size * 100) - underlying_pos.size
            stock_order = self.buy(data=underlying_data, size=shares_needed, **kwargs)
            orders.append(stock_order)
        
        # Buy put
        put_order = self.buy_put(data=put_data, size=size, **kwargs)
        orders.append(put_order)
        
        self._track_spread('protective_put', orders)
        return orders
    
    # Helper methods for option analysis
    
    def get_option_chain_data(self, symbol):
        '''Get all option data feeds for a given underlying symbol'''
        option_data = []
        for data in self.datas:
            if (hasattr(data, 'contract') and 
                hasattr(data.contract, 'p') and
                data.contract.p.symbol == symbol):
                option_data.append(data)
        return option_data
    
    def find_strike_data(self, symbol, expiry, strike, option_type):
        '''Find option data feed for specific contract parameters'''
        for data in self.datas:
            if (hasattr(data, 'contract') and 
                data.contract.p.symbol == symbol and
                data.contract.p.expiry == expiry and
                data.contract.p.strike == strike and
                data.contract.p.option_type.lower() == option_type.lower()):
                return data
        return None
    
    def get_atm_strike(self, symbol, underlying_price=None):
        '''Find at-the-money strike price for given underlying'''
        if underlying_price is None:
            # Try to get from underlying data
            for data in self.datas:
                if (hasattr(data, '_name') and data._name == symbol) or \
                   (hasattr(data, 'contract') and data.contract.p.symbol == symbol and
                    hasattr(data, 'underlying_price')):
                    try:
                        underlying_price = data.close[0]
                        break
                    except:
                        continue
        
        if underlying_price is None:
            return None
        
        # Find available strikes and pick closest
        strikes = set()
        for data in self.datas:
            if (hasattr(data, 'contract') and 
                data.contract.p.symbol == symbol):
                strikes.add(data.contract.p.strike)
        
        if not strikes:
            return None
        
        return min(strikes, key=lambda x: abs(x - underlying_price))
    
    def get_portfolio_greeks(self):
        '''Get portfolio Greeks from broker'''
        if hasattr(self.broker, 'get_portfolio_greeks'):
            return self.broker.get_portfolio_greeks()
        return None
    
    def calculate_max_loss(self, strategy_type, *args):
        '''Calculate maximum theoretical loss for option strategy'''
        # Implementation would depend on strategy type
        # This is a placeholder for strategy-specific calculations
        pass
    
    def calculate_max_profit(self, strategy_type, *args):
        '''Calculate maximum theoretical profit for option strategy'''
        # Implementation would depend on strategy type
        # This is a placeholder for strategy-specific calculations
        pass
    
    def calculate_breakeven(self, strategy_type, *args):
        '''Calculate breakeven point(s) for option strategy'''
        # Implementation would depend on strategy type
        # This is a placeholder for strategy-specific calculations
        pass
    
    # Internal tracking methods
    
    def _track_option_order(self, order, strategy_type):
        '''Track option orders by strategy type'''
        if strategy_type not in self.option_orders:
            self.option_orders[strategy_type] = []
        self.option_orders[strategy_type].append(order)
    
    def _track_spread(self, spread_type, orders):
        '''Track spread orders as a group'''
        if spread_type not in self.option_orders:
            self.option_orders[spread_type] = []
        self.option_orders[spread_type].append(orders)
    
    def notify_order(self, order):
        '''Override to handle option-specific order notifications'''
        super(OptionStrategy, self).notify_order(order)
        
        # Add option-specific handling if needed
        if hasattr(order.data, 'contract'):
            self._handle_option_order_notification(order)
    
    def _handle_option_order_notification(self, order):
        '''Handle option-specific order notifications'''
        # Can be overridden for custom option order handling
        pass


# Example option strategies

class SimpleCoveredCall(OptionStrategy):
    '''
    Example strategy: Simple covered call writing
    '''
    params = (
        ('call_dte', 30),      # Days to expiration for calls
        ('strike_pct', 1.05),  # Strike as percentage of current price
        ('hold_days', 21),     # Days to hold before closing
    )
    
    def __init__(self):
        super(SimpleCoveredCall, self).__init__()
        self.underlying_data = self.datas[0]  # Assume first data is underlying
        self.call_data = None  # Will be set based on parameters
        self.entry_date = None
        self.current_call_order = None
    
    def next(self):
        if not self.position:  # No underlying position
            # Buy underlying stock
            self.buy(data=self.underlying_data, size=100)
        
        elif self.call_data is None:  # Have stock, need to find call to sell
            # Find appropriate call option
            target_strike = self.underlying_data.close[0] * self.p.strike_pct
            self.call_data = self._find_call_option(target_strike)
            
            if self.call_data:
                # Sell covered call
                self.current_call_order = self.sell_call(data=self.call_data, size=1)
                self.entry_date = self.datetime.date(0)
        
        elif self.current_call_order and self.entry_date:
            # Check if it's time to close the call
            days_held = (self.datetime.date(0) - self.entry_date).days
            if days_held >= self.p.hold_days:
                # Close the call position
                call_position = self.getposition(self.call_data)
                if call_position.size < 0:  # Still short the call
                    self.buy_call(data=self.call_data, size=1)  # Buy to close
                
                # Reset for next cycle
                self.call_data = None
                self.current_call_order = None
                self.entry_date = None
    
    def _find_call_option(self, target_strike):
        '''Find call option with strike close to target'''
        # This would need to search through available option data
        # For now, return None (would need option chain data)
        return None


class LongStraddle(OptionStrategy):
    '''
    Example strategy: Long straddle on earnings announcements
    '''
    params = (
        ('entry_dte', 7),    # Enter straddle 7 days before expiration
        ('exit_dte', 1),     # Exit 1 day before expiration
    )
    
    def __init__(self):
        super(LongStraddle, self).__init__()
        self.underlying_data = self.datas[0]
        self.call_data = None
        self.put_data = None
        self.straddle_active = False
    
    def next(self):
        if not self.straddle_active:
            # Look for straddle entry opportunity
            atm_strike = self.get_atm_strike(self.underlying_data._name)
            if atm_strike:
                self.call_data = self.find_strike_data(
                    self.underlying_data._name, 
                    self._get_target_expiry(), 
                    atm_strike, 
                    'call'
                )
                self.put_data = self.find_strike_data(
                    self.underlying_data._name,
                    self._get_target_expiry(),
                    atm_strike,
                    'put'
                )
                
                if self.call_data and self.put_data:
                    # Enter long straddle
                    self.straddle(self.call_data, self.put_data, size=1, direction='long')
                    self.straddle_active = True
        
        else:
            # Check for exit conditions
            if self._should_exit_straddle():
                # Exit straddle
                call_pos = self.getposition(self.call_data)
                put_pos = self.getposition(self.put_data)
                
                if call_pos.size > 0:
                    self.sell_call(data=self.call_data, size=call_pos.size)
                if put_pos.size > 0:
                    self.sell_put(data=self.put_data, size=put_pos.size)
                
                self.straddle_active = False
                self.call_data = None
                self.put_data = None
    
    def _get_target_expiry(self):
        '''Get target expiration date for options'''
        # This would calculate appropriate expiration date
        # For now, return None (would need expiration calendar)
        return None
    
    def _should_exit_straddle(self):
        '''Determine if straddle should be exited'''
        # Check days to expiration or profit/loss targets
        return False  # Placeholder
