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
from ..feed import DataBase
from ..option import OptionContract
from ..optionpricing import BlackScholesModel
from ..utils.py3 import with_metaclass


class OptionDataBase(DataBase):
    '''
    Base class for option data feeds. Extends regular DataBase with
    option-specific functionality.
    '''
    
    params = (
        # Option contract parameters
        ('symbol', ''),
        ('expiry', None),
        ('strike', 0.0),
        ('option_type', 'call'),
        ('multiplier', 100),
        
        # Underlying data reference
        ('underlying_data', None),  # Reference to underlying asset data
        
        # Pricing model parameters
        ('pricing_model', None),  # Pricing model instance
        ('risk_free_rate', 0.02),  # Annual risk-free rate
        ('dividend_yield', 0.0),   # Annual dividend yield
        ('volatility', 0.25),      # Implied volatility override
        ('use_iv', False),         # Use implied volatility from data if available
        
        # Greeks calculation
        ('calculate_greeks', True),
    )
    
    # Additional lines for options data
    lines = ('bid', 'ask', 'impliedvol', 'delta', 'gamma', 'theta', 'vega', 'rho',
             'openinterest', 'underlying_price')
    
    def __init__(self):
        super(OptionDataBase, self).__init__()
        
        # Create option contract
        self.contract = OptionContract(
            symbol=self.p.symbol,
            expiry=self.p.expiry,
            strike=self.p.strike,
            option_type=self.p.option_type,
            multiplier=self.p.multiplier
        )
        
        # Initialize pricing model
        if self.p.pricing_model is None:
            self.pricing_model = BlackScholesModel()
        else:
            self.pricing_model = self.p.pricing_model
    
    def _load(self):
        '''
        Override to add option-specific processing
        '''
        # Load base data first
        if not super(OptionDataBase, self)._load():
            return False
        
        # Check if option is expired
        current_date = self.datetime.date(0)
        if self.contract.is_expired(current_date):
            # Option expired, set values to intrinsic value only
            self._set_expired_values()
        else:
            # Calculate theoretical values and Greeks
            self._calculate_option_values()
        
        return True
    
    def _set_expired_values(self):
        '''Set values for expired options'''
        underlying_price = self._get_underlying_price()
        intrinsic_value = self.contract.intrinsic_value(underlying_price)
        
        # Set all prices to intrinsic value
        self.lines.open[0] = intrinsic_value
        self.lines.high[0] = intrinsic_value
        self.lines.low[0] = intrinsic_value
        self.lines.close[0] = intrinsic_value
        
        # Zero out Greeks and other values
        if hasattr(self.lines, 'impliedvol'):
            self.lines.impliedvol[0] = 0.0
        if hasattr(self.lines, 'delta'):
            self.lines.delta[0] = 0.0
            self.lines.gamma[0] = 0.0
            self.lines.theta[0] = 0.0
            self.lines.vega[0] = 0.0
            self.lines.rho[0] = 0.0
    
    def _calculate_option_values(self):
        '''Calculate theoretical option values and Greeks'''
        if not self.p.calculate_greeks:
            return
        
        underlying_price = self._get_underlying_price()
        
        # Get volatility
        if self.p.use_iv and hasattr(self.lines, 'impliedvol') and self.lines.impliedvol[0] > 0:
            volatility = self.lines.impliedvol[0]
        else:
            volatility = self.p.volatility
        
        # Calculate theoretical price and Greeks
        try:
            theo_price, greeks = self.pricing_model.price(
                self.contract,
                underlying_price,
                volatility,
                self.p.risk_free_rate,
                self.p.dividend_yield
            )
            
            # Store Greeks in data lines
            if hasattr(self.lines, 'delta'):
                self.lines.delta[0] = greeks['delta']
                self.lines.gamma[0] = greeks['gamma']
                self.lines.theta[0] = greeks['theta']
                self.lines.vega[0] = greeks['vega']
                self.lines.rho[0] = greeks['rho']
            
            # Store underlying price
            if hasattr(self.lines, 'underlying_price'):
                self.lines.underlying_price[0] = underlying_price
                
        except Exception as e:
            # If calculation fails, set Greeks to zero
            if hasattr(self.lines, 'delta'):
                self.lines.delta[0] = 0.0
                self.lines.gamma[0] = 0.0
                self.lines.theta[0] = 0.0
                self.lines.vega[0] = 0.0
                self.lines.rho[0] = 0.0
    
    def _get_underlying_price(self):
        '''Get current underlying asset price'''
        if self.p.underlying_data is not None:
            try:
                return self.p.underlying_data.close[0]
            except (IndexError, AttributeError):
                pass
        
        # Fallback: try to estimate from option price (very rough)
        current_price = self.lines.close[0]
        if current_price > 0:
            if self.contract.is_call():
                return self.contract.p.strike + current_price
            else:
                return self.contract.p.strike - current_price
        
        return self.contract.p.strike  # Last resort


class OptionCSVData(OptionDataBase):
    '''
    CSV data feed for options data.
    
    Expected CSV format:
    Date,Time,Open,High,Low,Close,Volume,OpenInterest,Bid,Ask,ImpliedVol,UnderlyingPrice
    '''
    
    params = (
        ('bid', 8),           # Column index for bid price
        ('ask', 9),           # Column index for ask price  
        ('impliedvol', 10),   # Column index for implied volatility
        ('underlying_price', 11),  # Column index for underlying price
    )
    
    def _loadline(self, linetokens):
        '''Load a single line of CSV data'''
        # Load base OHLCV data
        if not super(OptionCSVData, self)._loadline(linetokens):
            return False
        
        # Load option-specific data
        try:
            if self.p.bid >= 0 and len(linetokens) > self.p.bid:
                self.lines.bid[0] = float(linetokens[self.p.bid])
            
            if self.p.ask >= 0 and len(linetokens) > self.p.ask:
                self.lines.ask[0] = float(linetokens[self.p.ask])
            
            if self.p.impliedvol >= 0 and len(linetokens) > self.p.impliedvol:
                self.lines.impliedvol[0] = float(linetokens[self.p.impliedvol])
            
            if self.p.underlying_price >= 0 and len(linetokens) > self.p.underlying_price:
                self.lines.underlying_price[0] = float(linetokens[self.p.underlying_price])
                
        except (ValueError, IndexError):
            # If we can't parse optional fields, continue with base data
            pass
        
        return True


class SyntheticOptionData(OptionDataBase):
    '''
    Synthetic option data generated from underlying asset data using
    pricing models. Useful for backtesting when historical option
    data is not available.
    '''
    
    params = (
        ('bid_ask_spread', 0.05),  # Bid-ask spread as fraction of mid price
        ('volume_model', None),    # Volume generation model
    )
    
    def _load(self):
        '''Generate synthetic option data'''
        # Check if underlying data is available
        if self.p.underlying_data is None:
            return False
        
        # Check if we have underlying data for current bar
        try:
            underlying_price = self.p.underlying_data.close[0]
            underlying_datetime = self.p.underlying_data.datetime[0]
        except (IndexError, AttributeError):
            return False
        
        # Set datetime from underlying
        self.lines.datetime[0] = underlying_datetime
        
        # Check if option is expired
        current_date = self.datetime.date(0)
        if self.contract.is_expired(current_date):
            self._set_expired_values()
            return True
        
        # Calculate theoretical option price
        try:
            theo_price, greeks = self.pricing_model.price(
                self.contract,
                underlying_price,
                self.p.volatility,
                self.p.risk_free_rate,
                self.p.dividend_yield
            )
            
            # Set OHLC values (simplified - using same price for all)
            self.lines.open[0] = theo_price
            self.lines.high[0] = theo_price * 1.02  # Simple high estimation
            self.lines.low[0] = theo_price * 0.98   # Simple low estimation
            self.lines.close[0] = theo_price
            
            # Set bid/ask based on spread
            spread = theo_price * self.p.bid_ask_spread
            self.lines.bid[0] = theo_price - spread / 2
            self.lines.ask[0] = theo_price + spread / 2
            
            # Set Greeks
            self.lines.delta[0] = greeks['delta']
            self.lines.gamma[0] = greeks['gamma']
            self.lines.theta[0] = greeks['theta']
            self.lines.vega[0] = greeks['vega']
            self.lines.rho[0] = greeks['rho']
            
            # Set underlying price
            self.lines.underlying_price[0] = underlying_price
            
            # Generate synthetic volume (simple model)
            self.lines.volume[0] = self._generate_volume(theo_price, greeks)
            
            # Open interest (static for now)
            self.lines.openinterest[0] = 1000  # Default value
            
            return True
            
        except Exception as e:
            return False
    
    def _generate_volume(self, price, greeks):
        '''Generate synthetic volume based on option characteristics'''
        # Simple volume model based on delta and price
        base_volume = 100
        
        # Higher volume for at-the-money options (delta around 0.5 for calls)
        delta_factor = 1 + (1 - abs(abs(greeks['delta']) - 0.5) * 2)
        
        # Higher volume for lower prices (more affordable)
        price_factor = max(0.1, 10 / max(price, 0.01))
        
        volume = int(base_volume * delta_factor * price_factor)
        return max(1, volume)  # Ensure at least 1


class OptionChain(object):
    '''
    Represents a complete option chain for an underlying asset.
    Manages multiple option contracts with different strikes and expirations.
    '''
    
    def __init__(self, symbol, underlying_data=None):
        self.symbol = symbol
        self.underlying_data = underlying_data
        self.contracts = {}  # key: (expiry, strike, option_type)
        self.data_feeds = {}  # option data feeds
    
    def add_contract(self, expiry, strike, option_type, data_feed=None):
        '''Add an option contract to the chain'''
        key = (expiry, strike, option_type.lower())
        
        if data_feed is None:
            # Create synthetic data feed
            contract = OptionContract(
                symbol=self.symbol,
                expiry=expiry,
                strike=strike,
                option_type=option_type
            )
            data_feed = SyntheticOptionData(
                symbol=self.symbol,
                expiry=expiry,
                strike=strike,
                option_type=option_type,
                underlying_data=self.underlying_data
            )
        
        self.contracts[key] = data_feed.contract
        self.data_feeds[key] = data_feed
    
    def get_contract(self, expiry, strike, option_type):
        '''Get option contract by parameters'''
        key = (expiry, strike, option_type.lower())
        return self.contracts.get(key)
    
    def get_data_feed(self, expiry, strike, option_type):
        '''Get option data feed by parameters'''
        key = (expiry, strike, option_type.lower())
        return self.data_feeds.get(key)
    
    def get_atm_contracts(self, expiry, underlying_price=None):
        '''Get at-the-money contracts for given expiry'''
        if underlying_price is None and self.underlying_data is not None:
            try:
                underlying_price = self.underlying_data.close[0]
            except:
                return None, None
        
        if underlying_price is None:
            return None, None
        
        # Find closest strike to underlying price
        strikes = set()
        for (exp, strike, opt_type) in self.contracts.keys():
            if exp == expiry:
                strikes.add(strike)
        
        if not strikes:
            return None, None
        
        closest_strike = min(strikes, key=lambda x: abs(x - underlying_price))
        
        call_key = (expiry, closest_strike, 'call')
        put_key = (expiry, closest_strike, 'put')
        
        call_contract = self.contracts.get(call_key)
        put_contract = self.contracts.get(put_key)
        
        return call_contract, put_contract
