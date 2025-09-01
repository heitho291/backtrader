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
import math
from copy import copy

from .utils.py3 import with_metaclass
from .metabase import MetaParams


class OptionContract(with_metaclass(MetaParams, object)):
    '''
    Represents an options contract with all necessary parameters for
    options pricing and backtesting.
    
    Params:
        - symbol: Underlying symbol (e.g., 'AAPL')
        - expiry: Expiration date (datetime.date or datetime.datetime)
        - strike: Strike price (float)
        - option_type: 'call' or 'put'
        - multiplier: Contract multiplier (default 100 for US options)
        - exchange: Exchange name (default 'SMART')
        - currency: Currency (default 'USD')
    '''
    
    params = (
        ('symbol', ''),
        ('expiry', None),
        ('strike', 0.0),
        ('option_type', 'call'),  # 'call' or 'put'
        ('multiplier', 100),
        ('exchange', 'SMART'),
        ('currency', 'USD'),
    )
    
    def __init__(self):
        super(OptionContract, self).__init__()
        self.validate()
    
    def validate(self):
        '''Validate contract parameters'''
        if not self.p.symbol:
            raise ValueError("Symbol must be specified")
        
        if self.p.expiry is None:
            raise ValueError("Expiry date must be specified")
        
        if self.p.strike <= 0:
            raise ValueError("Strike price must be positive")
        
        if self.p.option_type not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'")
    
    def days_to_expiry(self, current_date):
        '''Calculate days to expiry from current date'''
        if isinstance(self.p.expiry, datetime.datetime):
            expiry_date = self.p.expiry.date()
        else:
            expiry_date = self.p.expiry
            
        if isinstance(current_date, datetime.datetime):
            current_date = current_date.date()
            
        delta = expiry_date - current_date
        return max(0, delta.days)
    
    def is_expired(self, current_date):
        '''Check if option is expired'''
        return self.days_to_expiry(current_date) == 0
    
    def is_call(self):
        '''Check if this is a call option'''
        return self.p.option_type.lower() == 'call'
    
    def is_put(self):
        '''Check if this is a put option'''
        return self.p.option_type.lower() == 'put'
    
    def intrinsic_value(self, underlying_price):
        '''Calculate intrinsic value of the option'''
        if self.is_call():
            return max(0, underlying_price - self.p.strike)
        else:  # put
            return max(0, self.p.strike - underlying_price)
    
    def moneyness(self, underlying_price):
        '''Calculate moneyness (S/K for calls, K/S for puts)'''
        if self.is_call():
            return underlying_price / self.p.strike
        else:
            return self.p.strike / underlying_price
    
    def contract_name(self):
        '''Generate a standard contract name'''
        exp_str = self.p.expiry.strftime('%y%m%d') if hasattr(self.p.expiry, 'strftime') else str(self.p.expiry)
        option_code = 'C' if self.is_call() else 'P'
        strike_str = f"{self.p.strike:08.3f}".replace('.', '')
        return f"{self.p.symbol}{exp_str}{option_code}{strike_str}"
    
    def __str__(self):
        return (f"OptionContract({self.p.symbol} {self.p.expiry} "
                f"{self.p.strike} {self.p.option_type.upper()})")
    
    def __repr__(self):
        return self.__str__()


class OptionPosition(object):
    '''
    Keeps track of an option position including the contract details,
    quantity, and average cost basis.
    '''
    
    def __init__(self, contract, size=0, price=0.0):
        self.contract = contract
        self.size = size
        self.price = price if size else 0.0
        self.value = 0.0
        
    def update(self, size, price):
        '''Update position with new transaction'''
        if self.size == 0:
            # Opening new position
            self.size = size
            self.price = price
        elif (self.size > 0 and size > 0) or (self.size < 0 and size < 0):
            # Adding to existing position - calculate average price
            total_cost = (self.size * self.price) + (size * price)
            self.size += size
            self.price = total_cost / self.size if self.size != 0 else 0.0
        else:
            # Reducing or reversing position
            if abs(size) >= abs(self.size):
                # Closing or reversing
                remaining = size + self.size  # net position change
                if remaining == 0:
                    # Exact close
                    self.size = 0
                    self.price = 0.0
                else:
                    # Reversal
                    self.size = remaining
                    self.price = price
            else:
                # Partial close
                self.size += size
                # Keep same average price for remaining position
        
        return self.size, self.price
    
    def market_value(self, market_price):
        '''Calculate current market value of position'''
        return self.size * market_price * self.contract.p.multiplier
    
    def unrealized_pnl(self, market_price):
        '''Calculate unrealized P&L'''
        if self.size == 0:
            return 0.0
        
        cost_basis = self.size * self.price * self.contract.p.multiplier
        market_val = self.market_value(market_price)
        return market_val - cost_basis
    
    def __bool__(self):
        return self.size != 0
    
    __nonzero__ = __bool__
    
    def __len__(self):
        return abs(self.size)
