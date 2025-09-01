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

from .comminfo import CommInfoBase


class OptionCommissionInfo(CommInfoBase):
    '''
    Commission scheme specifically designed for options trading.
    
    Typical options commission structures:
    - Per contract fee (e.g., $0.65 per contract)
    - Plus percentage of premium or fixed minimum
    - Assignment/exercise fees
    - Different rates for opening vs closing
    '''
    
    params = (
        ('commission', 0.65),        # Per contract commission
        ('min_commission', 1.0),     # Minimum commission per order
        ('percentage', 0.0),         # Percentage of premium (if any)
        ('assignment_fee', 15.0),    # Fee for assignment/exercise
        ('closing_reduction', 0.5),  # Reduction factor for closing trades
        ('multiplier', 100),         # Options multiplier (typically 100)
    )
    
    def __init__(self):
        super(OptionCommissionInfo, self).__init__()
        # Options are not stocklike by default
        self._stocklike = False
        self._commtype = self.COMM_FIXED
    
    def getcommission(self, size, price):
        '''
        Calculate commission for options trade
        
        Args:
            size: Number of contracts (can be negative for short)
            price: Option premium per contract
            
        Returns:
            Commission amount
        '''
        contracts = abs(size)
        
        # Base commission per contract
        commission = contracts * self.p.commission
        
        # Add percentage of premium if specified
        if self.p.percentage > 0:
            premium_value = contracts * price * self.p.multiplier
            commission += premium_value * self.p.percentage
        
        # Apply minimum commission
        commission = max(commission, self.p.min_commission)
        
        return commission
    
    def getoperationcost(self, size, price):
        '''
        Calculate total cost of opening an options position
        
        For options, this includes:
        - Premium paid/received
        - Commission
        '''
        premium_cost = abs(size) * price * self.p.multiplier
        commission = self.getcommission(size, price)
        
        if size > 0:  # Buying options
            return premium_cost + commission
        else:  # Selling options
            return commission  # Premium is credited
    
    def getvalue(self, position, price):
        '''
        Calculate current market value of options position
        '''
        return position.size * price * self.p.multiplier
    
    def get_margin(self, price):
        '''
        Calculate margin requirement for options
        
        For long options: full premium paid (no additional margin)
        For short options: varies by strategy and underlying
        '''
        # Simplified margin calculation
        # Real implementations would be much more complex
        return price * self.p.multiplier * 0.2  # 20% of premium as rough estimate


class EquityOptionCommissionInfo(OptionCommissionInfo):
    '''
    Standard equity options commission structure
    '''
    params = (
        ('commission', 0.65),     # $0.65 per contract
        ('min_commission', 1.0),  # $1.00 minimum
        ('multiplier', 100),      # 100 shares per contract
    )


class IndexOptionCommissionInfo(OptionCommissionInfo):
    '''
    Index options commission structure
    Often has different fees due to cash settlement
    '''
    params = (
        ('commission', 0.75),     # Slightly higher per contract
        ('min_commission', 1.0),
        ('multiplier', 100),
        ('assignment_fee', 0.0),  # No assignment for cash-settled
    )


class WeeklyOptionCommissionInfo(OptionCommissionInfo):
    '''
    Weekly options commission structure
    Some brokers charge higher fees for weeklies
    '''
    params = (
        ('commission', 0.75),     # Higher fee for weeklies
        ('min_commission', 1.0),
        ('multiplier', 100),
    )


class PennyOptionCommissionInfo(OptionCommissionInfo):
    '''
    Commission structure for penny options
    (Options trading below $0.05)
    '''
    params = (
        ('commission', 0.50),     # Lower per contract for penny options
        ('min_commission', 0.50), # Lower minimum
        ('multiplier', 100),
    )
    
    def getcommission(self, size, price):
        '''Override to handle penny option pricing'''
        if price < 0.05:
            # Special pricing for penny options
            contracts = abs(size)
            return max(contracts * self.p.commission, self.p.min_commission)
        else:
            return super(PennyOptionCommissionInfo, self).getcommission(size, price)
