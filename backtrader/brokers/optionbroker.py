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
from collections import defaultdict

from ..brokers.bbroker import BackBroker
from ..option import OptionPosition, OptionContract
from ..order import Order
from ..utils.py3 import with_metaclass


class OptionBroker(BackBroker):
    '''
    Options-aware broker that extends BackBroker with option-specific
    functionality including:
    - Option position tracking
    - Expiration handling
    - Assignment/exercise simulation
    - Greeks-based portfolio analysis
    '''
    
    params = (
        # Option-specific parameters
        ('auto_exercise', True),       # Auto-exercise ITM options at expiry
        ('exercise_threshold', 0.01),  # Minimum ITM amount for auto-exercise
        ('assignment_prob', 1.0),      # Probability of assignment for short positions
        ('early_exercise', False),     # Enable early exercise for American options
        ('option_commission', 0.65),   # Per contract commission
        ('assignment_fee', 15.0),      # Assignment/exercise fee
    )
    
    def __init__(self):
        super(OptionBroker, self).__init__()
        # Option-specific tracking
        self.option_positions = defaultdict(lambda: defaultdict(OptionPosition))
        self.pending_exercises = []
        self.pending_assignments = []
        
    def start(self):
        super(OptionBroker, self).start()
        self.option_positions.clear()
        self.pending_exercises.clear()
        self.pending_assignments.clear()
    
    def next(self):
        '''Called on each bar - handle option expirations and assignments'''
        super(OptionBroker, self).next()
        
        # Check for option expirations
        self._check_expirations()
        
        # Process pending exercises and assignments
        self._process_exercises()
        self._process_assignments()
    
    def _check_expirations(self):
        '''Check for expiring options and handle automatic exercise/assignment'''
        current_date = self._get_current_date()
        
        # Find expiring options
        expiring_positions = []
        for data in self.datas:
            if hasattr(data, 'contract') and isinstance(data.contract, OptionContract):
                if data.contract.is_expired(current_date):
                    position = self.getposition(data)
                    if position.size != 0:
                        expiring_positions.append((data, position))
        
        # Handle each expiring position
        for data, position in expiring_positions:
            self._handle_expiration(data, position)
    
    def _handle_expiration(self, data, position):
        '''Handle expiration for a specific option position'''
        underlying_price = self._get_underlying_price(data)
        intrinsic_value = data.contract.intrinsic_value(underlying_price)
        
        if position.size > 0:  # Long position
            if intrinsic_value >= self.p.exercise_threshold and self.p.auto_exercise:
                self._schedule_exercise(data, position, intrinsic_value)
            else:
                # Option expires worthless
                self._expire_worthless(data, position)
        
        elif position.size < 0:  # Short position
            if intrinsic_value >= self.p.exercise_threshold:
                # Probable assignment
                if self._should_assign():
                    self._schedule_assignment(data, position, intrinsic_value)
                else:
                    # Assignment avoided, option expires
                    self._expire_worthless(data, position)
            else:
                # Option expires worthless (good for short seller)
                self._expire_worthless(data, position)
    
    def _schedule_exercise(self, data, position, intrinsic_value):
        '''Schedule an option exercise'''
        self.pending_exercises.append({
            'data': data,
            'position': position,
            'intrinsic_value': intrinsic_value,
            'exercise_date': self._get_current_date()
        })
    
    def _schedule_assignment(self, data, position, intrinsic_value):
        '''Schedule an option assignment'''
        self.pending_assignments.append({
            'data': data,
            'position': position,
            'intrinsic_value': intrinsic_value,
            'assignment_date': self._get_current_date()
        })
    
    def _process_exercises(self):
        '''Process pending option exercises'''
        for exercise in self.pending_exercises:
            self._execute_exercise(exercise)
        self.pending_exercises.clear()
    
    def _process_assignments(self):
        '''Process pending option assignments'''
        for assignment in self.pending_assignments:
            self._execute_assignment(assignment)
        self.pending_assignments.clear()
    
    def _execute_exercise(self, exercise):
        '''Execute an option exercise'''
        data = exercise['data']
        position = exercise['position']
        contract = data.contract
        
        # Calculate shares to receive/deliver
        shares = abs(position.size) * contract.p.multiplier
        underlying_price = self._get_underlying_price(data)
        
        # Close option position
        self._close_option_position(data, position)
        
        # Create underlying position
        if contract.is_call():
            # Exercise call: buy underlying at strike price
            cost = shares * contract.p.strike
            self.cash -= cost
            self.cash -= self.p.assignment_fee  # Exercise fee
            
            # Add underlying shares to portfolio
            self._add_underlying_position(contract.p.symbol, shares, contract.p.strike)
        
        else:  # put
            # Exercise put: sell underlying at strike price
            proceeds = shares * contract.p.strike
            self.cash += proceeds
            self.cash -= self.p.assignment_fee  # Exercise fee
            
            # Remove underlying shares from portfolio (or go short)
            self._add_underlying_position(contract.p.symbol, -shares, contract.p.strike)
    
    def _execute_assignment(self, assignment):
        '''Execute an option assignment'''
        data = assignment['data']
        position = assignment['position']
        contract = data.contract
        
        # Calculate shares to deliver/receive
        shares = abs(position.size) * contract.p.multiplier
        underlying_price = self._get_underlying_price(data)
        
        # Close option position (assignment)
        self._close_option_position(data, position)
        
        # Underlying position changes (opposite of exercise)
        if contract.is_call():
            # Assigned on short call: deliver underlying at strike price
            proceeds = shares * contract.p.strike
            self.cash += proceeds
            self.cash -= self.p.assignment_fee  # Assignment fee
            
            # Remove underlying shares (or go short)
            self._add_underlying_position(contract.p.symbol, -shares, contract.p.strike)
        
        else:  # put
            # Assigned on short put: buy underlying at strike price
            cost = shares * contract.p.strike
            self.cash -= cost
            self.cash -= self.p.assignment_fee  # Assignment fee
            
            # Add underlying shares
            self._add_underlying_position(contract.p.symbol, shares, contract.p.strike)
    
    def _close_option_position(self, data, position):
        '''Close an option position due to expiration/exercise/assignment'''
        # Set position to zero
        old_position = self.positions[data]
        old_position.size = 0
        old_position.price = 0.0
        
        # Remove from option positions tracking
        if hasattr(data, 'contract'):
            key = self._get_option_key(data.contract)
            if key in self.option_positions[data.contract.p.symbol]:
                del self.option_positions[data.contract.p.symbol][key]
    
    def _add_underlying_position(self, symbol, shares, price):
        '''Add underlying shares to portfolio (placeholder - needs underlying data feed)'''
        # This would need to be connected to the underlying asset's data feed
        # For now, just track the cash impact
        pass
    
    def _expire_worthless(self, data, position):
        '''Handle worthless option expiration'''
        # Close the position
        self._close_option_position(data, position)
        
        # No cash flows for worthless expiration
        # The loss is already reflected in the position value
    
    def _should_assign(self):
        '''Determine if assignment should occur (probabilistic)'''
        import random
        return random.random() < self.p.assignment_prob
    
    def _get_underlying_price(self, option_data):
        '''Get current underlying asset price'''
        if hasattr(option_data, 'underlying_price') and len(option_data.underlying_price):
            return option_data.underlying_price[0]
        
        # Fallback: estimate from option data
        if hasattr(option_data, 'contract'):
            return option_data.contract.p.strike  # Rough estimate
        
        return 100.0  # Default fallback
    
    def _get_current_date(self):
        '''Get current simulation date'''
        if self.datas:
            try:
                return self.datas[0].datetime.date(0)
            except:
                pass
        return datetime.date.today()
    
    def _get_option_key(self, contract):
        '''Generate unique key for option contract'''
        return (contract.p.expiry, contract.p.strike, contract.p.option_type)
    
    def submit(self, order):
        '''Override submit to handle option-specific logic'''
        # Check if this is an option order
        if hasattr(order.data, 'contract') and isinstance(order.data.contract, OptionContract):
            # Add option-specific validation
            if not self._validate_option_order(order):
                order.reject()
                return order
        
        return super(OptionBroker, self).submit(order)
    
    def _validate_option_order(self, order):
        '''Validate option-specific order requirements'''
        contract = order.data.contract
        current_date = self._get_current_date()
        
        # Check if option is already expired
        if contract.is_expired(current_date):
            return False
        
        # Check if sufficient buying power for margin requirements
        # (simplified - real implementation would be more complex)
        if order.isbuy():
            required_cash = abs(order.size) * order.data.close[0] * contract.p.multiplier
            if required_cash > self.cash:
                return False
        
        return True
    
    def getposition(self, data, clone=True):
        '''Override to handle option positions'''
        position = super(OptionBroker, self).getposition(data, clone)
        
        # If this is an option, also track in option-specific structure
        if hasattr(data, 'contract') and isinstance(data.contract, OptionContract):
            contract = data.contract
            symbol = contract.p.symbol
            key = self._get_option_key(contract)
            
            # Update option position tracking
            if position.size != 0:
                opt_pos = self.option_positions[symbol][key]
                if opt_pos.contract is None:
                    opt_pos.contract = contract
                opt_pos.size = position.size
                opt_pos.price = position.price
            elif key in self.option_positions[symbol]:
                # Position closed
                del self.option_positions[symbol][key]
        
        return position
    
    def get_portfolio_greeks(self, symbol=None):
        '''Calculate portfolio Greeks for all or specific underlying'''
        portfolio_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
        
        for data in self.datas:
            if hasattr(data, 'contract') and isinstance(data.contract, OptionContract):
                if symbol is None or data.contract.p.symbol == symbol:
                    position = self.getposition(data)
                    if position.size != 0:
                        # Add position Greeks
                        multiplier = data.contract.p.multiplier
                        try:
                            portfolio_greeks['delta'] += position.size * data.delta[0] * multiplier
                            portfolio_greeks['gamma'] += position.size * data.gamma[0] * multiplier
                            portfolio_greeks['theta'] += position.size * data.theta[0] * multiplier
                            portfolio_greeks['vega'] += position.size * data.vega[0] * multiplier
                            portfolio_greeks['rho'] += position.size * data.rho[0] * multiplier
                        except (AttributeError, IndexError):
                            # Greeks not available
                            pass
        
        return portfolio_greeks
    
    def get_option_positions(self, symbol=None):
        '''Get all option positions for a symbol or all symbols'''
        if symbol:
            return dict(self.option_positions.get(symbol, {}))
        else:
            return dict(self.option_positions)
