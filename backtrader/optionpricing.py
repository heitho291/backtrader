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

import math
import datetime
from copy import copy

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    # Fallback for when scipy is not available
    HAS_SCIPY = False
    class MockStats:
        class norm:
            @staticmethod
            def cdf(x):
                # Simple approximation for normal CDF when scipy not available
                return 0.5 * (1 + math.erf(x / math.sqrt(2)))
            
            @staticmethod
            def pdf(x):
                # Simple approximation for normal PDF when scipy not available
                return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    stats = MockStats()

from .utils.py3 import with_metaclass
from .metabase import MetaParams


class OptionPricingModel(with_metaclass(MetaParams, object)):
    '''
    Base class for option pricing models
    '''
    
    def price(self, contract, underlying_price, volatility, risk_free_rate, dividend_yield=0.0):
        '''
        Calculate theoretical option price
        
        Args:
            contract: OptionContract instance
            underlying_price: Current price of underlying asset
            volatility: Implied or historical volatility (annualized)
            risk_free_rate: Risk-free interest rate (annualized)
            dividend_yield: Dividend yield (annualized, default 0.0)
            
        Returns:
            tuple: (option_price, greeks_dict)
        '''
        raise NotImplementedError
    
    def implied_volatility(self, contract, underlying_price, option_price, 
                          risk_free_rate, dividend_yield=0.0):
        '''
        Calculate implied volatility using Newton-Raphson method
        '''
        raise NotImplementedError


class BlackScholesModel(OptionPricingModel):
    '''
    Black-Scholes option pricing model
    '''
    
    def price(self, contract, underlying_price, volatility, risk_free_rate, dividend_yield=0.0):
        '''
        Calculate Black-Scholes option price and Greeks
        '''
        S = underlying_price
        K = contract.p.strike
        T = contract.days_to_expiry(datetime.datetime.now()) / 365.25
        r = risk_free_rate
        q = dividend_yield
        sigma = volatility
        
        # Handle edge cases
        if T <= 0:
            # Expired option
            intrinsic = contract.intrinsic_value(S)
            return intrinsic, self._zero_greeks()
        
        if sigma <= 0:
            # Zero volatility
            if contract.is_call():
                if S > K:
                    return S - K * math.exp(-r * T), self._zero_greeks()
                else:
                    return 0.0, self._zero_greeks()
            else:  # put
                if S < K:
                    return K * math.exp(-r * T) - S, self._zero_greeks()
                else:
                    return 0.0, self._zero_greeks()
        
        # Calculate d1 and d2
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        # Standard normal CDF
        Nd1 = stats.norm.cdf(d1)
        Nd2 = stats.norm.cdf(d2)
        Nmd1 = stats.norm.cdf(-d1)
        Nmd2 = stats.norm.cdf(-d2)
        
        # Standard normal PDF
        nd1 = stats.norm.pdf(d1)
        
        # Discount factors
        df_r = math.exp(-r * T)
        df_q = math.exp(-q * T)
        
        if contract.is_call():
            # Call option price
            price = S * df_q * Nd1 - K * df_r * Nd2
            
            # Greeks
            delta = df_q * Nd1
            gamma = df_q * nd1 / (S * sigma * math.sqrt(T))
            theta = ((-S * df_q * nd1 * sigma / (2 * math.sqrt(T)) - 
                     r * K * df_r * Nd2 + q * S * df_q * Nd1) / 365.25)
            vega = S * df_q * nd1 * math.sqrt(T) / 100  # Per 1% vol change
            rho = K * T * df_r * Nd2 / 100  # Per 1% rate change
            
        else:  # put
            # Put option price
            price = K * df_r * Nmd2 - S * df_q * Nmd1
            
            # Greeks
            delta = -df_q * Nmd1
            gamma = df_q * nd1 / (S * sigma * math.sqrt(T))
            theta = ((-S * df_q * nd1 * sigma / (2 * math.sqrt(T)) + 
                     r * K * df_r * Nmd2 - q * S * df_q * Nmd1) / 365.25)
            vega = S * df_q * nd1 * math.sqrt(T) / 100  # Per 1% vol change
            rho = -K * T * df_r * Nmd2 / 100  # Per 1% rate change
        
        greeks = {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
        
        return max(0, price), greeks
    
    def implied_volatility(self, contract, underlying_price, option_price, 
                          risk_free_rate, dividend_yield=0.0, max_iterations=100, tolerance=1e-6):
        '''
        Calculate implied volatility using Newton-Raphson method
        '''
        if option_price <= 0:
            return 0.0
        
        # Initial guess
        vol = 0.3  # 30% initial guess
        
        for i in range(max_iterations):
            try:
                bs_price, greeks = self.price(contract, underlying_price, vol, 
                                            risk_free_rate, dividend_yield)
                
                price_diff = bs_price - option_price
                vega = greeks['vega'] * 100  # Convert back to per unit vol change
                
                if abs(price_diff) < tolerance:
                    return vol
                
                if vega == 0:
                    break
                
                # Newton-Raphson update
                vol_new = vol - price_diff / vega
                
                # Ensure vol stays positive and reasonable
                vol_new = max(0.001, min(5.0, vol_new))
                
                if abs(vol_new - vol) < tolerance:
                    return vol_new
                
                vol = vol_new
                
            except (ZeroDivisionError, ValueError, OverflowError):
                break
        
        return vol
    
    def _zero_greeks(self):
        '''Return zero Greeks for edge cases'''
        return {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }


class BinomialModel(OptionPricingModel):
    '''
    Binomial tree option pricing model (Cox-Ross-Rubinstein)
    '''
    
    params = (
        ('steps', 100),  # Number of time steps
    )
    
    def price(self, contract, underlying_price, volatility, risk_free_rate, dividend_yield=0.0):
        '''
        Calculate option price using binomial tree
        '''
        S = underlying_price
        K = contract.p.strike
        T = contract.days_to_expiry(datetime.datetime.now()) / 365.25
        r = risk_free_rate
        q = dividend_yield
        sigma = volatility
        n = self.p.steps
        
        if T <= 0:
            intrinsic = contract.intrinsic_value(S)
            return intrinsic, self._zero_greeks()
        
        # Time step
        dt = T / n
        
        # Up and down factors
        u = math.exp(sigma * math.sqrt(dt))
        d = 1 / u
        
        # Risk-neutral probability
        p = (math.exp((r - q) * dt) - d) / (u - d)
        
        # Initialize asset prices at maturity
        asset_prices = [S * (u ** (n - i)) * (d ** i) for i in range(n + 1)]
        
        # Initialize option values at maturity
        if contract.is_call():
            option_values = [max(0, price - K) for price in asset_prices]
        else:
            option_values = [max(0, K - price) for price in asset_prices]
        
        # Backward induction
        for step in range(n - 1, -1, -1):
            for i in range(step + 1):
                # Continuation value
                cont_value = math.exp(-r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
                
                # For American options, check early exercise
                asset_price = S * (u ** (step - i)) * (d ** i)
                if contract.is_call():
                    exercise_value = max(0, asset_price - K)
                else:
                    exercise_value = max(0, K - asset_price)
                
                # For European options, use continuation value only
                # For American options, use max of continuation and exercise
                option_values[i] = max(cont_value, exercise_value)
        
        # Calculate approximate Greeks using finite differences
        greeks = self._calculate_greeks_fd(contract, underlying_price, volatility, 
                                          risk_free_rate, dividend_yield)
        
        return option_values[0], greeks
    
    def _calculate_greeks_fd(self, contract, S, sigma, r, q):
        '''Calculate Greeks using finite differences'''
        # Small changes for finite differences
        dS = S * 0.01  # 1% change in underlying
        dsigma = 0.01  # 1% vol change
        dr = 0.0001   # 1 bp rate change
        dt = 1/365.25  # 1 day time change
        
        # Base price
        price0, _ = self.price(contract, S, sigma, r, q)
        
        # Delta (finite difference)
        try:
            price_up, _ = self.price(contract, S + dS, sigma, r, q)
            price_down, _ = self.price(contract, S - dS, sigma, r, q)
            delta = (price_up - price_down) / (2 * dS)
        except:
            delta = 0.0
        
        # Gamma (second derivative)
        try:
            gamma = (price_up - 2 * price0 + price_down) / (dS ** 2)
        except:
            gamma = 0.0
        
        # Vega
        try:
            price_vol_up, _ = self.price(contract, S, sigma + dsigma, r, q)
            vega = (price_vol_up - price0) / dsigma / 100
        except:
            vega = 0.0
        
        # Rho
        try:
            price_rate_up, _ = self.price(contract, S, sigma, r + dr, q)
            rho = (price_rate_up - price0) / dr / 100
        except:
            rho = 0.0
        
        # Theta (approximate using shorter time to expiry)
        try:
            # Create contract with 1 day less to expiry
            import datetime
            new_expiry = contract.p.expiry - datetime.timedelta(days=1)
            temp_contract = copy(contract)
            temp_contract.p.expiry = new_expiry
            price_theta, _ = self.price(temp_contract, S, sigma, r, q)
            theta = (price_theta - price0) / 365.25
        except:
            theta = 0.0
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def _zero_greeks(self):
        return {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
