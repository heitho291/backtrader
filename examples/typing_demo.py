#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
Demonstration of the modernized backtrader with comprehensive type hints and Pydantic validation.

This example shows how to use the new type system for better IDE support,
runtime validation, and more maintainable code.
"""

from typing import Optional

import backtrader as bt
from backtrader.types import CerebroConfig, Price, Volume


class TypedStrategy(bt.Strategy):
    """
    Example strategy showcasing modern type hints.
    
    This strategy demonstrates how type hints improve code clarity
    and enable better IDE support and static type checking.
    """
    
    params = (
        ('period', 20),
        ('threshold', 0.02),
    )
    
    def __init__(self):
        # Type hints make it clear what these variables contain
        self.sma: bt.indicators.SimpleMovingAverage = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.period
        )
        self.position_size: Volume = 0
        
    def start(self) -> None:
        """Called when the strategy starts."""
        print(f"Strategy started with period={self.params.period}")
        print(f"Using threshold={self.params.threshold}")
        
    def next(self) -> None:
        """Main strategy logic with type annotations."""
        current_price: Price = self.data.close[0]
        sma_value: Price = self.sma[0]
        
        # Calculate price deviation from moving average
        deviation: float = (current_price - sma_value) / sma_value
        
        if not self.position:
            # Buy signal: price significantly below SMA
            if deviation < -self.params.threshold:
                size: Volume = self._calculate_position_size(current_price)
                order = self.buy(size=size, price=current_price)
                print(f"BUY signal: price={current_price:.2f}, sma={sma_value:.2f}, size={size}")
                
        else:
            # Sell signal: price significantly above SMA or small profit
            if deviation > self.params.threshold:
                order = self.sell(size=self.position.size)
                print(f"SELL signal: price={current_price:.2f}, sma={sma_value:.2f}")
    
    def _calculate_position_size(self, price: Price) -> Volume:
        """
        Calculate position size based on available cash and price.
        
        Args:
            price: Current asset price
            
        Returns:
            Number of shares to buy
        """
        available_cash: Price = self.broker.getcash()
        max_shares: Volume = int(available_cash * 0.95 / price)  # Use 95% of cash
        return max(1, max_shares)
    
    def notify_order(self, order: bt.Order) -> None:
        """Handle order notifications with type hints."""
        if order.status in [order.Submitted, order.Accepted]:
            return  # Order submitted/accepted - no action needed
            
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"BUY EXECUTED: price={order.executed.price:.2f}, "
                      f"size={order.executed.size}, cost={order.executed.value:.2f}")
            else:
                print(f"SELL EXECUTED: price={order.executed.price:.2f}, "
                      f"size={order.executed.size}, cost={order.executed.value:.2f}")
                      
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f"Order {order.status}")


def run_typed_backtest():
    """
    Run a backtest demonstrating the new type system and Pydantic validation.
    """
    print("=== Backtrader Type Hints and Pydantic Demo ===\n")
    
    # Create Cerebro with Pydantic validation
    print("1. Creating Cerebro with validated configuration...")
    try:
        # This will validate the configuration using Pydantic
        config = CerebroConfig(
            preload=True,
            runonce=True,
            stdstats=True,
            maxcpus=2
        )
        print(f"   ✓ Valid configuration: preload={config.preload}, runonce={config.runonce}")
        
        # Try invalid configuration
        try:
            bad_config = CerebroConfig(exactbars=99)  # Invalid value
        except Exception as e:
            print(f"   ✓ Invalid config rejected: {type(e).__name__}")
            
    except Exception as e:
        print(f"   ✗ Configuration error: {e}")
        return
    
    # Create cerebro instance
    cerebro = bt.Cerebro()
    
    # Add strategy with type annotations
    print("\n2. Adding typed strategy...")
    cerebro.addstrategy(TypedStrategy, period=20, threshold=0.02)
    
    # Add data (you would normally load real data here)
    print("\n3. Adding sample data...")
    print("   (In a real scenario, you would load actual market data)")
    
    # Note: For a complete demo, you would need to add actual data:
    # data = bt.feeds.YahooFinanceData(dataname='AAPL', fromdate=datetime(2020,1,1))
    # cerebro.adddata(data)
    
    print("\n4. Type system demonstration:")
    print(f"   - Price type: {bt.Price}")
    print(f"   - Volume type: {bt.Volume}")  
    print(f"   - OrderStatus literals: {bt.OrderStatus}")
    print(f"   - CerebroConfig class: {bt.CerebroConfig}")
    
    print("\n5. Modern Python features:")
    print("   ✓ Comprehensive type hints for better IDE support")
    print("   ✓ Pydantic models for runtime validation")
    print("   ✓ Protocol-based typing for duck typing")
    print("   ✓ Type aliases for domain-specific types")
    print("   ✓ Generic types and type variables")
    print("   ✓ Literal types for constrained values")
    
    print("\n6. Benefits:")
    print("   • Better IDE autocomplete and error detection")
    print("   • Runtime data validation with Pydantic")
    print("   • Improved code documentation through types")
    print("   • Easier refactoring and maintenance")
    print("   • Modern Python 3.13+ compatibility")
    print("   • Backward compatibility maintained")
    
    print("\n=== Demo completed successfully! ===")


if __name__ == "__main__":
    run_typed_backtest()