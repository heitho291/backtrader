#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# Copyright (C) 2015-2025 Daniel Rodriguez
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
"""
Type definitions for the backtrader library.

This module provides comprehensive type hints for the entire backtrader
ecosystem, including protocols, type aliases, and validation models using
Pydantic for enhanced runtime type checking and data validation.
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeAlias,
    TypeVar,
    Union,
    runtime_checkable,
)

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing_extensions import TypedDict, NotRequired

# Import backtrader types for type checking
if False:  # TYPE_CHECKING equivalent but allows runtime access
    from backtrader import feed


# =============================================================================
# Basic Type Aliases
# =============================================================================

# Numeric types commonly used in trading
Numeric: TypeAlias = Union[int, float, Decimal]
Price: TypeAlias = Union[int, float, Decimal]
Volume: TypeAlias = Union[int, float]
Percentage: TypeAlias = Union[int, float]

# Date/Time types
DateType: TypeAlias = Union[datetime.date, datetime.datetime, str, float]
TimeType: TypeAlias = Union[datetime.time, datetime.datetime, str, float]
TimeDelta: TypeAlias = Union[datetime.timedelta, int, float]

# Generic type variables
T = TypeVar('T')
StrategyType = TypeVar('StrategyType', bound='Strategy')
IndicatorType = TypeVar('IndicatorType', bound='Indicator')
DataType = TypeVar('DataType', bound='feed.AbstractDataBase')


# =============================================================================
# Order Related Types
# =============================================================================

OrderStatus: TypeAlias = Literal[
    'Created', 'Submitted', 'Accepted', 'Partial', 'Completed',
    'Canceled', 'Expired', 'Margin', 'Rejected'
]

OrderType: TypeAlias = Literal[
    'Market', 'Limit', 'Stop', 'StopLimit', 'Close', 'StopTrail', 'StopTrailLimit'
]

OrderExecution: TypeAlias = Literal['Market', 'Close', 'None']

BuySell: TypeAlias = Literal['Buy', 'Sell']


class OrderData(BaseModel):
    """Pydantic model for order data validation."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
        populate_by_name=True,
    )
    
    status: OrderStatus
    order_type: OrderType = Field(alias='type')
    size: Volume
    price: Optional[Price] = None
    pricelimit: Optional[Price] = None
    trailpercent: Optional[Percentage] = None
    trailamount: Optional[Price] = None
    exectype: OrderExecution = 'Market'
    valid: Optional[DateType] = None
    tradeid: int = 0
    historical: bool = False
    
    @field_validator('size')
    @classmethod
    def validate_size(cls, v: Volume) -> Volume:
        if v == 0:
            raise ValueError("Order size cannot be zero")
        return v
    
    @field_validator('price', 'pricelimit', 'trailamount')
    @classmethod
    def validate_positive_price(cls, v: Optional[Price]) -> Optional[Price]:
        if v is not None and v < 0:
            raise ValueError("Price values must be non-negative")
        return v


# =============================================================================
# Data Feed Types
# =============================================================================

OHLCV: TypeAlias = Tuple[Price, Price, Price, Price, Volume]

TimeFrameType: TypeAlias = Literal[
    'Ticks', 'MicroSeconds', 'Seconds', 'Minutes', 'Days', 'Weeks', 'Months', 'Years'
]

Compression: TypeAlias = int


class BarData(TypedDict):
    """Type definition for a single bar of OHLCV data."""
    datetime_value: datetime.datetime
    open: Price
    high: Price
    low: Price
    close: Price
    volume: Volume
    openinterest: NotRequired[Volume]


class DataFeedConfig(BaseModel):
    """Configuration for data feeds with validation."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    timeframe: TimeFrameType = 'Days'
    compression: Compression = 1
    fromdate: Optional[DateType] = None
    todate: Optional[DateType] = None
    dataname: Optional[str] = None
    name: Optional[str] = None
    plotname: Optional[str] = None
    
    @field_validator('compression')
    @classmethod
    def validate_compression(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Compression must be at least 1")
        return v


# =============================================================================
# Strategy and Analysis Types
# =============================================================================

class ParameterDict(TypedDict, total=False):
    """Type definition for strategy parameters."""
    name: str
    value: Any
    doc: str
    

class StrategyParams(BaseModel):
    """Base class for strategy parameter validation."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra='allow',  # Allow additional parameters
        arbitrary_types_allowed=True,
    )


class PositionData(BaseModel):
    """Position information with validation."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    size: Volume
    price: Price
    upnl: Price = Field(default=0.0, description="Unrealized P&L")
    datetime_value: Optional[datetime.datetime] = Field(default=None, alias="datetime")
    
    @field_validator('size')
    @classmethod
    def validate_position_size(cls, v: Volume) -> Volume:
        # Position size can be positive (long), negative (short), or zero
        return v


# =============================================================================
# Indicator Types
# =============================================================================

class IndicatorConfig(BaseModel):
    """Configuration for indicators."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    period: int = Field(gt=0, description="Lookback period")
    plotname: str = ""
    subplot: bool = True
    plotlines: Dict[str, Any] = Field(default_factory=dict)
    plotyhlines: List[float] = Field(default_factory=list)


# =============================================================================
# Protocols for Duck Typing
# =============================================================================

@runtime_checkable
class Datasource(Protocol):
    """Protocol for data sources."""
    
    def islive(self) -> bool:
        """Check if data source provides live data."""
        ...
    
    def haslivedata(self) -> bool:
        """Check if live data is available."""
        ...
    
    def next(self) -> bool:
        """Move to next data point."""
        ...


@runtime_checkable
class Broker(Protocol):
    """Protocol for broker implementations."""
    
    def buy(self, size: Volume, price: Optional[Price] = None, **kwargs: Any) -> Any:
        """Place a buy order."""
        ...
    
    def sell(self, size: Volume, price: Optional[Price] = None, **kwargs: Any) -> Any:
        """Place a sell order."""
        ...
    
    def get_cash(self) -> Price:
        """Get available cash."""
        ...
    
    def get_value(self) -> Price:
        """Get total portfolio value."""
        ...


@runtime_checkable
class Strategy(Protocol):
    """Protocol for trading strategies."""
    
    def next(self) -> None:
        """Execute strategy logic for current bar."""
        ...
    
    def start(self) -> None:
        """Called when strategy starts."""
        ...
    
    def stop(self) -> None:
        """Called when strategy stops."""
        ...
    
    def notify_order(self, order: Any) -> None:
        """Handle order notifications."""
        ...
    
    def notify_trade(self, trade: Any) -> None:
        """Handle trade notifications."""
        ...


@runtime_checkable
class Indicator(Protocol):
    """Protocol for technical indicators."""
    
    def next(self) -> None:
        """Calculate indicator value for current period."""
        ...
    
    def once(self, start: int, end: int) -> None:
        """Vectorized calculation for range."""
        ...


@runtime_checkable
class Observer(Protocol):
    """Protocol for observers (stats collection)."""
    
    def next(self) -> None:
        """Update observer with current data."""
        ...


@runtime_checkable
class Analyzer(Protocol):
    """Protocol for analyzers (performance metrics)."""
    
    def start(self) -> None:
        """Initialize analyzer."""
        ...
    
    def stop(self) -> None:
        """Finalize analysis."""
        ...
    
    def get_analysis(self) -> Dict[str, Any]:
        """Get analysis results."""
        ...


# =============================================================================
# Cerebro Configuration Types
# =============================================================================

class CerebroConfig(BaseModel):
    """Configuration for Cerebro engine with comprehensive validation."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
    
    preload: bool = True
    runonce: bool = True
    live: bool = False
    exactbars: int = 1
    stdstats: bool = True
    oldbuysell: bool = False
    lookahead: int = 0
    optdatas: bool = True
    optreturn: bool = True
    objcache: bool = False
    writer_csv: bool = False
    tradehistory: bool = False
    maxcpus: Optional[int] = None
    
    @field_validator('exactbars')
    @classmethod
    def validate_exactbars(cls, v: int) -> int:
        if v < -1 or v > 2:
            raise ValueError("exactbars must be between -1 and 2")
        return v
    
    @field_validator('maxcpus')
    @classmethod
    def validate_maxcpus(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError("maxcpus must be at least 1")
        return v


# =============================================================================
# Trade and Performance Types
# =============================================================================

class TradeData(BaseModel):
    """Trade information with validation."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    tradeid: int
    size: Volume
    price: Price
    value: Price
    commission: Price = 0.0
    pnl: Price = 0.0
    pnlcomm: Price = 0.0
    justopened: bool = False
    isopen: bool = True
    isclosed: bool = False
    baropen: int = 0
    dtopen: Optional[datetime.datetime] = None
    barclose: int = 0
    dtclose: Optional[datetime.datetime] = None
    
    @field_validator('tradeid')
    @classmethod
    def validate_tradeid(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Trade ID must be non-negative")
        return v


class PerformanceMetrics(BaseModel):
    """Performance analysis results."""
    
    model_config = ConfigDict(validate_assignment=True)
    
    total_return: Percentage = 0.0
    sharpe_ratio: Optional[float] = None
    max_drawdown: Percentage = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_trade: Price = 0.0
    avg_win: Price = 0.0
    avg_loss: Price = 0.0
    profit_factor: Optional[float] = None
    
    @field_validator('total_trades', 'winning_trades', 'losing_trades')
    @classmethod
    def validate_trade_counts(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Trade counts must be non-negative")
        return v


# =============================================================================
# Function Type Aliases
# =============================================================================

StrategyClass: TypeAlias = Type[Strategy]
IndicatorClass: TypeAlias = Type[Indicator]
ObserverClass: TypeAlias = Type[Observer]
AnalyzerClass: TypeAlias = Type[Analyzer]

# Callback types
OrderCallback: TypeAlias = Callable[[Any], None]
TradeCallback: TypeAlias = Callable[[Any], None]
NotifyCallback: TypeAlias = Callable[..., None]

# Optimization types
OptimizationResult: TypeAlias = Tuple[StrategyParams, PerformanceMetrics]
OptimizationCallback: TypeAlias = Callable[[OptimizationResult], None]


# =============================================================================
# Utility Types
# =============================================================================

class VersionInfo(TypedDict):
    """Version information structure."""
    version: str
    btversion: str
    python_version: str
    platform: str


# Export commonly used types for convenience
__all__ = [
    # Basic types
    'Numeric', 'Price', 'Volume', 'Percentage',
    'DateType', 'TimeType', 'TimeDelta',
    
    # Order types
    'OrderStatus', 'OrderType', 'OrderExecution', 'BuySell',
    'OrderData',
    
    # Data types
    'OHLCV', 'TimeFrameType', 'Compression', 'BarData', 'DataFeedConfig',
    
    # Strategy types
    'ParameterDict', 'StrategyParams', 'PositionData',
    
    # Indicator types
    'IndicatorConfig',
    
    # Protocols
    'Datasource', 'Broker', 'Strategy', 'Indicator', 'Observer', 'Analyzer',
    
    # Configuration
    'CerebroConfig',
    
    # Trade types
    'TradeData', 'PerformanceMetrics',
    
    # Function types
    'StrategyClass', 'IndicatorClass', 'ObserverClass', 'AnalyzerClass',
    'OrderCallback', 'TradeCallback', 'NotifyCallback',
    'OptimizationResult', 'OptimizationCallback',
    
    # Utility
    'VersionInfo',
    
    # Type variables
    'T', 'StrategyType', 'IndicatorType', 'DataType',
]