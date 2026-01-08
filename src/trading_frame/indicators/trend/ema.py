"""Exponential Moving Average (EMA) indicator."""

import talib
import numpy as np
from typing import List, Optional, TYPE_CHECKING
from ..base import Indicator

if TYPE_CHECKING:
    from ...period import Period


class EMA(Indicator):
    """
    Exponential Moving Average (EMA)

    Weighted moving average that gives more importance to recent prices.
    More responsive to recent price changes than SMA.

    Formula: EMA = (Price - Previous EMA) × Multiplier + Previous EMA
    where Multiplier = 2 / (period + 1)

    Characteristics:
    - Less lag than SMA (more responsive to recent price changes)
    - More weight to recent prices, exponentially decreasing for older prices
    - More sensitive to price movements than SMA
    - Common periods: 9, 12, 20, 26, 50, 200

    Usage:
    - Price above EMA: Uptrend
    - Price below EMA: Downtrend
    - EMA crossovers: Trading signals
      * Fast EMA crosses above Slow EMA: Bullish signal
      * Fast EMA crosses below Slow EMA: Bearish signal
    - Support/Resistance: Price often bounces off EMA levels
    - Used in MACD (12 EMA - 26 EMA)

    Advantages over SMA:
    - Reacts faster to price changes
    - Better for short-term trading
    - Reduces lag in trend identification

    References:
        Classic technical analysis concept, widely used in trading systems
    """

    def __init__(self, period: int = 20, source: str = 'close_price'):
        """
        Initialize EMA indicator.

        Parameters:
            period: Number of periods for EMA calculation (default: 20)
                   Common values: 9, 12, 20, 26, 50, 100, 200
            source: Column name to use as input (default: 'close_price')
                   Can be 'close_price', 'open_price', or any other indicator

        Raises:
            ValueError: If period < 1
        """
        if period < 1:
            raise ValueError("EMA period must be at least 1")

        self.period = period
        self.source = source

    def requires_min_periods(self) -> int:
        """EMA needs at least 'period' periods for stable calculation."""
        return self.period

    def get_dependencies(self) -> List[str]:
        """EMA depends on the source column."""
        return [self.source]

    def get_normalization_type(self) -> str:
        """
        EMA normalization depends on source.
        If source is price, use price-based. Otherwise use minmax.
        """
        if self.source in ['open_price', 'high_price', 'low_price', 'close_price']:
            return 'price'
        return 'minmax'

    def calculate(self, periods: List['Period'], index: int) -> Optional[float]:
        """
        Calculate EMA for the period at index.

        Parameters:
            periods: List of all periods
            index: Index of period to calculate for

        Returns:
            EMA value or None if insufficient data
        """
        if not self.validate_periods(periods, index):
            return None

        # Extract values
        values = self._extract_values(periods, index, self.source)

        # Remove NaN values
        values = values[~np.isnan(values)]

        if len(values) < self.period:
            return None

        # Calculate EMA using TA-Lib
        ema_values = talib.EMA(values, timeperiod=self.period)

        # Return last value (current period)
        ema_value = ema_values[-1]

        if np.isnan(ema_value):
            return None

        return round(float(ema_value), 4)

    def __repr__(self) -> str:
        return f"EMA(period={self.period}, source='{self.source}')"
