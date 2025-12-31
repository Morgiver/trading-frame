"""Simple Moving Average (SMA) indicator."""

import talib
import numpy as np
from typing import List, Optional, TYPE_CHECKING
from ..base import Indicator

if TYPE_CHECKING:
    from ...period import Period


class SMA(Indicator):
    """
    Simple Moving Average (SMA)

    Average price over a specified number of periods.
    Used to smooth price data and identify trends.

    Formula: SMA = (P1 + P2 + ... + Pn) / n
    where P = price, n = period

    Characteristics:
    - Lags price action (more lag with longer periods)
    - Equal weight to all prices in the period
    - Smooth, easy to understand
    - Common periods: 20 (short-term), 50 (medium-term), 200 (long-term)

    Usage:
    - Price above SMA: Uptrend
    - Price below SMA: Downtrend
    - SMA crossovers: Trading signals
      * Fast SMA crosses above Slow SMA: Bullish (Golden Cross)
      * Fast SMA crosses below Slow SMA: Bearish (Death Cross)

    References:
        Classic technical analysis concept
    """

    def __init__(self, period: int = 20, source: str = 'close_price'):
        """
        Initialize SMA indicator.

        Parameters:
            period: Number of periods to average (default: 20)
                   Common values: 10, 20, 50, 100, 200
            source: Column name to use as input (default: 'close_price')
                   Can be 'close_price', 'open_price', or any other indicator

        Raises:
            ValueError: If period < 1
        """
        if period < 1:
            raise ValueError("SMA period must be at least 1")

        self.period = period
        self.source = source

    def requires_min_periods(self) -> int:
        """SMA needs at least 'period' periods."""
        return self.period

    def get_dependencies(self) -> List[str]:
        """SMA depends on the source column."""
        return [self.source]

    def calculate(self, periods: List['Period'], index: int) -> Optional[float]:
        """
        Calculate SMA for the period at index.

        Parameters:
            periods: List of all periods
            index: Index of period to calculate for

        Returns:
            SMA value or None if insufficient data
        """
        if not self.validate_periods(periods, index):
            return None

        # Extract values
        values = self._extract_values(periods, index, self.source)

        # Remove NaN values
        values = values[~np.isnan(values)]

        if len(values) < self.period:
            return None

        # Calculate SMA using TA-Lib
        sma_values = talib.SMA(values, timeperiod=self.period)

        # Return last value (current period)
        sma_value = sma_values[-1]

        if np.isnan(sma_value):
            return None

        return round(float(sma_value), 4)

    def __repr__(self) -> str:
        return f"SMA(period={self.period}, source='{self.source}')"
