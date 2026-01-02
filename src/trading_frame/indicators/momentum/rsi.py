"""Relative Strength Index (RSI) indicator."""

import talib
import numpy as np
from typing import List, Optional, TYPE_CHECKING
from ..base import Indicator

if TYPE_CHECKING:
    from ...period import Period


class RSI(Indicator):
    """
    Relative Strength Index (RSI)

    Measures the magnitude of recent price changes to evaluate
    overbought or oversold conditions.

    Range: 0-100
    - Above 70: Potentially overbought
    - Below 30: Potentially oversold
    - 50: Neutral

    Formula: RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss over period

    References:
        J. Welles Wilder Jr., "New Concepts in Technical Trading Systems" (1978)
    """

    def __init__(self, length: int = 14, source: str = 'close_price'):
        """
        Initialize RSI indicator.

        Parameters:
            length: Period for RSI calculation (default: 14)
                   Common values: 14 (standard), 9, 25
            source: Column name to use as input (default: 'close_price')
                   Can be any OHLC column or another indicator

        Raises:
            ValueError: If length < 2
        """
        if length < 2:
            raise ValueError("RSI length must be at least 2")

        self.length = length
        self.source = source

    def requires_min_periods(self) -> int:
        """RSI needs at least 'length' periods for proper calculation."""
        return self.length

    def get_dependencies(self) -> List[str]:
        """RSI depends on the source column."""
        return [self.source]

    def get_normalization_type(self) -> str:
        """RSI uses fixed range normalization (0-100)."""
        return 'fixed'

    def get_fixed_range(self) -> tuple:
        """RSI range is always 0-100."""
        return (0.0, 100.0)

    def calculate(self, periods: List['Period'], index: int) -> Optional[float]:
        """
        Calculate RSI for the period at index.

        Parameters:
            periods: List of all periods
            index: Index of period to calculate for

        Returns:
            RSI value (0-100) or None if insufficient data
        """
        if not self.validate_periods(periods, index):
            return None

        # Extract price values
        prices = self._extract_values(periods, index, self.source)

        # Remove NaN values
        prices = prices[~np.isnan(prices)]

        if len(prices) < self.length:
            return None

        # Calculate RSI using TA-Lib
        rsi_values = talib.RSI(prices, timeperiod=self.length)

        # Return last value (current period)
        rsi_value = rsi_values[-1]

        if np.isnan(rsi_value):
            return None

        return round(float(rsi_value), 2)

    def __repr__(self) -> str:
        return f"RSI(length={self.length}, source='{self.source}')"
