"""Average True Range (ATR) indicator."""

import talib
import numpy as np
from typing import List, Optional, TYPE_CHECKING
from ..base import Indicator

if TYPE_CHECKING:
    from ...period import Period


class ATR(Indicator):
    """
    Average True Range (ATR)

    Measures market volatility by calculating the average of true ranges over a period.
    Developed by J. Welles Wilder Jr.

    True Range (TR) is the greatest of:
    1. Current High - Current Low
    2. |Current High - Previous Close|
    3. |Current Low - Previous Close|

    ATR = Moving Average of TR over N periods (typically EMA)

    Characteristics:
    - Absolute measure of volatility (not relative like Bollinger Bands %)
    - Does not indicate price direction, only volatility
    - Higher ATR = Higher volatility
    - Lower ATR = Lower volatility
    - Always positive value
    - Measured in same units as price

    Usage:
    - Volatility assessment: High ATR = volatile market, Low ATR = quiet market
    - Position sizing: Use ATR to determine stop-loss distances
    - Breakout confirmation: High ATR often accompanies strong trends
    - Risk management: ATR-based stops (e.g., 2× ATR below entry)
    - Trend strength: Expanding ATR = strong trend, contracting ATR = weak trend

    Common Applications:
    - Stop-loss placement: Entry ± (multiplier × ATR)
    - Position sizing: Risk per trade / ATR
    - Chandelier Exit: High - (multiplier × ATR)
    - Keltner Channels: EMA ± (multiplier × ATR)

    Common Periods:
    - 14 periods (Wilder's original recommendation)
    - 7 periods (short-term, more responsive)
    - 21 periods (longer-term, smoother)

    References:
        J. Welles Wilder Jr., "New Concepts in Technical Trading Systems" (1978)
    """

    def __init__(self, period: int = 14):
        """
        Initialize ATR indicator.

        Parameters:
            period: Number of periods for ATR calculation (default: 14)
                   Common values: 7 (short-term), 14 (standard), 21 (long-term)

        Raises:
            ValueError: If period < 1
        """
        if period < 1:
            raise ValueError("ATR period must be at least 1")

        self.period = period

    def requires_min_periods(self) -> int:
        """ATR needs at least 'period + 1' periods (need previous close for TR)."""
        return self.period + 1

    def get_dependencies(self) -> List[str]:
        """ATR depends on high, low, and close prices."""
        return ['high_price', 'low_price', 'close_price']

    def get_normalization_type(self) -> str:
        """ATR uses price-based normalization since it's in price units."""
        return 'price'

    def calculate(self, periods: List['Period'], index: int) -> Optional[float]:
        """
        Calculate ATR for the period at index.

        Parameters:
            periods: List of all periods
            index: Index of period to calculate for

        Returns:
            ATR value or None if insufficient data
        """
        if not self.validate_periods(periods, index):
            return None

        # Need at least period + 1 for previous close
        if index < self.period:
            return None

        # Extract high, low, close arrays
        high_values = self._extract_values(periods, index, 'high_price')
        low_values = self._extract_values(periods, index, 'low_price')
        close_values = self._extract_values(periods, index, 'close_price')

        # Remove NaN values
        valid_mask = ~(np.isnan(high_values) | np.isnan(low_values) | np.isnan(close_values))
        high_values = high_values[valid_mask]
        low_values = low_values[valid_mask]
        close_values = close_values[valid_mask]

        if len(high_values) < self.period + 1:
            return None

        # Calculate ATR using TA-Lib
        atr_values = talib.ATR(high_values, low_values, close_values, timeperiod=self.period)

        # Return last value (current period)
        atr_value = atr_values[-1]

        if np.isnan(atr_value):
            return None

        return round(float(atr_value), 4)

    def __repr__(self) -> str:
        return f"ATR(period={self.period})"
