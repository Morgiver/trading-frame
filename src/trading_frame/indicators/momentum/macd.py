"""Moving Average Convergence Divergence (MACD) indicator."""

import talib
import numpy as np
from typing import List, Optional, TYPE_CHECKING
from ..base import Indicator

if TYPE_CHECKING:
    from ...period import Period


class MACD(Indicator):
    """
    Moving Average Convergence Divergence (MACD)

    Trend-following momentum indicator showing the relationship
    between two exponential moving averages.

    Components:
    - MACD Line: Fast EMA - Slow EMA
    - Signal Line: EMA of MACD Line
    - Histogram: MACD Line - Signal Line

    Signals:
    - MACD crosses above Signal: Bullish signal
    - MACD crosses below Signal: Bearish signal
    - Histogram > 0: Bullish momentum
    - Histogram < 0: Bearish momentum
    - Histogram expanding: Momentum increasing
    - Histogram contracting: Momentum decreasing

    References:
        Gerald Appel, "Technical Analysis: Power Tools for Active Investors" (2005)
    """

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        source: str = 'close_price'
    ):
        """
        Initialize MACD indicator.

        Parameters:
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line EMA period (default: 9)
            source: Column name to use as input (default: 'close_price')

        Raises:
            ValueError: If fast >= slow or signal < 1
        """
        if fast >= slow:
            raise ValueError("Fast period must be less than slow period")
        if signal < 1:
            raise ValueError("Signal period must be at least 1")

        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.source = source

    def requires_min_periods(self) -> int:
        """MACD needs slow + signal periods for proper calculation."""
        return self.slow + self.signal

    def get_dependencies(self) -> List[str]:
        """MACD depends on the source column."""
        return [self.source]

    def get_num_outputs(self) -> int:
        """MACD produces 3 outputs: line, signal, histogram."""
        return 3

    def calculate(self, periods: List['Period'], index: int) -> Optional[List[float]]:
        """
        Calculate MACD for the period at index.

        Parameters:
            periods: List of all periods
            index: Index of period to calculate for

        Returns:
            [macd_line, signal_line, histogram] or [None, None, None]
        """
        if not self.validate_periods(periods, index):
            return [None, None, None]

        # Extract price values
        prices = self._extract_values(periods, index, self.source)

        # Remove NaN values
        prices = prices[~np.isnan(prices)]

        if len(prices) < self.slow + self.signal:
            return [None, None, None]

        # Calculate MACD using TA-Lib
        macd_line, signal_line, histogram = talib.MACD(
            prices,
            fastperiod=self.fast,
            slowperiod=self.slow,
            signalperiod=self.signal
        )

        # Get last values (current period)
        macd = macd_line[-1]
        signal = signal_line[-1]
        hist = histogram[-1]

        if np.isnan(macd) or np.isnan(signal) or np.isnan(hist):
            return [None, None, None]

        return [
            round(float(macd), 4),
            round(float(signal), 4),
            round(float(hist), 4)
        ]

    def __repr__(self) -> str:
        return f"MACD(fast={self.fast}, slow={self.slow}, signal={self.signal}, source='{self.source}')"
