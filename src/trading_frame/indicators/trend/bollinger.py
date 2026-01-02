"""Bollinger Bands indicator."""

import talib
import numpy as np
from typing import List, Optional, TYPE_CHECKING
from ..base import Indicator

if TYPE_CHECKING:
    from ...period import Period


class BollingerBands(Indicator):
    """
    Bollinger Bands

    Volatility bands placed above and below a moving average.
    Bands expand during volatile periods and contract during quiet periods.

    Components:
    - Upper Band: SMA + (StdDev × multiplier)
    - Middle Band: SMA (typically 20-period)
    - Lower Band: SMA - (StdDev × multiplier)

    Interpretation:
    - Price touching upper band: Potential overbought condition
    - Price touching lower band: Potential oversold condition
    - Band squeeze (narrow bands): Low volatility, potential breakout coming
    - Band expansion (wide bands): High volatility, potential trend exhaustion
    - Walking the bands: Strong trend (price consistently near one band)

    Trading Signals:
    - Bollinger Bounce: Price bounces off bands (mean reversion strategy)
    - Bollinger Squeeze: Breakout after contraction
    - W-Bottoms: Price tests lower band twice (bullish)
    - M-Tops: Price tests upper band twice (bearish)

    References:
        John Bollinger, "Bollinger on Bollinger Bands" (2001)
    """

    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        source: str = 'close_price'
    ):
        """
        Initialize Bollinger Bands indicator.

        Parameters:
            period: SMA period (default: 20)
                   Common values: 20 (standard), 10, 50
            std_dev: Standard deviation multiplier (default: 2.0)
                    Common values: 2.0 (standard), 1.5, 2.5, 3.0
            source: Column name to use as input (default: 'close_price')

        Raises:
            ValueError: If period < 2 or std_dev <= 0
        """
        if period < 2:
            raise ValueError("Bollinger period must be at least 2")
        if std_dev <= 0:
            raise ValueError("Standard deviation multiplier must be positive")

        self.period = period
        self.std_dev = std_dev
        self.source = source

    def requires_min_periods(self) -> int:
        """Bollinger Bands need at least 'period' periods."""
        return self.period

    def get_dependencies(self) -> List[str]:
        """Bollinger Bands depend on the source column."""
        return [self.source]

    def get_num_outputs(self) -> int:
        """Bollinger Bands produce 3 outputs: upper, middle, lower."""
        return 3

    def get_normalization_type(self) -> str:
        """Bollinger Bands use price-based normalization (they represent prices)."""
        return 'price'

    def calculate(self, periods: List['Period'], index: int) -> Optional[List[float]]:
        """
        Calculate Bollinger Bands for the period at index.

        Parameters:
            periods: List of all periods
            index: Index of period to calculate for

        Returns:
            [upper_band, middle_band, lower_band] or [None, None, None]
        """
        if not self.validate_periods(periods, index):
            return [None, None, None]

        # Extract price values
        prices = self._extract_values(periods, index, self.source)

        # Remove NaN values
        prices = prices[~np.isnan(prices)]

        if len(prices) < self.period:
            return [None, None, None]

        # Calculate Bollinger Bands using TA-Lib
        upper, middle, lower = talib.BBANDS(
            prices,
            timeperiod=self.period,
            nbdevup=self.std_dev,
            nbdevdn=self.std_dev,
            matype=0  # 0 = SMA
        )

        # Get last values (current period)
        upper_val = upper[-1]
        middle_val = middle[-1]
        lower_val = lower[-1]

        if np.isnan(upper_val) or np.isnan(middle_val) or np.isnan(lower_val):
            return [None, None, None]

        return [
            round(float(upper_val), 4),
            round(float(middle_val), 4),
            round(float(lower_val), 4)
        ]

    def __repr__(self) -> str:
        return f"BollingerBands(period={self.period}, std_dev={self.std_dev}, source='{self.source}')"
