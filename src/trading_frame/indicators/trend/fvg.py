"""Fair Value Gap (FVG) indicator."""

import numpy as np
from typing import List, Optional, TYPE_CHECKING
from ..base import Indicator

if TYPE_CHECKING:
    from ...period import Period


class FVG(Indicator):
    """
    Fair Value Gap (FVG) Detector

    A Fair Value Gap occurs when there is a price imbalance between three consecutive candles,
    leaving a "gap" where no trading occurred. These gaps often act as support/resistance zones.

    Bullish FVG (Demand Zone):
    - Candle 1 (bearish): Creates a low
    - Candle 2: The gap candle
    - Candle 3 (bullish): Low of candle 3 > High of candle 1
    - Gap range: [High of candle 1, Low of candle 3]

    Bearish FVG (Supply Zone):
    - Candle 1 (bullish): Creates a high
    - Candle 2: The gap candle
    - Candle 3 (bearish): High of candle 3 < Low of candle 1
    - Gap range: [High of candle 3, Low of candle 1]

    The FVG is assigned to candle 2 (the middle candle where the gap occurs).

    Trading Applications:
    - Price often returns to fill FVG zones (mean reversion)
    - FVGs can act as support (bullish) or resistance (bearish)
    - Strong momentum moves often leave multiple FVGs
    - Unfilled FVGs indicate strong directional bias

    References:
        ICT (Inner Circle Trader) concepts, Smart Money Concepts (SMC)
    """

    def __init__(
        self,
        high_source: str = 'high_price',
        low_source: str = 'low_price'
    ):
        """
        Initialize FVG indicator.

        Parameters:
            high_source: Column name for high prices (default: 'high_price')
            low_source: Column name for low prices (default: 'low_price')
        """
        self.high_source = high_source
        self.low_source = low_source

        # Will be set by Frame via set_output_columns()
        self._output_columns = None

    def requires_min_periods(self) -> int:
        """FVG needs at least 3 periods (current + 2 historical)."""
        return 3

    def get_dependencies(self) -> List[str]:
        """FVG depends on high and low source columns."""
        return [self.high_source, self.low_source]

    def get_num_outputs(self) -> int:
        """FVG produces 2 outputs: fvg_high, fvg_low."""
        return 2

    def get_normalization_type(self) -> str:
        """FVG uses price-based normalization (they represent price levels)."""
        return 'price'

    def calculate(self, periods: List['Period'], index: int) -> Optional[List[Optional[float]]]:
        """
        Calculate FVG for the period at index.

        FVG is detected on candle 2 (middle candle) when comparing 3 consecutive candles.
        We check if the current period (index) is candle 3, and if so, we detect the FVG
        on candle 2 (index - 1).

        Parameters:
            periods: List of all periods
            index: Current index being calculated

        Returns:
            [fvg_high, fvg_low] for the CURRENT period (index)
            Returns [None, None] if no FVG at this period
        """
        if not self.validate_periods(periods, index):
            return [None, None]

        # We need at least 3 candles to detect FVG
        if index < 2:
            return [None, None]

        # Check if we are candle 3 (confirmation candle)
        # If yes, we detect FVG on candle 2 (middle candle at index - 1)
        candle_1_idx = index - 2  # First candle
        candle_2_idx = index - 1  # Middle candle (where FVG will be marked)
        candle_3_idx = index      # Current candle (confirmation)

        # Extract price data
        high_1 = periods[candle_1_idx]._data.get(self.high_source)
        low_1 = periods[candle_1_idx]._data.get(self.low_source)

        high_3 = periods[candle_3_idx]._data.get(self.high_source)
        low_3 = periods[candle_3_idx]._data.get(self.low_source)

        if any(v is None for v in [high_1, low_1, high_3, low_3]):
            return [None, None]

        # Detect Bullish FVG (Demand Zone)
        # Low of candle 3 > High of candle 1
        bullish_fvg = low_3 > high_1

        # Detect Bearish FVG (Supply Zone)
        # High of candle 3 < Low of candle 1
        bearish_fvg = high_3 < low_1

        # If we are at candle 3 and detect an FVG, mark it on candle 2
        if index == candle_3_idx and (bullish_fvg or bearish_fvg):
            if bullish_fvg:
                # Bullish FVG range: [High of candle 1, Low of candle 3]
                fvg_low = float(high_1)
                fvg_high = float(low_3)
                # Write to candle 2 (middle candle)
                periods[candle_2_idx]._data[self._output_columns[0]] = fvg_high
                periods[candle_2_idx]._data[self._output_columns[1]] = fvg_low
            elif bearish_fvg:
                # Bearish FVG range: [High of candle 3, Low of candle 1]
                fvg_low = float(high_3)
                fvg_high = float(low_1)
                # Write to candle 2 (middle candle)
                periods[candle_2_idx]._data[self._output_columns[0]] = fvg_high
                periods[candle_2_idx]._data[self._output_columns[1]] = fvg_low

        # Return the values for the CURRENT period (which may have been set earlier)
        high_value = periods[index]._data.get(self._output_columns[0])
        low_value = periods[index]._data.get(self._output_columns[1])

        return [high_value, low_value]

    def __repr__(self) -> str:
        return (
            f"FVG(high_source='{self.high_source}', low_source='{self.low_source}')"
        )
