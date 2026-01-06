"""Order Block (OB) indicator."""

import numpy as np
from typing import List, Optional, TYPE_CHECKING
from ..base import Indicator

if TYPE_CHECKING:
    from ...period import Period


class OrderBlock(Indicator):
    """
    Order Block (OB) Detector - Engulfing Pattern with Pivot

    Order Blocks are 2-candle engulfing patterns where one of the candles is a pivot point.

    Bullish Order Block (Demand Zone):
    - Bearish candle (red) completely engulfed by bullish candle (green)
    - One of the two candles must be a pivot point
    - OB zone: [low, high] of the engulfed bearish candle
    - Acts as potential support when price returns

    Bearish Order Block (Supply Zone):
    - Bullish candle (green) completely engulfed by bearish candle (red)
    - One of the two candles must be a pivot point
    - OB zone: [low, high] of the engulfed bullish candle
    - Acts as potential resistance when price returns

    Parameters:
        pivot_high_col: Column name for pivot highs from PivotPoints indicator (default: 'PIVOT_HIGH')
        pivot_low_col: Column name for pivot lows from PivotPoints indicator (default: 'PIVOT_LOW')

    Trading Applications:
    - Price often returns to OB zones (liquidity areas)
    - OBs act as strong support/resistance
    - Unmitigated OBs (price hasn't returned) are strongest
    - Break of OB can signal trend reversal

    References:
        ICT (Inner Circle Trader) concepts, Smart Money Concepts (SMC)
    """

    def __init__(
        self,
        pivot_high_col: str = 'PIVOT_HIGH',
        pivot_low_col: str = 'PIVOT_LOW',
        open_source: str = 'open_price',
        high_source: str = 'high_price',
        low_source: str = 'low_price',
        close_source: str = 'close_price'
    ):
        """
        Initialize OrderBlock indicator.

        Parameters:
            pivot_high_col: Column name for pivot highs (default: 'PIVOT_HIGH')
            pivot_low_col: Column name for pivot lows (default: 'PIVOT_LOW')
            open_source: Column name for open prices (default: 'open_price')
            high_source: Column name for high prices (default: 'high_price')
            low_source: Column name for low prices (default: 'low_price')
            close_source: Column name for close prices (default: 'close_price')
        """
        self.pivot_high_col = pivot_high_col
        self.pivot_low_col = pivot_low_col
        self.open_source = open_source
        self.high_source = high_source
        self.low_source = low_source
        self.close_source = close_source

        # Will be set by Frame via set_output_columns()
        self._output_columns = None

    def requires_min_periods(self) -> int:
        """OrderBlock needs at least 2 periods (for engulfing pattern)."""
        return 2

    def get_dependencies(self) -> List[str]:
        """OrderBlock depends on OHLC source columns and pivot columns."""
        return [
            self.open_source,
            self.high_source,
            self.low_source,
            self.close_source,
            self.pivot_high_col,
            self.pivot_low_col
        ]

    def get_num_outputs(self) -> int:
        """OrderBlock produces 2 outputs: ob_high, ob_low."""
        return 2

    def get_normalization_type(self) -> str:
        """OrderBlock uses price-based normalization (they represent price levels)."""
        return 'price'

    def _is_bullish_candle(self, period: 'Period') -> bool:
        """Check if candle is bullish (close > open)."""
        open_val = period._data.get(self.open_source)
        close_val = period._data.get(self.close_source)
        if open_val is None or close_val is None:
            return False
        return close_val > open_val

    def _is_bearish_candle(self, period: 'Period') -> bool:
        """Check if candle is bearish (close < open)."""
        open_val = period._data.get(self.open_source)
        close_val = period._data.get(self.close_source)
        if open_val is None or close_val is None:
            return False
        return close_val < open_val

    def _is_bullish_engulfing(self, prev: 'Period', curr: 'Period') -> bool:
        """
        Check if current candle bullish-engulfs previous candle.

        Bullish engulfing:
        - Previous candle is bearish (red)
        - Current candle is bullish (green)
        - Current candle completely engulfs previous candle
        """
        if not self._is_bearish_candle(prev) or not self._is_bullish_candle(curr):
            return False

        prev_open = prev._data.get(self.open_source)
        prev_close = prev._data.get(self.close_source)
        curr_open = curr._data.get(self.open_source)
        curr_close = curr._data.get(self.close_source)

        if any(v is None for v in [prev_open, prev_close, curr_open, curr_close]):
            return False

        # Current bullish candle engulfs previous bearish candle
        # curr_open < prev_close (starts below previous close)
        # curr_close > prev_open (ends above previous open)
        return curr_open < prev_close and curr_close > prev_open

    def _is_bearish_engulfing(self, prev: 'Period', curr: 'Period') -> bool:
        """
        Check if current candle bearish-engulfs previous candle.

        Bearish engulfing:
        - Previous candle is bullish (green)
        - Current candle is bearish (red)
        - Current candle completely engulfs previous candle
        """
        if not self._is_bullish_candle(prev) or not self._is_bearish_candle(curr):
            return False

        prev_open = prev._data.get(self.open_source)
        prev_close = prev._data.get(self.close_source)
        curr_open = curr._data.get(self.open_source)
        curr_close = curr._data.get(self.close_source)

        if any(v is None for v in [prev_open, prev_close, curr_open, curr_close]):
            return False

        # Current bearish candle engulfs previous bullish candle
        # curr_open > prev_close (starts above previous close)
        # curr_close < prev_open (ends below previous open)
        return curr_open > prev_close and curr_close < prev_open

    def _has_pivot(self, period: 'Period') -> bool:
        """Check if period has a pivot (high or low)."""
        pivot_high = period._data.get(self.pivot_high_col)
        pivot_low = period._data.get(self.pivot_low_col)
        return pivot_high is not None or pivot_low is not None

    def calculate(self, periods: List['Period'], index: int) -> Optional[List[Optional[float]]]:
        """
        Calculate OrderBlock for the period at index.

        Order Blocks are detected when:
        1. Current candle engulfs previous candle (bullish or bearish engulfing)
        2. One of the two candles (previous or current) is a pivot point

        The OB is marked on the ENGULFED candle (the one that got eaten).

        Parameters:
            periods: List of all periods
            index: Current index being calculated

        Returns:
            [ob_high, ob_low] for the CURRENT period (index)
            Returns [None, None] if no OB at this period
        """
        if not self.validate_periods(periods, index):
            return [None, None]

        # Need at least 2 periods for engulfing pattern
        if index < 1:
            return [None, None]

        prev = periods[index - 1]
        curr = periods[index]

        # Check for bullish engulfing (bearish candle engulfed by bullish)
        if self._is_bullish_engulfing(prev, curr):
            # One of the two candles must be a pivot
            if self._has_pivot(prev) or self._has_pivot(curr):
                # Mark the engulfed bearish candle as Bullish OB
                prev_high = prev._data.get(self.high_source)
                prev_low = prev._data.get(self.low_source)

                if prev_high is not None and prev_low is not None:
                    # Only mark if not already marked
                    if prev._data.get(self._output_columns[0]) is None:
                        prev._data[self._output_columns[0]] = float(prev_high)
                        prev._data[self._output_columns[1]] = float(prev_low)

        # Check for bearish engulfing (bullish candle engulfed by bearish)
        elif self._is_bearish_engulfing(prev, curr):
            # One of the two candles must be a pivot
            if self._has_pivot(prev) or self._has_pivot(curr):
                # Mark the engulfed bullish candle as Bearish OB
                prev_high = prev._data.get(self.high_source)
                prev_low = prev._data.get(self.low_source)

                if prev_high is not None and prev_low is not None:
                    # Only mark if not already marked
                    if prev._data.get(self._output_columns[0]) is None:
                        prev._data[self._output_columns[0]] = float(prev_high)
                        prev._data[self._output_columns[1]] = float(prev_low)

        # Return the values for the CURRENT period (which may have been set earlier)
        high_value = periods[index]._data.get(self._output_columns[0])
        low_value = periods[index]._data.get(self._output_columns[1])

        return [high_value, low_value]

    def __repr__(self) -> str:
        return (
            f"OrderBlock(pivot_high_col='{self.pivot_high_col}', pivot_low_col='{self.pivot_low_col}', "
            f"open_source='{self.open_source}', high_source='{self.high_source}', "
            f"low_source='{self.low_source}', close_source='{self.close_source}')"
        )
