"""Order Block (OB) indicator."""

import numpy as np
from typing import List, Optional, TYPE_CHECKING
from ..base import Indicator

if TYPE_CHECKING:
    from ...period import Period


class OrderBlock(Indicator):
    """
    Order Block (OB) Detector

    Order Blocks represent institutional buying or selling activity zones.
    They are identified as the last opposite-direction candle before a strong
    directional move (breakout).

    Bullish Order Block (Demand Zone):
    - Last bearish candle (close < open) before a bullish breakout
    - Breakout confirmed when price closes above the bearish candle's high
    - OB zone: [low, high] of the bearish candle
    - Acts as potential support when price returns

    Bearish Order Block (Supply Zone):
    - Last bullish candle (close > open) before a bearish breakout
    - Breakout confirmed when price closes below the bullish candle's low
    - OB zone: [low, high] of the bullish candle
    - Acts as potential resistance when price returns

    The indicator looks back to find the last opposite candle and marks it
    when a breakout is confirmed.

    Parameters:
        lookback: Maximum periods to search back for opposite candle (default: 10)
        min_body_pct: Minimum body size as % of candle range to filter noise (default: 0.3 = 30%)
        require_pivot: If True, require a pivot point in the 2-3 candles before OB (default: False)
        pivot_lookback: Candles to check for pivot before OB (default: 3)

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
        lookback: int = 10,
        min_body_pct: float = 0.3,
        require_pivot: bool = False,
        pivot_lookback: int = 3,
        open_source: str = 'open_price',
        high_source: str = 'high_price',
        low_source: str = 'low_price',
        close_source: str = 'close_price'
    ):
        """
        Initialize OrderBlock indicator.

        Parameters:
            lookback: Maximum periods to look back for opposite candle (default: 10)
            min_body_pct: Minimum body size as % of range (0-1, default: 0.3)
            require_pivot: Require pivot in preceding candles (default: False)
            pivot_lookback: Candles to check for pivot before OB (default: 3)
            open_source: Column name for open prices (default: 'open_price')
            high_source: Column name for high prices (default: 'high_price')
            low_source: Column name for low prices (default: 'low_price')
            close_source: Column name for close prices (default: 'close_price')

        Raises:
            ValueError: If lookback < 1 or min_body_pct < 0 or min_body_pct > 1 or pivot_lookback < 2
        """
        if lookback < 1:
            raise ValueError("lookback must be at least 1")
        if min_body_pct < 0 or min_body_pct > 1:
            raise ValueError("min_body_pct must be between 0 and 1")
        if pivot_lookback < 2:
            raise ValueError("pivot_lookback must be at least 2")

        self.lookback = lookback
        self.min_body_pct = min_body_pct
        self.require_pivot = require_pivot
        self.pivot_lookback = pivot_lookback
        self.open_source = open_source
        self.high_source = high_source
        self.low_source = low_source
        self.close_source = close_source

        # Will be set by Frame via set_output_columns()
        self._output_columns = None

    def requires_min_periods(self) -> int:
        """OrderBlock needs at least 2 periods (current + 1 for comparison)."""
        return 2

    def get_dependencies(self) -> List[str]:
        """OrderBlock depends on OHLC source columns."""
        return [self.open_source, self.high_source, self.low_source, self.close_source]

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

    def _get_body_size_pct(self, period: 'Period') -> Optional[float]:
        """Calculate body size as percentage of candle range."""
        open_val = period._data.get(self.open_source)
        close_val = period._data.get(self.close_source)
        high_val = period._data.get(self.high_source)
        low_val = period._data.get(self.low_source)

        if any(v is None for v in [open_val, close_val, high_val, low_val]):
            return None

        candle_range = high_val - low_val
        if candle_range == 0:
            return 0.0

        body_size = abs(close_val - open_val)
        return body_size / candle_range

    def _has_pivot_before(self, periods: List['Period'], ob_index: int) -> bool:
        """
        Check if there's a pivot (swing high or swing low) in the 2-3 candles
        before the potential Order Block candle.

        For bullish OB (bearish candle), look for swing low before it.
        For bearish OB (bullish candle), look for swing high before it.

        Parameters:
            periods: List of all periods
            ob_index: Index of the potential OB candle

        Returns:
            True if pivot found in preceding candles, False otherwise
        """
        # Need at least pivot_lookback candles before OB
        if ob_index < self.pivot_lookback:
            return False

        is_bullish_ob = self._is_bearish_candle(periods[ob_index])  # Bearish candle = Bullish OB
        is_bearish_ob = self._is_bullish_candle(periods[ob_index])  # Bullish candle = Bearish OB

        # Check the pivot_lookback candles immediately before the OB candle
        start_idx = max(0, ob_index - self.pivot_lookback)
        end_idx = ob_index  # Exclusive, so we check candles before OB

        for i in range(start_idx, end_idx):
            if is_bullish_ob:
                # For bullish OB, look for swing low (pivot low)
                if self._is_swing_low(periods, i):
                    return True
            elif is_bearish_ob:
                # For bearish OB, look for swing high (pivot high)
                if self._is_swing_high(periods, i):
                    return True

        return False

    def _is_swing_high(self, periods: List['Period'], index: int) -> bool:
        """
        Check if candle at index is a swing high (local high).
        Simple check: high is higher than previous and next candle highs.
        """
        if index < 1 or index >= len(periods) - 1:
            return False

        curr_high = periods[index]._data.get(self.high_source)
        prev_high = periods[index - 1]._data.get(self.high_source)
        next_high = periods[index + 1]._data.get(self.high_source)

        if any(v is None for v in [curr_high, prev_high, next_high]):
            return False

        return curr_high > prev_high and curr_high > next_high

    def _is_swing_low(self, periods: List['Period'], index: int) -> bool:
        """
        Check if candle at index is a swing low (local low).
        Simple check: low is lower than previous and next candle lows.
        """
        if index < 1 or index >= len(periods) - 1:
            return False

        curr_low = periods[index]._data.get(self.low_source)
        prev_low = periods[index - 1]._data.get(self.low_source)
        next_low = periods[index + 1]._data.get(self.low_source)

        if any(v is None for v in [curr_low, prev_low, next_low]):
            return False

        return curr_low < prev_low and curr_low < next_low

    def calculate(self, periods: List['Period'], index: int) -> Optional[List[Optional[float]]]:
        """
        Calculate OrderBlock for the period at index.

        Order Blocks are detected when:
        1. Current candle breaks in one direction
        2. We find the last opposite candle within lookback
        3. The opposite candle meets minimum body size requirement

        The OB is marked on the opposite candle (not the breakout candle).

        Parameters:
            periods: List of all periods
            index: Current index being calculated

        Returns:
            [ob_high, ob_low] for the CURRENT period (index)
            Returns [None, None] if no OB at this period
        """
        if not self.validate_periods(periods, index):
            return [None, None]

        # Need at least 2 periods (current + 1 for comparison)
        if index < 1:
            return [None, None]

        # Get current candle data
        current_close = periods[index]._data.get(self.close_source)
        if current_close is None:
            return [None, None]

        current_is_bullish = self._is_bullish_candle(periods[index])
        current_is_bearish = self._is_bearish_candle(periods[index])

        # Check if current candle has directional bias
        if not current_is_bullish and not current_is_bearish:
            return [None, None]

        # Look back for Order Block detection
        if current_is_bullish:
            # Looking for Bullish OB: last bearish candle before bullish breakout
            # Search back for bearish candles
            for i in range(index - 1, max(-1, index - self.lookback - 1), -1):
                if i < 0:
                    break

                # Check if this is a bearish candle
                if not self._is_bearish_candle(periods[i]):
                    continue

                # Check body size requirement
                body_pct = self._get_body_size_pct(periods[i])
                if body_pct is None or body_pct < self.min_body_pct:
                    continue

                # If pivot required, check for pivot in preceding candles
                if self.require_pivot:
                    if not self._has_pivot_before(periods, i):
                        continue

                # Check if current candle breaks above this bearish candle's high
                bearish_high = periods[i]._data.get(self.high_source)
                if bearish_high is None:
                    continue

                if current_close > bearish_high:
                    # Bullish breakout detected!
                    # Mark the bearish candle as Bullish Order Block
                    bearish_low = periods[i]._data.get(self.low_source)
                    if bearish_low is not None:
                        # Only mark if not already marked (don't overwrite existing OB)
                        if periods[i]._data.get(self._output_columns[0]) is None:
                            periods[i]._data[self._output_columns[0]] = float(bearish_high)
                            periods[i]._data[self._output_columns[1]] = float(bearish_low)
                        break  # Only mark the last (most recent) bearish candle

        elif current_is_bearish:
            # Looking for Bearish OB: last bullish candle before bearish breakout
            # Search back for bullish candles
            for i in range(index - 1, max(-1, index - self.lookback - 1), -1):
                if i < 0:
                    break

                # Check if this is a bullish candle
                if not self._is_bullish_candle(periods[i]):
                    continue

                # Check body size requirement
                body_pct = self._get_body_size_pct(periods[i])
                if body_pct is None or body_pct < self.min_body_pct:
                    continue

                # If pivot required, check for pivot in preceding candles
                if self.require_pivot:
                    if not self._has_pivot_before(periods, i):
                        continue

                # Check if current candle breaks below this bullish candle's low
                bullish_low = periods[i]._data.get(self.low_source)
                if bullish_low is None:
                    continue

                if current_close < bullish_low:
                    # Bearish breakout detected!
                    # Mark the bullish candle as Bearish Order Block
                    bullish_high = periods[i]._data.get(self.high_source)
                    if bullish_high is not None:
                        # Only mark if not already marked (don't overwrite existing OB)
                        if periods[i]._data.get(self._output_columns[0]) is None:
                            periods[i]._data[self._output_columns[0]] = float(bullish_high)
                            periods[i]._data[self._output_columns[1]] = float(bullish_low)
                        break  # Only mark the last (most recent) bullish candle

        # Return the values for the CURRENT period (which may have been set earlier)
        high_value = periods[index]._data.get(self._output_columns[0])
        low_value = periods[index]._data.get(self._output_columns[1])

        return [high_value, low_value]

    def __repr__(self) -> str:
        return (
            f"OrderBlock(lookback={self.lookback}, min_body_pct={self.min_body_pct}, "
            f"require_pivot={self.require_pivot}, pivot_lookback={self.pivot_lookback}, "
            f"open_source='{self.open_source}', high_source='{self.high_source}', "
            f"low_source='{self.low_source}', close_source='{self.close_source}')"
        )
