"""Pivot Points (Swing High/Low) indicator."""

import numpy as np
from typing import List, Optional, TYPE_CHECKING
from ..base import Indicator

if TYPE_CHECKING:
    from ...period import Period


class PivotPoints(Indicator):
    """
    Pivot Points (Swing High/Low) Detector

    Detects local highs and lows based on a configurable number of bars
    to the left and right of the pivot candidate.

    A Swing High is confirmed when the high price of the central bar is
    greater than all highs in the surrounding bars (left and right).
    A Swing Low is confirmed when the low price of the central bar is
    lower than all lows in the surrounding bars.

    Alternation Rule:
    - Pivots must alternate between High and Low
    - If two consecutive Highs occur without a Low between them, the newer High replaces the older
    - If two consecutive Lows occur without a High between them, the newer Low replaces the older

    Lag Behavior:
    - Pivots are confirmed 'right_bars' periods after the candidate bar
    - The pivot value is assigned to the candidate bar's period, not the confirmation bar
    - This creates a natural lag of 'right_bars' periods

    Example with left_bars=5, right_bars=2:
        Index:  117  118  119  120  121  122  123  124  125
        Price:   H    |    |    |    |    C    |    |    |

        At index 125, we can confirm if bar 122 is a pivot by comparing:
        - Bar 122's high/low against bars [117-121] (left) and [123-125] (right)

    References:
        Classic technical analysis concept used in Elliott Wave and pattern recognition
    """

    def __init__(
        self,
        left_bars: int = 5,
        right_bars: int = 2,
        high_source: str = 'high_price',
        low_source: str = 'low_price'
    ):
        """
        Initialize PivotPoints indicator.

        Parameters:
            left_bars: Number of bars to the left of pivot candidate (default: 5)
            right_bars: Number of bars to the right of pivot candidate (default: 2)
            high_source: Column name for high prices (default: 'high_price')
            low_source: Column name for low prices (default: 'low_price')

        Raises:
            ValueError: If left_bars < 1 or right_bars < 1
        """
        if left_bars < 1:
            raise ValueError("left_bars must be at least 1")
        if right_bars < 1:
            raise ValueError("right_bars must be at least 1")

        self.left_bars = left_bars
        self.right_bars = right_bars
        self.high_source = high_source
        self.low_source = low_source

        # Track last confirmed pivot state for alternation rule
        self._last_pivot_type = None  # 'high' or 'low'
        self._last_pivot_index = None  # Index where last pivot was confirmed

        # Will be set by Frame via set_output_columns()
        self._output_columns = None

    def requires_min_periods(self) -> int:
        """
        Minimum periods needed = left_bars + 1 (candidate) + right_bars.

        For example, left=5, right=2 requires 5+1+2 = 8 periods minimum.
        """
        return self.left_bars + 1 + self.right_bars

    def get_dependencies(self) -> List[str]:
        """PivotPoints depends on high and low source columns."""
        return [self.high_source, self.low_source]

    def get_num_outputs(self) -> int:
        """PivotPoints produce 2 outputs: high pivot price, low pivot price."""
        return 2

    def get_normalization_type(self) -> str:
        """Pivot points use price-based normalization (they represent prices)."""
        return 'price'

    def _is_swing_high(
        self,
        periods: List['Period'],
        candidate_idx: int,
        confirmation_idx: int
    ) -> bool:
        """
        Check if candidate_idx is a swing high confirmed at confirmation_idx.

        Parameters:
            periods: All periods
            candidate_idx: Index of pivot candidate
            confirmation_idx: Current index (candidate_idx + right_bars)

        Returns:
            True if swing high detected
        """
        candidate_high = periods[candidate_idx]._data.get(self.high_source)
        if candidate_high is None:
            return False

        # Check left bars
        for i in range(candidate_idx - self.left_bars, candidate_idx):
            if i < 0:
                continue
            compare_high = periods[i]._data.get(self.high_source)
            if compare_high is None:
                continue
            if compare_high >= candidate_high:
                return False

        # Check right bars
        for i in range(candidate_idx + 1, candidate_idx + self.right_bars + 1):
            if i >= len(periods):
                return False  # Not enough future data yet
            compare_high = periods[i]._data.get(self.high_source)
            if compare_high is None:
                continue
            if compare_high >= candidate_high:
                return False

        return True

    def _is_swing_low(
        self,
        periods: List['Period'],
        candidate_idx: int,
        confirmation_idx: int
    ) -> bool:
        """
        Check if candidate_idx is a swing low confirmed at confirmation_idx.

        Parameters:
            periods: All periods
            candidate_idx: Index of pivot candidate
            confirmation_idx: Current index (candidate_idx + right_bars)

        Returns:
            True if swing low detected
        """
        candidate_low = periods[candidate_idx]._data.get(self.low_source)
        if candidate_low is None:
            return False

        # Check left bars
        for i in range(candidate_idx - self.left_bars, candidate_idx):
            if i < 0:
                continue
            compare_low = periods[i]._data.get(self.low_source)
            if compare_low is None:
                continue
            if compare_low <= candidate_low:
                return False

        # Check right bars
        for i in range(candidate_idx + 1, candidate_idx + self.right_bars + 1):
            if i >= len(periods):
                return False  # Not enough future data yet
            compare_low = periods[i]._data.get(self.low_source)
            if compare_low is None:
                continue
            if compare_low <= candidate_low:
                return False

        return True

    def calculate(self, periods: List['Period'], index: int) -> Optional[List[Optional[float]]]:
        """
        Calculate pivot points for the period at index.

        This method is called for EACH period during frame updates. We check if the
        current period (index) can serve as the RIGHT edge to confirm a pivot that
        occurred 'right_bars' periods earlier.

        When a pivot is confirmed, we UPDATE the candidate period's data directly,
        but return values for the CURRENT period (index).

        Parameters:
            periods: List of all periods
            index: Current index being calculated

        Returns:
            [high_pivot_price, low_pivot_price] for the CURRENT period (index)
            Returns existing values if already set, or [None, None] if not a pivot
        """
        # Check if we have enough data to look for a pivot candidate
        if index >= self.left_bars + self.right_bars:
            # The candidate is 'right_bars' periods back from current index
            candidate_idx = index - self.right_bars

            # Ensure candidate has enough left context
            if candidate_idx >= self.left_bars:
                # Check for swing high
                is_high = self._is_swing_high(periods, candidate_idx, index)
                # Check for swing low
                is_low = self._is_swing_low(periods, candidate_idx, index)

                # Apply alternation rule
                if is_high and is_low:
                    # Both detected (rare but possible), respect last pivot type
                    if self._last_pivot_type == 'high':
                        # Last was high, so accept the low
                        is_high = False
                    else:
                        # Last was low (or None), so accept the high
                        is_low = False

                if is_high:
                    candidate_high = periods[candidate_idx]._data.get(self.high_source)

                    # Alternation rule: if last was also a high, replace it
                    if self._last_pivot_type == 'high' and self._last_pivot_index is not None:
                        # Clear the previous high
                        periods[self._last_pivot_index]._data[self._output_columns[0]] = None

                    # Update the CANDIDATE period, not the current one
                    periods[candidate_idx]._data[self._output_columns[0]] = float(candidate_high)
                    self._last_pivot_type = 'high'
                    self._last_pivot_index = candidate_idx

                if is_low:
                    candidate_low = periods[candidate_idx]._data.get(self.low_source)

                    # Alternation rule: if last was also a low, replace it
                    if self._last_pivot_type == 'low' and self._last_pivot_index is not None:
                        # Clear the previous low
                        periods[self._last_pivot_index]._data[self._output_columns[1]] = None

                    # Update the CANDIDATE period, not the current one
                    periods[candidate_idx]._data[self._output_columns[1]] = float(candidate_low)
                    self._last_pivot_type = 'low'
                    self._last_pivot_index = candidate_idx

        # Return the values for the CURRENT period (which may have been set earlier)
        high_value = periods[index]._data.get(self._output_columns[0])
        low_value = periods[index]._data.get(self._output_columns[1])

        return [high_value, low_value]

    def __repr__(self) -> str:
        return (
            f"PivotPoints(left_bars={self.left_bars}, right_bars={self.right_bars}, "
            f"high_source='{self.high_source}', low_source='{self.low_source}')"
        )
