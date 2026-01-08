"""SMA Crossover indicator - Example of composite indicator."""

from typing import List, Optional, TYPE_CHECKING
from ..base import Indicator

if TYPE_CHECKING:
    from ...period import Period


class SMACrossover(Indicator):
    """
    SMA Crossover Detector (Composite Indicator Example)

    Detects crossovers between fast and slow Simple Moving Averages.
    This is a composite indicator that depends on two SMA indicators.

    Signals:
    - +1: Golden Cross (fast SMA crosses above slow SMA) - Bullish
    - -1: Death Cross (fast SMA crosses below slow SMA) - Bearish
    -  0: No crossover

    This indicator demonstrates how to build composite indicators that
    depend on other indicators by reading directly from period._data.

    References:
        Classic technical analysis concept used in trend following systems
    """

    def __init__(self, fast: int = 20, slow: int = 50):
        """
        Initialize SMACrossover indicator.

        Parameters:
            fast: Fast SMA period (default: 20)
            slow: Slow SMA period (default: 50)

        Raises:
            ValueError: If fast >= slow
        """
        if fast >= slow:
            raise ValueError("Fast period must be less than slow period")

        self.fast = fast
        self.slow = slow

    def requires_min_periods(self) -> int:
        """Needs at least slow + 1 periods (for previous values)."""
        return self.slow + 1

    def get_dependencies(self) -> List[str]:
        """
        Depends on SMA indicators with naming convention.

        This demonstrates the dependency system - Frame will ensure
        these columns exist before allowing this indicator to be added.
        """
        return [f'SMA_{self.fast}', f'SMA_{self.slow}']

    def calculate(self, periods: List['Period'], index: int) -> Optional[int]:
        """
        Calculate crossover signal for the period at index.

        Parameters:
            periods: List of all periods
            index: Index of period to calculate for

        Returns:
            +1 (golden cross), -1 (death cross), 0 (no crossover), or None
        """
        if not self.validate_periods(periods, index):
            return None

        if index == 0:
            return 0  # No crossover on first period

        # Read current values from period._data
        curr_fast = periods[index]._data.get(f'SMA_{self.fast}')
        curr_slow = periods[index]._data.get(f'SMA_{self.slow}')

        # Read previous values
        prev_fast = periods[index - 1]._data.get(f'SMA_{self.fast}')
        prev_slow = periods[index - 1]._data.get(f'SMA_{self.slow}')

        # Check if all values are available
        if None in [curr_fast, curr_slow, prev_fast, prev_slow]:
            return 0

        # Detect Golden Cross (fast crosses above slow)
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            return 1

        # Detect Death Cross (fast crosses below slow)
        if prev_fast >= prev_slow and curr_fast < curr_slow:
            return -1

        # No crossover
        return 0

    def get_normalization_type(self) -> str:
        """Fixed range normalization (-1 to +1)."""
        return 'fixed'

    def get_fixed_range(self) -> tuple:
        """Crossover range is -1 (bearish) to +1 (bullish)."""
        return (-1.0, 1.0)

    def __repr__(self) -> str:
        return f"SMACrossover(fast={self.fast}, slow={self.slow})"
