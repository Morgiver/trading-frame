"""Base class for all technical indicators."""

from abc import ABC, abstractmethod
from typing import List, Union, Optional, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..period import Period


class Indicator(ABC):
    """
    Base class for all technical indicators.

    Indicators calculate derived values from OHLCV data or other indicators,
    using TA-Lib for calculations when possible.
    """

    @abstractmethod
    def calculate(self, periods: List['Period'], index: int) -> Union[Any, List[Any]]:
        """
        Calculate indicator value(s) for a specific period.

        Parameters:
            periods: List of all periods in the frame (for historical access)
            index: Index of the period to calculate for

        Returns:
            Single value OR list of values (for multi-column indicators)
            Returns None or [None, ...] if insufficient data
        """
        pass

    @abstractmethod
    def requires_min_periods(self) -> int:
        """
        Minimum number of periods needed for calculation.

        Returns:
            Minimum periods required (e.g., 14 for RSI-14)
        """
        pass

    def get_dependencies(self) -> List[str]:
        """
        Return list of column names this indicator depends on.

        Returns:
            List of column names (e.g., ['close_price', 'RSI_14'])
            Default is empty list (no dependencies)
        """
        return []

    def get_num_outputs(self) -> int:
        """
        Number of output values this indicator produces.

        Returns:
            1 for single-column indicators (RSI, SMA)
            >1 for multi-column indicators (MACD=3, Bollinger=3)
        """
        return 1

    def validate_periods(self, periods: List['Period'], index: int) -> bool:
        """
        Check if we have enough periods for calculation.

        Parameters:
            periods: List of all periods
            index: Current period index

        Returns:
            True if calculation is possible, False otherwise
        """
        return index >= self.requires_min_periods() - 1

    def _extract_values(
        self,
        periods: List['Period'],
        index: int,
        column: str,
        lookback: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract values from periods for TA-Lib calculations.

        Parameters:
            periods: List of all periods
            index: Current period index
            column: Column name to extract (e.g., 'close_price', 'RSI_14')
            lookback: Number of periods to look back (None = all available)

        Returns:
            numpy array of float64 values (NaN for missing values)
        """
        if lookback is None:
            start_idx = 0
        else:
            start_idx = max(0, index - lookback + 1)

        values = []
        for p in periods[start_idx:index + 1]:
            val = p._data.get(column)
            if val is not None:
                values.append(float(val))
            else:
                values.append(np.nan)

        return np.array(values, dtype=np.float64)
