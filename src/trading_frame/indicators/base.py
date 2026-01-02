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

    def get_normalization_type(self) -> str:
        """
        Return the normalization strategy for this indicator.

        Returns:
            'minmax' - Min-Max normalization (default)
            'fixed' - Fixed range normalization (e.g., RSI 0-100)
            'price' - Share OHLC min-max range
            'none' - No normalization
        """
        return 'minmax'

    def get_fixed_range(self) -> Optional[tuple]:
        """
        Return fixed range for normalization if type is 'fixed'.

        Returns:
            Tuple (min, max) or None
        """
        return None

    def normalize(
        self,
        values: Union[float, List[float], np.ndarray],
        all_values: Optional[np.ndarray] = None,
        price_range: Optional[tuple] = None
    ) -> Union[float, List[float], np.ndarray]:
        """
        Normalize indicator value(s) based on normalization type.

        Parameters:
            values: Value(s) to normalize (single value, list, or array)
            all_values: All values across periods for min-max calculation
            price_range: (min, max) tuple for price-based normalization

        Returns:
            Normalized value(s) in range [0, 1]
        """
        norm_type = self.get_normalization_type()

        # Handle None values
        if values is None:
            return None
        if isinstance(values, (list, np.ndarray)):
            if all(v is None for v in values):
                return values

        # Fixed range normalization (e.g., RSI 0-100)
        if norm_type == 'fixed':
            fixed_range = self.get_fixed_range()
            if fixed_range is None:
                raise ValueError("Fixed range not defined for indicator")
            min_val, max_val = fixed_range
            if isinstance(values, (list, np.ndarray)):
                return [(v - min_val) / (max_val - min_val) if v is not None else None for v in values]
            return (values - min_val) / (max_val - min_val) if values is not None else None

        # Price-based normalization (share OHLC range)
        elif norm_type == 'price':
            if price_range is None:
                raise ValueError("Price range required for price-based normalization")
            min_val, max_val = price_range
            if max_val == min_val:
                return 0.0 if isinstance(values, (int, float)) else [0.0] * len(values)
            if isinstance(values, (list, np.ndarray)):
                return [(v - min_val) / (max_val - min_val) if v is not None else None for v in values]
            return (values - min_val) / (max_val - min_val) if values is not None else None

        # Min-Max normalization on indicator's own values
        elif norm_type == 'minmax':
            if all_values is None:
                raise ValueError("All values required for min-max normalization")
            # Filter out None/NaN values
            valid_values = all_values[~np.isnan(all_values)]
            if len(valid_values) == 0:
                return None if isinstance(values, (int, float)) else [None] * len(values)
            min_val = float(np.min(valid_values))
            max_val = float(np.max(valid_values))
            if max_val == min_val:
                return 0.0 if isinstance(values, (int, float)) else [0.0] * len(values)
            if isinstance(values, (list, np.ndarray)):
                return [(v - min_val) / (max_val - min_val) if v is not None else None for v in values]
            return (values - min_val) / (max_val - min_val) if values is not None else None

        # No normalization
        elif norm_type == 'none':
            return values

        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")

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
