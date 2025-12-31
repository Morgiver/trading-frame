"""Period data structure for aggregated trading data."""

from datetime import datetime
from decimal import Decimal
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .candle import Candle
    from .frame import Frame


class Period:
    """
    Represents an aggregated period of trading data.

    A Period aggregates multiple candles into a single time range with OHLCV data.
    """

    def __init__(
        self,
        frame: 'Frame',
        open_date: datetime,
        close_date: Optional[datetime] = None
    ) -> None:
        """
        Initialize a Period.

        Parameters:
            frame: Parent Frame that owns this period
            open_date: Opening datetime of the period
            close_date: Closing datetime of the period (can be None initially)
        """
        self.frame = frame
        self._data = {
            'open_date': open_date,
            'close_date': close_date,
            'open_price': None,
            'high_price': None,
            'low_price': None,
            'close_price': None,
            'volume': Decimal('0')
        }

    def __getattr__(self, name: str):
        """Allow attribute-style access to data dictionary."""
        if name.startswith('_'):
            # Avoid recursion for private attributes
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        if '_data' in self.__dict__ and name in self._data:
            return self._data[name]

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value) -> None:
        """Allow attribute-style setting of data dictionary."""
        if name in ('frame', '_data'):
            # Set internal attributes directly
            super().__setattr__(name, value)
        elif '_data' in self.__dict__ and name in self._data:
            # Set data dictionary values
            self._data[name] = value
        else:
            # Default behavior for new attributes
            super().__setattr__(name, value)

    def __repr__(self) -> str:
        """String representation of the Period."""
        return (
            f"Period(open_date={self.open_date.isoformat()}, "
            f"close_date={self.close_date.isoformat() if self.close_date else 'None'}, "
            f"open={self.open_price}, high={self.high_price}, "
            f"low={self.low_price}, close={self.close_price}, "
            f"volume={self.volume})"
        )

    def update(self, candle: 'Candle') -> None:
        """
        Update period with new candle data.

        Parameters:
            candle: Candle to aggregate into this period
        """
        # Initialize open price if this is the first candle
        if self.open_price is None:
            self.open_price = candle.open_price

        # Update high price
        if self.high_price is None or candle.high_price > self.high_price:
            self.high_price = candle.high_price

        # Update low price
        if self.low_price is None or candle.low_price < self.low_price:
            self.low_price = candle.low_price

        # Always update close price to latest candle's close
        self.close_price = candle.close_price

        # Accumulate volume using Decimal for precision
        self.volume += candle.volume

    def to_dict(self) -> dict:
        """
        Convert Period to dictionary.

        Returns:
            Dictionary containing all period data
        """
        return self._data.copy()

    def to_numpy(self):
        """
        Convert Period to numpy array.

        Returns:
            numpy.ndarray: Array with [open, high, low, close, volume]
        """
        import numpy as np
        return np.array([
            float(self.open_price) if self.open_price is not None else np.nan,
            float(self.high_price) if self.high_price is not None else np.nan,
            float(self.low_price) if self.low_price is not None else np.nan,
            float(self.close_price) if self.close_price is not None else np.nan,
            float(self.volume)
        ], dtype=np.float64)
