"""Candle data structure for trading data."""

from datetime import datetime
from decimal import Decimal
from typing import Union


class Candle:
    """
    Represents a single OHLCV candle.

    A Candle is a aggregated trading data over a specific period containing:
    - Opening, High, Low, Closing prices
    - Volume traded during the period
    """

    def __init__(
        self,
        date: Union[str, int, float, datetime],
        open: float,
        high: float,
        low: float,
        close: float,
        volume: Union[float, Decimal],
        date_format: str = '%Y-%m-%dT%H:%M:%S.%fZ'
    ) -> None:
        """
        Initialize a Candle.

        Parameters:
            date: Opening date of the candle (str, timestamp, or datetime)
            open: Opening price
            high: Highest price during the period
            low: Lowest price during the period
            close: Closing price
            volume: Total volume traded (supports Decimal for precision)
            date_format: Format string for date parsing (if date is str)

        Raises:
            ValueError: If date format is invalid or prices are negative
            TypeError: If date type is not supported
        """
        # Parse date
        if isinstance(date, datetime):
            self.date = date
        elif isinstance(date, str):
            try:
                self.date = datetime.strptime(date, date_format)
            except ValueError as e:
                raise ValueError(f"Invalid date format. Expected '{date_format}': {e}")
        elif isinstance(date, (int, float)):
            try:
                self.date = datetime.fromtimestamp(date)
            except (ValueError, OSError) as e:
                raise ValueError(f"Invalid timestamp: {e}")
        else:
            raise TypeError(f"Date must be str, int, float, or datetime, got {type(date)}")

        # Validate prices
        if open < 0 or high < 0 or low < 0 or close < 0:
            raise ValueError("Prices cannot be negative")

        if high < low:
            raise ValueError(f"High price ({high}) cannot be lower than low price ({low})")

        if high < max(open, close) or low > min(open, close):
            raise ValueError("High/Low must contain open and close prices")

        # Convert volume to Decimal for precision
        if isinstance(volume, Decimal):
            volume_decimal = volume
        else:
            volume_decimal = Decimal(str(volume))

        if volume_decimal < 0:
            raise ValueError("Volume cannot be negative")

        self.open_price = float(open)
        self.high_price = float(high)
        self.low_price = float(low)
        self.close_price = float(close)
        self.volume = volume_decimal

    def __repr__(self) -> str:
        """String representation of the Candle."""
        return (
            f"Candle(date={self.date.isoformat()}, "
            f"open={self.open_price}, high={self.high_price}, "
            f"low={self.low_price}, close={self.close_price}, "
            f"volume={self.volume})"
        )

    def to_dict(self) -> dict:
        """
        Convert Candle to dictionary.

        Returns:
            Dictionary containing all candle data
        """
        return {
            'date': self.date,
            'open_price': self.open_price,
            'high_price': self.high_price,
            'low_price': self.low_price,
            'close_price': self.close_price,
            'volume': self.volume
        }
