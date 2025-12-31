"""Unit tests for Period class."""

import pytest
from datetime import datetime
from trading_frame import Candle, Period, Frame


class TestPeriodInitialization:
    """Test Period initialization."""

    def test_init(self):
        """Test basic initialization."""
        frame = Frame()
        open_date = datetime(2025, 1, 1, 0, 0, 0)
        close_date = datetime(2025, 1, 1, 0, 5, 0)

        period = Period(frame, open_date, close_date)

        assert period.frame is frame
        assert period.open_date == open_date
        assert period.close_date == close_date
        assert period.open_price is None
        assert period.high_price is None
        assert period.low_price is None
        assert period.close_price is None
        assert period.volume == 0.0

    def test_init_without_close_date(self):
        """Test initialization without close_date."""
        frame = Frame()
        open_date = datetime(2025, 1, 1, 0, 0, 0)

        period = Period(frame, open_date)

        assert period.close_date is None


class TestPeriodUpdate:
    """Test Period update logic."""

    def test_update_first_candle(self):
        """Test updating period with first candle."""
        frame = Frame()
        period = Period(frame, datetime(2025, 1, 1, 0, 0, 0))

        candle = Candle(
            datetime(2025, 1, 1, 0, 0, 0),
            100.0, 110.0, 90.0, 105.0, 1000.0
        )

        period.update(candle)

        assert period.open_price == 100.0
        assert period.high_price == 110.0
        assert period.low_price == 90.0
        assert period.close_price == 105.0
        assert period.volume == 1000.0

    def test_update_multiple_candles(self):
        """Test updating period with multiple candles."""
        frame = Frame()
        period = Period(frame, datetime(2025, 1, 1, 0, 0, 0))

        # First candle
        candle1 = Candle(
            datetime(2025, 1, 1, 0, 0, 0),
            100.0, 110.0, 90.0, 105.0, 1000.0
        )
        period.update(candle1)

        # Second candle with higher high
        candle2 = Candle(
            datetime(2025, 1, 1, 0, 1, 0),
            105.0, 120.0, 100.0, 115.0, 500.0
        )
        period.update(candle2)

        assert period.open_price == 100.0  # Unchanged
        assert period.high_price == 120.0  # Updated to higher
        assert period.low_price == 90.0    # Unchanged (still lowest)
        assert period.close_price == 115.0  # Updated to latest
        assert period.volume == 1500.0     # Accumulated

        # Third candle with lower low
        candle3 = Candle(
            datetime(2025, 1, 1, 0, 2, 0),
            115.0, 118.0, 85.0, 95.0, 750.0
        )
        period.update(candle3)

        assert period.open_price == 100.0  # Unchanged
        assert period.high_price == 120.0  # Unchanged (still highest)
        assert period.low_price == 85.0    # Updated to lower
        assert period.close_price == 95.0   # Updated to latest
        assert period.volume == 2250.0     # Accumulated


class TestPeriodMethods:
    """Test Period methods."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        frame = Frame()
        open_date = datetime(2025, 1, 1, 0, 0, 0)
        close_date = datetime(2025, 1, 1, 0, 5, 0)

        period = Period(frame, open_date, close_date)

        candle = Candle(
            datetime(2025, 1, 1, 0, 0, 0),
            100.0, 110.0, 90.0, 105.0, 1000.0
        )
        period.update(candle)

        result = period.to_dict()

        assert result['open_date'] == open_date
        assert result['close_date'] == close_date
        assert result['open_price'] == 100.0
        assert result['high_price'] == 110.0
        assert result['low_price'] == 90.0
        assert result['close_price'] == 105.0
        assert result['volume'] == 1000.0

    def test_attribute_access(self):
        """Test attribute-style access to data."""
        frame = Frame()
        period = Period(frame, datetime(2025, 1, 1, 0, 0, 0))

        # Test getting attributes
        assert period.volume == 0.0

        # Test setting attributes
        period.volume = 1500.0
        assert period.volume == 1500.0

    def test_invalid_attribute_access(self):
        """Test accessing invalid attributes raises AttributeError."""
        frame = Frame()
        period = Period(frame, datetime(2025, 1, 1, 0, 0, 0))

        with pytest.raises(AttributeError):
            _ = period.invalid_attribute

    def test_repr(self):
        """Test string representation."""
        frame = Frame()
        open_date = datetime(2025, 1, 1, 0, 0, 0)
        close_date = datetime(2025, 1, 1, 0, 5, 0)

        period = Period(frame, open_date, close_date)

        candle = Candle(open_date, 100.0, 110.0, 90.0, 105.0, 1000.0)
        period.update(candle)

        repr_str = repr(period)

        assert 'Period' in repr_str
        assert '2025-01-01' in repr_str
