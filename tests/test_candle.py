"""Unit tests for Candle class."""

import pytest
from datetime import datetime
from trading_frame import Candle


class TestCandleInitialization:
    """Test Candle initialization with different date formats."""

    def test_init_with_datetime(self):
        """Test initialization with datetime object."""
        dt = datetime(2025, 1, 1, 0, 0, 0)
        candle = Candle(dt, 100.0, 110.0, 90.0, 105.0, 1000.0)

        assert candle.date == dt
        assert candle.open_price == 100.0
        assert candle.high_price == 110.0
        assert candle.low_price == 90.0
        assert candle.close_price == 105.0
        assert candle.volume == 1000.0

    def test_init_with_string_default_format(self):
        """Test initialization with string date (default format)."""
        candle = Candle(
            '2025-01-01T12:30:45.000Z',
            100.0, 110.0, 90.0, 105.0, 1000.0
        )

        assert candle.date.year == 2025
        assert candle.date.month == 1
        assert candle.date.day == 1
        assert candle.date.hour == 12
        assert candle.date.minute == 30

    def test_init_with_string_custom_format(self):
        """Test initialization with custom date format."""
        candle = Candle(
            '2025-01-01 12:30:45',
            100.0, 110.0, 90.0, 105.0, 1000.0,
            date_format='%Y-%m-%d %H:%M:%S'
        )

        assert candle.date.year == 2025
        assert candle.date.hour == 12

    def test_init_with_timestamp_int(self):
        """Test initialization with integer timestamp."""
        timestamp = 1704067200  # 2024-01-01 00:00:00 UTC
        candle = Candle(timestamp, 100.0, 110.0, 90.0, 105.0, 1000.0)

        assert candle.date == datetime.fromtimestamp(timestamp)

    def test_init_with_timestamp_float(self):
        """Test initialization with float timestamp."""
        timestamp = 1704067200.5
        candle = Candle(timestamp, 100.0, 110.0, 90.0, 105.0, 1000.0)

        assert candle.date == datetime.fromtimestamp(timestamp)


class TestCandleValidation:
    """Test Candle validation logic."""

    def test_invalid_date_string_format(self):
        """Test that invalid date format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid date format"):
            Candle('invalid-date', 100.0, 110.0, 90.0, 105.0, 1000.0)

    def test_invalid_date_type(self):
        """Test that unsupported date type raises TypeError."""
        with pytest.raises(TypeError, match="Date must be"):
            Candle([2025, 1, 1], 100.0, 110.0, 90.0, 105.0, 1000.0)

    def test_negative_prices(self):
        """Test that negative prices raise ValueError."""
        dt = datetime(2025, 1, 1)

        with pytest.raises(ValueError, match="Prices cannot be negative"):
            Candle(dt, -100.0, 110.0, 90.0, 105.0, 1000.0)

        with pytest.raises(ValueError, match="Prices cannot be negative"):
            Candle(dt, 100.0, -110.0, 90.0, 105.0, 1000.0)

    def test_high_lower_than_low(self):
        """Test that high < low raises ValueError."""
        dt = datetime(2025, 1, 1)

        with pytest.raises(ValueError, match="High price .* cannot be lower than low price"):
            Candle(dt, 100.0, 90.0, 110.0, 105.0, 1000.0)

    def test_high_not_containing_open_close(self):
        """Test that high must be >= max(open, close)."""
        dt = datetime(2025, 1, 1)

        with pytest.raises(ValueError, match="High/Low must contain"):
            Candle(dt, 100.0, 95.0, 90.0, 105.0, 1000.0)

    def test_low_not_containing_open_close(self):
        """Test that low must be <= min(open, close)."""
        dt = datetime(2025, 1, 1)

        with pytest.raises(ValueError, match="High/Low must contain"):
            Candle(dt, 100.0, 110.0, 101.0, 105.0, 1000.0)

    def test_negative_volume(self):
        """Test that negative volume raises ValueError."""
        dt = datetime(2025, 1, 1)

        with pytest.raises(ValueError, match="Volume cannot be negative"):
            Candle(dt, 100.0, 110.0, 90.0, 105.0, -1000.0)


class TestCandleMethods:
    """Test Candle methods."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        dt = datetime(2025, 1, 1, 12, 0, 0)
        candle = Candle(dt, 100.0, 110.0, 90.0, 105.0, 1000.0)

        result = candle.to_dict()

        assert result['date'] == dt
        assert result['open_price'] == 100.0
        assert result['high_price'] == 110.0
        assert result['low_price'] == 90.0
        assert result['close_price'] == 105.0
        assert result['volume'] == 1000.0

    def test_repr(self):
        """Test string representation."""
        dt = datetime(2025, 1, 1, 12, 0, 0)
        candle = Candle(dt, 100.0, 110.0, 90.0, 105.0, 1000.0)

        repr_str = repr(candle)

        assert 'Candle' in repr_str
        assert '2025-01-01' in repr_str
        assert 'open=100.0' in repr_str
        assert 'high=110.0' in repr_str
