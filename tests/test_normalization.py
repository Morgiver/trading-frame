"""Tests for to_normalize() method."""

import pytest
import numpy as np
from datetime import datetime
from decimal import Decimal
from trading_frame import Candle, TimeFrame


class TestNormalization:
    """Test normalization of OHLCV data."""

    def test_normalize_basic(self):
        """Test basic normalization with known values."""
        frame = TimeFrame('1T', max_periods=10)

        # Create candles with known price range [100, 200]
        for i in range(5):
            candle = Candle(
                date=datetime(2024, 1, 1, 0, i),
                open=100.0 + i * 20,
                high=120.0 + i * 20,
                low=100.0 + i * 20,
                close=110.0 + i * 20,
                volume=Decimal('1000.0')
            )
            frame.feed(candle)

        normalized = frame.to_normalize()

        # Check shape
        assert normalized.shape == (5, 5)

        # Check all values are in [0, 1]
        assert np.all(normalized >= 0.0)
        assert np.all(normalized <= 1.0)

        # Check no NaN in valid data
        assert not np.isnan(normalized).any()

    def test_normalize_price_range(self):
        """Test that prices are normalized using unified min-max."""
        frame = TimeFrame('1T', max_periods=10)

        # Price range: 50 to 150
        candle1 = Candle(
            date=datetime(2024, 1, 1, 0, 0),
            open=50.0,
            high=60.0,
            low=50.0,
            close=55.0,
            volume=Decimal('1000.0')
        )
        candle2 = Candle(
            date=datetime(2024, 1, 1, 0, 1),
            open=140.0,
            high=150.0,
            low=135.0,
            close=145.0,
            volume=Decimal('2000.0')
        )

        frame.feed(candle1)
        frame.feed(candle2)

        normalized = frame.to_normalize()

        # Min price (50) should normalize to 0
        # Max price (150) should normalize to 1
        # Range = 100
        assert normalized[0, 0] == pytest.approx(0.0, abs=0.01)  # open=50
        assert normalized[1, 1] == pytest.approx(1.0, abs=0.01)  # high=150

    def test_normalize_volume_independent(self):
        """Test that volume is normalized independently from price."""
        frame = TimeFrame('1T', max_periods=10)

        # Different volume range than price
        candle1 = Candle(
            date=datetime(2024, 1, 1, 0, 0),
            open=100.0,
            high=110.0,
            low=100.0,
            close=105.0,
            volume=Decimal('500.0')
        )
        candle2 = Candle(
            date=datetime(2024, 1, 1, 0, 1),
            open=100.0,
            high=110.0,
            low=100.0,
            close=105.0,
            volume=Decimal('1500.0')
        )

        frame.feed(candle1)
        frame.feed(candle2)

        normalized = frame.to_normalize()

        # Volume should be normalized in its own range
        assert normalized[0, 4] == pytest.approx(0.0, abs=0.01)  # min volume
        assert normalized[1, 4] == pytest.approx(1.0, abs=0.01)  # max volume

    def test_normalize_empty_frame(self):
        """Test normalization with empty frame."""
        frame = TimeFrame('1T', max_periods=10)
        normalized = frame.to_normalize()

        assert normalized.shape == (0, 5)

    def test_normalize_constant_prices(self):
        """Test normalization when all prices are the same."""
        frame = TimeFrame('1T', max_periods=10)

        # All prices are 100
        for i in range(3):
            candle = Candle(
                date=datetime(2024, 1, 1, 0, i),
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=Decimal('1000.0')
            )
            frame.feed(candle)

        normalized = frame.to_normalize()

        # When range is 0, should handle gracefully (divide by 1.0)
        assert normalized.shape == (3, 5)
        # Prices should all be 0 (since range is 0, (100-100)/1.0 = 0)
        assert np.all(normalized[:, :4] == 0.0)

    def test_normalize_constant_volume(self):
        """Test normalization when all volumes are the same."""
        frame = TimeFrame('1T', max_periods=10)

        # All volumes are 1000
        for i in range(3):
            candle = Candle(
                date=datetime(2024, 1, 1, 0, i),
                open=100.0 + i * 10,
                high=110.0 + i * 10,
                low=100.0 + i * 10,
                close=105.0 + i * 10,
                volume=Decimal('1000.0')
            )
            frame.feed(candle)

        normalized = frame.to_normalize()

        # Volume should be 0 when constant
        assert np.all(normalized[:, 4] == 0.0)

    def test_normalize_dtype(self):
        """Test that normalized data has correct dtype."""
        frame = TimeFrame('1T', max_periods=10)

        for i in range(3):
            candle = Candle(
                date=datetime(2024, 1, 1, 0, i),
                open=100.0,
                high=110.0,
                low=95.0,
                close=105.0,
                volume=Decimal('1000.0')
            )
            frame.feed(candle)

        normalized = frame.to_normalize()

        assert normalized.dtype == np.float64

    def test_normalize_preserves_ohlc_relationships(self):
        """Test that OHLC relationships are preserved after normalization."""
        frame = TimeFrame('1T', max_periods=10)

        candle = Candle(
            date=datetime(2024, 1, 1, 0, 0),
            open=100.0,
            high=120.0,
            low=90.0,
            close=110.0,
            volume=Decimal('1000.0')
        )
        frame.feed(candle)

        normalized = frame.to_normalize()

        # High should still be highest, low should be lowest
        row = normalized[0]
        open_n, high_n, low_n, close_n, vol_n = row

        assert high_n >= open_n
        assert high_n >= close_n
        assert low_n <= open_n
        assert low_n <= close_n


class TestNormalizationIntegration:
    """Integration tests for normalization."""

    def test_normalize_for_ml_pipeline(self):
        """Test typical ML pipeline usage."""
        frame = TimeFrame('5T', max_periods=100)

        # Fill with realistic data
        from datetime import timedelta
        base_date = datetime(2024, 1, 1, 0, 0)
        for i in range(50):
            candle = Candle(
                date=base_date + timedelta(minutes=i * 5),
                open=50000.0 + i * 100,
                high=50100.0 + i * 100,
                low=49900.0 + i * 100,
                close=50050.0 + i * 100,
                volume=Decimal(str(1000.0 + i * 10))
            )
            frame.feed(candle)

        # Get normalized data
        normalized = frame.to_normalize()

        # Should be ready for ML
        assert normalized.shape == (50, 5)
        assert normalized.dtype == np.float64
        assert np.all(normalized >= 0.0)
        assert np.all(normalized <= 1.0)
        assert not np.isnan(normalized).any()

    def test_normalize_multiple_exports(self):
        """Test that normalization can be called multiple times."""
        frame = TimeFrame('1T', max_periods=10)

        for i in range(5):
            candle = Candle(
                date=datetime(2024, 1, 1, 0, i),
                open=100.0 + i,
                high=110.0 + i,
                low=95.0 + i,
                close=105.0 + i,
                volume=Decimal('1000.0')
            )
            frame.feed(candle)

        # Export twice
        norm1 = frame.to_normalize()
        norm2 = frame.to_normalize()

        # Should be identical
        assert np.array_equal(norm1, norm2)
