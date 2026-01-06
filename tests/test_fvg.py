"""Tests for FVG (Fair Value Gap) indicator."""

import pytest
from datetime import datetime, timedelta
from trading_frame import Candle, TimeFrame
from trading_frame.indicators.trend.fvg import FVG


def create_candle(date, open_val=100, high=105, low=95, close=102, volume=1000):
    """Helper to create a test candle with automatic open/close adjustment."""
    # Ensure open and close are within high/low range
    open_val = max(low, min(high, open_val))
    close = max(low, min(high, close))

    return Candle(
        date=date,
        open=open_val,
        high=high,
        low=low,
        close=close,
        volume=volume
    )


class TestFVGBasic:
    """Basic functionality tests."""

    def test_initialization(self):
        """Test FVG initialization."""
        fvg = FVG()
        assert fvg.high_source == 'high_price'
        assert fvg.low_source == 'low_price'

    def test_initialization_custom_sources(self):
        """Test FVG with custom sources."""
        fvg = FVG(
            high_source='custom_high',
            low_source='custom_low'
        )
        assert fvg.high_source == 'custom_high'
        assert fvg.low_source == 'custom_low'

    def test_min_periods(self):
        """Test minimum periods calculation."""
        fvg = FVG()
        assert fvg.requires_min_periods() == 3  # Need 3 candles

    def test_dependencies(self):
        """Test dependency declaration."""
        fvg = FVG()
        deps = fvg.get_dependencies()
        assert 'high_price' in deps
        assert 'low_price' in deps

    def test_num_outputs(self):
        """Test number of outputs."""
        fvg = FVG()
        assert fvg.get_num_outputs() == 2  # fvg_high, fvg_low

    def test_normalization_type(self):
        """Test normalization type."""
        fvg = FVG()
        assert fvg.get_normalization_type() == 'price'


class TestFVGDetection:
    """FVG detection tests."""

    def test_bullish_fvg_detection(self):
        """Test bullish FVG detection (demand zone)."""
        frame = TimeFrame('1T', max_periods=50)
        fvg = FVG()

        base_date = datetime(2024, 1, 1, 12, 0)

        # Bullish FVG pattern:
        # Candle 1: High=100, Low=90
        # Candle 2: The gap candle (will have FVG marked)
        # Candle 3: High=120, Low=105 (Low > Candle1.High = gap!)
        # Gap range: [100, 105]

        candles_data = [
            {'high': 100, 'low': 90},   # Candle 1
            {'high': 103, 'low': 97},   # Candle 2 (FVG will be here)
            {'high': 120, 'low': 105},  # Candle 3 (confirmation)
        ]

        for i, data in enumerate(candles_data):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=data['high'], low=data['low'])
            frame.feed(candle)

        # Add indicator
        frame.add_indicator(fvg, ['FVG_HIGH', 'FVG_LOW'])

        # Check that candle 2 (index 1) has the bullish FVG
        assert frame.periods[1].FVG_HIGH == 105.0  # Low of candle 3
        assert frame.periods[1].FVG_LOW == 100.0   # High of candle 1

    def test_bearish_fvg_detection(self):
        """Test bearish FVG detection (supply zone)."""
        frame = TimeFrame('1T', max_periods=50)
        fvg = FVG()

        base_date = datetime(2024, 1, 1, 12, 0)

        # Bearish FVG pattern:
        # Candle 1: High=110, Low=100
        # Candle 2: The gap candle (will have FVG marked)
        # Candle 3: High=95, Low=85 (High < Candle1.Low = gap!)
        # Gap range: [95, 100]

        candles_data = [
            {'high': 110, 'low': 100},  # Candle 1
            {'high': 105, 'low': 98},   # Candle 2 (FVG will be here)
            {'high': 95, 'low': 85},    # Candle 3 (confirmation)
        ]

        for i, data in enumerate(candles_data):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=data['high'], low=data['low'])
            frame.feed(candle)

        # Add indicator
        frame.add_indicator(fvg, ['FVG_HIGH', 'FVG_LOW'])

        # Check that candle 2 (index 1) has the bearish FVG
        assert frame.periods[1].FVG_HIGH == 100.0  # Low of candle 1
        assert frame.periods[1].FVG_LOW == 95.0    # High of candle 3

    def test_no_fvg_when_no_gap(self):
        """Test that no FVG is detected when there's no gap."""
        frame = TimeFrame('1T', max_periods=50)
        fvg = FVG()

        base_date = datetime(2024, 1, 1, 12, 0)

        # No gap pattern - continuous price ranges
        candles_data = [
            {'high': 105, 'low': 95},
            {'high': 110, 'low': 100},
            {'high': 115, 'low': 105},
        ]

        for i, data in enumerate(candles_data):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=data['high'], low=data['low'])
            frame.feed(candle)

        # Add indicator
        frame.add_indicator(fvg, ['FVG_HIGH', 'FVG_LOW'])

        # No FVG should be detected
        assert frame.periods[1].FVG_HIGH is None
        assert frame.periods[1].FVG_LOW is None


class TestFVGEdgeCases:
    """Edge cases tests."""

    def test_insufficient_periods(self):
        """Test with insufficient periods."""
        frame = TimeFrame('1T', max_periods=50)
        fvg = FVG()

        base_date = datetime(2024, 1, 1, 12, 0)

        # Only 2 candles
        for i in range(2):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=110, low=90)
            frame.feed(candle)

        # Add indicator
        frame.add_indicator(fvg, ['FVG_HIGH', 'FVG_LOW'])

        # No FVG should be detected with < 3 candles
        for period in frame.periods:
            assert period.FVG_HIGH is None
            assert period.FVG_LOW is None

    def test_exact_boundary_no_gap(self):
        """Test exact boundary (touching prices, no gap)."""
        frame = TimeFrame('1T', max_periods=50)
        fvg = FVG()

        base_date = datetime(2024, 1, 1, 12, 0)

        # Candle 1: High=100, Low=90
        # Candle 2: Middle
        # Candle 3: Low=100 (exactly touching, NOT > 100, so NO gap)
        candles_data = [
            {'high': 100, 'low': 90},
            {'high': 103, 'low': 97},
            {'high': 110, 'low': 100},  # Low=100, not > 100
        ]

        for i, data in enumerate(candles_data):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=data['high'], low=data['low'])
            frame.feed(candle)

        # Add indicator
        frame.add_indicator(fvg, ['FVG_HIGH', 'FVG_LOW'])

        # No FVG because Low(3) = High(1), not > High(1)
        assert frame.periods[1].FVG_HIGH is None
        assert frame.periods[1].FVG_LOW is None


class TestFVGIntegration:
    """Integration tests with Frame."""

    def test_add_to_existing_periods(self):
        """Test adding FVG to frame with existing periods."""
        frame = TimeFrame('1T', max_periods=50)
        base_date = datetime(2024, 1, 1, 12, 0)

        # Feed data first (bullish FVG)
        candles_data = [
            {'high': 100, 'low': 90},
            {'high': 103, 'low': 97},
            {'high': 120, 'low': 105},
        ]

        for i, data in enumerate(candles_data):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=data['high'], low=data['low'])
            frame.feed(candle)

        # Add indicator after data is loaded
        fvg = FVG()
        frame.add_indicator(fvg, ['FVG_HIGH', 'FVG_LOW'])

        # Verify calculation happened
        assert frame.periods[1].FVG_HIGH == 105.0
        assert frame.periods[1].FVG_LOW == 100.0

    def test_remove_indicator(self):
        """Test removing FVG indicator."""
        frame = TimeFrame('1T', max_periods=50)
        fvg = FVG()

        base_date = datetime(2024, 1, 1, 12, 0)
        candles_data = [
            {'high': 100, 'low': 90},
            {'high': 103, 'low': 97},
            {'high': 120, 'low': 105},
        ]

        for i, data in enumerate(candles_data):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=data['high'], low=data['low'])
            frame.feed(candle)

        frame.add_indicator(fvg, ['FVG_HIGH', 'FVG_LOW'])

        # Remove indicator
        frame.remove_indicator(['FVG_HIGH', 'FVG_LOW'])

        # Columns should be removed
        assert 'FVG_HIGH' not in frame.periods[0]._data
        assert 'FVG_LOW' not in frame.periods[0]._data

    def test_export_to_pandas(self):
        """Test exporting frame with FVG to pandas."""
        frame = TimeFrame('1T', max_periods=50)
        fvg = FVG()

        base_date = datetime(2024, 1, 1, 12, 0)
        candles_data = [
            {'high': 100, 'low': 90},
            {'high': 103, 'low': 97},
            {'high': 120, 'low': 105},
        ]

        for i, data in enumerate(candles_data):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=data['high'], low=data['low'])
            frame.feed(candle)

        frame.add_indicator(fvg, ['FVG_HIGH', 'FVG_LOW'])

        # Export to pandas
        df = frame.to_pandas()

        # Check columns exist
        assert 'FVG_HIGH' in df.columns
        assert 'FVG_LOW' in df.columns

        # Check values
        assert df.loc[1, 'FVG_HIGH'] == 105.0
        assert df.loc[1, 'FVG_LOW'] == 100.0

    def test_export_to_numpy(self):
        """Test exporting frame with FVG to numpy."""
        frame = TimeFrame('1T', max_periods=50)
        fvg = FVG()

        base_date = datetime(2024, 1, 1, 12, 0)
        candles_data = [
            {'high': 100, 'low': 90},
            {'high': 103, 'low': 97},
            {'high': 120, 'low': 105},
        ]

        for i, data in enumerate(candles_data):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=data['high'], low=data['low'])
            frame.feed(candle)

        frame.add_indicator(fvg, ['FVG_HIGH', 'FVG_LOW'])

        # Export to numpy
        arr = frame.to_numpy()

        # Should have OHLCV (5) + FVG_HIGH (1) + FVG_LOW (1) = 7 columns
        assert arr.shape[1] == 7
        assert arr.shape[0] == 3  # 3 periods

    def test_normalization(self):
        """Test that FVG normalizes correctly with price range."""
        frame = TimeFrame('1T', max_periods=50)
        fvg = FVG()

        base_date = datetime(2024, 1, 1, 12, 0)
        candles_data = [
            {'high': 100, 'low': 90},
            {'high': 103, 'low': 97},
            {'high': 120, 'low': 105},
        ]

        for i, data in enumerate(candles_data):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=data['high'], low=data['low'])
            frame.feed(candle)

        frame.add_indicator(fvg, ['FVG_HIGH', 'FVG_LOW'])

        # Export normalized
        normalized = frame.to_normalize()

        # All values should be in [0, 1] range (or NaN)
        import numpy as np
        valid_values = normalized[~np.isnan(normalized)]
        assert np.all(valid_values >= 0.0)
        assert np.all(valid_values <= 1.0)

    def test_custom_column_names_integration(self):
        """Test that custom column names work end-to-end in Frame."""
        frame = TimeFrame('1T', max_periods=50)
        fvg = FVG()

        base_date = datetime(2024, 1, 1, 12, 0)
        candles_data = [
            {'high': 100, 'low': 90},
            {'high': 103, 'low': 97},
            {'high': 120, 'low': 105},
        ]

        for i, data in enumerate(candles_data):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=data['high'], low=data['low'])
            frame.feed(candle)

        # Use custom column names when adding indicator
        frame.add_indicator(fvg, ['GAP_HIGH', 'GAP_LOW'])

        # Export to pandas and check column names
        df = frame.to_pandas()
        assert 'GAP_HIGH' in df.columns
        assert 'GAP_LOW' in df.columns
        assert 'FVG_HIGH' not in df.columns
        assert 'FVG_LOW' not in df.columns

        # Verify values are assigned correctly
        assert df.iloc[1]['GAP_HIGH'] == 105.0
        assert df.iloc[1]['GAP_LOW'] == 100.0


class TestFVGMultiple:
    """Test multiple FVG detection."""

    def test_multiple_fvgs_in_sequence(self):
        """Test detecting multiple FVGs in a sequence."""
        frame = TimeFrame('1T', max_periods=50)
        fvg = FVG()

        base_date = datetime(2024, 1, 1, 12, 0)

        # Create pattern with 2 bullish FVGs
        candles_data = [
            # First FVG
            {'high': 100, 'low': 90},   # Candle 1
            {'high': 103, 'low': 97},   # Candle 2 (FVG #1 here)
            {'high': 120, 'low': 105},  # Candle 3 (confirmation)
            # Second FVG
            {'high': 125, 'low': 115},  # Candle 1
            {'high': 128, 'low': 122},  # Candle 2 (FVG #2 here)
            {'high': 145, 'low': 130},  # Candle 3 (confirmation)
        ]

        for i, data in enumerate(candles_data):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=data['high'], low=data['low'])
            frame.feed(candle)

        frame.add_indicator(fvg, ['FVG_HIGH', 'FVG_LOW'])

        # Check first FVG at index 1
        assert frame.periods[1].FVG_HIGH == 105.0
        assert frame.periods[1].FVG_LOW == 100.0

        # Check second FVG at index 4
        assert frame.periods[4].FVG_HIGH == 130.0
        assert frame.periods[4].FVG_LOW == 125.0
