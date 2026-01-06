"""Tests for OrderBlock indicator."""

import pytest
from datetime import datetime, timedelta
from trading_frame import Candle, TimeFrame
from trading_frame.indicators.trend.order_block import OrderBlock


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


class TestOrderBlockBasic:
    """Basic functionality tests."""

    def test_initialization(self):
        """Test OrderBlock initialization."""
        ob = OrderBlock()
        assert ob.lookback == 10
        assert ob.min_body_pct == 0.3
        assert ob.open_source == 'open_price'
        assert ob.high_source == 'high_price'
        assert ob.low_source == 'low_price'
        assert ob.close_source == 'close_price'

    def test_initialization_custom_params(self):
        """Test OrderBlock with custom parameters."""
        ob = OrderBlock(
            lookback=5,
            min_body_pct=0.5,
            open_source='custom_open',
            high_source='custom_high',
            low_source='custom_low',
            close_source='custom_close'
        )
        assert ob.lookback == 5
        assert ob.min_body_pct == 0.5
        assert ob.open_source == 'custom_open'

    def test_invalid_lookback(self):
        """Test that lookback must be >= 1."""
        with pytest.raises(ValueError, match="lookback must be at least 1"):
            OrderBlock(lookback=0)

    def test_invalid_min_body_pct(self):
        """Test that min_body_pct must be between 0 and 1."""
        with pytest.raises(ValueError, match="min_body_pct must be between 0 and 1"):
            OrderBlock(min_body_pct=-0.1)
        with pytest.raises(ValueError, match="min_body_pct must be between 0 and 1"):
            OrderBlock(min_body_pct=1.5)

    def test_min_periods(self):
        """Test minimum periods calculation."""
        ob = OrderBlock(lookback=10)
        assert ob.requires_min_periods() == 2  # Only need current + 1 for comparison

    def test_dependencies(self):
        """Test dependency declaration."""
        ob = OrderBlock()
        deps = ob.get_dependencies()
        assert 'open_price' in deps
        assert 'high_price' in deps
        assert 'low_price' in deps
        assert 'close_price' in deps

    def test_num_outputs(self):
        """Test number of outputs."""
        ob = OrderBlock()
        assert ob.get_num_outputs() == 2  # ob_high, ob_low

    def test_normalization_type(self):
        """Test normalization type."""
        ob = OrderBlock()
        assert ob.get_normalization_type() == 'price'


class TestOrderBlockDetection:
    """Order Block detection tests."""

    def test_bullish_ob_detection(self):
        """Test bullish OB detection (last bearish candle before bullish breakout)."""
        frame = TimeFrame('1T', max_periods=50)
        ob = OrderBlock(lookback=5, min_body_pct=0.3)

        base_date = datetime(2024, 1, 1, 12, 0)

        # Pattern:
        # Candle 0: Bearish (this will be the OB)
        # Candle 1: Some candles...
        # Candle 2: Bullish breakout (close > candle 0 high)

        candles_data = [
            {'open': 105, 'high': 110, 'low': 95, 'close': 98},   # Bearish candle (OB candidate)
            {'open': 99, 'high': 104, 'low': 96, 'close': 101},   # Neutral/small move
            {'open': 102, 'high': 120, 'low': 100, 'close': 115}, # Bullish breakout (close > 110)
        ]

        for i, data in enumerate(candles_data):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(
                date,
                open_val=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close']
            )
            frame.feed(candle)

        # Add indicator
        frame.add_indicator(ob, ['OB_HIGH', 'OB_LOW'])

        # Check that candle 0 (bearish) is marked as Bullish OB
        assert frame.periods[0].OB_HIGH == 110.0  # High of bearish candle
        assert frame.periods[0].OB_LOW == 95.0    # Low of bearish candle

    def test_bearish_ob_detection(self):
        """Test bearish OB detection (last bullish candle before bearish breakout)."""
        frame = TimeFrame('1T', max_periods=50)
        ob = OrderBlock(lookback=5, min_body_pct=0.3)

        base_date = datetime(2024, 1, 1, 12, 0)

        # Pattern:
        # Candle 0: Bullish (this will be the OB)
        # Candle 1: Some candles...
        # Candle 2: Bearish breakout (close < candle 0 low)

        candles_data = [
            {'open': 95, 'high': 110, 'low': 90, 'close': 105},  # Bullish candle (OB candidate)
            {'open': 104, 'high': 108, 'low': 101, 'close': 103}, # Neutral/small move
            {'open': 102, 'high': 105, 'low': 80, 'close': 85},  # Bearish breakout (close < 90)
        ]

        for i, data in enumerate(candles_data):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(
                date,
                open_val=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close']
            )
            frame.feed(candle)

        # Add indicator
        frame.add_indicator(ob, ['OB_HIGH', 'OB_LOW'])

        # Check that candle 0 (bullish) is marked as Bearish OB
        assert frame.periods[0].OB_HIGH == 110.0  # High of bullish candle
        assert frame.periods[0].OB_LOW == 90.0    # Low of bullish candle

    def test_no_ob_without_breakout(self):
        """Test that no OB is detected without a breakout."""
        frame = TimeFrame('1T', max_periods=50)
        ob = OrderBlock(lookback=5, min_body_pct=0.3)

        base_date = datetime(2024, 1, 1, 12, 0)

        # Pattern: No clear breakout
        candles_data = [
            {'open': 100, 'high': 110, 'low': 95, 'close': 98},   # Bearish
            {'open': 99, 'high': 104, 'low': 96, 'close': 101},   # Small bullish
            {'open': 102, 'high': 108, 'low': 100, 'close': 106}, # Bullish but doesn't break above 110
        ]

        for i, data in enumerate(candles_data):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(
                date,
                open_val=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close']
            )
            frame.feed(candle)

        # Add indicator
        frame.add_indicator(ob, ['OB_HIGH', 'OB_LOW'])

        # No OB should be detected
        for period in frame.periods:
            assert period.OB_HIGH is None
            assert period.OB_LOW is None

    def test_min_body_size_filter(self):
        """Test that small body candles are filtered out."""
        frame = TimeFrame('1T', max_periods=50)
        ob = OrderBlock(lookback=5, min_body_pct=0.5)  # Require 50% body

        base_date = datetime(2024, 1, 1, 12, 0)

        # Pattern with doji-like bearish candle (small body)
        candles_data = [
            {'open': 102, 'high': 110, 'low': 95, 'close': 100},  # Bearish but small body (2 / 15 = 13%)
            {'open': 99, 'high': 104, 'low': 96, 'close': 101},
            {'open': 102, 'high': 120, 'low': 100, 'close': 115}, # Bullish breakout
        ]

        for i, data in enumerate(candles_data):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(
                date,
                open_val=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close']
            )
            frame.feed(candle)

        # Add indicator
        frame.add_indicator(ob, ['OB_HIGH', 'OB_LOW'])

        # No OB should be detected (body too small)
        assert frame.periods[0].OB_HIGH is None
        assert frame.periods[0].OB_LOW is None


class TestOrderBlockEdgeCases:
    """Edge cases tests."""

    def test_insufficient_periods(self):
        """Test with insufficient periods."""
        frame = TimeFrame('1T', max_periods=50)
        ob = OrderBlock(lookback=5)

        base_date = datetime(2024, 1, 1, 12, 0)

        # Only 1 candle
        candle = create_candle(base_date, open_val=100, high=110, low=95, close=98)
        frame.feed(candle)

        # Add indicator
        frame.add_indicator(ob, ['OB_HIGH', 'OB_LOW'])

        # No OB should be detected with < 2 candles
        assert frame.periods[0].OB_HIGH is None
        assert frame.periods[0].OB_LOW is None

    def test_lookback_limit(self):
        """Test that lookback is respected."""
        frame = TimeFrame('1T', max_periods=50)
        ob = OrderBlock(lookback=2, min_body_pct=0.3)

        base_date = datetime(2024, 1, 1, 12, 0)

        # Pattern: Bearish at index 0, but lookback=2 from index 5
        # So index 0 should NOT be detected
        candles_data = [
            {'open': 105, 'high': 110, 'low': 95, 'close': 98},   # Bearish (index 0)
            {'open': 99, 'high': 104, 'low': 96, 'close': 101},   # index 1
            {'open': 102, 'high': 106, 'low': 100, 'close': 104}, # index 2
            {'open': 103, 'high': 107, 'low': 101, 'close': 105}, # index 3 (bearish, within lookback)
            {'open': 106, 'high': 108, 'low': 102, 'close': 103}, # Bearish
            {'open': 104, 'high': 125, 'low': 102, 'close': 120}, # Bullish breakout (index 5)
        ]

        for i, data in enumerate(candles_data):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(
                date,
                open_val=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close']
            )
            frame.feed(candle)

        # Add indicator
        frame.add_indicator(ob, ['OB_HIGH', 'OB_LOW'])

        # Index 0 should NOT be marked (beyond lookback)
        assert frame.periods[0].OB_HIGH is None

        # Index 4 (last bearish within lookback) should be marked
        assert frame.periods[4].OB_HIGH == 108.0


class TestOrderBlockIntegration:
    """Integration tests with Frame."""

    def test_add_to_existing_periods(self):
        """Test adding OrderBlock to frame with existing periods."""
        frame = TimeFrame('1T', max_periods=50)
        base_date = datetime(2024, 1, 1, 12, 0)

        # Feed data first
        candles_data = [
            {'open': 105, 'high': 110, 'low': 95, 'close': 98},
            {'open': 99, 'high': 104, 'low': 96, 'close': 101},
            {'open': 102, 'high': 120, 'low': 100, 'close': 115},
        ]

        for i, data in enumerate(candles_data):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(
                date,
                open_val=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close']
            )
            frame.feed(candle)

        # Add indicator after data is loaded
        ob = OrderBlock(lookback=5, min_body_pct=0.3)
        frame.add_indicator(ob, ['OB_HIGH', 'OB_LOW'])

        # Verify calculation happened
        assert frame.periods[0].OB_HIGH == 110.0
        assert frame.periods[0].OB_LOW == 95.0

    def test_export_to_pandas(self):
        """Test exporting frame with OrderBlock to pandas."""
        frame = TimeFrame('1T', max_periods=50)
        ob = OrderBlock(lookback=5, min_body_pct=0.3)

        base_date = datetime(2024, 1, 1, 12, 0)
        candles_data = [
            {'open': 105, 'high': 110, 'low': 95, 'close': 98},
            {'open': 99, 'high': 104, 'low': 96, 'close': 101},
            {'open': 102, 'high': 120, 'low': 100, 'close': 115},
        ]

        for i, data in enumerate(candles_data):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(
                date,
                open_val=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close']
            )
            frame.feed(candle)

        frame.add_indicator(ob, ['OB_HIGH', 'OB_LOW'])

        # Export to pandas
        df = frame.to_pandas()

        # Check columns exist
        assert 'OB_HIGH' in df.columns
        assert 'OB_LOW' in df.columns

        # Check values
        assert df.loc[0, 'OB_HIGH'] == 110.0
        assert df.loc[0, 'OB_LOW'] == 95.0

    def test_custom_column_names(self):
        """Test custom column names work correctly."""
        frame = TimeFrame('1T', max_periods=50)
        ob = OrderBlock(lookback=5, min_body_pct=0.3)

        base_date = datetime(2024, 1, 1, 12, 0)
        candles_data = [
            {'open': 105, 'high': 110, 'low': 95, 'close': 98},
            {'open': 99, 'high': 104, 'low': 96, 'close': 101},
            {'open': 102, 'high': 120, 'low': 100, 'close': 115},
        ]

        for i, data in enumerate(candles_data):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(
                date,
                open_val=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close']
            )
            frame.feed(candle)

        # Use custom column names
        frame.add_indicator(ob, ['BLOCK_HIGH', 'BLOCK_LOW'])

        df = frame.to_pandas()
        assert 'BLOCK_HIGH' in df.columns
        assert 'BLOCK_LOW' in df.columns
        assert df.loc[0, 'BLOCK_HIGH'] == 110.0


class TestOrderBlockMultiple:
    """Test multiple OB detection."""

    def test_multiple_obs_same_trend(self):
        """Test detecting multiple OBs in same trend."""
        frame = TimeFrame('1T', max_periods=50)
        ob = OrderBlock(lookback=5, min_body_pct=0.3)

        base_date = datetime(2024, 1, 1, 12, 0)

        # Multiple bullish breakouts
        candles_data = [
            # First OB
            {'open': 105, 'high': 110, 'low': 95, 'close': 98},   # Bearish OB
            {'open': 99, 'high': 104, 'low': 96, 'close': 101},
            {'open': 102, 'high': 120, 'low': 100, 'close': 115}, # Breakout
            # Second OB
            {'open': 118, 'high': 122, 'low': 112, 'close': 114}, # Bearish OB
            {'open': 115, 'high': 119, 'low': 113, 'close': 117},
            {'open': 118, 'high': 135, 'low': 116, 'close': 130}, # Breakout
        ]

        for i, data in enumerate(candles_data):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(
                date,
                open_val=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close']
            )
            frame.feed(candle)

        frame.add_indicator(ob, ['OB_HIGH', 'OB_LOW'])

        # Check both OBs detected
        assert frame.periods[0].OB_HIGH == 110.0
        assert frame.periods[3].OB_HIGH == 122.0
