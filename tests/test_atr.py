"""
Tests for ATR (Average True Range) indicator
"""

import pytest
from trading_frame import Candle, TimeFrame
from trading_frame.indicators.volatility.atr import ATR
from datetime import datetime, timedelta


def create_candle(date, open_price, high, low, close, volume=1000):
    """Helper to create a candle with OHLC"""
    return Candle(
        date=date,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume
    )


def test_atr_basic():
    """Test basic ATR calculation"""
    frame = TimeFrame('1D', max_periods=100)
    atr = ATR(period=5)
    frame.add_indicator(atr, ['ATR_5'])

    base_date = datetime(2025, 1, 1)

    # Create candles with varying ranges
    candles_data = [
        # (open, high, low, close)
        (100, 105, 95, 102),
        (102, 108, 100, 105),
        (105, 110, 103, 107),
        (107, 112, 105, 110),
        (110, 115, 108, 112),
        (112, 118, 110, 115),
        (115, 120, 113, 117),
    ]

    for i, (o, h, l, c) in enumerate(candles_data):
        candle = create_candle(base_date + timedelta(days=i), o, h, l, c)
        frame.feed(candle)

    # ATR should exist after period + 1 periods
    assert frame.periods[4].ATR_5 is None, "ATR should be None before period+1"
    assert frame.periods[5].ATR_5 is not None, "ATR should exist after period+1"

    # ATR should be positive
    atr_value = frame.periods[6].ATR_5
    assert atr_value > 0, "ATR should be positive"


def test_atr_increasing_volatility():
    """Test ATR increases with increasing volatility"""
    frame = TimeFrame('1D', max_periods=100)
    atr = ATR(period=5)
    frame.add_indicator(atr, ['ATR_5'])

    base_date = datetime(2025, 1, 1)

    # First period: low volatility (small ranges)
    for i in range(10):
        candle = create_candle(
            base_date + timedelta(days=i),
            open_price=100,
            high=102,
            low=98,
            close=100 + (i % 2)  # Alternate slightly
        )
        frame.feed(candle)

    low_volatility_atr = frame.periods[9].ATR_5

    # Second period: high volatility (large ranges)
    for i in range(10, 20):
        candle = create_candle(
            base_date + timedelta(days=i),
            open_price=100,
            high=110,
            low=90,
            close=100 + ((i - 10) % 2) * 5
        )
        frame.feed(candle)

    high_volatility_atr = frame.periods[19].ATR_5

    assert high_volatility_atr > low_volatility_atr, "ATR should increase with volatility"


def test_atr_decreasing_volatility():
    """Test ATR decreases with decreasing volatility"""
    frame = TimeFrame('1D', max_periods=100)
    atr = ATR(period=5)
    frame.add_indicator(atr, ['ATR_5'])

    base_date = datetime(2025, 1, 1)

    # First period: high volatility
    for i in range(10):
        candle = create_candle(
            base_date + timedelta(days=i),
            open_price=100,
            high=115,
            low=85,
            close=100 + (i % 2) * 10
        )
        frame.feed(candle)

    high_volatility_atr = frame.periods[9].ATR_5

    # Second period: low volatility
    for i in range(10, 20):
        candle = create_candle(
            base_date + timedelta(days=i),
            open_price=100,
            high=101,
            low=99,
            close=100
        )
        frame.feed(candle)

    low_volatility_atr = frame.periods[19].ATR_5

    assert low_volatility_atr < high_volatility_atr, "ATR should decrease as volatility decreases"


def test_atr_gap_up():
    """Test ATR with gap up (tests Previous Close component of True Range)"""
    frame = TimeFrame('1D', max_periods=100)
    atr = ATR(period=5)
    frame.add_indicator(atr, ['ATR_5'])

    base_date = datetime(2025, 1, 1)

    # Stable prices
    for i in range(8):
        candle = create_candle(
            base_date + timedelta(days=i),
            open_price=100,
            high=102,
            low=98,
            close=100
        )
        frame.feed(candle)

    atr_before_gap = frame.periods[7].ATR_5

    # Gap up: close at 100, next day opens and stays at 120
    candle = create_candle(
        base_date + timedelta(days=8),
        open_price=120,
        high=122,
        low=118,
        close=120
    )
    frame.feed(candle)

    atr_after_gap = frame.periods[8].ATR_5

    # ATR should increase due to gap (High - Previous Close = 122 - 100 = 22)
    assert atr_after_gap > atr_before_gap, "ATR should increase after gap up"


def test_atr_different_periods():
    """Test ATR with different periods"""
    frame = TimeFrame('1D', max_periods=100)
    atr_short = ATR(period=5)
    atr_long = ATR(period=20)
    frame.add_indicator(atr_short, ['ATR_5'])
    frame.add_indicator(atr_long, ['ATR_20'])

    base_date = datetime(2025, 1, 1)

    # Create volatile market
    for i in range(30):
        volatility = 10 + (i % 5) * 2  # Varying volatility
        candle = create_candle(
            base_date + timedelta(days=i),
            open_price=100,
            high=100 + volatility,
            low=100 - volatility,
            close=100 + (i % 3 - 1) * 2
        )
        frame.feed(candle)

    # Short-period ATR should be more responsive
    last_period = frame.periods[-1]
    assert last_period.ATR_5 is not None
    assert last_period.ATR_20 is not None

    # Both should be positive
    assert last_period.ATR_5 > 0
    assert last_period.ATR_20 > 0


def test_atr_insufficient_data():
    """Test ATR with insufficient data"""
    frame = TimeFrame('1D', max_periods=100)
    atr = ATR(period=14)
    frame.add_indicator(atr, ['ATR_14'])

    base_date = datetime(2025, 1, 1)

    # Only 10 periods (need 15 for period=14)
    for i in range(10):
        close = 100 + i
        candle = create_candle(
            base_date + timedelta(days=i),
            open_price=close - 1,
            high=close + 5,
            low=close - 5,
            close=close
        )
        frame.feed(candle)

    # All ATR values should be None
    for period in frame.periods:
        assert period.ATR_14 is None, "ATR should be None with insufficient data"


def test_atr_validation():
    """Test ATR parameter validation"""
    with pytest.raises(ValueError):
        ATR(period=0)

    with pytest.raises(ValueError):
        ATR(period=-1)


def test_atr_trending_market():
    """Test ATR in trending market"""
    frame = TimeFrame('1D', max_periods=100)
    atr = ATR(period=10)
    frame.add_indicator(atr, ['ATR_10'])

    base_date = datetime(2025, 1, 1)

    # Strong uptrend with expanding ranges
    for i in range(20):
        base_price = 100 + i * 2
        range_size = 5 + i * 0.5
        candle = create_candle(
            base_date + timedelta(days=i),
            open_price=base_price,
            high=base_price + range_size,
            low=base_price - range_size / 2,
            close=base_price + range_size / 2
        )
        frame.feed(candle)

    # ATR should increase in trending market with expanding ranges
    atr_values = [p.ATR_10 for p in frame.periods if p.ATR_10 is not None]
    assert len(atr_values) > 5
    assert atr_values[-1] > atr_values[0], "ATR should increase with expanding ranges"


def test_atr_consolidation():
    """Test ATR in consolidating market (low volatility)"""
    frame = TimeFrame('1D', max_periods=100)
    atr = ATR(period=10)
    frame.add_indicator(atr, ['ATR_10'])

    base_date = datetime(2025, 1, 1)

    # First: normal volatility
    for i in range(15):
        candle = create_candle(
            base_date + timedelta(days=i),
            open_price=100,
            high=108,
            low=92,
            close=100 + (i % 3 - 1) * 2
        )
        frame.feed(candle)

    normal_atr = frame.periods[14].ATR_10

    # Then: tight consolidation
    for i in range(15, 30):
        candle = create_candle(
            base_date + timedelta(days=i),
            open_price=100,
            high=101,
            low=99,
            close=100
        )
        frame.feed(candle)

    consolidation_atr = frame.periods[29].ATR_10

    assert consolidation_atr < normal_atr, "ATR should decrease during consolidation"
    # ATR decays slowly, so after only 15 periods it won't be that low yet
    assert consolidation_atr < normal_atr * 0.7, "ATR should decrease significantly"


def test_atr_repr():
    """Test ATR string representation"""
    atr = ATR(period=14)
    assert repr(atr) == "ATR(period=14)"
