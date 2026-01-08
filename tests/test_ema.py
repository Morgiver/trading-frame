"""
Tests for EMA (Exponential Moving Average) indicator
"""

import pytest
from trading_frame import Candle, TimeFrame
from trading_frame.indicators.trend.ema import EMA
from datetime import datetime, timedelta


def create_candle(date, close, volume=1000):
    """Helper to create a candle with specified close price"""
    # Make sure high/low contain open and close
    open_price = close - 1
    high = max(open_price, close) + 1
    low = min(open_price, close) - 1

    return Candle(
        date=date,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume
    )


def test_ema_basic():
    """Test basic EMA calculation"""
    frame = TimeFrame('1D', max_periods=100)
    ema = EMA(period=5, source='close_price')
    frame.add_indicator(ema, ['EMA_5'])

    base_date = datetime(2025, 1, 1)

    # Simple increasing prices
    prices = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118]

    for i, price in enumerate(prices):
        candle = create_candle(base_date + timedelta(days=i), close=price)
        frame.feed(candle)

    # Check that EMA exists after enough periods
    assert frame.periods[4].EMA_5 is not None, "EMA should exist after 5 periods"
    assert frame.periods[9].EMA_5 is not None, "EMA should exist at last period"

    # EMA should be between min and max price
    last_ema = frame.periods[9].EMA_5
    assert 100 <= last_ema <= 118, "EMA should be within price range"


def test_ema_vs_sma():
    """Test that EMA is more responsive than SMA"""
    from trading_frame.indicators.trend.sma import SMA

    frame = TimeFrame('1D', max_periods=100)
    ema = EMA(period=10, source='close_price')
    sma = SMA(period=10, source='close_price')
    frame.add_indicator(ema, ['EMA_10'])
    frame.add_indicator(sma, ['SMA_10'])

    base_date = datetime(2025, 1, 1)

    # Stable prices then sudden jump
    prices = [100] * 10 + [120] * 5

    for i, price in enumerate(prices):
        candle = create_candle(base_date + timedelta(days=i), close=price)
        frame.feed(candle)

    # After sudden jump, EMA should react faster than SMA
    # EMA should be closer to new price (120) than SMA
    last_period = frame.periods[-1]
    ema_value = last_period.EMA_10
    sma_value = last_period.SMA_10

    assert ema_value is not None and sma_value is not None
    assert ema_value > sma_value, "EMA should be higher (more responsive) after price increase"
    assert ema_value > 105, "EMA should have reacted to price jump"


def test_ema_downtrend():
    """Test EMA in downtrend"""
    frame = TimeFrame('1D', max_periods=100)
    ema = EMA(period=5, source='close_price')
    frame.add_indicator(ema, ['EMA_5'])

    base_date = datetime(2025, 1, 1)

    # Decreasing prices
    prices = [120, 118, 116, 114, 112, 110, 108, 106, 104, 102]

    for i, price in enumerate(prices):
        candle = create_candle(base_date + timedelta(days=i), close=price)
        frame.feed(candle)

    # EMA should decrease
    ema_values = [p.EMA_5 for p in frame.periods if p.EMA_5 is not None]

    # Check that EMA is generally decreasing
    assert ema_values[0] > ema_values[-1], "EMA should decrease in downtrend"


def test_ema_different_periods():
    """Test EMA with different periods"""
    frame = TimeFrame('1D', max_periods=100)
    ema_fast = EMA(period=5, source='close_price')
    ema_slow = EMA(period=20, source='close_price')
    frame.add_indicator(ema_fast, ['EMA_5'])
    frame.add_indicator(ema_slow, ['EMA_20'])

    base_date = datetime(2025, 1, 1)

    # Generate 30 periods with trending prices
    prices = list(range(100, 130))

    for i, price in enumerate(prices):
        candle = create_candle(base_date + timedelta(days=i), close=price)
        frame.feed(candle)

    # Fast EMA should be closer to current price than slow EMA
    last_period = frame.periods[-1]
    assert last_period.EMA_5 is not None
    assert last_period.EMA_20 is not None

    current_price = 129
    distance_fast = abs(current_price - last_period.EMA_5)
    distance_slow = abs(current_price - last_period.EMA_20)

    assert distance_fast < distance_slow, "Fast EMA should be closer to current price"


def test_ema_insufficient_data():
    """Test EMA with insufficient data"""
    frame = TimeFrame('1D', max_periods=100)
    ema = EMA(period=10, source='close_price')
    frame.add_indicator(ema, ['EMA_10'])

    base_date = datetime(2025, 1, 1)

    # Only 5 periods (need 10)
    for i in range(5):
        candle = create_candle(base_date + timedelta(days=i), close=100 + i)
        frame.feed(candle)

    # All EMA values should be None
    for period in frame.periods:
        assert period.EMA_10 is None, "EMA should be None with insufficient data"


def test_ema_golden_cross():
    """Test EMA golden cross scenario (fast crosses above slow)"""
    frame = TimeFrame('1D', max_periods=100)
    ema_fast = EMA(period=5, source='close_price')
    ema_slow = EMA(period=10, source='close_price')
    frame.add_indicator(ema_fast, ['EMA_5'])
    frame.add_indicator(ema_slow, ['EMA_10'])

    base_date = datetime(2025, 1, 1)

    # Downtrend then sharp uptrend
    prices = [120, 118, 116, 114, 112, 110, 108, 106, 104, 102] + \
             [104, 108, 112, 116, 120, 124, 128, 132, 136, 140]

    for i, price in enumerate(prices):
        candle = create_candle(base_date + timedelta(days=i), close=price)
        frame.feed(candle)

    # Find where fast crosses above slow
    crossed = False
    for i in range(10, len(frame.periods) - 1):
        curr = frame.periods[i]
        prev = frame.periods[i - 1]

        if curr.EMA_5 is not None and curr.EMA_10 is not None:
            if prev.EMA_5 is not None and prev.EMA_10 is not None:
                if prev.EMA_5 <= prev.EMA_10 and curr.EMA_5 > curr.EMA_10:
                    crossed = True
                    break

    assert crossed, "Fast EMA should cross above slow EMA (golden cross)"


def test_ema_validation():
    """Test EMA parameter validation"""
    with pytest.raises(ValueError):
        EMA(period=0)

    with pytest.raises(ValueError):
        EMA(period=-1)


def test_ema_with_custom_source():
    """Test EMA with custom source column"""
    from trading_frame.indicators.trend.sma import SMA

    frame = TimeFrame('1D', max_periods=100)

    # First calculate SMA
    sma = SMA(period=5, source='close_price')
    frame.add_indicator(sma, ['SMA_5'])

    # Then calculate EMA of SMA
    ema_of_sma = EMA(period=5, source='SMA_5')
    frame.add_indicator(ema_of_sma, ['EMA_SMA'])

    base_date = datetime(2025, 1, 1)

    for i in range(20):
        candle = create_candle(base_date + timedelta(days=i), close=100 + i)
        frame.feed(candle)

    # EMA of SMA should exist after enough periods
    # SMA needs 5 periods, then EMA needs 5 periods of SMA
    # First 4 periods: no SMA
    # Period 4+: SMA exists
    # Period 8+: EMA of SMA exists (has 5 SMA values from periods 4-8)
    assert frame.periods[3].SMA_5 is None, "SMA should be None before 5 periods"
    assert frame.periods[4].SMA_5 is not None, "SMA should exist at period 5"
    assert frame.periods[7].EMA_SMA is None, "EMA_SMA should be None with only 4 SMA values"
    assert frame.periods[8].EMA_SMA is not None, "EMA_SMA should exist with 5 SMA values"


def test_ema_repr():
    """Test EMA string representation"""
    ema = EMA(period=20, source='close_price')
    assert repr(ema) == "EMA(period=20, source='close_price')"
