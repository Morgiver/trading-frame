"""Unit tests for Frame class."""

import pytest
from datetime import datetime
from trading_frame import Candle, Frame


class TestFrameInitialization:
    """Test Frame initialization."""

    def test_init_default(self):
        """Test default initialization."""
        frame = Frame()

        assert frame.max_periods == 250
        assert len(frame.periods) == 0
        assert 'new_period' in frame._event_channels
        assert 'update' in frame._event_channels
        assert 'close' in frame._event_channels

    def test_init_custom_max_periods(self):
        """Test initialization with custom max_periods."""
        frame = Frame(max_periods=100)

        assert frame.max_periods == 100

    def test_init_invalid_max_periods(self):
        """Test that invalid max_periods raises ValueError."""
        with pytest.raises(ValueError, match="max_periods must be at least 1"):
            Frame(max_periods=0)

        with pytest.raises(ValueError, match="max_periods must be at least 1"):
            Frame(max_periods=-5)


class TestFrameEvents:
    """Test Frame event system."""

    def test_on_register_callback(self):
        """Test registering event callbacks."""
        frame = Frame()
        callback_called = []

        def callback(*args, **kwargs):
            callback_called.append(True)

        frame.on('new_period', callback)

        assert len(frame._event_channels['new_period']) == 1

    def test_on_invalid_channel(self):
        """Test registering callback on invalid channel raises ValueError."""
        frame = Frame()

        with pytest.raises(ValueError, match="Invalid channel"):
            frame.on('invalid_channel', lambda: None)

    def test_emit_event(self):
        """Test emitting events calls callbacks."""
        frame = Frame()
        callback_data = []

        def callback(arg1, arg2):
            callback_data.append((arg1, arg2))

        frame.on('update', callback)
        frame.emit('update', 'value1', 'value2')

        assert len(callback_data) == 1
        assert callback_data[0] == ('value1', 'value2')

    def test_emit_multiple_callbacks(self):
        """Test emitting event calls all registered callbacks."""
        frame = Frame()
        call_count = []

        def callback1():
            call_count.append(1)

        def callback2():
            call_count.append(2)

        frame.on('close', callback1)
        frame.on('close', callback2)
        frame.emit('close')

        assert len(call_count) == 2
        assert 1 in call_count
        assert 2 in call_count


class TestFrameFeed:
    """Test Frame feed functionality."""

    def test_feed_single_candle(self):
        """Test feeding a single candle creates a period."""
        frame = Frame()
        candle = Candle(
            datetime(2025, 1, 1, 0, 0, 0),
            100.0, 110.0, 90.0, 105.0, 1000.0
        )

        frame.feed(candle)

        assert len(frame.periods) == 1
        assert frame.periods[0].open_price == 100.0

    def test_feed_invalid_type(self):
        """Test feeding non-Candle raises TypeError."""
        frame = Frame()

        with pytest.raises(TypeError, match="Expected Candle instance"):
            frame.feed("not a candle")

    def test_feed_respects_max_periods(self):
        """Test that feed respects max_periods limit."""
        frame = Frame(max_periods=3)

        for i in range(5):
            candle = Candle(
                datetime(2025, 1, 1, 0, i, 0),
                100.0, 110.0, 90.0, 105.0, 1000.0
            )
            frame.feed(candle)

        # Should only keep last 3 periods
        assert len(frame.periods) == 3

    def test_feed_triggers_new_period_event(self):
        """Test that feeding triggers new_period event."""
        frame = Frame()
        events = []

        def on_new_period(f):
            events.append('new_period')

        frame.on('new_period', on_new_period)

        candle = Candle(
            datetime(2025, 1, 1, 0, 0, 0),
            100.0, 110.0, 90.0, 105.0, 1000.0
        )
        frame.feed(candle)

        assert 'new_period' in events

    def test_feed_triggers_close_event(self):
        """Test that feeding a second period triggers close event."""
        frame = Frame()
        events = []

        def on_close(f):
            events.append('close')

        frame.on('close', on_close)

        # First candle
        candle1 = Candle(
            datetime(2025, 1, 1, 0, 0, 0),
            100.0, 110.0, 90.0, 105.0, 1000.0
        )
        frame.feed(candle1)

        assert len(events) == 0  # No close yet

        # Second candle (will trigger new period, thus close)
        candle2 = Candle(
            datetime(2025, 1, 1, 0, 1, 0),
            105.0, 115.0, 95.0, 110.0, 500.0
        )
        frame.feed(candle2)

        assert 'close' in events


class TestFrameUpdatePeriod:
    """Test Frame update_period method."""

    def test_update_period_no_periods_raises(self):
        """Test that update_period without periods raises RuntimeError."""
        frame = Frame()
        candle = Candle(
            datetime(2025, 1, 1, 0, 0, 0),
            100.0, 110.0, 90.0, 105.0, 1000.0
        )

        with pytest.raises(RuntimeError, match="No periods to update"):
            frame.update_period(candle)
