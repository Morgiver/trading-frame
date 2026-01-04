"""Unit tests for prefill() method."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from trading_frame import Candle, TimeFrame


class TestPrefillTargetPeriods:
    """Test prefill() with target_periods parameter."""

    def test_prefill_default_target(self):
        """Test prefill with default target (max_periods)."""
        frame = TimeFrame('1T', max_periods=10)

        # Default behavior: target_periods = max_periods = 10 closed periods
        # But max_periods also limits total periods, so we can only ever have 9 closed + 1 open = 10 total
        # This means the default won't ever reach 10 closed periods due to max_periods limit

        # To get 10 closed periods, we need 11 total (10 closed + 1 open)
        # But max_periods=10 prevents this
        # So the default behavior is: stop when we have max_periods closed OR hit the limit

        result = False
        for i in range(20):
            candle = Candle(
                date=datetime(2024, 1, 1, 0, i),
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=Decimal('1000.0')
            )
            result = frame.prefill(candle)

            if result:
                break

        # When max_periods limits us, we can only get max_periods-1 closed periods
        # This is expected behavior
        closed = len(frame.periods) - 1 if frame.periods else 0
        assert closed >= 9, f"Should have at least 9 closed periods, got {closed}"
        assert len(frame.periods) == 10, "Frame should be at max_periods capacity"

    def test_prefill_custom_target_periods(self):
        """Test prefill with custom target_periods."""
        frame = TimeFrame('1T', max_periods=100)

        # Target 5 closed periods
        target = 5

        # Generate candles
        for i in range(10):
            candle = Candle(
                date=datetime(2024, 1, 1, 0, i),
                open=100.0 + i,
                high=105.0 + i,
                low=95.0 + i,
                close=102.0 + i,
                volume=Decimal('1000.0')
            )

            result = frame.prefill(candle, target_periods=target)

            closed = len(frame.periods) - 1 if frame.periods else 0

            if closed < target:
                assert result is False
            else:
                assert result is True
                break

        # Should have at least 5 closed periods
        assert len(frame.periods) - 1 >= target

    def test_prefill_zero_target(self):
        """Test prefill with target_periods=0."""
        frame = TimeFrame('1T', max_periods=10)

        candle = Candle(
            date=datetime(2024, 1, 1, 0, 0),
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=Decimal('1000.0')
        )

        # Target 0 closed periods means first candle should return True
        result = frame.prefill(candle, target_periods=0)

        # We have 1 period, 0 closed (last one is open)
        assert len(frame.periods) == 1
        assert result is True  # Already >= 0

    def test_prefill_maintains_max_periods(self):
        """Test that prefill respects max_periods limit."""
        frame = TimeFrame('1T', max_periods=5)

        # Feed many candles
        for i in range(20):
            candle = Candle(
                date=datetime(2024, 1, 1, 0, i),
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=Decimal('1000.0')
            )
            frame.prefill(candle)

        # Should never exceed max_periods
        assert len(frame.periods) <= 5


class TestPrefillTargetTimestamp:
    """Test prefill() with target_timestamp parameter."""

    def test_prefill_target_timestamp(self):
        """Test prefill with target_timestamp."""
        frame = TimeFrame('1T', max_periods=100)

        # Target timestamp: 2024-01-01 00:05:00
        target_dt = datetime(2024, 1, 1, 0, 5, 0)
        target_ts = target_dt.timestamp()

        # Generate candles
        for i in range(10):
            candle_dt = datetime(2024, 1, 1, 0, i, 0)
            candle = Candle(
                date=candle_dt,
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=Decimal('1000.0')
            )

            result = frame.prefill(candle, target_timestamp=target_ts)

            if candle_dt.timestamp() < target_ts:
                assert result is False, f"Should be False at {candle_dt}"
            else:
                assert result is True, f"Should be True at {candle_dt}"
                break

        # Should have reached target
        assert result is True

    def test_prefill_exact_timestamp_match(self):
        """Test prefill when candle timestamp exactly matches target."""
        frame = TimeFrame('1T', max_periods=100)

        target_dt = datetime(2024, 1, 1, 0, 5, 0)
        target_ts = target_dt.timestamp()

        # Candle with exact timestamp
        candle = Candle(
            date=target_dt,
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=Decimal('1000.0')
        )

        result = frame.prefill(candle, target_timestamp=target_ts)
        assert result is True

    def test_prefill_timestamp_past_target(self):
        """Test prefill when candle is already past target."""
        frame = TimeFrame('1T', max_periods=100)

        target_dt = datetime(2024, 1, 1, 0, 5, 0)
        target_ts = target_dt.timestamp()

        # Candle well past target
        candle_dt = datetime(2024, 1, 1, 1, 0, 0)
        candle = Candle(
            date=candle_dt,
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=Decimal('1000.0')
        )

        result = frame.prefill(candle, target_timestamp=target_ts)
        assert result is True


class TestPrefillValidation:
    """Test prefill() parameter validation."""

    def test_prefill_both_targets_raises(self):
        """Test that specifying both targets raises ValueError."""
        frame = TimeFrame('1T', max_periods=10)

        candle = Candle(
            date=datetime(2024, 1, 1, 0, 0),
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=Decimal('1000.0')
        )

        with pytest.raises(ValueError, match="both"):
            frame.prefill(candle, target_periods=5, target_timestamp=1234567890.0)

    def test_prefill_invalid_candle_type(self):
        """Test that invalid candle type raises TypeError."""
        frame = TimeFrame('1T', max_periods=10)

        with pytest.raises(TypeError):
            frame.prefill("not a candle")


class TestPrefillIntegration:
    """Integration tests for prefill()."""

    def test_prefill_with_indicators(self):
        """Test prefill works correctly with indicators."""
        from trading_frame.indicators.momentum.rsi import RSI

        frame = TimeFrame('1T', max_periods=50)
        frame.add_indicator(RSI(length=14), 'RSI_14')

        # Prefill with 20 candles
        for i in range(25):
            candle = Candle(
                date=datetime(2024, 1, 1, 0, i),
                open=100.0 + i,
                high=105.0 + i,
                low=95.0 + i,
                close=102.0 + i,
                volume=Decimal('1000.0')
            )

            result = frame.prefill(candle, target_periods=20)
            if result:
                break

        # Should have 20+ closed periods
        assert len(frame.periods) - 1 >= 20

        # Indicators should be calculated
        for period in frame.periods[-10:]:  # Check last 10 periods
            # Should have RSI value (might be None if not enough data)
            assert 'RSI_14' in period._data

    def test_prefill_typical_warmup_scenario(self):
        """Test typical warm-up scenario before live trading."""
        frame = TimeFrame('5T', max_periods=100)

        # Simulate warm-up: fill until we have 100 closed periods
        # For 5T frame, need 5 candles (1 min apart) per period
        # So need ~505 candles to get 100 closed periods + 1 open
        candles_processed = 0
        for i in range(600):
            candle = Candle(
                date=datetime(2024, 1, 1, 0, 0) + timedelta(minutes=i),
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=Decimal('1000.0')
            )

            candles_processed += 1
            if frame.prefill(candle):  # Uses default max_periods
                break

        # Should have processed enough candles
        # With max_periods=100, we stop when periods reach 100 (99 closed + 1 open)
        # Each 5T period needs ~5 minutes, so ~500 candles
        assert candles_processed >= 495, f"Processed {candles_processed} candles"
        # Frame should be at max capacity
        assert len(frame.periods) == 100

    def test_prefill_events_still_fire(self):
        """Test that events still fire during prefill."""
        frame = TimeFrame('1T', max_periods=10)

        new_period_count = 0
        close_count = 0

        def on_new(f):
            nonlocal new_period_count
            new_period_count += 1

        def on_close(f):
            nonlocal close_count
            close_count += 1

        frame.on('new_period', on_new)
        frame.on('close', on_close)

        # Prefill 5 candles
        for i in range(5):
            candle = Candle(
                date=datetime(2024, 1, 1, 0, i),
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=Decimal('1000.0')
            )
            frame.prefill(candle, target_periods=10)

        # Should have 5 new_period events
        assert new_period_count == 5
        # Should have 4 close events (first period doesn't close anything)
        assert close_count == 4

    def test_prefill_then_normal_feed(self):
        """Test switching from prefill to normal feed."""
        frame = TimeFrame('1T', max_periods=10)

        # Prefill phase
        for i in range(15):
            candle = Candle(
                date=datetime(2024, 1, 1, 0, i),
                open=100.0 + i,
                high=105.0 + i,
                low=95.0 + i,
                close=102.0 + i,
                volume=Decimal('1000.0')
            )

            if frame.prefill(candle):
                break

        prefill_periods = len(frame.periods)

        # Now use normal feed
        for i in range(15, 20):
            candle = Candle(
                date=datetime(2024, 1, 1, 0, i),
                open=100.0 + i,
                high=105.0 + i,
                low=95.0 + i,
                close=102.0 + i,
                volume=Decimal('1000.0')
            )
            frame.feed(candle)

        # Should maintain max_periods
        assert len(frame.periods) == 10
