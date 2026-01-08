"""Unit tests for prefill() method."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from trading_frame import Candle, TimeFrame, InsufficientDataError


class TestPrefillDefault:
    """Test prefill() with default behavior (fill to max_periods)."""

    def test_prefill_default_fills_to_capacity(self):
        """Test prefill fills to max_periods by default."""
        frame = TimeFrame('1T', max_periods=10)

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

        # Should complete when frame is full
        assert result is True
        assert len(frame.periods) == 10

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


class TestPrefillWithTimestamp:
    """Test prefill() with target_timestamp."""

    def test_prefill_timestamp_relaxed_mode(self):
        """Test prefill with timestamp and require_full=False."""
        frame = TimeFrame('1T', max_periods=100)

        # Target timestamp at minute 10
        target_dt = datetime(2024, 1, 1, 0, 10, 0)
        target_ts = target_dt.timestamp()

        result = False
        for i in range(15):
            candle = Candle(
                date=datetime(2024, 1, 1, 0, i),
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=Decimal('1000.0')
            )

            result = frame.prefill(candle, target_timestamp=target_ts, require_full=False)

            if result:
                break

        # Should stop at timestamp (even if not full)
        assert result is True
        assert len(frame.periods) < 100  # Not full

    def test_prefill_timestamp_strict_mode_sufficient_data(self):
        """Test prefill with timestamp and require_full=True when data is sufficient."""
        frame = TimeFrame('5T', max_periods=20)

        # Target: 20 periods = 100 minutes
        # Provide 120 minutes of data
        target_dt = datetime(2024, 1, 1, 2, 0, 0)  # 120 minutes from midnight
        target_ts = target_dt.timestamp()

        result = False
        for i in range(150):
            candle = Candle(
                date=datetime(2024, 1, 1, 0, 0) + timedelta(minutes=i),
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=Decimal('1000.0')
            )

            result = frame.prefill(candle, target_timestamp=target_ts, require_full=True)

            if result:
                break

        # Should complete successfully
        assert result is True
        assert len(frame.periods) == 20  # Full

    def test_prefill_timestamp_strict_mode_insufficient_data_raises(self):
        """Test prefill raises when insufficient data at timestamp."""
        frame = TimeFrame('1T', max_periods=100)

        # Target timestamp at minute 10 (but need 100 periods)
        target_dt = datetime(2024, 1, 1, 0, 10, 0)
        target_ts = target_dt.timestamp()

        # Only 15 candles - will reach timestamp before filling frame
        with pytest.raises(InsufficientDataError, match="only have.*periods"):
            for i in range(15):
                candle = Candle(
                    date=datetime(2024, 1, 1, 0, i),
                    open=100.0,
                    high=105.0,
                    low=95.0,
                    close=102.0,
                    volume=Decimal('1000.0')
                )

                frame.prefill(candle, target_timestamp=target_ts, require_full=True)

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

        # Relaxed mode - should stop even if not full
        result = frame.prefill(candle, target_timestamp=target_ts, require_full=False)
        assert result is True


class TestPrefillValidation:
    """Test prefill() parameter validation."""

    def test_prefill_invalid_candle_type(self):
        """Test that invalid candle type raises TypeError."""
        frame = TimeFrame('1T', max_periods=10)

        with pytest.raises(TypeError):
            frame.prefill("not a candle")


class TestPrefillIntegration:
    """Integration tests for prefill()."""

    def test_prefill_typical_warmup_scenario(self):
        """Test typical warm-up scenario before live trading."""
        frame = TimeFrame('5T', max_periods=100)

        # Warm-up: fill frame to capacity
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
            if frame.prefill(candle):
                break

        # Should have processed enough candles
        assert candles_processed >= 495
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
            frame.prefill(candle)

        # Should have 5 new_period events
        assert new_period_count == 5
        # Should have 4 close events
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

        # Should be full
        assert len(frame.periods) == 10

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

    def test_prefill_production_scenario(self):
        """Test recommended production warm-up pattern."""
        frame = TimeFrame('5T', max_periods=100)

        # Production: fill until target time with validation
        target_dt = datetime(2024, 1, 1, 12, 0, 0)
        target_ts = target_dt.timestamp()

        # Generate sufficient data (12 hours = 720 minutes)
        for i in range(750):
            candle = Candle(
                date=datetime(2024, 1, 1, 0, 0) + timedelta(minutes=i),
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=Decimal('1000.0')
            )

            try:
                result = frame.prefill(candle, target_timestamp=target_ts, require_full=True)
                if result:
                    break
            except InsufficientDataError as e:
                pytest.fail(f"Should have enough data: {e}")

        # Should be full at target time
        assert len(frame.periods) == 100
        # Last candle should be >= target time
        assert frame.periods[-1].open_date.timestamp() <= target_ts
