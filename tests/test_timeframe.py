"""Unit tests for TimeFrame class."""

import pytest
from datetime import datetime, timedelta
from trading_frame import Candle, TimeFrame


class TestTimeFrameInitialization:
    """Test TimeFrame initialization."""

    def test_init_default(self):
        """Test default initialization (5 minutes)."""
        tf = TimeFrame()

        assert tf.length == 5
        assert tf.alias == 'T'

    def test_init_custom_periods(self):
        """Test initialization with custom period length."""
        tf = TimeFrame(periods_length='15T')

        assert tf.length == 15
        assert tf.alias == 'T'

    def test_init_seconds(self):
        """Test initialization with seconds."""
        tf = TimeFrame(periods_length='30S')

        assert tf.length == 30
        assert tf.alias == 'S'

    def test_init_hours(self):
        """Test initialization with hours."""
        tf = TimeFrame(periods_length='4H')

        assert tf.length == 4
        assert tf.alias == 'H'

    def test_init_days(self):
        """Test initialization with days."""
        tf = TimeFrame(periods_length='1D')

        assert tf.length == 1
        assert tf.alias == 'D'

    def test_init_weeks(self):
        """Test initialization with weeks."""
        tf = TimeFrame(periods_length='2W')

        assert tf.length == 2
        assert tf.alias == 'W'

    def test_init_months(self):
        """Test initialization with months."""
        tf = TimeFrame(periods_length='3M')

        assert tf.length == 3
        assert tf.alias == 'M'

    def test_init_years(self):
        """Test initialization with years."""
        tf = TimeFrame(periods_length='1Y')

        assert tf.length == 1
        assert tf.alias == 'Y'

    def test_init_invalid_format(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid periods_length"):
            TimeFrame(periods_length='X')

    def test_init_invalid_range(self):
        """Test that invalid time range raises ValueError."""
        with pytest.raises(ValueError, match="Invalid time range"):
            TimeFrame(periods_length='5X')

    def test_init_invalid_number(self):
        """Test that non-integer number raises ValueError."""
        with pytest.raises(ValueError, match="Number part must be an integer"):
            TimeFrame(periods_length='abc5T')

    def test_init_zero_length(self):
        """Test that zero length raises ValueError."""
        with pytest.raises(ValueError, match="Period length must be at least 1"):
            TimeFrame(periods_length='0T')


class TestTimeFrameDefineCloseDate:
    """Test TimeFrame close date calculation."""

    def test_define_close_date_seconds(self):
        """Test close date calculation for seconds."""
        tf = TimeFrame(periods_length='5S')
        candle = Candle(
            datetime(2025, 1, 1, 0, 0, 12, 500000),
            100.0, 110.0, 90.0, 105.0, 1000.0
        )

        close_date = tf.define_close_date(candle)

        # Should align to 5-second boundary: 00:00:10 -> 00:00:14.999999
        assert close_date == datetime(2025, 1, 1, 0, 0, 14, 999999)

    def test_define_close_date_minutes(self):
        """Test close date calculation for minutes."""
        tf = TimeFrame(periods_length='5T')
        candle = Candle(
            datetime(2025, 1, 1, 0, 7, 30),
            100.0, 110.0, 90.0, 105.0, 1000.0
        )

        close_date = tf.define_close_date(candle)

        # Should align to 5-minute boundary: 00:05:00 -> 00:09:59.999999
        assert close_date == datetime(2025, 1, 1, 0, 9, 59, 999999)

    def test_define_close_date_hours(self):
        """Test close date calculation for hours."""
        tf = TimeFrame(periods_length='4H')
        candle = Candle(
            datetime(2025, 1, 1, 5, 30, 0),
            100.0, 110.0, 90.0, 105.0, 1000.0
        )

        close_date = tf.define_close_date(candle)

        # Should align to 4-hour boundary: 04:00:00 -> 07:59:59.999999
        assert close_date == datetime(2025, 1, 1, 7, 59, 59, 999999)

    def test_define_close_date_days(self):
        """Test close date calculation for days."""
        tf = TimeFrame(periods_length='1D')
        candle = Candle(
            datetime(2025, 1, 1, 12, 0, 0),
            100.0, 110.0, 90.0, 105.0, 1000.0
        )

        close_date = tf.define_close_date(candle)

        # Should align to 1-day boundary: 00:00:00 -> 23:59:59.999999
        assert close_date.date() == datetime(2025, 1, 1).date()
        assert close_date.hour == 23
        assert close_date.minute == 59

    def test_define_close_date_weeks(self):
        """Test close date calculation for weeks."""
        tf = TimeFrame(periods_length='1W')
        # Wednesday, Jan 1, 2025
        candle = Candle(
            datetime(2025, 1, 1, 12, 0, 0),
            100.0, 110.0, 90.0, 105.0, 1000.0
        )

        close_date = tf.define_close_date(candle)

        # Should align to week starting Monday Dec 30, 2024
        # and end on Sunday Jan 5, 2025 23:59:59.999999
        assert close_date.month == 1
        assert close_date.day == 5  # Sunday

    def test_define_close_date_months(self):
        """Test close date calculation for months."""
        tf = TimeFrame(periods_length='1M')
        candle = Candle(
            datetime(2025, 1, 15, 12, 0, 0),
            100.0, 110.0, 90.0, 105.0, 1000.0
        )

        close_date = tf.define_close_date(candle)

        # Should align to 1-month boundary: Jan 1 -> Jan 31 23:59:59.999999
        assert close_date.year == 2025
        assert close_date.month == 1
        assert close_date.day == 31

    def test_define_close_date_years(self):
        """Test close date calculation for years."""
        tf = TimeFrame(periods_length='1Y')
        candle = Candle(
            datetime(2025, 6, 15, 12, 0, 0),
            100.0, 110.0, 90.0, 105.0, 1000.0
        )

        close_date = tf.define_close_date(candle)

        # Should align to 1-year boundary: 2025-01-01 -> 2025-12-31 23:59:59.999999
        assert close_date.year == 2025
        assert close_date.month == 12
        assert close_date.day == 31


class TestTimeFrameIsNewPeriod:
    """Test TimeFrame new period detection."""

    def test_is_new_period_first_candle(self):
        """Test that first candle always creates new period."""
        tf = TimeFrame(periods_length='5T')
        candle = Candle(
            datetime(2025, 1, 1, 0, 0, 0),
            100.0, 110.0, 90.0, 105.0, 1000.0
        )

        assert tf.is_new_period(candle) is True

    def test_is_new_period_within_same_period(self):
        """Test that candle within period doesn't create new period."""
        tf = TimeFrame(periods_length='5T')

        # First candle at 00:00:00
        candle1 = Candle(
            datetime(2025, 1, 1, 0, 0, 0),
            100.0, 110.0, 90.0, 105.0, 1000.0
        )
        tf.feed(candle1)

        # Second candle at 00:02:00 (within same 5-minute period)
        candle2 = Candle(
            datetime(2025, 1, 1, 0, 2, 0),
            105.0, 115.0, 95.0, 110.0, 500.0
        )

        assert tf.is_new_period(candle2) is False

    def test_is_new_period_crosses_boundary(self):
        """Test that candle crossing boundary creates new period."""
        tf = TimeFrame(periods_length='5T')

        # First candle at 00:02:00
        candle1 = Candle(
            datetime(2025, 1, 1, 0, 2, 0),
            100.0, 110.0, 90.0, 105.0, 1000.0
        )
        tf.feed(candle1)

        # Second candle at 00:05:00 (crosses into next 5-minute period)
        candle2 = Candle(
            datetime(2025, 1, 1, 0, 5, 0),
            105.0, 115.0, 95.0, 110.0, 500.0
        )

        assert tf.is_new_period(candle2) is True


class TestTimeFrameFeed:
    """Test TimeFrame feed functionality."""

    def test_feed_creates_aligned_periods(self):
        """Test that periods are properly aligned to time boundaries."""
        tf = TimeFrame(periods_length='5T')

        # Feed candles at various times
        candles = [
            Candle(datetime(2025, 1, 1, 0, 2, 0), 100.0, 110.0, 90.0, 105.0, 1000.0),
            Candle(datetime(2025, 1, 1, 0, 3, 0), 105.0, 115.0, 95.0, 110.0, 500.0),
            Candle(datetime(2025, 1, 1, 0, 7, 0), 110.0, 120.0, 100.0, 115.0, 750.0),
        ]

        for candle in candles:
            tf.feed(candle)

        # Should have 2 periods: [00:00-00:04], [00:05-00:09]
        assert len(tf.periods) == 2

        # First period should aggregate first two candles
        assert tf.periods[0].open_price == 100.0
        assert tf.periods[0].close_price == 110.0
        assert tf.periods[0].volume == 1500.0

        # Second period should have third candle
        assert tf.periods[1].open_price == 110.0
        assert tf.periods[1].volume == 750.0

    def test_feed_multiple_time_ranges(self):
        """Test feeding with different time ranges."""
        for range_str, delta in [
            ('30S', timedelta(seconds=30)),
            ('15T', timedelta(minutes=15)),
            ('2H', timedelta(hours=2)),
            ('1D', timedelta(days=1)),
        ]:
            tf = TimeFrame(periods_length=range_str)

            start_date = datetime(2025, 1, 1, 0, 0, 0)
            candle1 = Candle(start_date, 100.0, 110.0, 90.0, 105.0, 1000.0)
            candle2 = Candle(start_date + delta, 105.0, 115.0, 95.0, 110.0, 500.0)

            tf.feed(candle1)
            tf.feed(candle2)

            # Should create 2 periods
            assert len(tf.periods) == 2
