"""TimeFrame implementation for time-based period aggregation."""

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from .frame import Frame
from .candle import Candle


class TimeFrame(Frame):
    """
    Time-based Frame that creates periods based on time intervals.

    Supports: seconds, minutes, hours, days, weeks, months, and years.
    """

    ACCEPTED_RANGES = {
        'S': 'second',
        'T': 'minute',
        'H': 'hour',
        'D': 'day',
        'W': 'week',
        'M': 'month',
        'Y': 'year'
    }

    def __init__(
        self,
        periods_length: str = '5T',
        max_periods: int = 250
    ) -> None:
        """
        Initialize a TimeFrame.

        Parameters:
            periods_length: Time interval (e.g., '5T' for 5 minutes, '1H' for 1 hour,
                          '1W' for 1 week, '1M' for 1 month, '1Y' for 1 year)
            max_periods: Maximum number of periods to keep in memory

        Raises:
            ValueError: If periods_length format is invalid
        """
        super().__init__(max_periods)

        if len(periods_length) < 2:
            raise ValueError(
                f"Invalid periods_length '{periods_length}'. "
                f"Expected format: '<number><range>' (e.g., '5T', '1H')"
            )

        self.alias = periods_length[-1]

        if self.alias not in self.ACCEPTED_RANGES:
            raise ValueError(
                f"Invalid time range '{self.alias}'. "
                f"Accepted ranges: {list(self.ACCEPTED_RANGES.keys())}"
            )

        try:
            self.length = int(periods_length[:-1])
        except ValueError:
            raise ValueError(
                f"Invalid periods_length '{periods_length}'. "
                f"Number part must be an integer."
            )

        if self.length < 1:
            raise ValueError("Period length must be at least 1")

    def is_new_period(self, candle: Candle) -> bool:
        """
        Determine if candle should start a new period.

        Parameters:
            candle: Candle to evaluate

        Returns:
            True if candle's date is after the current period's close date
        """
        if not self.periods:
            return True

        return candle.date > self.periods[-1].close_date

    def define_close_date(self, candle: Candle) -> datetime:
        """
        Calculate the closing datetime for a period.

        Parameters:
            candle: Candle that triggered period creation

        Returns:
            Closing datetime for the period
        """
        open_date = candle.date

        # Calculate close date based on time range
        if self.alias == 'S':
            # Seconds
            open_date = open_date.replace(microsecond=0)
            zeroing = open_date.second % self.length
            close_date = (open_date - timedelta(seconds=zeroing)) + \
                        timedelta(seconds=self.length, microseconds=-1)

        elif self.alias == 'T':
            # Minutes
            open_date = open_date.replace(second=0, microsecond=0)
            zeroing = open_date.minute % self.length
            close_date = (open_date - timedelta(minutes=zeroing)) + \
                        timedelta(minutes=self.length, microseconds=-1)

        elif self.alias == 'H':
            # Hours
            open_date = open_date.replace(minute=0, second=0, microsecond=0)
            zeroing = open_date.hour % self.length
            close_date = (open_date - timedelta(hours=zeroing)) + \
                        timedelta(hours=self.length, microseconds=-1)

        elif self.alias == 'D':
            # Days
            open_date = open_date.replace(hour=0, minute=0, second=0, microsecond=0)
            zeroing = open_date.day % self.length
            close_date = (open_date - timedelta(days=zeroing)) + \
                        timedelta(days=self.length, microseconds=-1)

        elif self.alias == 'W':
            # Weeks (Monday = 0, Sunday = 6)
            open_date = open_date.replace(hour=0, minute=0, second=0, microsecond=0)
            # Go to Monday of current week
            days_since_monday = open_date.weekday()
            monday = open_date - timedelta(days=days_since_monday)
            # Calculate which week number we're in
            week_num = (monday - datetime(monday.year, 1, 1)).days // 7
            zeroing = week_num % self.length
            aligned_monday = monday - timedelta(weeks=zeroing)
            close_date = aligned_monday + timedelta(weeks=self.length, microseconds=-1)

        elif self.alias == 'M':
            # Months
            open_date = open_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            zeroing = (open_date.month - 1) % self.length
            aligned_month = open_date - relativedelta(months=zeroing)
            close_date = aligned_month + relativedelta(months=self.length, microseconds=-1)

        elif self.alias == 'Y':
            # Years
            open_date = open_date.replace(
                month=1, day=1, hour=0, minute=0, second=0, microsecond=0
            )
            zeroing = open_date.year % self.length
            aligned_year = open_date.replace(year=open_date.year - zeroing)
            close_date = aligned_year + relativedelta(years=self.length, microseconds=-1)

        return close_date
