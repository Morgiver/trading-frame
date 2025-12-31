"""Base Frame class for organizing trading data into periods."""

from typing import Callable, List
from .candle import Candle
from .period import Period


class Frame:
    """
    Base class for organizing trading data into aggregated periods.

    A Frame manages a collection of periods and provides an event system
    for reacting to period creation, updates, and closures.
    """

    def __init__(self, max_periods: int = 250) -> None:
        """
        Initialize a Frame.

        Parameters:
            max_periods: Maximum number of periods to keep in memory
        """
        if max_periods < 1:
            raise ValueError("max_periods must be at least 1")

        self.max_periods = max_periods
        self.periods: List[Period] = []
        self._event_channels = {
            'new_period': [],
            'update': [],
            'close': []
        }

    def on(self, channel: str, callback: Callable) -> None:
        """
        Register a callback for an event channel.

        Parameters:
            channel: Event channel name ('new_period', 'update', or 'close')
            callback: Function to call when event is emitted

        Raises:
            ValueError: If channel is not valid
        """
        if channel not in self._event_channels:
            raise ValueError(
                f"Invalid channel '{channel}'. "
                f"Valid channels: {list(self._event_channels.keys())}"
            )

        self._event_channels[channel].append(callback)

    def emit(self, channel: str, *args, **kwargs) -> None:
        """
        Emit an event to all registered callbacks.

        Parameters:
            channel: Event channel to emit to
            *args: Positional arguments to pass to callbacks
            **kwargs: Keyword arguments to pass to callbacks
        """
        if channel in self._event_channels:
            for callback in self._event_channels[channel]:
                callback(*args, **kwargs)

    def is_new_period(self, candle: Candle) -> bool:
        """
        Determine if a candle should start a new period.

        This method should be overridden in subclasses to implement
        specific period boundary logic.

        Parameters:
            candle: Candle to evaluate

        Returns:
            True if candle should start a new period
        """
        return True

    def define_close_date(self, candle: Candle):
        """
        Define the closing date for a new period.

        This method should be overridden in subclasses to implement
        specific close date logic.

        Parameters:
            candle: Candle that triggered period creation

        Returns:
            Closing datetime for the period (or None)
        """
        return None

    def create_new_period(self, candle: Candle) -> None:
        """
        Create a new period and add it to the periods list.

        Parameters:
            candle: Candle to initialize the new period with
        """
        close_date = self.define_close_date(candle)
        new_period = Period(self, candle.date, close_date)
        new_period.update(candle)

        self.periods.append(new_period)
        self.emit('new_period', self)

    def update_period(self, candle: Candle) -> None:
        """
        Update the last period with new candle data.

        Parameters:
            candle: Candle to add to the current period
        """
        if not self.periods:
            raise RuntimeError("No periods to update")

        self.periods[-1].update(candle)
        self.emit('update', self)

    def feed(self, candle: Candle) -> None:
        """
        Feed a new candle into the frame.

        Automatically creates new periods or updates existing ones
        based on is_new_period logic.

        Parameters:
            candle: Candle to process

        Raises:
            TypeError: If candle is not a Candle instance
        """
        if not isinstance(candle, Candle):
            raise TypeError(f"Expected Candle instance, got {type(candle)}")

        # Check if we need to create a new period
        if not self.periods or self.is_new_period(candle):
            # If we have periods, emit close event for the previous one
            if self.periods:
                self.emit('close', self)

            self.create_new_period(candle)
        else:
            self.update_period(candle)

        # Maintain max_periods limit
        if len(self.periods) > self.max_periods:
            self.periods.pop(0)

    def to_numpy(self):
        """
        Convert all periods to numpy array.

        Returns:
            numpy.ndarray: 2D array where each row is [open, high, low, close, volume]
        """
        import numpy as np
        if not self.periods:
            return np.array([], dtype=np.float64).reshape(0, 5)
        
        return np.array([period.to_numpy() for period in self.periods], dtype=np.float64)

    def to_pandas(self):
        """
        Convert all periods to pandas DataFrame.

        Returns:
            pandas.DataFrame: DataFrame with columns [open_date, close_date, open_price, 
                            high_price, low_price, close_price, volume]
        """
        import pandas as pd
        if not self.periods:
            return pd.DataFrame(columns=[
                'open_date', 'close_date', 'open_price', 
                'high_price', 'low_price', 'close_price', 'volume'
            ])
        
        data = [period.to_dict() for period in self.periods]
        df = pd.DataFrame(data)
        
        # Convert volume from Decimal to float for pandas compatibility
        df['volume'] = df['volume'].astype(float)
        
        return df
