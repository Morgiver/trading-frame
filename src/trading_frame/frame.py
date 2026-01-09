"""Base Frame class for organizing trading data into periods."""

from typing import Callable, List, Dict, Union, Tuple, Optional
from .candle import Candle
from .period import Period


class Frame:
    """
    Base class for organizing trading data into aggregated periods.

    A Frame manages a collection of periods and provides an event system
    for reacting to period creation, updates, and closures.

    Also supports adding technical indicators that automatically calculate
    values for all periods.
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
            channel: Event channel name
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

        # Update OHLCV
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

    def prefill(self, candle: Candle, target_timestamp: float = None, require_full: bool = True) -> bool:
        """
        Feed candle and check if prefill target is reached.

        Use this method during warm-up phase to fill the frame before live trading.
        Always aims to fill the frame to max_periods capacity.

        Parameters:
            candle: Candle to process
            target_timestamp: Stop when candle timestamp >= this value (optional)
            require_full: If True with timestamp, raises error if max_periods not reached at timestamp
                         If False, just stops at timestamp regardless of period count (default: True)

        Returns:
            True if target is reached, False otherwise

        Raises:
            InsufficientDataError: If require_full=True and frame not full at target timestamp
            TypeError: If candle is not a Candle instance

        Modes:
            1. Default (fill to capacity):
               prefill(candle)
               → Stop when max_periods reached

            2. Fill until timestamp (relaxed):
               prefill(candle, target_timestamp=ts, require_full=False)
               → Stop when timestamp reached (any number of periods)

            3. Fill until timestamp (validated - RECOMMENDED):
               prefill(candle, target_timestamp=ts, require_full=True)
               → Stop when timestamp reached, RAISE if not at max_periods

        Example:
            # Fill until max_periods (default)
            for candle in historical_data:
                if frame.prefill(candle):
                    break  # Frame is full

            # Fill until timestamp with validation (RECOMMENDED)
            target_ts = datetime(2024, 1, 1, 12, 0).timestamp()
            try:
                for candle in historical_data:
                    if frame.prefill(candle, target_timestamp=target_ts):
                        break  # Frame is full at target date
            except InsufficientDataError as e:
                print(f"Not enough historical data: {e}")

            # Fill until timestamp without validation (relaxed)
            for candle in historical_data:
                if frame.prefill(candle, target_timestamp=target_ts, require_full=False):
                    break  # Reached target date (may not be full)
        """
        # Feed the candle
        self.feed(candle)

        # Count closed periods
        closed_periods = len(self.periods) - 1 if self.periods else 0
        is_at_capacity = len(self.periods) >= self.max_periods

        # Mode 1: No timestamp - fill until max_periods
        if target_timestamp is None:
            return is_at_capacity

        # Mode 2 & 3: With timestamp
        candle_ts = candle.date.timestamp()

        # If we haven't reached timestamp yet, keep going
        if candle_ts < target_timestamp:
            return False

        # Reached timestamp
        if require_full and not is_at_capacity:
            # Strict mode: MUST be at max_periods at timestamp
            from .exceptions import InsufficientDataError
            raise InsufficientDataError(
                f"Reached target timestamp but only have {len(self.periods)} periods "
                f"(max_periods: {self.max_periods}). Need more historical data before target date."
            )

        # Target timestamp reached (and either full or require_full=False)
        return True

    def to_numpy(self):
        """
        Convert all periods to numpy array.

        Returns:
            numpy.ndarray: 2D array with OHLCV data
        """
        import numpy as np
        if not self.periods:
            return np.array([], dtype=np.float64).reshape(0, 5)

        data = []
        for period in self.periods:
            row = [
                float(period.open_price) if period.open_price is not None else np.nan,
                float(period.high_price) if period.high_price is not None else np.nan,
                float(period.low_price) if period.low_price is not None else np.nan,
                float(period.close_price) if period.close_price is not None else np.nan,
                float(period.volume)
            ]
            data.append(row)

        return np.array(data, dtype=np.float64)

    def to_pandas(self):
        """
        Convert all periods to pandas DataFrame.

        Returns:
            pandas.DataFrame: DataFrame with OHLCV columns and DateTimeIndex
        """
        import pandas as pd
        if not self.periods:
            base_cols = ['open_date', 'close_date', 'open_price',
                         'high_price', 'low_price', 'close_price', 'volume']
            return pd.DataFrame(columns=base_cols)

        data = [period.to_dict() for period in self.periods]
        df = pd.DataFrame(data)

        # Convert volume from Decimal to float
        df['volume'] = df['volume'].astype(float)

        # Set open_date as DateTimeIndex
        df.set_index('open_date', inplace=True)

        return df

    def to_normalize(self):
        """
        Convert all periods to normalized numpy array.

        Normalization strategies:
        - OHLC: Unified Min-Max normalization across all price values
        - Volume: Independent Min-Max normalization across all volumes

        Returns:
            numpy.ndarray: 2D array with normalized OHLCV values in range [0, 1]
        """
        import numpy as np

        if not self.periods:
            return np.array([], dtype=np.float64).reshape(0, 5)

        # Step 1: Extract all OHLC and Volume values
        price_values = []
        volume_values = []

        for period in self.periods:
            if period.open_price is not None:
                price_values.append(float(period.open_price))
            if period.high_price is not None:
                price_values.append(float(period.high_price))
            if period.low_price is not None:
                price_values.append(float(period.low_price))
            if period.close_price is not None:
                price_values.append(float(period.close_price))
            volume_values.append(float(period.volume))

        # Step 2: Calculate Price and Volume ranges
        price_array = np.array(price_values)
        volume_array = np.array(volume_values)

        price_min = float(np.min(price_array)) if len(price_array) > 0 else 0.0
        price_max = float(np.max(price_array)) if len(price_array) > 0 else 1.0
        volume_min = float(np.min(volume_array)) if len(volume_array) > 0 else 0.0
        volume_max = float(np.max(volume_array)) if len(volume_array) > 0 else 1.0

        # Avoid division by zero
        price_range = price_max - price_min if price_max != price_min else 1.0
        volume_range = volume_max - volume_min if volume_max != volume_min else 1.0

        # Step 3: Build normalized data
        data = []
        for period in self.periods:
            row = [
                (float(period.open_price) - price_min) / price_range if period.open_price is not None else np.nan,
                (float(period.high_price) - price_min) / price_range if period.high_price is not None else np.nan,
                (float(period.low_price) - price_min) / price_range if period.low_price is not None else np.nan,
                (float(period.close_price) - price_min) / price_range if period.close_price is not None else np.nan,
                (float(period.volume) - volume_min) / volume_range
            ]
            data.append(row)

        return np.array(data, dtype=np.float64)
