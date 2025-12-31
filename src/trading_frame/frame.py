"""Base Frame class for organizing trading data into periods."""

from typing import Callable, List, Dict, Union
from .candle import Candle
from .period import Period
from .indicators.base import Indicator


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
            'close': [],
            'indicator_added': [],
            'indicator_removed': []
        }
        self.indicators: Dict[Union[str, tuple], Indicator] = {}

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

    def add_indicator(
        self,
        indicator: Indicator,
        column_name: Union[str, List[str]]
    ) -> None:
        """
        Add an indicator to the frame.

        This will:
        1. Register the indicator
        2. Add the column(s) to all existing periods
        3. Calculate values for all existing periods

        Parameters:
            indicator: Indicator instance to add
            column_name: Single name (str) or list of names for multi-column

        Raises:
            ValueError: If dependencies not met or column already exists
        """
        # Validate dependencies
        dependencies = indicator.get_dependencies()
        for dep in dependencies:
            # Check if dependency exists in any period's _data
            if self.periods and dep not in self.periods[0]._data:
                raise ValueError(
                    f"Dependency '{dep}' not found. "
                    f"Add dependent indicators first or use valid column name."
                )

        # Validate column_name format
        is_multi = isinstance(column_name, list)
        num_outputs = indicator.get_num_outputs()

        if is_multi:
            if len(column_name) != num_outputs:
                raise ValueError(
                    f"Indicator produces {num_outputs} outputs but "
                    f"{len(column_name)} column names provided"
                )
            columns = column_name
            registry_key = tuple(column_name)
        else:
            if num_outputs > 1:
                raise ValueError(
                    f"Indicator produces {num_outputs} outputs. "
                    f"Provide a list of {num_outputs} column names."
                )
            columns = [column_name]
            registry_key = column_name

        # Check for duplicate column names
        for col in columns:
            for existing_key in self.indicators.keys():
                existing_cols = [existing_key] if isinstance(existing_key, str) else list(existing_key)
                if col in existing_cols:
                    raise ValueError(f"Column '{col}' already exists")

        # Register indicator
        self.indicators[registry_key] = indicator

        # Add columns to all existing periods and calculate
        for i, period in enumerate(self.periods):
            # Initialize columns
            for col in columns:
                period._data[col] = None

            # Calculate value(s)
            result = indicator.calculate(self.periods, i)

            # Assign to columns
            if is_multi:
                if result is not None and len(result) == len(columns):
                    for col, val in zip(columns, result):
                        period._data[col] = val
            else:
                period._data[columns[0]] = result

        # Emit event
        self.emit('indicator_added', self, columns)

    def remove_indicator(self, column_name: Union[str, List[str]]) -> None:
        """
        Remove an indicator and its data from all periods.

        Parameters:
            column_name: Single name or list of names to remove

        Raises:
            ValueError: If indicator not found
        """
        # Determine registry key
        if isinstance(column_name, list):
            registry_key = tuple(column_name)
            columns = column_name
        else:
            registry_key = column_name
            columns = [column_name]

        if registry_key not in self.indicators:
            raise ValueError(f"Indicator '{column_name}' not found")

        # Remove from registry
        del self.indicators[registry_key]

        # Remove columns from all periods
        for period in self.periods:
            for col in columns:
                if col in period._data:
                    del period._data[col]

        # Emit event
        self.emit('indicator_removed', self, columns)

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

        # Initialize all indicator columns in new period
        for registry_key in self.indicators.keys():
            columns = [registry_key] if isinstance(registry_key, str) else list(registry_key)
            for col in columns:
                new_period._data[col] = None

        new_period.update(candle)
        self.periods.append(new_period)

        # Calculate indicators for new period
        self._recalculate_indicators_for_period(len(self.periods) - 1)

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

        # Recalculate all indicators for the updated period
        self._recalculate_indicators_for_period(len(self.periods) - 1)

        self.emit('update', self)

    def _recalculate_indicators_for_period(self, index: int) -> None:
        """
        Recalculate all indicators for a specific period.

        Parameters:
            index: Index of the period to recalculate
        """
        for registry_key, indicator in self.indicators.items():
            columns = [registry_key] if isinstance(registry_key, str) else list(registry_key)

            # Calculate
            result = indicator.calculate(self.periods, index)

            # Assign
            if len(columns) > 1:
                # Multi-column
                if result is not None and len(result) == len(columns):
                    for col, val in zip(columns, result):
                        self.periods[index]._data[col] = val
            else:
                # Single column
                self.periods[index]._data[columns[0]] = result

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
            numpy.ndarray: 2D array with OHLCV + all indicators
        """
        import numpy as np
        if not self.periods:
            # Base columns + indicator columns
            n_cols = 5 + len(self._get_all_indicator_columns())
            return np.array([], dtype=np.float64).reshape(0, n_cols)

        data = []
        for period in self.periods:
            row = [
                float(period.open_price) if period.open_price is not None else np.nan,
                float(period.high_price) if period.high_price is not None else np.nan,
                float(period.low_price) if period.low_price is not None else np.nan,
                float(period.close_price) if period.close_price is not None else np.nan,
                float(period.volume)
            ]

            # Add indicator values
            for col_name in sorted(self._get_all_indicator_columns()):
                value = period._data.get(col_name)
                row.append(float(value) if value is not None else np.nan)

            data.append(row)

        return np.array(data, dtype=np.float64)

    def _get_all_indicator_columns(self) -> List[str]:
        """Get flat list of all indicator column names."""
        columns = []
        for key in self.indicators.keys():
            if isinstance(key, str):
                columns.append(key)
            else:
                columns.extend(list(key))
        return columns

    def to_pandas(self):
        """
        Convert all periods to pandas DataFrame.

        Returns:
            pandas.DataFrame: DataFrame with OHLCV + all indicator columns
        """
        import pandas as pd
        if not self.periods:
            base_cols = ['open_date', 'close_date', 'open_price',
                         'high_price', 'low_price', 'close_price', 'volume']
            all_cols = base_cols + sorted(self._get_all_indicator_columns())
            return pd.DataFrame(columns=all_cols)

        data = [period.to_dict() for period in self.periods]
        df = pd.DataFrame(data)

        # Convert volume from Decimal to float
        df['volume'] = df['volume'].astype(float)

        # Convert indicator values to float (handle None)
        for col_name in self._get_all_indicator_columns():
            if col_name in df.columns:
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

        return df
