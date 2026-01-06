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

        # Inform indicator of its output column names
        indicator.set_output_columns(column_name)

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

    def to_normalize(self):
        """
        Convert all periods to normalized numpy array.

        Normalization strategies:
        - OHLC + Price-based indicators: Unified Min-Max across OHLC and all
          price-based indicators (SMA, Bollinger Bands, etc.)
          This ensures consistent scaling when indicators can exceed price range.
        - Volume: Independent Min-Max normalization across all volumes
        - Indicators: Each indicator defines its own normalization strategy
          * RSI: Fixed range 0-100 → [0, 1]
          * SMA (price-based): Shares unified price min-max range
          * MACD: Min-Max on its own values
          * Bollinger Bands: Shares unified price min-max range

        Returns:
            numpy.ndarray: 2D array with normalized values in range [0, 1]
                          (price-based indicators may slightly exceed [0,1] if
                          their values extend beyond OHLC before unification)
        """
        import numpy as np

        if not self.periods:
            n_cols = 5 + len(self._get_all_indicator_columns())
            return np.array([], dtype=np.float64).reshape(0, n_cols)

        # Step 1: Extract all OHLC and Volume values
        price_values = []  # Will include OHLC + price-based indicators
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

        # Step 2: Add price-based indicator values to price range calculation
        # This ensures Bollinger Bands, SMA, etc. are included in min/max
        for col_name in sorted(self._get_all_indicator_columns()):
            # Find indicator
            indicator = None
            for key, ind in self.indicators.items():
                if isinstance(key, str) and key == col_name:
                    indicator = ind
                    break
                elif isinstance(key, tuple) and col_name in key:
                    indicator = ind
                    break

            # If price-based, include in price range calculation
            if indicator and indicator.get_normalization_type() == 'price':
                for period in self.periods:
                    val = period._data.get(col_name)
                    if val is not None:
                        price_values.append(float(val))

        # Step 3: Calculate Price and Volume ranges
        price_array = np.array(price_values)
        volume_array = np.array(volume_values)

        price_min = float(np.min(price_array)) if len(price_array) > 0 else 0.0
        price_max = float(np.max(price_array)) if len(price_array) > 0 else 1.0
        volume_min = float(np.min(volume_array)) if len(volume_array) > 0 else 0.0
        volume_max = float(np.max(volume_array)) if len(volume_array) > 0 else 1.0

        # Avoid division by zero
        price_range = price_max - price_min if price_max != price_min else 1.0
        volume_range = volume_max - volume_min if volume_max != volume_min else 1.0

        # Step 4: Collect all indicator values for min-max indicators
        indicator_arrays = {}
        for col_name in sorted(self._get_all_indicator_columns()):
            values = []
            for period in self.periods:
                val = period._data.get(col_name)
                if val is not None:
                    values.append(float(val))
                else:
                    values.append(np.nan)
            indicator_arrays[col_name] = np.array(values, dtype=np.float64)

        # Step 5: Build normalized data
        data = []
        for period in self.periods:
            # Normalize OHLC
            row = [
                (float(period.open_price) - price_min) / price_range if period.open_price is not None else np.nan,
                (float(period.high_price) - price_min) / price_range if period.high_price is not None else np.nan,
                (float(period.low_price) - price_min) / price_range if period.low_price is not None else np.nan,
                (float(period.close_price) - price_min) / price_range if period.close_price is not None else np.nan,
                # Normalize Volume
                (float(period.volume) - volume_min) / volume_range
            ]

            # Normalize indicators
            for col_name in sorted(self._get_all_indicator_columns()):
                value = period._data.get(col_name)

                # Find the indicator instance
                indicator = None
                for key, ind in self.indicators.items():
                    if isinstance(key, str) and key == col_name:
                        indicator = ind
                        break
                    elif isinstance(key, tuple) and col_name in key:
                        indicator = ind
                        break

                if indicator is None:
                    # Shouldn't happen, but fallback to raw value
                    row.append(float(value) if value is not None else np.nan)
                    continue

                # Normalize based on indicator's strategy
                normalized = indicator.normalize(
                    values=value,
                    all_values=indicator_arrays[col_name],
                    price_range=(price_min, price_max)
                )

                row.append(float(normalized) if normalized is not None else np.nan)

            data.append(row)

        return np.array(data, dtype=np.float64)
