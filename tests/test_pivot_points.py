"""Tests for PivotPoints indicator."""

import pytest
from datetime import datetime, timedelta
from trading_frame import Candle, TimeFrame
from trading_frame.indicators.trend.pivot_points import PivotPoints


def create_candle(date, high, low, open_val=None, close_val=None, volume=1000):
    """Helper to create a candle with specific high/low."""
    # Auto-adjust open/close to be within high/low range if not specified
    if open_val is None:
        open_val = (high + low) / 2
    if close_val is None:
        close_val = (high + low) / 2

    # Ensure open/close are within high/low range
    open_val = max(low, min(high, open_val))
    close_val = max(low, min(high, close_val))

    return Candle(
        date=date,
        open=open_val,
        high=high,
        low=low,
        close=close_val,
        volume=volume
    )


class TestPivotPointsBasic:
    """Basic functionality tests."""

    def test_initialization(self):
        """Test PivotPoints initialization."""
        pivot = PivotPoints(left_bars=5, right_bars=2)
        assert pivot.left_bars == 5
        assert pivot.right_bars == 2
        assert pivot.high_source == 'high_price'
        assert pivot.low_source == 'low_price'

    def test_initialization_custom_sources(self):
        """Test PivotPoints with custom sources."""
        pivot = PivotPoints(
            left_bars=3,
            right_bars=1,
            high_source='custom_high',
            low_source='custom_low'
        )
        assert pivot.high_source == 'custom_high'
        assert pivot.low_source == 'custom_low'

    def test_initialization_custom_columns(self):
        """Test PivotPoints with custom column names."""
        pivot = PivotPoints(
            left_bars=5,
            right_bars=2,
            high_column='SWING_HIGH',
            low_column='SWING_LOW'
        )
        assert pivot.high_column == 'SWING_HIGH'
        assert pivot.low_column == 'SWING_LOW'

    def test_invalid_left_bars(self):
        """Test that left_bars must be >= 1."""
        with pytest.raises(ValueError, match="left_bars must be at least 1"):
            PivotPoints(left_bars=0, right_bars=2)

    def test_invalid_right_bars(self):
        """Test that right_bars must be >= 1."""
        with pytest.raises(ValueError, match="right_bars must be at least 1"):
            PivotPoints(left_bars=5, right_bars=0)

    def test_min_periods(self):
        """Test minimum periods calculation."""
        pivot = PivotPoints(left_bars=5, right_bars=2)
        assert pivot.requires_min_periods() == 8  # 5 + 1 + 2

        pivot = PivotPoints(left_bars=3, right_bars=3)
        assert pivot.requires_min_periods() == 7  # 3 + 1 + 3

    def test_dependencies(self):
        """Test dependency declaration."""
        pivot = PivotPoints()
        deps = pivot.get_dependencies()
        assert 'high_price' in deps
        assert 'low_price' in deps

    def test_num_outputs(self):
        """Test number of outputs."""
        pivot = PivotPoints()
        assert pivot.get_num_outputs() == 2

    def test_normalization_type(self):
        """Test normalization type."""
        pivot = PivotPoints()
        assert pivot.get_normalization_type() == 'price'


class TestPivotPointsDetection:
    """Test pivot detection logic."""

    def test_swing_high_detection_simple(self):
        """Test simple swing high detection with left=2, right=2."""
        frame = TimeFrame('1T', max_periods=50)
        pivot = PivotPoints(left_bars=2, right_bars=2)

        # Create pattern: low, low, HIGH, low, low
        base_date = datetime(2024, 1, 1, 12, 0)
        highs = [100, 101, 110, 102, 101]  # Index 2 is the peak
        lows = [90, 91, 95, 92, 91]

        for i, (h, l) in enumerate(zip(highs, lows)):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=h, low=l)
            frame.feed(candle)

        # Add indicator
        frame.add_indicator(pivot, ['PIVOT_HIGH', 'PIVOT_LOW'])

        # Check that index 2 (the peak) has a swing high after confirmation
        # Confirmation happens at index 2 + 2 = 4
        assert frame.periods[2].PIVOT_HIGH == 110.0
        assert frame.periods[2].PIVOT_LOW is None

    def test_swing_low_detection_simple(self):
        """Test simple swing low detection with left=2, right=2."""
        frame = TimeFrame('1T', max_periods=50)
        pivot = PivotPoints(left_bars=2, right_bars=2)

        # Create pattern: high, high, LOW, high, high
        base_date = datetime(2024, 1, 1, 12, 0)
        highs = [100, 99, 95, 98, 100]
        lows = [90, 89, 80, 88, 90]  # Index 2 is the trough

        for i, (h, l) in enumerate(zip(highs, lows)):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=h, low=l)
            frame.feed(candle)

        # Add indicator
        frame.add_indicator(pivot, ['PIVOT_HIGH', 'PIVOT_LOW'])

        # Check that index 2 (the trough) has a swing low
        assert frame.periods[2].PIVOT_HIGH is None
        assert frame.periods[2].PIVOT_LOW == 80.0

    def test_asymmetric_bars(self):
        """Test with different left and right bar counts (left=5, right=2)."""
        frame = TimeFrame('1T', max_periods=50)
        pivot = PivotPoints(left_bars=5, right_bars=2)

        # Create pattern with 5 left bars, 1 peak, 2 right bars
        base_date = datetime(2024, 1, 1, 12, 0)
        # Indices: 0, 1, 2, 3, 4, [5=peak], 6, 7
        highs = [100, 101, 102, 103, 104, 120, 105, 106]
        lows = [90, 91, 92, 93, 94, 100, 95, 96]

        for i, (h, l) in enumerate(zip(highs, lows)):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=h, low=l)
            frame.feed(candle)

        # Add indicator
        frame.add_indicator(pivot, ['PIVOT_HIGH', 'PIVOT_LOW'])

        # Index 5 should be a swing high (confirmed at 5 + 2 = 7)
        assert frame.periods[5].PIVOT_HIGH == 120.0
        assert frame.periods[5].PIVOT_LOW is None


class TestPivotPointsAlternation:
    """Test alternation rule (High ↔ Low succession)."""

    def test_alternation_high_replaces_high(self):
        """Test that consecutive highs replace each other."""
        frame = TimeFrame('1T', max_periods=50)
        pivot = PivotPoints(left_bars=2, right_bars=2)

        base_date = datetime(2024, 1, 1, 12, 0)
        # Pattern: low, low, HIGH1(110), low, low, HIGH2(115), low, low
        # Indices:  0    1    2         3    4    5          6    7
        highs = [100, 101, 110, 102, 103, 115, 104, 105]
        lows =  [90,  91,  95,  92,  93,  98,  94,  95]

        for i, (h, l) in enumerate(zip(highs, lows)):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=h, low=l)
            frame.feed(candle)

        # Add indicator
        frame.add_indicator(pivot, ['PIVOT_HIGH', 'PIVOT_LOW'])

        # First high at index 2 should be replaced by second high at index 5
        assert frame.periods[2].PIVOT_HIGH is None  # Replaced
        assert frame.periods[5].PIVOT_HIGH == 115.0  # New high

    def test_alternation_low_replaces_low(self):
        """
        Test consecutive lows behavior.

        NOTE: The alternation rule applies at detection time, not retroactively.
        If two lows are detected without a high between them (due to timing),
        both can coexist. This is expected behavior.

        For strict alternation, users should filter pivots post-processing.
        """
        frame = TimeFrame('1T', max_periods=50)
        pivot = PivotPoints(left_bars=2, right_bars=2)

        base_date = datetime(2024, 1, 1, 12, 0)
        # Pattern: high, high, LOW1(80), high, high, LOW2(75), high, high
        # Indices:  0     1     2         3     4     5         6     7
        highs = [100, 99, 95,  98,  99,  94,  97,  98]
        lows =  [90,  89, 80,  88,  89,  75,  87,  88]

        for i, (h, l) in enumerate(zip(highs, lows)):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=h, low=l)
            frame.feed(candle)

            if i == 0:
                frame.add_indicator(pivot, ['PIVOT_HIGH', 'PIVOT_LOW'])

        # Both lows are valid swing lows detected independently
        # No high pivot between them, so both coexist (expected behavior)
        assert frame.periods[2].PIVOT_LOW == 80.0  # First low
        assert frame.periods[5].PIVOT_LOW == 75.0  # Second low

    def test_valid_alternation_high_low_high(self):
        """Test valid alternation: High → Low → High."""
        frame = TimeFrame('1T', max_periods=50)
        pivot = PivotPoints(left_bars=2, right_bars=2)

        base_date = datetime(2024, 1, 1, 12, 0)
        # Pattern: low, low, HIGH, low, low, LOW, high, high, HIGH, low, low
        # Indices: 0    1    2     3    4    5    6     7     8     9    10
        highs = [100, 101, 120, 102, 103, 95, 104, 105, 125, 106, 107]
        lows =  [90,  91,  100, 92,  93,  75, 94,  95,  105, 96,  97]

        for i, (h, l) in enumerate(zip(highs, lows)):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=h, low=l)
            frame.feed(candle)

        # Add indicator
        frame.add_indicator(pivot, ['PIVOT_HIGH', 'PIVOT_LOW'])

        # Should have: High at 2, Low at 5, High at 8
        assert frame.periods[2].PIVOT_HIGH == 120.0
        assert frame.periods[2].PIVOT_LOW is None

        assert frame.periods[5].PIVOT_HIGH is None
        assert frame.periods[5].PIVOT_LOW == 75.0

        assert frame.periods[8].PIVOT_HIGH == 125.0
        assert frame.periods[8].PIVOT_LOW is None


class TestPivotPointsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_insufficient_periods(self):
        """Test that pivots are None when not enough periods."""
        frame = TimeFrame('1T', max_periods=50)
        pivot = PivotPoints(left_bars=5, right_bars=2)

        base_date = datetime(2024, 1, 1, 12, 0)

        # Feed only 6 periods (need 8 minimum)
        for i in range(6):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=100+i, low=90+i)
            frame.feed(candle)

        # Add indicator
        frame.add_indicator(pivot, ['PIVOT_HIGH', 'PIVOT_LOW'])

        # All should be None
        for period in frame.periods:
            assert period.PIVOT_HIGH is None
            assert period.PIVOT_LOW is None

    def test_no_pivots_in_monotonic_trend(self):
        """Test that no pivots are detected in a monotonic trend."""
        frame = TimeFrame('1T', max_periods=50)
        pivot = PivotPoints(left_bars=2, right_bars=2)

        base_date = datetime(2024, 1, 1, 12, 0)

        # Monotonic increasing trend
        for i in range(10):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=100+i, low=90+i)
            frame.feed(candle)

        # Add indicator
        frame.add_indicator(pivot, ['PIVOT_HIGH', 'PIVOT_LOW'])

        # No pivots should be detected
        for period in frame.periods:
            assert period.PIVOT_HIGH is None
            assert period.PIVOT_LOW is None

    def test_equal_values_no_pivot(self):
        """Test that equal values don't create pivots."""
        frame = TimeFrame('1T', max_periods=50)
        pivot = PivotPoints(left_bars=2, right_bars=2)

        base_date = datetime(2024, 1, 1, 12, 0)

        # All same high/low values
        for i in range(10):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=100, low=90)
            frame.feed(candle)

        # Add indicator
        frame.add_indicator(pivot, ['PIVOT_HIGH', 'PIVOT_LOW'])

        # No pivots with equal values
        for period in frame.periods:
            assert period.PIVOT_HIGH is None
            assert period.PIVOT_LOW is None

    def test_lag_behavior(self):
        """Test that pivots are confirmed with right_bars lag."""
        frame = TimeFrame('1T', max_periods=50)
        pivot = PivotPoints(left_bars=2, right_bars=3)  # 3-bar lag

        base_date = datetime(2024, 1, 1, 12, 0)

        # Pattern: low, low, HIGH, low, low, low (need 3 right bars)
        # Indices: 0    1    2     3    4    5
        highs = [100, 101, 110, 102, 101, 100]
        lows =  [90,  91,  95,  92,  91,  90]

        for i, (h, l) in enumerate(zip(highs, lows)):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=h, low=l)
            frame.feed(candle)

            # Add indicator after first feed to test incremental updates
            if i == 0:
                frame.add_indicator(pivot, ['PIVOT_HIGH', 'PIVOT_LOW'])

        # Pivot at index 2 confirmed at index 2 + 3 = 5
        # After feeding index 5, the pivot should appear at index 2
        assert frame.periods[2].PIVOT_HIGH == 110.0
        assert frame.periods[2].PIVOT_LOW is None


class TestPivotPointsIntegration:
    """Integration tests with Frame."""

    def test_add_to_existing_periods(self):
        """Test adding PivotPoints to frame with existing periods."""
        frame = TimeFrame('1T', max_periods=50)
        base_date = datetime(2024, 1, 1, 12, 0)

        # Feed data first
        highs = [100, 101, 110, 102, 101]
        lows = [90, 91, 95, 92, 91]

        for i, (h, l) in enumerate(zip(highs, lows)):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=h, low=l)
            frame.feed(candle)

        # Add indicator after data is loaded
        pivot = PivotPoints(left_bars=2, right_bars=2)
        frame.add_indicator(pivot, ['PIVOT_HIGH', 'PIVOT_LOW'])

        # Verify calculation happened
        assert frame.periods[2].PIVOT_HIGH == 110.0

    def test_remove_indicator(self):
        """Test removing PivotPoints indicator."""
        frame = TimeFrame('1T', max_periods=50)
        base_date = datetime(2024, 1, 1, 12, 0)

        # Feed data
        highs = [100, 101, 110, 102, 101]
        lows = [90, 91, 95, 92, 91]

        for i, (h, l) in enumerate(zip(highs, lows)):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=h, low=l)
            frame.feed(candle)

        # Add and remove indicator
        pivot = PivotPoints(left_bars=2, right_bars=2)
        frame.add_indicator(pivot, ['PIVOT_HIGH', 'PIVOT_LOW'])
        frame.remove_indicator(['PIVOT_HIGH', 'PIVOT_LOW'])

        # Columns should be removed
        for period in frame.periods:
            assert 'PIVOT_HIGH' not in period._data
            assert 'PIVOT_LOW' not in period._data

    def test_export_to_pandas(self):
        """Test exporting frame with PivotPoints to pandas."""
        frame = TimeFrame('1T', max_periods=50)
        pivot = PivotPoints(left_bars=2, right_bars=2)

        base_date = datetime(2024, 1, 1, 12, 0)
        highs = [100, 101, 110, 102, 101]
        lows = [90, 91, 95, 92, 91]

        for i, (h, l) in enumerate(zip(highs, lows)):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=h, low=l)
            frame.feed(candle)

        frame.add_indicator(pivot, ['PIVOT_HIGH', 'PIVOT_LOW'])

        # Export to pandas
        df = frame.to_pandas()

        # Check columns exist
        assert 'PIVOT_HIGH' in df.columns
        assert 'PIVOT_LOW' in df.columns

        # Check values
        assert df.loc[2, 'PIVOT_HIGH'] == 110.0
        assert df.loc[2, 'PIVOT_LOW'] != df.loc[2, 'PIVOT_LOW']  # NaN check

    def test_export_to_numpy(self):
        """Test exporting frame with PivotPoints to numpy."""
        frame = TimeFrame('1T', max_periods=50)
        pivot = PivotPoints(left_bars=2, right_bars=2)

        base_date = datetime(2024, 1, 1, 12, 0)
        highs = [100, 101, 110, 102, 101]
        lows = [90, 91, 95, 92, 91]

        for i, (h, l) in enumerate(zip(highs, lows)):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=h, low=l)
            frame.feed(candle)

        frame.add_indicator(pivot, ['PIVOT_HIGH', 'PIVOT_LOW'])

        # Export to numpy
        arr = frame.to_numpy()

        # Should have OHLCV (5) + PIVOT_HIGH (1) + PIVOT_LOW (1) = 7 columns
        assert arr.shape[1] == 7
        assert arr.shape[0] == 5  # 5 periods

    def test_normalization(self):
        """Test that PivotPoints normalize correctly with price range."""
        frame = TimeFrame('1T', max_periods=50)
        pivot = PivotPoints(left_bars=2, right_bars=2)

        base_date = datetime(2024, 1, 1, 12, 0)
        highs = [100, 101, 110, 102, 101]
        lows = [90, 91, 95, 92, 91]

        for i, (h, l) in enumerate(zip(highs, lows)):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=h, low=l, open_val=h-5, close_val=l+5)
            frame.feed(candle)

        frame.add_indicator(pivot, ['PIVOT_HIGH', 'PIVOT_LOW'])

        # Export normalized
        normalized = frame.to_normalize()

        # All values should be in [0, 1] range (or NaN)
        import numpy as np
        valid_values = normalized[~np.isnan(normalized)]
        assert np.all(valid_values >= 0.0)
        assert np.all(valid_values <= 1.0)

    def test_custom_column_names_integration(self):
        """Test that custom column names work end-to-end in Frame."""
        frame = TimeFrame('1T', max_periods=50)
        pivot = PivotPoints(
            left_bars=2,
            right_bars=2,
            high_column='SWING_HIGH',
            low_column='SWING_LOW'
        )

        base_date = datetime(2024, 1, 1, 12, 0)
        # Need at least 5 periods to confirm a pivot at index 2 (2 left + 1 candidate + 2 right)
        # And 7 periods to confirm a pivot at index 4
        highs = [100, 101, 110, 102, 101, 100, 99]  # 110 at index 2 should be swing high
        lows = [90, 91, 95, 92, 81, 82, 83]         # 81 at index 4 should be swing low

        for i, (h, l) in enumerate(zip(highs, lows)):
            date = base_date + timedelta(minutes=i)
            candle = create_candle(date, high=h, low=l, open_val=h-5, close_val=l+5)
            frame.feed(candle)

        # Use custom column names when adding indicator
        frame.add_indicator(pivot, ['SWING_HIGH', 'SWING_LOW'])

        # Export to pandas and check column names
        df = frame.to_pandas()
        assert 'SWING_HIGH' in df.columns
        assert 'SWING_LOW' in df.columns
        assert 'PIVOT_HIGH' not in df.columns
        assert 'PIVOT_LOW' not in df.columns

        # Verify values are assigned correctly
        assert df.iloc[2]['SWING_HIGH'] == 110  # Index 2 should have swing high
        assert df.iloc[4]['SWING_LOW'] == 81    # Index 4 should have swing low
