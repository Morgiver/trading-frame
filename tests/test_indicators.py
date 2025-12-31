"""Unit tests for indicator system."""

import pytest
from datetime import datetime
from decimal import Decimal
from trading_frame import Candle, TimeFrame
from trading_frame.indicators.momentum.rsi import RSI
from trading_frame.indicators.momentum.macd import MACD
from trading_frame.indicators.trend.sma import SMA
from trading_frame.indicators.trend.bollinger import BollingerBands


class TestIndicatorSystem:
    """Test the indicator integration with Frame."""

    def setup_method(self):
        """Create a frame with sample data."""
        self.frame = TimeFrame('1T', max_periods=100)

        # Generate 50 candles with trending data
        base_price = 100.0
        for i in range(50):
            # Create uptrend
            open_p = base_price + (i * 0.5)
            close_p = open_p + 0.3
            high_p = close_p + 0.2
            low_p = open_p - 0.1

            candle = Candle(
                date=datetime(2024, 1, 1, 0, i),
                open=open_p,
                high=high_p,
                low=low_p,
                close=close_p,
                volume=Decimal('1000.0')
            )
            self.frame.feed(candle)

    def test_add_single_column_indicator(self):
        """Test adding a single-column indicator."""
        rsi = RSI(length=14)
        self.frame.add_indicator(rsi, 'RSI_14')

        # Check indicator registered
        assert 'RSI_14' in self.frame.indicators

        # Check column exists in periods
        for period in self.frame.periods:
            assert 'RSI_14' in period._data

        # Check values calculated (after warmup)
        assert self.frame.periods[-1]._data['RSI_14'] is not None

    def test_add_multi_column_indicator(self):
        """Test adding a multi-column indicator."""
        macd = MACD(fast=12, slow=26, signal=9)
        self.frame.add_indicator(macd, ['MACD_LINE', 'MACD_SIGNAL', 'MACD_HIST'])

        # Check indicator registered with tuple key
        registry_key = ('MACD_LINE', 'MACD_SIGNAL', 'MACD_HIST')
        assert registry_key in self.frame.indicators

        # Check all columns exist
        for period in self.frame.periods:
            assert 'MACD_LINE' in period._data
            assert 'MACD_SIGNAL' in period._data
            assert 'MACD_HIST' in period._data

        # Check values calculated
        last_period = self.frame.periods[-1]
        assert last_period._data['MACD_LINE'] is not None

    def test_indicator_dependency_validation(self):
        """Test that dependencies are validated."""
        # Try to add dependent indicator before base
        dependent_sma = SMA(period=5, source='RSI_14')

        with pytest.raises(ValueError, match="Dependency 'RSI_14' not found"):
            self.frame.add_indicator(dependent_sma, 'RSI_SMA')

        # Add base indicator first
        self.frame.add_indicator(RSI(14), 'RSI_14')

        # Now dependent should work
        self.frame.add_indicator(dependent_sma, 'RSI_SMA')
        assert 'RSI_SMA' in self.frame.indicators

    def test_indicator_column_conflict(self):
        """Test that duplicate column names are rejected."""
        self.frame.add_indicator(RSI(14), 'RSI_14')

        # Try to add another with same name
        with pytest.raises(ValueError, match="Column 'RSI_14' already exists"):
            self.frame.add_indicator(SMA(20), 'RSI_14')

    def test_indicator_output_count_mismatch(self):
        """Test validation of output count vs column names."""
        # MACD produces 3 outputs
        macd = MACD()

        # Try with single column name
        with pytest.raises(ValueError, match="produces 3 outputs"):
            self.frame.add_indicator(macd, 'MACD')

        # Try with wrong number of columns
        with pytest.raises(ValueError, match="produces 3 outputs"):
            self.frame.add_indicator(macd, ['MACD_LINE', 'MACD_SIGNAL'])

    def test_remove_indicator(self):
        """Test removing an indicator."""
        self.frame.add_indicator(RSI(14), 'RSI_14')
        assert 'RSI_14' in self.frame.indicators

        self.frame.remove_indicator('RSI_14')

        # Check removed from registry
        assert 'RSI_14' not in self.frame.indicators

        # Check removed from periods
        for period in self.frame.periods:
            assert 'RSI_14' not in period._data

    def test_remove_multi_column_indicator(self):
        """Test removing a multi-column indicator."""
        self.frame.add_indicator(
            MACD(),
            ['MACD_LINE', 'MACD_SIGNAL', 'MACD_HIST']
        )

        self.frame.remove_indicator(['MACD_LINE', 'MACD_SIGNAL', 'MACD_HIST'])

        # Check all columns removed
        for period in self.frame.periods:
            assert 'MACD_LINE' not in period._data
            assert 'MACD_SIGNAL' not in period._data
            assert 'MACD_HIST' not in period._data

    def test_indicator_recalculation_on_update(self):
        """Test that indicators recalculate on period update."""
        self.frame.add_indicator(RSI(14), 'RSI_14')

        # Get current RSI value
        old_rsi = self.frame.periods[-1]._data['RSI_14']

        # Feed new candle that updates current period
        candle = Candle(
            date=datetime(2024, 1, 1, 0, 49, 30),
            open=120.0,
            high=125.0,
            low=119.0,
            close=124.0,
            volume=Decimal('2000.0')
        )
        self.frame.feed(candle)

        # RSI should have recalculated
        new_rsi = self.frame.periods[-1]._data['RSI_14']
        assert new_rsi != old_rsi

    def test_indicators_in_new_periods(self):
        """Test that indicators calculate for new periods."""
        self.frame.add_indicator(SMA(20), 'SMA_20')

        initial_count = len(self.frame.periods)

        # Add new candle that creates new period
        candle = Candle(
            date=datetime(2024, 1, 1, 1, 0),
            open=125.0,
            high=126.0,
            low=124.0,
            close=125.5,
            volume=Decimal('1500.0')
        )
        self.frame.feed(candle)

        # New period created
        assert len(self.frame.periods) == initial_count + 1

        # New period has indicator column
        assert 'SMA_20' in self.frame.periods[-1]._data

    def test_indicator_events(self):
        """Test that indicator events are emitted."""
        added_columns = []
        removed_columns = []

        def on_added(frame, columns):
            added_columns.extend(columns)

        def on_removed(frame, columns):
            removed_columns.extend(columns)

        self.frame.on('indicator_added', on_added)
        self.frame.on('indicator_removed', on_removed)

        # Add indicator
        self.frame.add_indicator(RSI(14), 'RSI_14')
        assert 'RSI_14' in added_columns

        # Remove indicator
        self.frame.remove_indicator('RSI_14')
        assert 'RSI_14' in removed_columns

    def test_multiple_indicators(self):
        """Test frame with multiple indicators."""
        self.frame.add_indicator(RSI(14), 'RSI_14')
        self.frame.add_indicator(SMA(20), 'SMA_20')
        self.frame.add_indicator(SMA(50), 'SMA_50')
        self.frame.add_indicator(
            BollingerBands(period=20),
            ['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER']
        )

        # Check all registered
        assert len(self.frame.indicators) == 4

        # Check all columns in periods
        last_period = self.frame.periods[-1]
        assert 'RSI_14' in last_period._data
        assert 'SMA_20' in last_period._data
        assert 'SMA_50' in last_period._data
        assert 'BB_UPPER' in last_period._data
        assert 'BB_MIDDLE' in last_period._data
        assert 'BB_LOWER' in last_period._data


class TestRSI:
    """Test RSI indicator."""

    def test_rsi_calculation(self):
        """Test RSI calculates valid values."""
        frame = TimeFrame('1T', max_periods=50)

        # Generate data
        for i in range(30):
            candle = Candle(
                date=datetime(2024, 1, 1, 0, i),
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=Decimal('1000.0')
            )
            frame.feed(candle)

        rsi = RSI(length=14)
        frame.add_indicator(rsi, 'RSI_14')

        # RSI should be between 0 and 100
        rsi_value = frame.periods[-1]._data['RSI_14']
        assert rsi_value is not None
        assert 0 <= rsi_value <= 100

    def test_rsi_requires_min_periods(self):
        """Test RSI minimum period requirement."""
        rsi = RSI(length=14)
        assert rsi.requires_min_periods() == 14


class TestMACD:
    """Test MACD indicator."""

    def test_macd_calculation(self):
        """Test MACD calculates three values."""
        frame = TimeFrame('1T', max_periods=100)

        # Generate data
        for i in range(50):
            candle = Candle(
                date=datetime(2024, 1, 1, 0, i),
                open=100.0 + i * 0.5,
                high=101.0 + i * 0.5,
                low=99.0 + i * 0.5,
                close=100.5 + i * 0.5,
                volume=Decimal('1000.0')
            )
            frame.feed(candle)

        macd = MACD(fast=12, slow=26, signal=9)
        frame.add_indicator(macd, ['MACD_LINE', 'MACD_SIGNAL', 'MACD_HIST'])

        # All three values should be calculated
        last_period = frame.periods[-1]
        assert last_period._data['MACD_LINE'] is not None
        assert last_period._data['MACD_SIGNAL'] is not None
        assert last_period._data['MACD_HIST'] is not None

    def test_macd_num_outputs(self):
        """Test MACD reports 3 outputs."""
        macd = MACD()
        assert macd.get_num_outputs() == 3


class TestSMA:
    """Test SMA indicator."""

    def test_sma_calculation(self):
        """Test SMA calculates correctly."""
        frame = TimeFrame('1T', max_periods=30)

        # Generate data with known values
        for i in range(25):
            price = 100.0
            candle = Candle(
                date=datetime(2024, 1, 1, 0, i),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=Decimal('1000.0')
            )
            frame.feed(candle)

        sma = SMA(period=20)
        frame.add_indicator(sma, 'SMA_20')

        # SMA of constant 100 should be ~100
        sma_value = frame.periods[-1]._data['SMA_20']
        assert sma_value is not None
        assert 99.0 <= sma_value <= 101.0

    def test_sma_dependent_indicator(self):
        """Test SMA can use other indicators as source."""
        frame = TimeFrame('1T', max_periods=50)

        # Generate data
        for i in range(40):
            candle = Candle(
                date=datetime(2024, 1, 1, 0, i),
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=Decimal('1000.0')
            )
            frame.feed(candle)

        # Add RSI first
        frame.add_indicator(RSI(14), 'RSI_14')

        # Then SMA of RSI
        rsi_sma = SMA(period=5, source='RSI_14')
        frame.add_indicator(rsi_sma, 'RSI_SMA')

        # Check dependencies
        assert 'RSI_14' in rsi_sma.get_dependencies()

        # Check calculated
        assert frame.periods[-1]._data['RSI_SMA'] is not None


class TestBollingerBands:
    """Test Bollinger Bands indicator."""

    def test_bollinger_calculation(self):
        """Test Bollinger Bands calculates three bands."""
        frame = TimeFrame('1T', max_periods=50)

        # Generate data
        for i in range(30):
            candle = Candle(
                date=datetime(2024, 1, 1, 0, i),
                open=100.0 + i * 0.3,
                high=101.0 + i * 0.3,
                low=99.0 + i * 0.3,
                close=100.5 + i * 0.3,
                volume=Decimal('1000.0')
            )
            frame.feed(candle)

        bb = BollingerBands(period=20, std_dev=2)
        frame.add_indicator(bb, ['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER'])

        # All three bands should exist
        last_period = frame.periods[-1]
        upper = last_period._data['BB_UPPER']
        middle = last_period._data['BB_MIDDLE']
        lower = last_period._data['BB_LOWER']

        assert upper is not None
        assert middle is not None
        assert lower is not None

        # Upper > Middle > Lower
        assert upper > middle > lower

    def test_bollinger_num_outputs(self):
        """Test Bollinger Bands reports 3 outputs."""
        bb = BollingerBands()
        assert bb.get_num_outputs() == 3


class TestExportsWithIndicators:
    """Test numpy/pandas exports with indicators."""

    def test_to_numpy_with_indicators(self):
        """Test numpy export includes indicators."""
        frame = TimeFrame('1T', max_periods=50)

        # Generate data
        for i in range(30):
            candle = Candle(
                date=datetime(2024, 1, 1, 0, i),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=Decimal('1000.0')
            )
            frame.feed(candle)

        # Add indicators
        frame.add_indicator(RSI(14), 'RSI_14')
        frame.add_indicator(SMA(20), 'SMA_20')

        # Export
        arr = frame.to_numpy()

        # Should have OHLCV (5) + 2 indicators = 7 columns
        assert arr.shape[1] == 7
        assert arr.shape[0] == len(frame.periods)

    def test_to_pandas_with_indicators(self):
        """Test pandas export includes indicators."""
        frame = TimeFrame('1T', max_periods=50)

        # Generate data
        for i in range(30):
            candle = Candle(
                date=datetime(2024, 1, 1, 0, i),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=Decimal('1000.0')
            )
            frame.feed(candle)

        # Add multi-column indicator
        frame.add_indicator(
            MACD(),
            ['MACD_LINE', 'MACD_SIGNAL', 'MACD_HIST']
        )

        # Export
        df = frame.to_pandas()

        # Should have indicator columns
        assert 'MACD_LINE' in df.columns
        assert 'MACD_SIGNAL' in df.columns
        assert 'MACD_HIST' in df.columns

        # Check values are numeric
        assert df['MACD_LINE'].dtype == float
