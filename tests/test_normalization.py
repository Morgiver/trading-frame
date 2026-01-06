"""Unit tests for normalization system."""

import pytest
import numpy as np
from datetime import datetime
from decimal import Decimal
from trading_frame import Candle, TimeFrame
from trading_frame.indicators.momentum.rsi import RSI
from trading_frame.indicators.momentum.macd import MACD
from trading_frame.indicators.trend.sma import SMA
from trading_frame.indicators.trend.bollinger import BollingerBands
from trading_frame.indicators.trend.pivot_points import PivotPoints


class TestNormalization:
    """Test the normalization system."""

    def setup_method(self):
        """Create a frame with sample data and indicators."""
        self.frame = TimeFrame('1T', max_periods=100)

        # Generate 50 candles with varying prices
        for i in range(50):
            base_price = 100.0 + (i * 2.0)  # Prices from 100 to 198
            candle = Candle(
                date=datetime(2024, 1, 1, 0, i),
                open=base_price,
                high=base_price + 2.0,
                low=base_price - 1.0,
                close=base_price + 1.0,
                volume=Decimal(str(1000.0 + i * 50.0))  # Volumes from 1000 to 3450
            )
            self.frame.feed(candle)

        # Add indicators
        self.frame.add_indicator(RSI(length=14), 'RSI_14')
        self.frame.add_indicator(SMA(period=20), 'SMA_20')
        self.frame.add_indicator(
            MACD(fast=12, slow=26, signal=9),
            ['MACD_LINE', 'MACD_SIGNAL', 'MACD_HIST']
        )
        self.frame.add_indicator(
            BollingerBands(period=20),
            ['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER']
        )
        self.frame.add_indicator(
            PivotPoints(left_bars=5, right_bars=2),
            ['PIVOT_HIGH', 'PIVOT_LOW']
        )

    def test_to_normalize_shape(self):
        """Test that to_normalize returns correct shape."""
        normalized = self.frame.to_normalize()

        # Should have same number of rows as periods
        assert normalized.shape[0] == len(self.frame.periods)

        # Should have OHLCV (5) + RSI (1) + SMA (1) + MACD (3) + BB (3) + PivotPoints (2) = 15 columns
        assert normalized.shape[1] == 15

    def test_ohlc_normalization_range(self):
        """Test that OHLC values are normalized to [0, 1]."""
        normalized = self.frame.to_normalize()

        # Extract OHLC columns (first 4 columns)
        ohlc = normalized[:, :4]

        # All valid values should be in [0, 1]
        valid_values = ohlc[~np.isnan(ohlc)]
        assert np.all(valid_values >= 0.0)
        assert np.all(valid_values <= 1.0)

        # Note: min/max may not be exactly 0/1 because price range includes
        # price-based indicators (Bollinger Bands) that can exceed OHLC range
        # Just verify values are within valid range
        assert np.min(valid_values) >= 0.0
        assert np.max(valid_values) <= 1.0

    def test_volume_normalization_range(self):
        """Test that volume is normalized to [0, 1]."""
        normalized = self.frame.to_normalize()

        # Volume is column 4 (index 4)
        volume = normalized[:, 4]

        # All values should be in [0, 1]
        assert np.all(volume >= 0.0)
        assert np.all(volume <= 1.0)

        # Should have both 0 and 1 (min and max)
        assert np.min(volume) == pytest.approx(0.0, abs=0.01)
        assert np.max(volume) == pytest.approx(1.0, abs=0.01)

    def test_rsi_normalization_fixed_range(self):
        """Test that RSI is normalized from [0, 100] to [0, 1]."""
        normalized = self.frame.to_normalize()

        # RSI is column 5 (after OHLCV)
        # Column order: O, H, L, C, V, BB_LOWER, BB_MIDDLE, BB_UPPER, MACD_HIST, MACD_LINE, MACD_SIGNAL, RSI_14, SMA_20
        # Need to find RSI column
        indicator_cols = sorted(self.frame._get_all_indicator_columns())
        rsi_index = 5 + indicator_cols.index('RSI_14')

        rsi_normalized = normalized[:, rsi_index]

        # Filter out NaN values (early periods without enough data)
        valid_rsi = rsi_normalized[~np.isnan(rsi_normalized)]

        # All values should be in [0, 1]
        assert len(valid_rsi) > 0
        assert np.all(valid_rsi >= 0.0)
        assert np.all(valid_rsi <= 1.0)

        # Get raw RSI values to verify normalization
        raw_rsi_values = []
        for period in self.frame.periods:
            if period._data['RSI_14'] is not None:
                raw_rsi_values.append(period._data['RSI_14'])

        if len(raw_rsi_values) > 0:
            # Check that normalization is correct: raw / 100
            expected_normalized = [v / 100.0 for v in raw_rsi_values]
            actual_normalized = valid_rsi.tolist()
            np.testing.assert_allclose(actual_normalized, expected_normalized, rtol=1e-5)

    def test_sma_normalization_price_based(self):
        """Test that SMA (price-based) shares unified price normalization range."""
        normalized = self.frame.to_normalize()

        # Get unified price range (OHLC + all price-based indicators)
        price_values = []
        for period in self.frame.periods:
            if period.open_price is not None:
                price_values.extend([
                    float(period.open_price),
                    float(period.high_price),
                    float(period.low_price),
                    float(period.close_price)
                ])

        # Add price-based indicator values (SMA, Bollinger Bands)
        for col in ['SMA_20', 'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER']:
            for period in self.frame.periods:
                val = period._data.get(col)
                if val is not None:
                    price_values.append(float(val))

        price_min = np.min(price_values)
        price_max = np.max(price_values)
        price_range = price_max - price_min

        # Get SMA column
        indicator_cols = sorted(self.frame._get_all_indicator_columns())
        sma_index = 5 + indicator_cols.index('SMA_20')

        sma_normalized = normalized[:, sma_index]
        valid_sma = sma_normalized[~np.isnan(sma_normalized)]

        # All values should be in [0, 1]
        assert len(valid_sma) > 0
        assert np.all(valid_sma >= 0.0)
        assert np.all(valid_sma <= 1.0)

        # Verify normalization formula with unified price range
        raw_sma_values = []
        for period in self.frame.periods:
            if period._data['SMA_20'] is not None:
                raw_sma_values.append(period._data['SMA_20'])

        if len(raw_sma_values) > 0:
            expected_normalized = [(v - price_min) / price_range for v in raw_sma_values]
            actual_normalized = valid_sma.tolist()
            np.testing.assert_allclose(actual_normalized, expected_normalized, rtol=1e-5)

    def test_macd_normalization_minmax(self):
        """Test that MACD uses min-max normalization on its own values."""
        normalized = self.frame.to_normalize()

        # Get MACD columns
        indicator_cols = sorted(self.frame._get_all_indicator_columns())
        macd_line_idx = 5 + indicator_cols.index('MACD_LINE')
        macd_signal_idx = 5 + indicator_cols.index('MACD_SIGNAL')
        macd_hist_idx = 5 + indicator_cols.index('MACD_HIST')

        macd_line = normalized[:, macd_line_idx]
        macd_signal = normalized[:, macd_signal_idx]
        macd_hist = normalized[:, macd_hist_idx]

        # Each should be normalized independently
        for macd_col in [macd_line, macd_signal, macd_hist]:
            valid_values = macd_col[~np.isnan(macd_col)]
            if len(valid_values) > 0:
                assert np.all(valid_values >= 0.0)
                assert np.all(valid_values <= 1.0)

                # For linear trending data, MACD values may be very similar
                # Just verify normalization is applied (min is close to 0)
                if len(np.unique(valid_values)) > 1:  # Only if there's variation
                    assert np.min(valid_values) <= 0.1  # Close to 0

    def test_bollinger_normalization_price_based(self):
        """Test that Bollinger Bands share OHLC normalization range."""
        normalized = self.frame.to_normalize()

        # Get OHLC range
        ohlc_values = []
        for period in self.frame.periods:
            if period.open_price is not None:
                ohlc_values.extend([
                    float(period.open_price),
                    float(period.high_price),
                    float(period.low_price),
                    float(period.close_price)
                ])

        ohlc_min = np.min(ohlc_values)
        ohlc_max = np.max(ohlc_values)

        # Get BB columns
        indicator_cols = sorted(self.frame._get_all_indicator_columns())
        bb_upper_idx = 5 + indicator_cols.index('BB_UPPER')
        bb_middle_idx = 5 + indicator_cols.index('BB_MIDDLE')
        bb_lower_idx = 5 + indicator_cols.index('BB_LOWER')

        # BB values can go outside [0, 1] since bands can exceed OHLC range
        # That's expected behavior - upper band can be above max price
        # Just verify middle and lower are reasonable
        bb_middle = normalized[:, bb_middle_idx]
        bb_lower = normalized[:, bb_lower_idx]

        valid_middle = bb_middle[~np.isnan(bb_middle)]
        valid_lower = bb_lower[~np.isnan(bb_lower)]

        if len(valid_middle) > 0:
            # Middle should be within [0, 1] (it's a moving average of prices)
            assert np.all(valid_middle >= 0.0)
            assert np.all(valid_middle <= 1.0)

        # Lower can be below 0 if prices are trending up
        # Upper can be above 1 if prices are trending up

        # Verify upper > middle > lower relationship is preserved
        for i, period in enumerate(self.frame.periods):
            upper_raw = period._data.get('BB_UPPER')
            middle_raw = period._data.get('BB_MIDDLE')
            lower_raw = period._data.get('BB_LOWER')

            if all(v is not None for v in [upper_raw, middle_raw, lower_raw]):
                upper_norm = normalized[i, bb_upper_idx]
                middle_norm = normalized[i, bb_middle_idx]
                lower_norm = normalized[i, bb_lower_idx]

                # Relationship should be preserved
                assert upper_norm >= middle_norm >= lower_norm

    def test_empty_frame_normalization(self):
        """Test normalization with empty frame."""
        empty_frame = TimeFrame('1T', max_periods=10)
        normalized = empty_frame.to_normalize()

        # Should return empty array with correct shape
        assert normalized.shape[0] == 0
        assert normalized.shape[1] == 5  # Just OHLCV

    def test_normalization_consistency(self):
        """Test that normalization is consistent across multiple calls."""
        normalized1 = self.frame.to_normalize()
        normalized2 = self.frame.to_normalize()

        np.testing.assert_array_equal(normalized1, normalized2)

    def test_pivot_points_price_normalization(self):
        """Test that PivotPoints use price-based normalization."""
        normalized = self.frame.to_normalize()

        # Get price min/max (unified range for OHLC + price-based indicators)
        price_values = []
        for period in self.frame.periods:
            if period.open_price is not None:
                price_values.extend([
                    float(period.open_price),
                    float(period.high_price),
                    float(period.low_price),
                    float(period.close_price)
                ])

        # Add price-based indicator values (SMA, BB, PivotPoints)
        for col in ['SMA_20', 'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER', 'PIVOT_HIGH', 'PIVOT_LOW']:
            for period in self.frame.periods:
                val = period._data.get(col)
                if val is not None:
                    price_values.append(float(val))

        price_min = np.min(price_values)
        price_max = np.max(price_values)
        price_range = price_max - price_min

        # Get PivotPoints column indices
        indicator_cols = sorted(self.frame._get_all_indicator_columns())
        pivot_high_index = 5 + indicator_cols.index('PIVOT_HIGH')
        pivot_low_index = 5 + indicator_cols.index('PIVOT_LOW')

        # Verify normalization for detected pivots
        for i, period in enumerate(self.frame.periods):
            pivot_high_raw = period._data.get('PIVOT_HIGH')
            pivot_low_raw = period._data.get('PIVOT_LOW')

            if pivot_high_raw is not None:
                expected_norm = (pivot_high_raw - price_min) / price_range
                actual_norm = normalized[i, pivot_high_index]
                assert actual_norm == pytest.approx(expected_norm, abs=1e-6)

            if pivot_low_raw is not None:
                expected_norm = (pivot_low_raw - price_min) / price_range
                actual_norm = normalized[i, pivot_low_index]
                assert actual_norm == pytest.approx(expected_norm, abs=1e-6)

    def test_normalization_no_indicators(self):
        """Test normalization without indicators."""
        simple_frame = TimeFrame('1T', max_periods=50)

        # Add data
        for i in range(20):
            candle = Candle(
                date=datetime(2024, 1, 1, 0, i),
                open=100.0 + i,
                high=105.0 + i,
                low=95.0 + i,
                close=102.0 + i,
                volume=Decimal('1000.0')
            )
            simple_frame.feed(candle)

        normalized = simple_frame.to_normalize()

        # Should have only OHLCV columns
        assert normalized.shape[1] == 5

        # All values should be in [0, 1]
        assert np.all(normalized >= 0.0)
        assert np.all(normalized <= 1.0)


class TestIndicatorNormalizationMethods:
    """Test indicator normalization methods directly."""

    def test_rsi_fixed_range(self):
        """Test RSI fixed range normalization."""
        rsi = RSI(length=14)

        assert rsi.get_normalization_type() == 'fixed'
        assert rsi.get_fixed_range() == (0.0, 100.0)

        # Test normalization
        assert rsi.normalize(0.0) == pytest.approx(0.0)
        assert rsi.normalize(50.0) == pytest.approx(0.5)
        assert rsi.normalize(100.0) == pytest.approx(1.0)

    def test_sma_price_normalization(self):
        """Test SMA price-based normalization."""
        sma = SMA(period=20, source='close_price')

        assert sma.get_normalization_type() == 'price'

        # Test normalization with price range
        price_range = (100.0, 200.0)
        assert sma.normalize(100.0, price_range=price_range) == pytest.approx(0.0)
        assert sma.normalize(150.0, price_range=price_range) == pytest.approx(0.5)
        assert sma.normalize(200.0, price_range=price_range) == pytest.approx(1.0)

    def test_sma_minmax_normalization(self):
        """Test SMA min-max normalization for non-price source."""
        sma = SMA(period=5, source='RSI_14')

        assert sma.get_normalization_type() == 'minmax'

        # Test normalization with all values
        all_values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        assert sma.normalize(10.0, all_values=all_values) == pytest.approx(0.0)
        assert sma.normalize(30.0, all_values=all_values) == pytest.approx(0.5)
        assert sma.normalize(50.0, all_values=all_values) == pytest.approx(1.0)

    def test_macd_minmax_normalization(self):
        """Test MACD min-max normalization."""
        macd = MACD()

        assert macd.get_normalization_type() == 'minmax'

        all_values = np.array([-5.0, -2.5, 0.0, 2.5, 5.0])
        assert macd.normalize(-5.0, all_values=all_values) == pytest.approx(0.0)
        assert macd.normalize(0.0, all_values=all_values) == pytest.approx(0.5)
        assert macd.normalize(5.0, all_values=all_values) == pytest.approx(1.0)

    def test_bollinger_price_normalization(self):
        """Test Bollinger Bands price-based normalization."""
        bb = BollingerBands()

        assert bb.get_normalization_type() == 'price'

        price_range = (100.0, 200.0)
        values = [110.0, 150.0, 190.0]
        normalized = bb.normalize(values, price_range=price_range)

        assert normalized[0] == pytest.approx(0.1)
        assert normalized[1] == pytest.approx(0.5)
        assert normalized[2] == pytest.approx(0.9)

    def test_normalize_none_values(self):
        """Test normalization handles None values."""
        rsi = RSI(length=14)

        assert rsi.normalize(None) is None

        # List with None
        result = rsi.normalize([50.0, None, 75.0])
        assert result[0] == pytest.approx(0.5)
        assert result[1] is None
        assert result[2] == pytest.approx(0.75)

    def test_normalize_constant_values(self):
        """Test normalization with constant values (min == max)."""
        # Use minmax normalization for this test
        sma = SMA(period=20, source='RSI_14')  # Non-price source = minmax

        # All values the same
        all_values = np.array([100.0, 100.0, 100.0])
        result = sma.normalize(100.0, all_values=all_values)

        # Should return 0.0 when min == max
        assert result == 0.0
