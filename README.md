# Trading Frame

A Python package to manage and organize trading data into various frame types.

## Overview

Trading Frame provides a clean API to aggregate raw trading data (candles) into organized periods using different time-based frames.

## Features

- **Candle**: Raw OHLCV trading data structure with robust validation
- **Period**: Aggregated data over a time range with Decimal precision for volumes
- **Frame**: Base class for data aggregation with event system
- **TimeFrame**: Time-based period aggregation (seconds, minutes, hours, days, weeks, months, years)
- **Indicators**: Technical indicators using TA-Lib (RSI, MACD, SMA, Bollinger Bands, Pivot Points, and more)
- **Export**: Convert frames to NumPy arrays or Pandas DataFrames with indicator columns
- **Normalization**: Intelligent normalization strategies for ML (OHLC, Volume, and indicator-specific)
- **Prefill**: Efficient warm-up mechanism to fill frames before live trading

## Examples

The `examples/` directory contains complete demonstrations of all indicators using real market data:

- **`rsi_demo.py`** - RSI indicator with overbought/oversold levels
- **`macd_demo.py`** - MACD with bullish/bearish crossover detection
- **`sma_demo.py`** - Multiple moving averages (SMA 20, 50, 200) with Golden/Death Cross
- **`bollinger_demo.py`** - Bollinger Bands with volatility analysis
- **`pivot_points_demo.py`** - Swing High/Low detection for support/resistance

All examples use **Yahoo Finance** to download real **Nasdaq Composite (^IXIC)** data and visualize results with **mplfinance**.

```bash
# Run any example
python examples/rsi_demo.py
python examples/macd_demo.py
python examples/pivot_points_demo.py
```

## Installation

### From GitHub (recommended)

```bash
pip install git+https://github.com/Morgiver/trading-frame.git
```

### For Development

```bash
git clone https://github.com/Morgiver/trading-frame.git
cd trading-frame
pip install -e .
```

### Install Example Dependencies

```bash
pip install yfinance mplfinance matplotlib
```

## Usage

### Basic Example

```python
from trading_frame import Candle, TimeFrame
from datetime import datetime

# Create a 5-minute time frame
frame = TimeFrame(periods_length='5T', max_periods=100)

# Feed candle data
candle = Candle(
    date='2025-01-01T00:00:00.000Z',
    open=50000.0,
    high=50100.0,
    low=49900.0,
    close=50050.0,
    volume=1000.0
)

frame.feed(candle)

# Access periods
for period in frame.periods:
    print(f"Open: {period.open_price}, Close: {period.close_price}, Volume: {period.volume}")
```

### Warm-Up with Prefill

```python
from datetime import datetime
from trading_frame import InsufficientDataError

# Create frame
frame = TimeFrame('5T', max_periods=100)

# Warm-up: fill frame before live trading
# Option 1: Fill until max_periods (default)
for candle in historical_data:
    if frame.prefill(candle):
        break  # Frame is ready with 100 periods

# Option 2: Fill until specific timestamp (relaxed mode)
target_ts = datetime(2024, 1, 1, 12, 0).timestamp()
for candle in historical_data:
    if frame.prefill(candle, target_timestamp=target_ts, require_full=False):
        break  # Reached target date (may not be full)

# Option 3: Fill until timestamp WITH validation (recommended for production)
target_ts = datetime(2024, 1, 1, 12, 0).timestamp()
try:
    for candle in historical_data:
        # Ensure frame is full at target timestamp
        if frame.prefill(candle, target_timestamp=target_ts, require_full=True):
            break  # Frame is full at target date
except InsufficientDataError as e:
    print(f"Not enough historical data: {e}")
    # Handle insufficient data (fetch more, adjust start time, etc.)

# Now switch to live trading
for candle in live_data:
    frame.feed(candle)  # Normal operation
```

### Event Handling

```python
# Register callbacks for period events
def on_new_period(frame):
    print(f"New period created! Total periods: {len(frame.periods)}")

def on_period_close(frame):
    last_period = frame.periods[-2] if len(frame.periods) > 1 else None
    if last_period:
        print(f"Period closed: {last_period.close_price}")

frame.on('new_period', on_new_period)
frame.on('close', on_period_close)
```

### Technical Indicators

```python
from trading_frame.indicators.momentum.rsi import RSI
from trading_frame.indicators.momentum.macd import MACD
from trading_frame.indicators.trend.sma import SMA
from trading_frame.indicators.trend.bollinger import BollingerBands
from trading_frame.indicators.trend.pivot_points import PivotPoints

# Add single-column indicator
frame.add_indicator(RSI(length=14), 'RSI_14')

# Add multi-column indicator
frame.add_indicator(
    MACD(fast=12, slow=26, signal=9),
    ['MACD_LINE', 'MACD_SIGNAL', 'MACD_HIST']
)

# Add moving averages
frame.add_indicator(SMA(period=20), 'SMA_20')
frame.add_indicator(SMA(period=50), 'SMA_50')

# Add Bollinger Bands
frame.add_indicator(
    BollingerBands(period=20, std_dev=2),
    ['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER']
)

# Add Pivot Points (Swing High/Low detection)
frame.add_indicator(
    PivotPoints(left_bars=5, right_bars=2),
    ['PIVOT_HIGH', 'PIVOT_LOW']
)

# Access indicator values
for period in frame.periods:
    print(f"RSI: {period.RSI_14}, SMA20: {period.SMA_20}")
    if period.PIVOT_HIGH:
        print(f"  Swing High detected at {period.PIVOT_HIGH}")
    if period.PIVOT_LOW:
        print(f"  Swing Low detected at {period.PIVOT_LOW}")

# Dependent indicators (indicator calculated from another indicator)
frame.add_indicator(SMA(period=5, source='RSI_14'), 'RSI_SMA')

# Remove indicator
frame.remove_indicator('RSI_14')
```

### Pivot Points (Swing High/Low Detection)

Pivot Points detect local highs and lows based on surrounding bars:

```python
from trading_frame.indicators.trend.pivot_points import PivotPoints

# Create indicator with asymmetric confirmation
pivot = PivotPoints(
    left_bars=5,   # Compare with 5 bars to the left
    right_bars=2   # Confirm with 2 bars to the right
)

frame.add_indicator(pivot, ['PIVOT_HIGH', 'PIVOT_LOW'])

# Access pivot points
for i, period in enumerate(frame.periods):
    if period.PIVOT_HIGH is not None:
        print(f"Swing High at index {i}: {period.PIVOT_HIGH}")
    if period.PIVOT_LOW is not None:
        print(f"Swing Low at index {i}: {period.PIVOT_LOW}")
```

**How it works:**

- **Swing High**: Detected when `high[0]` is greater than all `high[-left_bars..-1]` and `high[1..right_bars]`
- **Swing Low**: Detected when `low[0]` is lower than all `low[-left_bars..-1]` and `low[1..right_bars]`
- **Confirmation Lag**: Pivots are confirmed `right_bars` periods after the candidate
  - With `right_bars=2`, a pivot at index 100 is confirmed at index 102
  - The pivot value is assigned to the candidate bar (100), not the confirmation bar (102)

**Use cases:**

- Support/Resistance level identification
- Elliott Wave analysis
- Pattern recognition (Head & Shoulders, Double Tops/Bottoms)
- Trend reversal detection
- Entry/Exit signal generation

### Export to NumPy/Pandas

```python
# Export to NumPy array (includes indicators)
import numpy as np
data = frame.to_numpy()  # Returns [[open, high, low, close, volume, RSI_14, SMA_20], ...]

# Export to Pandas DataFrame (includes indicators)
import pandas as pd
df = frame.to_pandas()  # Returns DataFrame with OHLCV + indicator columns
print(df.head())
```

### Normalized Data Export

Trading Frame provides intelligent normalization for machine learning applications:

```python
# Export normalized data to [0, 1] range
normalized = frame.to_normalize()  # NumPy array with normalized values

# Different normalization strategies by data type:
# - OHLC: Min-Max across all prices
# - Volume: Min-Max across all volumes
# - RSI: Fixed range [0-100] → [0-1]
# - SMA (price-based): Shares OHLC min-max range
# - MACD: Min-Max on its own values
# - Bollinger Bands: Shares OHLC min-max range
```

Each indicator defines its own normalization strategy:
- **Fixed Range**: RSI (0-100) normalized to [0, 1]
- **Price-Based**: SMA, Bollinger Bands share the OHLC normalization range
- **Min-Max**: MACD, and non-price indicators use their own value ranges

This ensures semantically correct normalization for different indicator types.

## Supported Time Ranges

| Alias | Description | Example |
|-------|-------------|---------|
| `S` | Second | `'30S'` = 30-second periods |
| `T` | Minute | `'5T'` = 5-minute periods |
| `H` | Hour | `'4H'` = 4-hour periods |
| `D` | Day | `'1D'` = 1-day periods |
| `W` | Week | `'1W'` = 1-week periods (Monday-Sunday) |
| `M` | Month | `'1M'` = 1-month periods |
| `Y` | Year | `'1Y'` = 1-year periods |

## Features

### Decimal Precision

Volumes use Python's `Decimal` type for precise accumulation, avoiding floating-point errors:

```python
from decimal import Decimal

candle = Candle(
    date=datetime.now(),
    open=100.0,
    high=110.0,
    low=90.0,
    close=105.0,
    volume=Decimal('0.00000001')  # Precise volume
)
```

### Robust Validation

All inputs are validated:
- Date formats (string, timestamp, datetime objects)
- Price consistency (high >= low, high contains open/close)
- Non-negative prices and volumes

## Development

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=trading_frame

# Run specific test file
pytest tests/test_timeframe.py -v
```

### Project Structure

```
trading-frame/
├── src/trading_frame/
│   ├── __init__.py
│   ├── candle.py      # OHLCV data structure
│   ├── period.py      # Aggregated period
│   ├── frame.py       # Base frame class
│   ├── timeframe.py   # Time-based frame
│   └── indicators/    # Technical indicators
│       ├── base.py    # Base indicator class
│       ├── momentum/  # RSI, MACD, etc.
│       ├── trend/     # SMA, Bollinger Bands, etc.
│       └── volume/    # Volume indicators
├── tests/
│   ├── test_candle.py
│   ├── test_period.py
│   ├── test_frame.py
│   ├── test_timeframe.py
│   └── test_indicators.py
└── pyproject.toml
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

This is a private repository. For bug reports or feature requests, please use GitHub Issues.
