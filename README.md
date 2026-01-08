# Trading Frame

A Python package to manage and organize trading data into various frame types.

## ⚠️ Breaking Change (v0.4.0)

**Technical indicators have been moved to a separate package:** [trading-indicators](https://github.com/Morgiver/trading-indicators)

If you were using indicators from `trading-frame`, you need to install the new package:

```bash
pip install git+https://github.com/Morgiver/trading-indicators.git
```

### Migration Guide

**Old way (v0.3.x):**
```python
from trading_frame import Frame
from trading_frame.indicators import RSI, SMA, MACD
frame.add_indicator(RSI(14), 'RSI_14')
```

**New way (v0.4.x):**
```python
from trading_frame import Frame
from trading_indicators import RSI, SMA, MACD
rsi = RSI(frame=frame, length=14, column_name='RSI_14')
```

The new architecture provides:
- 🔄 **Automatic synchronization** with frame events
- 📊 **Period-by-period calculation** (more efficient)
- 🎯 **Simpler API** with automatic frame binding
- 🔗 **Composite indicators** support

See [trading-indicators documentation](https://github.com/Morgiver/trading-indicators) for details.

---

## Overview

Trading Frame provides a clean API to aggregate raw trading data (candles) into organized periods using different time-based frames.

## Features

- **Candle**: Raw OHLCV trading data structure with robust validation
- **Period**: Aggregated data over a time range with Decimal precision for volumes
- **Frame**: Base class for data aggregation with event system
- **TimeFrame**: Time-based period aggregation (seconds, minutes, hours, days, weeks, months, years)
- **Export**: Convert frames to NumPy arrays or Pandas DataFrames
- **Normalization**: ML-ready normalized data with Min-Max scaling
- **Prefill**: Efficient warm-up mechanism to fill frames before live trading

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

### Export to NumPy/Pandas

```python
# Export to NumPy array
import numpy as np
data = frame.to_numpy()  # Returns [[open, high, low, close, volume], ...]

# Export to Pandas DataFrame
import pandas as pd
df = frame.to_pandas()  # Returns DataFrame with OHLCV columns
print(df.head())
```

### Normalized Data for Machine Learning

Trading Frame provides normalized OHLCV data ready for ML models:

```python
# Export normalized data to [0, 1] range
normalized = frame.to_normalize()  # NumPy array with normalized values

# Normalization strategies:
# - OHLC: Unified Min-Max across all price values
# - Volume: Independent Min-Max across all volumes
# All values scaled to [0, 1] range

# Perfect for feeding into neural networks, LSTMs, etc.
import torch
tensor = torch.from_numpy(normalized)
```

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
│   └── exceptions.py  # Custom exceptions
├── tests/
│   ├── test_candle.py
│   ├── test_period.py
│   ├── test_frame.py
│   ├── test_timeframe.py
│   └── test_prefill.py
└── pyproject.toml
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

This is a private repository. For bug reports or feature requests, please use GitHub Issues.

## Related Projects

- [trading-indicators](https://github.com/Morgiver/trading-indicators) - Technical indicators with automatic frame synchronization
- [trading-asset-view](https://github.com/Morgiver/trading-asset-view) - Multi-timeframe orchestration layer
