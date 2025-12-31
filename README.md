# Trading Frame

A Python package to manage and organize trading data into various frame types.

## Overview

Trading Frame provides a clean API to aggregate raw trading data (candles) into organized periods using different time-based frames.

## Features

- **Candle**: Raw OHLCV trading data structure with robust validation
- **Period**: Aggregated data over a time range with Decimal precision for volumes
- **Frame**: Base class for data aggregation with event system
- **TimeFrame**: Time-based period aggregation (seconds, minutes, hours, days, weeks, months, years)
- **Export**: Convert frames to NumPy arrays or Pandas DataFrames

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
│   └── timeframe.py   # Time-based frame
├── tests/
│   ├── test_candle.py
│   ├── test_period.py
│   ├── test_frame.py
│   └── test_timeframe.py
└── pyproject.toml
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

This is a private repository. For bug reports or feature requests, please use GitHub Issues.
