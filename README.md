# Trading Frame

A Python package to manage and organize trading data into various frame types.

## Overview

Trading Frame provides a clean API to aggregate raw trading data (candles) into organized periods using different time-based frames.

## Features

- **Candle**: Raw OHLCV trading data structure
- **Period**: Aggregated data over a time range
- **Frame**: Base class for data aggregation
- **TimeFrame**: Time-based period aggregation (seconds, minutes, hours, days, weeks, months, years)

## Installation

```bash
pip install -e .
```

## Usage

```python
from trading_frame import Candle, TimeFrame

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
    print(f"Open: {period.open_price}, Close: {period.close_price}")
```

## Supported Time Ranges

- `S`: Second
- `T`: Minute
- `H`: Hour
- `D`: Day
- `W`: Week
- `M`: Month
- `Y`: Year

## Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=trading_frame
```

## License

MIT License - see LICENSE file for details.
