"""
Bollinger Bands Indicator Demo with Real Market Data

Demonstrates Bollinger Bands indicator using real Nasdaq data from Yahoo Finance.
Visualizes volatility bands and price action relative to bands.
"""

import yfinance as yf
import pandas as pd
import mplfinance as mpf

from trading_frame import Candle, TimeFrame
from trading_frame.indicators import BollingerBands


def main():
    print("=" * 60)
    print("Bollinger Bands Demo - Nasdaq Composite")
    print("=" * 60)

    # 1. Download Nasdaq data
    print("\n1. Downloading ^IXIC data from Yahoo Finance...")
    ticker = "^IXIC"
    data = yf.download(ticker, period="6mo", interval="1d", progress=False)
    print(f"   Downloaded {len(data)} daily candles")

    # 2. Create TimeFrame
    print("\n2. Creating TimeFrame...")
    frame = TimeFrame('1D', max_periods=250)

    # 3. Add Bollinger Bands indicator
    print("3. Adding Bollinger Bands indicator (20, 2.0)...")
    bb = BollingerBands(period=20, std_dev=2.0)
    frame.add_indicator(bb, ['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER'])

    # 4. Feed data
    print("4. Processing candles...")
    for date, row in data.iterrows():
        candle = Candle(
            date=date,
            open=row['Open'].item(),
            high=row['High'].item(),
            low=row['Low'].item(),
            close=row['Close'].item(),
            volume=row['Volume'].item()
        )
        frame.feed(candle)

    # 5. Analyze band touches
    upper_touches = 0
    lower_touches = 0

    for period in frame.periods:
        if period.BB_UPPER and period.BB_LOWER:
            # Price touches upper band
            if period.high_price >= period.BB_UPPER * 0.999:
                upper_touches += 1
            # Price touches lower band
            if period.low_price <= period.BB_LOWER * 1.001:
                lower_touches += 1

    print(f"\n5. Results:")
    print(f"   Upper band touches: {upper_touches}")
    print(f"   Lower band touches: {lower_touches}")

    # 6. Export to pandas
    print("\n6. Creating visualization...")
    df = frame.to_pandas()
    df.set_index('open_date', inplace=True)
    df.rename(columns={
        'open_price': 'Open',
        'high_price': 'High',
        'low_price': 'Low',
        'close_price': 'Close',
        'volume': 'Volume'
    }, inplace=True)

    # 7. Create chart with Bollinger Bands
    apds = [
        mpf.make_addplot(df['BB_UPPER'], color='red', width=1, linestyle='--'),
        mpf.make_addplot(df['BB_MIDDLE'], color='blue', width=1.5),
        mpf.make_addplot(df['BB_LOWER'], color='green', width=1, linestyle='--'),
    ]

    mpf.plot(
        df,
        type='candle',
        style='charles',
        title=f'{ticker} with Bollinger Bands(20,2)\nRed=Upper | Blue=Middle | Green=Lower',
        ylabel='Price',
        volume=True,
        addplot=apds,
        figsize=(16, 9),
        warn_too_much_data=300
    )

    print("   Chart displayed!")
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
