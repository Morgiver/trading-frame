"""
SMA Indicator Demo with Real Market Data

Demonstrates SMA indicator using real Nasdaq data from Yahoo Finance.
Visualizes multiple moving averages for trend identification.
"""

import yfinance as yf
import pandas as pd
import mplfinance as mpf

from trading_frame import Candle, TimeFrame
from trading_frame.indicators import SMA


def main():
    print("=" * 60)
    print("SMA Demo - Nasdaq-100 ETF")
    print("=" * 60)

    # 1. Download Nasdaq data (1 year for SMA 200)
    print("\n1. Downloading QQQ (Nasdaq-100 ETF) data from Yahoo Finance...")
    ticker = "QQQ"
    data = yf.download(ticker, period="1y", interval="1d", progress=False)
    print(f"   Downloaded {len(data)} daily candles")

    # 2. Create TimeFrame
    print("\n2. Creating TimeFrame...")
    frame = TimeFrame('1D', max_periods=250)

    # 3. Add multiple SMA indicators
    print("3. Adding SMA indicators (20, 50, 200)...")
    frame.add_indicator(SMA(period=20), 'SMA_20')
    frame.add_indicator(SMA(period=50), 'SMA_50')
    frame.add_indicator(SMA(period=200), 'SMA_200')

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

    # 5. Detect Golden Cross / Death Cross
    golden_cross = 0
    death_cross = 0

    for i in range(1, len(frame.periods)):
        prev = frame.periods[i-1]
        curr = frame.periods[i]

        if prev.SMA_50 and prev.SMA_200 and curr.SMA_50 and curr.SMA_200:
            # Golden Cross: SMA50 crosses above SMA200
            if prev.SMA_50 < prev.SMA_200 and curr.SMA_50 > curr.SMA_200:
                golden_cross += 1
            # Death Cross: SMA50 crosses below SMA200
            elif prev.SMA_50 > prev.SMA_200 and curr.SMA_50 < curr.SMA_200:
                death_cross += 1

    print(f"\n5. Results:")
    print(f"   Golden Cross (SMA50 > SMA200): {golden_cross}")
    print(f"   Death Cross (SMA50 < SMA200):  {death_cross}")

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

    # 7. Create chart with SMAs
    apds = [
        mpf.make_addplot(df['SMA_20'], color='blue', width=1.5),
        mpf.make_addplot(df['SMA_50'], color='orange', width=1.5),
        mpf.make_addplot(df['SMA_200'], color='red', width=2),
    ]

    mpf.plot(
        df,
        type='candle',
        style='charles',
        title=f'{ticker} with SMA(20,50,200)\nBlue=20 | Orange=50 | Red=200',
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
