"""
PivotPoints Indicator Demo with Real Market Data

Demonstrates PivotPoints indicator using real Nasdaq data from Yahoo Finance.
Visualizes swing highs and lows on a candlestick chart.
"""

import yfinance as yf
import pandas as pd
import mplfinance as mpf
from datetime import datetime

from trading_frame import Candle, TimeFrame
from trading_frame.indicators.trend.pivot_points import PivotPoints


def main():
    print("=" * 60)
    print("PivotPoints Demo - Real Market Data")
    print("=" * 60)

    # 1. Download real Nasdaq data from Yahoo Finance
    print("\n1. Downloading ^IXIC (Nasdaq Composite) data from Yahoo Finance...")
    ticker = "^IXIC"
    data = yf.download(ticker, period="3mo", interval="1d", progress=False)

    print(f"   Downloaded {len(data)} daily candles")

    # 2. Create TimeFrame
    print("\n2. Creating TimeFrame...")
    frame = TimeFrame('1D', max_periods=250)

    # 3. Add PivotPoints indicator
    print("3. Adding PivotPoints indicator (left=5, right=2)...")
    pivot = PivotPoints(left_bars=5, right_bars=2)
    frame.add_indicator(pivot, ['PIVOT_HIGH', 'PIVOT_LOW'])

    # 4. Feed data to frame
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

    # 5. Count pivots
    highs = sum(1 for p in frame.periods if p.PIVOT_HIGH is not None)
    lows = sum(1 for p in frame.periods if p.PIVOT_LOW is not None)

    print(f"\n5. Results:")
    print(f"   Swing Highs: {highs}")
    print(f"   Swing Lows:  {lows}")
    print(f"   Total Pivots: {highs + lows}")

    # 6. Export to pandas
    print("\n6. Creating visualization...")
    df = frame.to_pandas()
    df.set_index('open_date', inplace=True)

    # Rename columns for mplfinance compatibility
    df.rename(columns={
        'open_price': 'Open',
        'high_price': 'High',
        'low_price': 'Low',
        'close_price': 'Close',
        'volume': 'Volume'
    }, inplace=True)

    # 7. Prepare pivot markers
    pivot_highs = pd.Series(index=df.index, dtype=float)
    pivot_lows = pd.Series(index=df.index, dtype=float)

    for idx, row in df.iterrows():
        if pd.notna(row['PIVOT_HIGH']):
            pivot_highs.loc[idx] = row['PIVOT_HIGH']
        if pd.notna(row['PIVOT_LOW']):
            pivot_lows.loc[idx] = row['PIVOT_LOW']

    # 8. Create chart
    addplots = []

    if pivot_highs.notna().any():
        addplots.append(mpf.make_addplot(
            pivot_highs,
            type='scatter',
            markersize=150,
            marker='v',
            color='red'
        ))

    if pivot_lows.notna().any():
        addplots.append(mpf.make_addplot(
            pivot_lows,
            type='scatter',
            markersize=150,
            marker='^',
            color='lime'
        ))

    mpf.plot(
        df,
        type='candle',
        style='charles',
        title=f'{ticker} - PivotPoints (left=5, right=2)\nRed=High | Green=Low',
        ylabel='Price',
        volume=True,
        addplot=addplots if addplots else None,
        figsize=(16, 9),
        warn_too_much_data=300
    )

    print("   Chart displayed!")
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
