"""
MACD Indicator Demo with Real Market Data

Demonstrates MACD indicator using real Nasdaq data from Yahoo Finance.
Visualizes trend momentum with MACD line, signal line, and histogram.
"""

import yfinance as yf
import pandas as pd
import mplfinance as mpf

from trading_frame import Candle, TimeFrame
from trading_frame.indicators import MACD


def main():
    print("=" * 60)
    print("MACD Demo - Nasdaq Composite")
    print("=" * 60)

    # 1. Download Nasdaq data
    print("\n1. Downloading ^IXIC data from Yahoo Finance...")
    ticker = "^IXIC"
    data = yf.download(ticker, period="6mo", interval="1d", progress=False)
    print(f"   Downloaded {len(data)} daily candles")

    # 2. Create TimeFrame
    print("\n2. Creating TimeFrame...")
    frame = TimeFrame('1D', max_periods=250)

    # 3. Add MACD indicator
    print("3. Adding MACD indicator (12, 26, 9)...")
    macd = MACD(fast=12, slow=26, signal=9)
    frame.add_indicator(macd, ['MACD_LINE', 'MACD_SIGNAL', 'MACD_HIST'])

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

    # 5. Analyze MACD crossovers
    bullish_cross = 0
    bearish_cross = 0

    for i in range(1, len(frame.periods)):
        prev = frame.periods[i-1]
        curr = frame.periods[i]

        if prev.MACD_LINE and prev.MACD_SIGNAL and curr.MACD_LINE and curr.MACD_SIGNAL:
            # Bullish crossover: MACD crosses above signal
            if prev.MACD_LINE < prev.MACD_SIGNAL and curr.MACD_LINE > curr.MACD_SIGNAL:
                bullish_cross += 1
            # Bearish crossover: MACD crosses below signal
            elif prev.MACD_LINE > prev.MACD_SIGNAL and curr.MACD_LINE < curr.MACD_SIGNAL:
                bearish_cross += 1

    print(f"\n5. Results:")
    print(f"   Bullish crossovers: {bullish_cross}")
    print(f"   Bearish crossovers: {bearish_cross}")

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

    # 7. Create MACD subplot (panel 2 because panel 1 is volume)
    apds = [
        mpf.make_addplot(df['MACD_LINE'], panel=2, color='blue', ylabel='MACD'),
        mpf.make_addplot(df['MACD_SIGNAL'], panel=2, color='red'),
        mpf.make_addplot(df['MACD_HIST'], panel=2, type='bar', color='gray', alpha=0.5),
    ]

    mpf.plot(
        df,
        type='candle',
        style='charles',
        title=f'{ticker} with MACD(12,26,9)\nBlue=MACD | Red=Signal | Gray=Histogram',
        ylabel='Price',
        volume=True,
        addplot=apds,
        figsize=(16, 9),
        panel_ratios=(3, 1, 1),
        warn_too_much_data=300
    )

    print("   Chart displayed!")
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
