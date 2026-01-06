"""
RSI Indicator Demo with Real Market Data

Demonstrates RSI indicator using real Nasdaq data from Yahoo Finance.
Visualizes overbought/oversold conditions.
"""

import yfinance as yf
import pandas as pd
import mplfinance as mpf

from trading_frame import Candle, TimeFrame
from trading_frame.indicators import RSI


def main():
    print("=" * 60)
    print("RSI Demo - Nasdaq-100 ETF")
    print("=" * 60)

    # 1. Download Nasdaq data
    print("\n1. Downloading QQQ (Nasdaq-100 ETF) data from Yahoo Finance...")
    ticker = "QQQ"
    data = yf.download(ticker, period="3mo", interval="1d", progress=False)
    print(f"   Downloaded {len(data)} daily candles")

    # 2. Create TimeFrame
    print("\n2. Creating TimeFrame...")
    frame = TimeFrame('1D', max_periods=250)

    # 3. Add RSI indicator
    print("3. Adding RSI indicator (period=14)...")
    rsi = RSI(length=14)
    frame.add_indicator(rsi, 'RSI_14')

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

    # 5. Analyze RSI
    rsi_values = [p.RSI_14 for p in frame.periods if p.RSI_14 is not None]
    overbought = sum(1 for v in rsi_values if v > 70)
    oversold = sum(1 for v in rsi_values if v < 30)

    print(f"\n5. Results:")
    print(f"   Overbought (RSI > 70): {overbought} periods")
    print(f"   Oversold (RSI < 30):   {oversold} periods")

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

    # 7. Create RSI subplot (panel 2 because panel 1 is volume)
    apds = [
        mpf.make_addplot(df['RSI_14'], panel=2, color='purple', ylabel='RSI'),
        mpf.make_addplot([70]*len(df), panel=2, color='red', linestyle='--', width=0.7),
        mpf.make_addplot([30]*len(df), panel=2, color='green', linestyle='--', width=0.7),
    ]

    mpf.plot(
        df,
        type='candle',
        style='charles',
        title=f'{ticker} with RSI(14)\nRed line = Overbought (70) | Green line = Oversold (30)',
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
