"""
Order Block (OB) Indicator Demo with Real Market Data

Demonstrates OrderBlock indicator using real Nasdaq data from Yahoo Finance.
Visualizes institutional buying/selling zones (order blocks) on a candlestick chart.
"""

import yfinance as yf
import pandas as pd
import mplfinance as mpf
from datetime import datetime

from trading_frame import Candle, TimeFrame
from trading_frame.indicators.trend.order_block import OrderBlock


def main():
    print("=" * 60)
    print("Order Block (OB) Demo - Real Market Data")
    print("=" * 60)

    # 1. Download real Nasdaq data from Yahoo Finance
    print("\n1. Downloading ^IXIC (Nasdaq Composite) data from Yahoo Finance...")
    ticker = "^IXIC"
    data = yf.download(ticker, period="3mo", interval="1d", progress=False)

    print(f"   Downloaded {len(data)} daily candles")

    # 2. Create TimeFrame
    print("\n2. Creating TimeFrame...")
    frame = TimeFrame('1D', max_periods=250)

    # 3. Add OrderBlock indicator
    print("3. Adding OrderBlock indicator (lookback=10, min_body=30%)...")
    ob = OrderBlock(lookback=10, min_body_pct=0.3)
    frame.add_indicator(ob, ['OB_HIGH', 'OB_LOW'])

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

    # 5. Count Order Blocks
    ob_count = sum(1 for p in frame.periods if p.OB_HIGH is not None)

    print(f"\n5. Results:")
    print(f"   Total Order Blocks detected: {ob_count}")

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

    # 7. Prepare OB visualization
    ob_zones = []

    for idx, row in df.iterrows():
        if pd.notna(row['OB_HIGH']) and pd.notna(row['OB_LOW']):
            ob_high = row['OB_HIGH']
            ob_low = row['OB_LOW']

            ob_zones.append({
                'date': idx,
                'high': ob_high,
                'low': ob_low
            })

    # Create additional plots for OB zones
    addplots = []

    # Plot OB zones as markers
    if ob_zones:
        # Create series for high and low bounds
        ob_high_series = pd.Series(index=df.index, dtype=float)
        ob_low_series = pd.Series(index=df.index, dtype=float)

        for zone in ob_zones:
            ob_high_series.loc[zone['date']] = zone['high']
            ob_low_series.loc[zone['date']] = zone['low']

        # Add markers at OB locations
        addplots.append(mpf.make_addplot(
            ob_high_series,
            type='scatter',
            markersize=100,
            marker='_',
            color='orange',
            alpha=0.8
        ))
        addplots.append(mpf.make_addplot(
            ob_low_series,
            type='scatter',
            markersize=100,
            marker='_',
            color='orange',
            alpha=0.8
        ))

    mpf.plot(
        df,
        type='candle',
        style='charles',
        title=f'{ticker} - Order Blocks\nOrange lines mark institutional zones',
        ylabel='Price',
        volume=True,
        addplot=addplots if addplots else None,
        figsize=(16, 9),
        warn_too_much_data=300
    )

    print("   Chart displayed!")

    # Show some OB details
    if ob_zones:
        print(f"\n7. Order Block Details (first 5):")
        for i, zone in enumerate(ob_zones[:5]):
            print(f"   OB #{i+1}: Date={zone['date'].strftime('%Y-%m-%d')}, "
                  f"Range=[{zone['low']:.2f}, {zone['high']:.2f}], "
                  f"Size={zone['high']-zone['low']:.2f}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
