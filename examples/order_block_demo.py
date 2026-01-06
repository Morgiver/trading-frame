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
from trading_frame.indicators.trend.pivot_points import PivotPoints


def main():
    print("=" * 60)
    print("Order Block (OB) Demo - Real Market Data")
    print("=" * 60)

    # 1. Download real Nasdaq data from Yahoo Finance
    print("\n1. Downloading QQQ (Nasdaq-100 ETF) data from Yahoo Finance...")
    ticker = "QQQ"
    data = yf.download(ticker, period="5d", interval="15m", progress=False)

    print(f"   Downloaded {len(data)} 15-minute candles")

    # 2. Create TimeFrame
    print("\n2. Creating TimeFrame...")
    frame = TimeFrame('15T', max_periods=500)

    # 3. Add PivotPoints indicator (required for OrderBlock pivot filter)
    print("3. Adding PivotPoints indicator (left=5, right=2)...")
    pivot = PivotPoints(left_bars=5, right_bars=2)
    frame.add_indicator(pivot, ['PIVOT_HIGH', 'PIVOT_LOW'])

    # 4. Add OrderBlock indicator with pivot filter
    print("4. Adding OrderBlock indicator (lookback=10, min_body=30%, require_pivot=True)...")
    ob = OrderBlock(lookback=10, min_body_pct=0.3, require_pivot=True, pivot_lookback=3)
    frame.add_indicator(ob, ['OB_HIGH', 'OB_LOW'])

    # 5. Feed data to frame
    print("5. Processing candles...")
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

    # 6. Count Pivots and Order Blocks
    pivot_count = sum(1 for p in frame.periods if p.PIVOT_HIGH is not None or p.PIVOT_LOW is not None)
    ob_count = sum(1 for p in frame.periods if p.OB_HIGH is not None)

    print(f"\n6. Results:")
    print(f"   Pivots detected: {pivot_count}")
    print(f"   Order Blocks detected: {ob_count}")

    # 7. Export to pandas
    print("\n7. Creating visualization...")
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

    # 8. Prepare Pivot and OB visualization
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

    # Prepare pivot markers
    pivot_highs = pd.Series(index=df.index, dtype=float)
    pivot_lows = pd.Series(index=df.index, dtype=float)

    for idx, row in df.iterrows():
        if pd.notna(row['PIVOT_HIGH']):
            pivot_highs.loc[idx] = row['PIVOT_HIGH']
        if pd.notna(row['PIVOT_LOW']):
            pivot_lows.loc[idx] = row['PIVOT_LOW']

    # Create additional plots for pivots and OB zones
    addplots = []

    # Plot pivots
    if pivot_highs.notna().any():
        addplots.append(mpf.make_addplot(
            pivot_highs,
            type='scatter',
            markersize=100,
            marker='v',
            color='red',
            alpha=0.6
        ))

    if pivot_lows.notna().any():
        addplots.append(mpf.make_addplot(
            pivot_lows,
            type='scatter',
            markersize=100,
            marker='^',
            color='lime',
            alpha=0.6
        ))

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
        title=f'{ticker} - Order Blocks with Pivot Filter (15min)\nRed/Green=Pivots | Orange=Order Blocks',
        ylabel='Price',
        volume=True,
        addplot=addplots if addplots else None,
        figsize=(16, 9),
        warn_too_much_data=500
    )

    print("   Chart displayed!")

    # Show some OB details
    if ob_zones:
        print(f"\n8. Order Block Details (first 5):")
        for i, zone in enumerate(ob_zones[:5]):
            print(f"   OB #{i+1}: DateTime={zone['date'].strftime('%Y-%m-%d %H:%M')}, "
                  f"Range=[{zone['low']:.2f}, {zone['high']:.2f}], "
                  f"Size={zone['high']-zone['low']:.2f}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
