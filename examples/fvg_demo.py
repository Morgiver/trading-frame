"""
Fair Value Gap (FVG) Indicator Demo with Real Market Data

Demonstrates FVG indicator using real Nasdaq data from Yahoo Finance.
Visualizes bullish (demand) and bearish (supply) Fair Value Gaps on a candlestick chart.
"""

import yfinance as yf
import pandas as pd
import mplfinance as mpf
from datetime import datetime

from trading_frame import Candle, TimeFrame
from trading_frame.indicators.trend.fvg import FVG


def main():
    print("=" * 60)
    print("Fair Value Gap (FVG) Demo - Real Market Data")
    print("=" * 60)

    # 1. Download real Nasdaq data from Yahoo Finance
    print("\n1. Downloading ^IXIC (Nasdaq Composite) data from Yahoo Finance...")
    ticker = "^IXIC"
    data = yf.download(ticker, period="3mo", interval="1d", progress=False)

    print(f"   Downloaded {len(data)} daily candles")

    # 2. Create TimeFrame
    print("\n2. Creating TimeFrame...")
    frame = TimeFrame('1D', max_periods=250)

    # 3. Add FVG indicator
    print("3. Adding FVG indicator...")
    fvg = FVG()
    frame.add_indicator(fvg, ['FVG_HIGH', 'FVG_LOW'])

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

    # 5. Count FVGs
    fvg_count = sum(1 for p in frame.periods if p.FVG_HIGH is not None)
    bullish_count = 0
    bearish_count = 0

    for p in frame.periods:
        if p.FVG_HIGH is not None and p.FVG_LOW is not None:
            # Bullish FVG: FVG_HIGH > FVG_LOW (demand zone)
            # Bearish FVG: FVG_HIGH < FVG_LOW would be wrong, but actually both cases have FVG_HIGH > FVG_LOW
            # Let's check by looking at the range direction
            # In our implementation:
            # Bullish: fvg_low = high_1, fvg_high = low_3 (where low_3 > high_1)
            # Bearish: fvg_low = high_3, fvg_high = low_1 (where high_3 < low_1)
            # Both should have fvg_high > fvg_low
            # To differentiate: we'd need to look at candle direction or store FVG type
            # For now, we'll just count total FVGs
            pass

    print(f"\n5. Results:")
    print(f"   Total FVGs detected: {fvg_count}")

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

    # 7. Prepare FVG visualization
    # We'll draw rectangles for FVG zones
    fvg_zones = []

    for idx, row in df.iterrows():
        if pd.notna(row['FVG_HIGH']) and pd.notna(row['FVG_LOW']):
            fvg_high = row['FVG_HIGH']
            fvg_low = row['FVG_LOW']

            # Create a series for the FVG zone
            # We'll use a filled area between FVG_LOW and FVG_HIGH
            fvg_zones.append({
                'date': idx,
                'high': fvg_high,
                'low': fvg_low
            })

    # Create additional plots for FVG zones
    addplots = []

    # Plot FVG zones as shaded areas
    if fvg_zones:
        # Create series for high and low bounds
        fvg_high_series = pd.Series(index=df.index, dtype=float)
        fvg_low_series = pd.Series(index=df.index, dtype=float)

        for zone in fvg_zones:
            fvg_high_series.loc[zone['date']] = zone['high']
            fvg_low_series.loc[zone['date']] = zone['low']

        # Add markers at FVG locations
        addplots.append(mpf.make_addplot(
            fvg_high_series,
            type='scatter',
            markersize=100,
            marker='_',
            color='purple',
            alpha=0.8
        ))
        addplots.append(mpf.make_addplot(
            fvg_low_series,
            type='scatter',
            markersize=100,
            marker='_',
            color='purple',
            alpha=0.8
        ))

    mpf.plot(
        df,
        type='candle',
        style='charles',
        title=f'{ticker} - Fair Value Gaps (FVG)\nPurple lines mark FVG zones',
        ylabel='Price',
        volume=True,
        addplot=addplots if addplots else None,
        figsize=(16, 9),
        warn_too_much_data=300
    )

    print("   Chart displayed!")

    # Show some FVG details
    if fvg_zones:
        print(f"\n7. FVG Details (first 5):")
        for i, zone in enumerate(fvg_zones[:5]):
            print(f"   FVG #{i+1}: Date={zone['date'].strftime('%Y-%m-%d')}, "
                  f"Range=[{zone['low']:.2f}, {zone['high']:.2f}], "
                  f"Size={zone['high']-zone['low']:.2f}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
