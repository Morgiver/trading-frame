"""
Demonstration of Composite Indicators with Automatic Dependency Resolution

This example shows how to create and use composite indicators (indicators that
depend on other indicators) with automatic dependency resolution.
"""

import sys
import io

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from datetime import datetime, timedelta
from trading_frame import TimeFrame, Candle
from trading_frame.indicators import SMA, RSI, SMACrossover


def create_sample_data(n=200):
    """Create sample market data with trend changes."""
    candles = []
    base_date = datetime(2024, 1, 1)

    # Simulate market with trend changes
    for i in range(n):
        # Create price movement with trends
        if i < 50:
            # Initial downtrend
            base_price = 120 - i * 0.3
        elif i < 100:
            # Consolidation
            base_price = 105 + (i % 10) * 0.2
        else:
            # Uptrend
            base_price = 105 + (i - 100) * 0.4

        # Add some noise
        noise = (i % 7 - 3) * 0.5

        candles.append(Candle(
            date=base_date + timedelta(minutes=i * 5),
            open=base_price + noise,
            high=base_price + abs(noise) + 1,
            low=base_price - abs(noise) - 1,
            close=base_price + noise * 0.5,
            volume=1000 + (i % 100) * 10
        ))

    return candles


def demo_manual_order():
    """Example 1: Manual addition in correct order."""
    print("=" * 70)
    print("EXAMPLE 1: Manual Addition (Correct Order)")
    print("=" * 70)

    frame = TimeFrame('5T', max_periods=250)
    candles = create_sample_data(150)

    # Feed historical data
    for candle in candles:
        frame.feed(candle)

    print(f"\nLoaded {len(frame.periods)} periods of historical data")

    # Add indicators manually in CORRECT order (dependencies first)
    print("\n1. Adding SMA(20)...")
    frame.add_indicator(SMA(period=20))

    print("2. Adding SMA(50)...")
    frame.add_indicator(SMA(period=50))

    print("3. Adding SMACrossover(20, 50) - depends on SMA_20 and SMA_50...")
    frame.add_indicator(SMACrossover(fast=20, slow=50))

    print("4. Adding RSI(14) - independent indicator...")
    frame.add_indicator(RSI(length=14))

    # Show results
    last_period = frame.periods[-1]
    print(f"\n✓ All indicators added successfully!")
    print(f"\nLast period values:")
    print(f"  SMA_20:           {last_period._data.get('SMA_20'):.2f}")
    print(f"  SMA_50:           {last_period._data.get('SMA_50'):.2f}")
    print(f"  SMACrossover_20:  {last_period._data.get('SMACrossover_20')}")
    print(f"  RSI_14:           {last_period._data.get('RSI_14'):.2f}")


def demo_auto_resolution():
    """Example 2: Automatic dependency resolution (any order!)."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: Automatic Resolution (Any Order!)")
    print("=" * 70)

    frame = TimeFrame('5T', max_periods=250)
    candles = create_sample_data(150)

    # Feed historical data
    for candle in candles:
        frame.feed(candle)

    print(f"\nLoaded {len(frame.periods)} periods of historical data")

    # Add indicators in WRONG order - will be auto-resolved! 🎉
    print("\nAdding indicators in WRONG order (dependencies last)...")
    print("  - SMACrossover(20, 50)  <- Needs SMA_20, SMA_50")
    print("  - RSI(14)               <- Independent")
    print("  - SMA(50)               <- Needed by crossover")
    print("  - SMA(20)               <- Needed by crossover")

    mapping = frame.add_indicators_auto(
        SMACrossover(fast=20, slow=50),  # Depends on SMA_20, SMA_50
        RSI(14),                          # Independent
        SMA(50),                          # Needed by crossover
        SMA(20)                           # Needed by crossover
    )

    print(f"\n✓ Auto-resolved and added in correct order!")
    print(f"\nActual addition order (resolved automatically):")
    for i, (indicator, col_name) in enumerate(mapping.items(), 1):
        deps = indicator.get_dependencies()
        deps_str = f" (depends on: {', '.join(deps)})" if deps else " (no dependencies)"
        print(f"  {i}. {indicator.__class__.__name__} → {col_name}{deps_str}")

    # Show results
    last_period = frame.periods[-1]
    print(f"\nLast period values:")
    print(f"  SMA_20:           {last_period._data.get('SMA_20'):.2f}")
    print(f"  SMA_50:           {last_period._data.get('SMA_50'):.2f}")
    print(f"  SMACrossover_20:  {last_period._data.get('SMACrossover_20')}")
    print(f"  RSI_14:           {last_period._data.get('RSI_14'):.2f}")


def demo_crossover_signals():
    """Example 3: Finding crossover signals."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 3: Detecting SMA Crossover Signals")
    print("=" * 70)

    frame = TimeFrame('5T', max_periods=250)
    candles = create_sample_data(200)

    # Feed historical data
    for candle in candles:
        frame.feed(candle)

    # Add indicators (order doesn't matter!)
    frame.add_indicators_auto(
        SMACrossover(fast=10, slow=30),
        SMA(10),
        SMA(30)
    )

    # Find all crossover signals
    golden_crosses = []
    death_crosses = []

    for i, period in enumerate(frame.periods):
        signal = period._data.get('SMACrossover_10')
        if signal == 1:
            golden_crosses.append((i, period.open_date))
        elif signal == -1:
            death_crosses.append((i, period.open_date))

    print(f"\n📊 Found {len(golden_crosses)} Golden Cross signals (bullish)")
    for idx, date in golden_crosses[:3]:  # Show first 3
        print(f"   Period {idx}: {date.strftime('%Y-%m-%d %H:%M')}")

    print(f"\n📉 Found {len(death_crosses)} Death Cross signals (bearish)")
    for idx, date in death_crosses[:3]:  # Show first 3
        print(f"   Period {idx}: {date.strftime('%Y-%m-%d %H:%M')}")


def demo_auto_naming():
    """Example 4: Automatic column naming."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 4: Automatic Column Naming")
    print("=" * 70)

    frame = TimeFrame('5T', max_periods=100)
    candles = create_sample_data(80)

    for candle in candles:
        frame.feed(candle)

    # Add indicators WITHOUT specifying column names - auto-generated!
    print("\nAdding indicators with automatic naming...")

    # Use add_indicators_auto for everything
    sma20 = SMA(period=20)
    sma10 = SMA(period=10)
    rsi14 = RSI(length=14)
    crossover = SMACrossover(fast=10, slow=20)

    mapping = frame.add_indicators_auto(
        sma20,
        rsi14,
        crossover,  # Composite - depends on SMA_10, SMA_20
        sma10
    )

    for indicator in mapping:
        print(f"  {indicator} → '{mapping[indicator]}'")

    print("\n✓ All column names generated automatically!")
    print(f"\nAvailable columns: {sorted(frame.periods[-1]._data.keys())}")


def demo_error_handling():
    """Example 5: Error handling for missing dependencies."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 5: Error Handling")
    print("=" * 70)

    frame = TimeFrame('5T', max_periods=100)
    candles = create_sample_data(60)

    for candle in candles:
        frame.feed(candle)

    # Try to add crossover WITHOUT dependencies
    print("\nTrying to add SMACrossover without SMA dependencies...")
    try:
        frame.add_indicator(SMACrossover(fast=20, slow=50))
        print("  ✗ This should have failed!")
    except ValueError as e:
        print(f"  ✓ Error caught correctly: {e}")

    print("\nNow adding with auto-resolution (will work!)...")
    mapping = frame.add_indicators_auto(
        SMACrossover(fast=20, slow=50),
        SMA(20),
        SMA(50)
    )
    print(f"  ✓ Added successfully with {len(mapping)} indicators")


if __name__ == '__main__':
    demo_manual_order()
    demo_auto_resolution()
    demo_crossover_signals()
    demo_auto_naming()
    demo_error_handling()

    print("\n\n" + "=" * 70)
    print("🎉 All examples completed successfully!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • Use add_indicators_auto() for worry-free indicator addition")
    print("  • Dependencies are resolved automatically (topological sort)")
    print("  • Column names are auto-generated from indicator parameters")
    print("  • Composite indicators read from period._data directly")
    print("  • Errors are caught early with helpful messages")
    print("\nNext Steps:")
    print("  • Create your own composite indicators")
    print("  • Use SMACrossover as a template")
    print("  • Combine multiple indicators for complex strategies")
    print("=" * 70)
