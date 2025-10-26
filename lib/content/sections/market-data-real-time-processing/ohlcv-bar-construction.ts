export const ohlcvBarConstruction = {
    title: 'OHLCV Bar Construction',
    id: 'ohlcv-bar-construction',
    content: `
# OHLCV Bar Construction

## Introduction

OHLCV bars (Open, High, Low, Close, Volume) are the fundamental building blocks of technical analysis and algorithmic trading. While tick data provides raw granularity, bars aggregate ticks into meaningful time windows that reveal market structure, trends, and patterns. Constructing bars correctly is crucial - errors in bar construction can invalidate entire trading strategies and backtests.

**Why Bar Construction Matters:**
- **Trading Strategies**: 90%+ of strategies use bar data, not ticks
- **Visualization**: Candlestick charts require properly constructed bars
- **Performance**: Bars reduce data volume by 100-1000× vs ticks
- **Backtesting**: Historical analysis relies on accurate bar construction
- **Real-time**: Live bars must match historical for strategy consistency

**Bar Construction in Production:**
- **Exchanges**: Publish official 1-min bars for thousands of symbols
- **Data Vendors**: Bloomberg, Refinitiv provide bars at multiple timeframes
- **Trading Platforms**: TradingView generates bars on-the-fly from ticks
- **Quant Firms**: Build custom bar types (volume, tick, dollar bars)

By the end of this section, you'll understand:
- Time-based, volume-based, and tick-based bars
- Algorithms for real-time bar construction
- Handling partial bars and late ticks
- Alternative bar types (Renko, Range, Kagi)
- Production-grade bar builders

---

## OHLCV Components

### Bar Structure

\`\`\`python
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

@dataclass
class Bar:
    """OHLCV bar with complete information"""
    symbol: str
    timestamp: datetime  # Bar start time
    interval: str  # "1min", "5min", "1hour", "1day"
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    
    # Additional fields
    trade_count: int = 0
    vwap: Optional[Decimal] = None  # Volume-weighted average price
    bid_volume: int = 0  # Buy-initiated volume
    ask_volume: int = 0  # Sell-initiated volume
    
    @property
    def body(self) -> Decimal:
        """Candlestick body size"""
        return abs(self.close - self.open)
    
    @property
    def upper_wick(self) -> Decimal:
        """Upper wick/shadow length"""
        return self.high - max(self.open, self.close)
    
    @property
    def lower_wick(self) -> Decimal:
        """Lower wick/shadow length"""
        return min(self.open, self.close) - self.low
    
    @property
    def is_bullish(self) -> bool:
        """Green candle (close > open)"""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """Red candle (close < open)"""
        return self.close < self.open
    
    @property
    def range(self) -> Decimal:
        """High-low range"""
        return self.high - self.low

# Example bar
bar = Bar(
    symbol="AAPL",
    timestamp=datetime(2024, 1, 15, 9, 30, 0),
    interval="1min",
    open=Decimal("150.25"),
    high=Decimal("150.35"),
    low=Decimal("150.20"),
    close=Decimal("150.30"),
    volume=12500,
    trade_count=234,
    vwap=Decimal("150.275")
)

print(f"Body: {bar.body}")  # 0.05
print(f"Range: {bar.range}")  # 0.15
print(f"Bullish: {bar.is_bullish}")  # True
\`\`\`

---

## Time-Based Bars

### Real-Time Time Bar Builder

\`\`\`python
from collections import defaultdict
from typing import Dict, List, Callable
import asyncio

class TimeBarBuilder:
    """Build time-based bars from tick stream"""
    
    def __init__(self, interval_seconds: int = 60):
        self.interval_seconds = interval_seconds
        
        # Current bar being built (symbol -> bar data)
        self.current_bars: Dict[str, dict] = {}
        
        # Completed bars callback
        self.bar_callbacks: List[Callable] = []
        
        # Stats
        self.bars_completed = 0
        self.ticks_processed = 0
    
    def register_callback(self, callback: Callable):
        """Register callback for completed bars"""
        self.bar_callbacks.append(callback)
    
    async def process_tick(self, tick: 'QuoteTick') -> Optional[Bar]:
        """Process tick and return completed bar if any"""
        symbol = tick.symbol
        timestamp = tick.exchange_timestamp
        
        # Determine bar window
        bar_start = self._get_bar_start(timestamp)
        
        # Initialize bar if needed
        if symbol not in self.current_bars:
            self._init_bar(symbol, bar_start, tick)
        
        current_bar = self.current_bars[symbol]
        
        # Check if tick belongs to current bar
        if bar_start == current_bar['start']:
            # Update current bar
            self._update_bar(current_bar, tick)
            self.ticks_processed += 1
            return None
        else:
            # Tick belongs to new bar - complete current bar
            completed_bar = self._complete_bar(symbol, current_bar)
            
            # Start new bar
            self._init_bar(symbol, bar_start, tick)
            
            # Execute callbacks
            for callback in self.bar_callbacks:
                await callback(completed_bar)
            
            self.bars_completed += 1
            return completed_bar
    
    def _get_bar_start(self, timestamp: datetime) -> datetime:
        """Calculate bar start time for timestamp"""
        # Round down to interval boundary
        seconds = (timestamp - datetime(1970, 1, 1)).total_seconds()
        bar_seconds = int(seconds // self.interval_seconds) * self.interval_seconds
        return datetime.utcfromtimestamp(bar_seconds)
    
    def _init_bar(self, symbol: str, bar_start: datetime, tick: 'QuoteTick'):
        """Initialize new bar"""
        mid_price = tick.mid_price
        
        self.current_bars[symbol] = {
            'start': bar_start,
            'symbol': symbol,
            'open': mid_price,
            'high': mid_price,
            'low': mid_price,
            'close': mid_price,
            'volume': tick.bid_size + tick.ask_size,
            'trade_count': 1,
            'vwap_numerator': float(mid_price) * (tick.bid_size + tick.ask_size),
            'vwap_denominator': tick.bid_size + tick.ask_size
        }
    
    def _update_bar(self, bar_data: dict, tick: 'QuoteTick'):
        """Update bar with new tick"""
        mid_price = tick.mid_price
        
        # Update high/low
        if mid_price > bar_data['high']:
            bar_data['high'] = mid_price
        if mid_price < bar_data['low']:
            bar_data['low'] = mid_price
        
        # Update close
        bar_data['close'] = mid_price
        
        # Update volume
        tick_volume = tick.bid_size + tick.ask_size
        bar_data['volume'] += tick_volume
        bar_data['trade_count'] += 1
        
        # Update VWAP
        bar_data['vwap_numerator'] += float(mid_price) * tick_volume
        bar_data['vwap_denominator'] += tick_volume
    
    def _complete_bar(self, symbol: str, bar_data: dict) -> Bar:
        """Create completed Bar object"""
        vwap = (Decimal(str(bar_data['vwap_numerator'])) / 
                Decimal(str(bar_data['vwap_denominator'])))
        
        return Bar(
            symbol=symbol,
            timestamp=bar_data['start'],
            interval=f"{self.interval_seconds}s",
            open=bar_data['open'],
            high=bar_data['high'],
            low=bar_data['low'],
            close=bar_data['close'],
            volume=bar_data['volume'],
            trade_count=bar_data['trade_count'],
            vwap=vwap
        )
    
    def force_complete_all(self) -> List[Bar]:
        """Force complete all partial bars (e.g., at market close)"""
        completed = []
        for symbol, bar_data in list(self.current_bars.items()):
            bar = self._complete_bar(symbol, bar_data)
            completed.append(bar)
        self.current_bars.clear()
        return completed

# Usage
async def bar_handler(bar: Bar):
    print(f"{bar.symbol} {bar.timestamp}: O={bar.open} H={bar.high} "
          f"L={bar.low} C={bar.close} V={bar.volume}")

builder = TimeBarBuilder(interval_seconds=60)  # 1-minute bars
builder.register_callback(bar_handler)

# Process tick stream
async def process_stream():
    for tick in tick_stream:
        completed_bar = await builder.process_tick(tick)
        # completed_bar is None until bar completes
\`\`\`

### Handling Late Ticks

\`\`\`python
class LateTickHandler:
    """Handle ticks that arrive late (after bar closed)"""
    
    def __init__(self, late_window_seconds: int = 5):
        self.late_window_seconds = late_window_seconds
        # Store completed bars temporarily for updates
        self.recent_bars: Dict[tuple, Bar] = {}  # (symbol, timestamp) -> Bar
        self.late_tick_count = 0
    
    def store_completed_bar(self, bar: Bar):
        """Store bar for potential late updates"""
        key = (bar.symbol, bar.timestamp)
        self.recent_bars[key] = bar
        
        # Clean old bars
        cutoff = datetime.now() - timedelta(seconds=self.late_window_seconds * 2)
        to_delete = [
            k for k, b in self.recent_bars.items()
            if b.timestamp < cutoff
        ]
        for k in to_delete:
            del self.recent_bars[k]
    
    def handle_late_tick(self, tick: 'QuoteTick', bar_start: datetime) -> Optional[Bar]:
        """Handle tick that belongs to already-closed bar"""
        key = (tick.symbol, bar_start)
        
        if key not in self.recent_bars:
            # Bar too old, discard tick
            self.late_tick_count += 1
            return None
        
        # Update bar
        bar = self.recent_bars[key]
        mid_price = tick.mid_price
        
        # May need to update high/low
        updated = False
        if mid_price > bar.high:
            bar.high = mid_price
            updated = True
        if mid_price < bar.low:
            bar.low = mid_price
            updated = True
        
        # Always update close (most recent tick)
        if tick.exchange_timestamp > bar.timestamp:
            bar.close = mid_price
            updated = True
        
        # Update volume
        bar.volume += tick.bid_size + tick.ask_size
        bar.trade_count += 1
        
        self.late_tick_count += 1
        
        return bar if updated else None
\`\`\`

---

## Volume-Based Bars

### Volume Bar Builder

\`\`\`python
class VolumeBarBuilder:
    """Build bars based on volume instead of time"""
    
    def __init__(self, volume_threshold: int = 10000):
        self.volume_threshold = volume_threshold
        self.current_bars: Dict[str, dict] = {}
        self.bar_callbacks: List[Callable] = []
    
    async def process_tick(self, tick: 'QuoteTick') -> Optional[Bar]:
        """Process tick and complete bar when volume threshold reached"""
        symbol = tick.symbol
        
        if symbol not in self.current_bars:
            self._init_bar(symbol, tick)
        
        current_bar = self.current_bars[symbol]
        tick_volume = tick.bid_size + tick.ask_size
        
        # Update bar
        self._update_bar(current_bar, tick)
        
        # Check if volume threshold reached
        if current_bar['volume'] >= self.volume_threshold:
            # Complete bar
            bar = self._complete_bar(symbol, current_bar)
            
            # Start new bar with overflow volume
            overflow_volume = current_bar['volume'] - self.volume_threshold
            self._init_bar(symbol, tick)
            self.current_bars[symbol]['volume'] = overflow_volume
            
            for callback in self.bar_callbacks:
                await callback(bar)
            
            return bar
        
        return None
    
    def _init_bar(self, symbol: str, tick: 'QuoteTick'):
        """Initialize new volume bar"""
        mid_price = tick.mid_price
        
        self.current_bars[symbol] = {
            'start_time': tick.exchange_timestamp,
            'symbol': symbol,
            'open': mid_price,
            'high': mid_price,
            'low': mid_price,
            'close': mid_price,
            'volume': 0,
            'trade_count': 0,
            'vwap_numerator': 0.0,
            'vwap_denominator': 0
        }
    
    def _update_bar(self, bar_data: dict, tick: 'QuoteTick'):
        """Update volume bar"""
        mid_price = tick.mid_price
        tick_volume = tick.bid_size + tick.ask_size
        
        if mid_price > bar_data['high']:
            bar_data['high'] = mid_price
        if mid_price < bar_data['low']:
            bar_data['low'] = mid_price
        
        bar_data['close'] = mid_price
        bar_data['volume'] += tick_volume
        bar_data['trade_count'] += 1
        bar_data['vwap_numerator'] += float(mid_price) * tick_volume
        bar_data['vwap_denominator'] += tick_volume
    
    def _complete_bar(self, symbol: str, bar_data: dict) -> Bar:
        """Complete volume bar"""
        vwap = Decimal(str(bar_data['vwap_numerator'] / bar_data['vwap_denominator']))
        
        return Bar(
            symbol=symbol,
            timestamp=bar_data['start_time'],
            interval=f"{self.volume_threshold}v",
            open=bar_data['open'],
            high=bar_data['high'],
            low=bar_data['low'],
            close=bar_data['close'],
            volume=bar_data['volume'],
            trade_count=bar_data['trade_count'],
            vwap=vwap
        )
\`\`\`

---

## Tick-Based Bars

\`\`\`python
class TickBarBuilder:
    """Build bars based on number of ticks (not time or volume)"""
    
    def __init__(self, tick_threshold: int = 100):
        self.tick_threshold = tick_threshold
        self.current_bars: Dict[str, dict] = {}
        self.bar_callbacks: List[Callable] = []
    
    async def process_tick(self, tick: 'QuoteTick') -> Optional[Bar]:
        """Process tick and complete bar when tick count reached"""
        symbol = tick.symbol
        
        if symbol not in self.current_bars:
            self._init_bar(symbol, tick)
        
        current_bar = self.current_bars[symbol]
        self._update_bar(current_bar, tick)
        
        if current_bar['tick_count'] >= self.tick_threshold:
            bar = self._complete_bar(symbol, current_bar)
            self._init_bar(symbol, tick)
            
            for callback in self.bar_callbacks:
                await callback(bar)
            
            return bar
        
        return None
\`\`\`

---

## Alternative Bar Types

### Dollar Bars

Bars based on dollar volume (price × volume).

\`\`\`python
class DollarBarBuilder:
    """Build bars based on dollar volume"""
    
    def __init__(self, dollar_threshold: Decimal = Decimal("1000000")):
        self.dollar_threshold = dollar_threshold  # $1M per bar
        self.current_bars: Dict[str, dict] = {}
    
    async def process_tick(self, tick: 'QuoteTick') -> Optional[Bar]:
        symbol = tick.symbol
        
        if symbol not in self.current_bars:
            self._init_bar(symbol, tick)
        
        current_bar = self.current_bars[symbol]
        
        # Calculate dollar volume for this tick
        mid_price = tick.mid_price
        tick_volume = tick.bid_size + tick.ask_size
        dollar_volume = mid_price * Decimal(tick_volume)
        
        # Update bar
        self._update_bar(current_bar, tick, dollar_volume)
        
        # Check threshold
        if current_bar['dollar_volume'] >= self.dollar_threshold:
            bar = self._complete_bar(symbol, current_bar)
            self._init_bar(symbol, tick)
            
            return bar
        
        return None
    
    def _init_bar(self, symbol: str, tick: 'QuoteTick'):
        mid_price = tick.mid_price
        self.current_bars[symbol] = {
            'start_time': tick.exchange_timestamp,
            'symbol': symbol,
            'open': mid_price,
            'high': mid_price,
            'low': mid_price,
            'close': mid_price,
            'volume': 0,
            'dollar_volume': Decimal('0'),
            'trade_count': 0
        }
    
    def _update_bar(self, bar_data: dict, tick: 'QuoteTick', dollar_volume: Decimal):
        mid_price = tick.mid_price
        
        if mid_price > bar_data['high']:
            bar_data['high'] = mid_price
        if mid_price < bar_data['low']:
            bar_data['low'] = mid_price
        
        bar_data['close'] = mid_price
        bar_data['volume'] += tick.bid_size + tick.ask_size
        bar_data['dollar_volume'] += dollar_volume
        bar_data['trade_count'] += 1
\`\`\`

### Renko Bars

Price-based bars that filter out time and volume.

\`\`\`python
class RenkoBarBuilder:
    """Build Renko bars (brick-based on price moves)"""
    
    def __init__(self, brick_size: Decimal = Decimal("0.50")):
        self.brick_size = brick_size
        self.current_price = None
        self.bricks: List[dict] = []
    
    async def process_tick(self, tick: 'QuoteTick') -> List[Bar]:
        """Process tick and return completed bricks"""
        mid_price = tick.mid_price
        
        if self.current_price is None:
            self.current_price = mid_price
            return []
        
        # Calculate price move in brick units
        price_change = mid_price - self.current_price
        bricks_moved = int(price_change / self.brick_size)
        
        if abs(bricks_moved) == 0:
            return []
        
        # Create bricks
        new_bricks = []
        for i in range(abs(bricks_moved)):
            if bricks_moved > 0:
                # Up brick
                open_price = self.current_price
                close_price = self.current_price + self.brick_size
                new_bricks.append(Bar(
                    symbol=tick.symbol,
                    timestamp=tick.exchange_timestamp,
                    interval="renko",
                    open=open_price,
                    high=close_price,
                    low=open_price,
                    close=close_price,
                    volume=0  # Renko ignores volume
                ))
                self.current_price = close_price
            else:
                # Down brick
                open_price = self.current_price
                close_price = self.current_price - self.brick_size
                new_bricks.append(Bar(
                    symbol=tick.symbol,
                    timestamp=tick.exchange_timestamp,
                    interval="renko",
                    open=open_price,
                    high=open_price,
                    low=close_price,
                    close=close_price,
                    volume=0
                ))
                self.current_price = close_price
        
        return new_bricks
\`\`\`

---

## Production Bar Builder

\`\`\`python
from enum import Enum

class BarType(Enum):
    TIME = "time"
    VOLUME = "volume"
    TICK = "tick"
    DOLLAR = "dollar"
    RENKO = "renko"

class UnifiedBarBuilder:
    """Production-grade bar builder supporting multiple bar types"""
    
    def __init__(self, bar_type: BarType, **params):
        self.bar_type = bar_type
        
        # Initialize appropriate builder
        if bar_type == BarType.TIME:
            self.builder = TimeBarBuilder(params.get('interval_seconds', 60))
        elif bar_type == BarType.VOLUME:
            self.builder = VolumeBarBuilder(params.get('volume_threshold', 10000))
        elif bar_type == BarType.TICK:
            self.builder = TickBarBuilder(params.get('tick_threshold', 100))
        elif bar_type == BarType.DOLLAR:
            self.builder = DollarBarBuilder(params.get('dollar_threshold', Decimal('1000000')))
        elif bar_type == BarType.RENKO:
            self.builder = RenkoBarBuilder(params.get('brick_size', Decimal('0.50')))
        
        # Statistics
        self.bars_completed = 0
        self.ticks_processed = 0
    
    async def process_tick(self, tick: 'QuoteTick'):
        """Process tick through appropriate builder"""
        result = await self.builder.process_tick(tick)
        self.ticks_processed += 1
        
        if result:
            if isinstance(result, list):
                self.bars_completed += len(result)
            else:
                self.bars_completed += 1
        
        return result
    
    def get_stats(self) -> dict:
        return {
            'bar_type': self.bar_type.value,
            'bars_completed': self.bars_completed,
            'ticks_processed': self.ticks_processed,
            'ticks_per_bar': (self.ticks_processed / self.bars_completed 
                             if self.bars_completed > 0 else 0)
        }

# Usage - create different bar types
time_bars = UnifiedBarBuilder(BarType.TIME, interval_seconds=60)
volume_bars = UnifiedBarBuilder(BarType.VOLUME, volume_threshold=10000)
dollar_bars = UnifiedBarBuilder(BarType.DOLLAR, dollar_threshold=Decimal('500000'))

# Process same tick stream through all builders
for tick in tick_stream:
    await time_bars.process_tick(tick)
    await volume_bars.process_tick(tick)
    await dollar_bars.process_tick(tick)
\`\`\`

---

## Best Practices

1. **Always use bar start time** - Not end time or mid-point
2. **Handle partial bars** - Last bar of day may be incomplete
3. **Account for late ticks** - Keep recent bars for updates
4. **Calculate VWAP correctly** - Use (price × volume) / total_volume
5. **Store trade count** - Useful for filtering low-activity bars
6. **Time zone consistency** - Always use UTC
7. **Validate bars** - Ensure high >= low, volume > 0
8. **Document bar construction** - Crucial for reproducibility

---

## Bar Types Comparison

| Bar Type | Pros | Cons | Use Case |
|----------|------|------|----------|
| Time | Standard, easy to understand | Irregular activity | General trading |
| Volume | Adapts to market activity | Variable time per bar | High-frequency |
| Tick | Activity-based | Ignores size | Microstructure |
| Dollar | Accounts for price × volume | Complex | Institutional |
| Renko | Filters noise | Loses timing info | Trend following |

---

## Next Steps

Now you can:
1. **Build time bars** for standard technical analysis
2. **Implement volume bars** for activity-based strategies
3. **Create custom bar types** for specific strategies
4. **Handle real-time construction** with late ticks
5. **Backtest with proper bars** - matching live construction

Bar construction is fundamental to trading - every candlestick chart starts here.
`,
};

