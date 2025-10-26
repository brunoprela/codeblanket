export const marketMakingStrategies = {
    title: 'Market Making Strategies',
    slug: 'market-making-strategies',
    description:
        'Master market making: bid-ask spread management, inventory risk, adverse selection, and high-frequency market making',
    content: `
# Market Making Strategies

## Introduction: Liquidity Provision as a Business

Market makers are the invisible backbone of financial markets. Every time you place an order, a market maker is often on the other side, providing liquidity by continuously quoting bid and ask prices. Firms like Citadel Securities (27% of US equity volume), Virtu Financial, and Jane Street have built multi-billion dollar businesses on market making.

**What you'll learn:**
- Bid-ask spread dynamics and optimal pricing
- Inventory risk management and position skewing
- Adverse selection and information asymmetry
- Order flow toxicity and smart order routing
- High-frequency market making infrastructure

**Why this matters for engineers:**
- Market making requires sophisticated real-time systems
- Microsecond latency directly impacts profitability
- Risk management is mission-critical (bankruptcy risk from runaway inventory)
- High Sharpe ratios (2-5) but requires significant capital and infrastructure

**Performance Characteristics:**
- **Sharpe Ratio**: 2-5 (excellent risk-adjusted returns)
- **Win Rate**: 80-95% (most trades profitable, but small margins)
- **Holding Period**: Seconds to minutes (high turnover)
- **Capital Requirement**: $10M+ (meaningful scale)
- **Infrastructure Cost**: $50K-$500K/month (co-location, data feeds, technology)

---

## The Market Making Business Model

### How Market Makers Profit

**Core Equation:**
\`\`\`
Profit = Spread Revenue - Inventory Cost - Adverse Selection - Operational Costs
\`\`\`

**Revenue Sources:**
1. **Bid-Ask Spread**: Buy at bid, sell at ask, capture spread
2. **Rebates**: Exchange rebates for providing liquidity ($0.0020-$0.0030 per share)
3. **Payment for Order Flow**: Retail brokers pay for their orders

**Cost Sources:**
1. **Inventory Risk**: Price moves while holding inventory
2. **Adverse Selection**: Informed traders trade against you
3. **Technology**: Co-location, data feeds, infrastructure
4. **Regulatory**: Compliance, market data fees

\`\`\`python
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
from collections import deque
import logging

class Side(Enum):
    """Order side"""
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    """Order type"""
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    CANCEL = "CANCEL"

@dataclass
class Quote:
    """
    Bid/Ask quote
    """
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    timestamp: datetime
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price"""
        return (self.bid_price + self.ask_price) / 2
    
    @property
    def spread(self) -> float:
        """Calculate spread"""
        return self.ask_price - self.bid_price
    
    @property
    def spread_bps(self) -> float:
        """Calculate spread in basis points"""
        return (self.spread / self.mid_price) * 10000

@dataclass
class Trade:
    """
    Executed trade
    """
    timestamp: datetime
    price: float
    size: int
    side: Side
    is_aggressive: bool  # True if we lifted offer or hit bid
    
@dataclass
class InventoryPosition:
    """
    Current inventory position
    """
    symbol: str
    quantity: int  # Positive = long, negative = short
    avg_price: float
    unrealized_pnl: float
    realized_pnl: float
    
    def mark_to_market(self, current_price: float) -> float:
        """Calculate current unrealized P&L"""
        self.unrealized_pnl = (current_price - self.avg_price) * self.quantity
        return self.unrealized_pnl
    
    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)"""
        return self.realized_pnl + self.unrealized_pnl

class MarketMakingEngine:
    """
    Core market making engine
    
    Responsibilities:
    1. Quote management (bid/ask pricing)
    2. Inventory risk management
    3. P&L tracking
    4. Risk limits enforcement
    """
    
    def __init__(self,
                 symbol: str,
                 target_spread_bps: float = 5.0,
                 max_inventory: int = 1000,
                 max_position_value: float = 100_000,
                 inventory_skew_factor: float = 0.5):
        """
        Initialize market making engine
        
        Args:
            symbol: Trading symbol
            target_spread_bps: Target bid-ask spread in basis points
            max_inventory: Maximum inventory (shares)
            max_position_value: Maximum position value (dollars)
            inventory_skew_factor: How much to skew quotes per unit inventory
        """
        self.symbol = symbol
        self.target_spread_bps = target_spread_bps
        self.max_inventory = max_inventory
        self.max_position_value = max_position_value
        self.inventory_skew_factor = inventory_skew_factor
        
        # State
        self.inventory = 0
        self.avg_cost = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        
        # Order management
        self.active_bid_order = None
        self.active_ask_order = None
        
        # Trade history
        self.trades: List[Trade] = []
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def calculate_fair_value(self, market_data: Dict) -> float:
        """
        Calculate fair value estimate
        
        Can use:
        - Mid price
        - Microprice (weighted by size)
        - VWAP
        - External signals
        
        Args:
            market_data: Current market data
            
        Returns:
            Fair value estimate
        """
        bid = market_data['bid']
        ask = market_data['ask']
        bid_size = market_data['bid_size']
        ask_size = market_data['ask_size']
        
        # Microprice: weighted by opposite side size
        # If bid_size >> ask_size, more buying pressure → fair value closer to ask
        total_size = bid_size + ask_size
        if total_size > 0:
            microprice = (bid * ask_size + ask * bid_size) / total_size
        else:
            microprice = (bid + ask) / 2
        
        return microprice
    
    def calculate_target_spread(self,
                                fair_value: float,
                                volatility: float,
                                volume: float,
                                order_flow_imbalance: float) -> float:
        """
        Calculate optimal spread dynamically
        
        Factors:
        1. Volatility: Wider spread in high vol
        2. Volume: Tighter spread in high volume (more competition)
        3. Order flow: Widen if imbalanced (toxic flow)
        4. Inventory: Widen to slow trading when at limits
        
        Args:
            fair_value: Current fair value
            volatility: Recent volatility
            volume: Recent volume
            order_flow_imbalance: Buy vs sell imbalance
            
        Returns:
            Target spread in dollars
        """
        # Base spread (in bps)
        base_spread_bps = self.target_spread_bps
        
        # Volatility adjustment: wider spread in high vol
        avg_volatility = 0.02  # 2% daily vol average
        vol_adjustment = volatility / avg_volatility
        
        # Volume adjustment: tighter in high volume
        avg_volume = 1_000_000  # shares
        volume_adjustment = np.sqrt(avg_volume / max(volume, 1))
        
        # Order flow adjustment: widen if imbalanced (toxic)
        imbalance_adjustment = 1 + abs(order_flow_imbalance) * 2
        
        # Inventory adjustment: widen when near limits
        inventory_pct = abs(self.inventory) / self.max_inventory
        inventory_adjustment = 1 + inventory_pct
        
        # Combined adjustment
        adjusted_spread_bps = (
            base_spread_bps * 
            vol_adjustment * 
            volume_adjustment * 
            imbalance_adjustment * 
            inventory_adjustment
        )
        
        # Convert to dollars
        spread_dollars = fair_value * (adjusted_spread_bps / 10000)
        
        # Ensure minimum spread (tick size)
        min_spread = 0.01  # $0.01 for stocks > $1
        spread_dollars = max(spread_dollars, min_spread)
        
        return spread_dollars
    
    def calculate_inventory_skew(self, fair_value: float) -> Tuple[float, float]:
        """
        Calculate how much to skew bid/ask based on inventory
        
        Logic:
        - Long inventory (positive): Shift both bid/ask DOWN to encourage selling
        - Short inventory (negative): Shift both bid/ask UP to encourage buying
        - Zero inventory: No skew
        
        Args:
            fair_value: Current fair value
            
        Returns:
            (bid_skew, ask_skew) in dollars
        """
        # Inventory as percentage of max
        inventory_pct = self.inventory / self.max_inventory
        
        # Skew in basis points
        skew_bps = inventory_pct * self.inventory_skew_factor * 100
        
        # Convert to dollars
        skew_dollars = fair_value * (skew_bps / 10000)
        
        # Skew is same for both bid and ask (shifts entire spread)
        return -skew_dollars, -skew_dollars
    
    def generate_quotes(self, market_data: Dict) -> Optional[Quote]:
        """
        Generate bid/ask quotes
        
        Args:
            market_data: Current market data
            
        Returns:
            Quote object or None if should not quote
        """
        # Calculate fair value
        fair_value = self.calculate_fair_value(market_data)
        
        # Calculate target spread
        spread = self.calculate_target_spread(
            fair_value=fair_value,
            volatility=market_data.get('volatility', 0.02),
            volume=market_data.get('volume', 1_000_000),
            order_flow_imbalance=market_data.get('imbalance', 0.0)
        )
        
        # Calculate inventory skew
        bid_skew, ask_skew = self.calculate_inventory_skew(fair_value)
        
        # Calculate bid/ask
        half_spread = spread / 2
        bid = fair_value - half_spread + bid_skew
        ask = fair_value + half_spread + ask_skew
        
        # Round to tick size
        tick_size = 0.01
        bid = np.floor(bid / tick_size) * tick_size
        ask = np.ceil(ask / tick_size) * tick_size
        
        # Check if we should quote (risk limits)
        if not self.should_quote(market_data):
            return None
        
        # Determine quote sizes
        bid_size, ask_size = self.calculate_quote_sizes()
        
        quote = Quote(
            bid_price=bid,
            ask_price=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            timestamp=datetime.now()
        )
        
        self.logger.info(
            f"Quote: {quote.bid_price:.2f} x {quote.ask_price:.2f} "
            f"(spread: {quote.spread_bps:.1f}bps, inventory: {self.inventory})"
        )
        
        return quote
    
    def calculate_quote_sizes(self) -> Tuple[int, int]:
        """
        Calculate bid/ask sizes based on inventory
        
        Returns:
            (bid_size, ask_size)
        """
        base_size = 100  # shares
        
        # Reduce size when inventory is high
        inventory_pct = abs(self.inventory) / self.max_inventory
        size_reduction = 1 - (inventory_pct * 0.5)  # Reduce up to 50%
        
        adjusted_size = int(base_size * size_reduction)
        adjusted_size = max(adjusted_size, 1)  # Minimum 1 share
        
        # Asymmetric sizing based on inventory
        if self.inventory > 0:  # Long inventory
            bid_size = int(adjusted_size * 0.5)  # Smaller bid
            ask_size = int(adjusted_size * 1.5)  # Larger ask
        elif self.inventory < 0:  # Short inventory
            bid_size = int(adjusted_size * 1.5)  # Larger bid
            ask_size = int(adjusted_size * 0.5)  # Smaller ask
        else:  # Neutral
            bid_size = adjusted_size
            ask_size = adjusted_size
        
        return max(bid_size, 1), max(ask_size, 1)
    
    def should_quote(self, market_data: Dict) -> bool:
        """
        Determine if should quote in current conditions
        
        Don't quote if:
        1. Inventory at limit
        2. Market halted
        3. Extreme volatility
        4. Near close
        
        Args:
            market_data: Current market data
            
        Returns:
            True if should quote
        """
        # Check inventory limits
        if abs(self.inventory) >= self.max_inventory:
            self.logger.warning("Inventory at limit, not quoting")
            return False
        
        # Check position value limit
        current_price = market_data.get('mid_price', 0)
        position_value = abs(self.inventory * current_price)
        if position_value >= self.max_position_value:
            self.logger.warning("Position value at limit, not quoting")
            return False
        
        # Check if market halted
        if market_data.get('halted', False):
            self.logger.warning("Market halted, not quoting")
            return False
        
        # Check extreme volatility
        volatility = market_data.get('volatility', 0)
        if volatility > 0.10:  # 10% daily vol
            self.logger.warning("Extreme volatility, not quoting")
            return False
        
        return True
    
    def handle_trade_execution(self,
                               side: Side,
                               price: float,
                               size: int,
                               is_aggressive: bool = False):
        """
        Handle trade execution and update inventory
        
        Args:
            side: Side we traded (BUY or SELL)
            price: Execution price
            size: Execution size
            is_aggressive: True if we were aggressor
        """
        trade = Trade(
            timestamp=datetime.now(),
            price=price,
            size=size,
            side=side,
            is_aggressive=is_aggressive
        )
        
        self.trades.append(trade)
        
        # Update inventory
        if side == Side.BUY:
            # Bought shares (long)
            old_inventory = self.inventory
            new_inventory = old_inventory + size
            
            # Update average cost
            if old_inventory >= 0:
                # Adding to long or covering short
                total_cost = old_inventory * self.avg_cost + size * price
                self.avg_cost = total_cost / new_inventory if new_inventory != 0 else price
            else:
                # Covering short position
                if new_inventory >= 0:
                    # Fully covered, calculate realized P&L
                    covered_size = min(size, abs(old_inventory))
                    self.realized_pnl += covered_size * (self.avg_cost - price)
                    
                    remaining_size = size - covered_size
                    if remaining_size > 0:
                        self.avg_cost = price
                else:
                    # Partially covered
                    self.avg_cost = (old_inventory * self.avg_cost + size * price) / new_inventory
            
            self.inventory = new_inventory
            
        else:  # SELL
            # Sold shares (short)
            old_inventory = self.inventory
            new_inventory = old_inventory - size
            
            # Update average cost
            if old_inventory <= 0:
                # Adding to short or covering long
                total_cost = abs(old_inventory) * self.avg_cost + size * price
                self.avg_cost = total_cost / abs(new_inventory) if new_inventory != 0 else price
            else:
                # Covering long position
                if new_inventory <= 0:
                    # Fully covered, calculate realized P&L
                    covered_size = min(size, abs(old_inventory))
                    self.realized_pnl += covered_size * (price - self.avg_cost)
                    
                    remaining_size = size - covered_size
                    if remaining_size > 0:
                        self.avg_cost = price
                else:
                    # Partially covered
                    self.avg_cost = (old_inventory * self.avg_cost - size * price) / new_inventory
            
            self.inventory = new_inventory
        
        self.logger.info(
            f"Trade executed: {side.value} {size} @ {price:.2f}, "
            f"New inventory: {self.inventory}, "
            f"Realized P&L: ${self.realized_pnl: .2f
}"
        )
    
    def calculate_pnl(self, current_price: float) -> Dict[str, float]:
"""
        Calculate comprehensive P & L

Args:
current_price: Current market price

Returns:
            Dict with P & L metrics
"""
        # Unrealized P & L
if self.inventory > 0:
    self.unrealized_pnl = (current_price - self.avg_cost) * self.inventory
        elif self.inventory < 0:
self.unrealized_pnl = (self.avg_cost - current_price) * abs(self.inventory)
        else:
self.unrealized_pnl = 0
        
        # Total P & L
total_pnl = self.realized_pnl + self.unrealized_pnl
        
        # Calculate spread revenue(sum of all spread captures)
spread_revenue = 0
buy_trades = [t for t in self.trades if t.side == Side.BUY and not t.is_aggressive]
sell_trades = [t for t in self.trades if t.side == Side.SELL and not t.is_aggressive]
        
        # Pair buy and sell trades to calculate spread captured
for buy, sell in zip(buy_trades, sell_trades):
    if buy.timestamp < sell.timestamp:
        spread_captured = sell.price - buy.price
spread_revenue += spread_captured * min(buy.size, sell.size)

return {
    'realized_pnl': self.realized_pnl,
    'unrealized_pnl': self.unrealized_pnl,
    'total_pnl': total_pnl,
    'spread_revenue': spread_revenue,
    'num_trades': len(self.trades),
    'inventory': self.inventory,
    'avg_cost': self.avg_cost
}

# Example usage
if __name__ == "__main__":
    # Initialize market maker
mm = MarketMakingEngine(
    symbol = "AAPL",
    target_spread_bps = 5.0,
    max_inventory = 1000,
    max_position_value = 200_000,
    inventory_skew_factor = 0.5
)
    
    # Simulate market data
market_data = {
    'bid': 150.00,
    'ask': 150.05,
    'bid_size': 100,
    'ask_size': 100,
    'mid_price': 150.025,
    'volatility': 0.02,
    'volume': 1_000_000,
    'imbalance': 0.1,
    'halted': False
}
    
    # Generate quote
quote = mm.generate_quotes(market_data)
if quote:
    print(f"\\n=== Market Making Quote ===")
print(f"Bid: ${quote.bid_price:.2f} x {quote.bid_size}")
print(f"Ask: ${quote.ask_price:.2f} x {quote.ask_size}")
print(f"Spread: {quote.spread_bps:.2f} bps")
print(f"Mid: ${quote.mid_price:.2f}")
    
    # Simulate trade
mm.handle_trade_execution(Side.BUY, 150.00, 100, is_aggressive = False)
mm.handle_trade_execution(Side.SELL, 150.05, 100, is_aggressive = False)
    
    # Calculate P & L
pnl = mm.calculate_pnl(150.025)
print(f"\\n=== P&L Report ===")
print(f"Realized P&L: ${pnl['realized_pnl']:.2f}")
print(f"Unrealized P&L: ${pnl['unrealized_pnl']:.2f}")
print(f"Total P&L: ${pnl['total_pnl']:.2f}")
print(f"Spread Revenue: ${pnl['spread_revenue']:.2f}")
print(f"Number of Trades: {pnl['num_trades']}")
\`\`\`

---

## Adverse Selection and Information Asymmetry

### The Adverse Selection Problem

**Definition**: Adverse selection occurs when informed traders systematically trade against your quotes before you can adjust them.

**Example:**
1. News breaks: Company announces earnings beat
2. HFT sees news in 1 microsecond
3. Market maker quotes still stale (haven't updated yet)
4. HFT buys at your (now too low) ask
5. You're stuck long at below-market price
6. Result: Loss

**Cost of Adverse Selection:**
- Can consume 30-50% of gross spread revenue
- Worse for slower market makers
- Mitigated by fast reactions and predictive models

\`\`\`python
class AdverseSelectionProtection:
    """
    Protect against adverse selection
    
    Strategies:
    1. Fast quote updates (<1ms)
    2. Predictive models (detect informed flow)
    3. Order flow toxicity detection
    4. Strategic quote cancellation
    """
    
    def __init__(self):
        self.recent_trades = deque(maxlen=1000)
        self.cancel_threshold = 0.7  # Toxicity threshold
        
    def calculate_order_flow_toxicity(self,
                                     recent_trades: List[Trade],
                                     time_window: timedelta = timedelta(seconds=10)) -> float:
        """
        Calculate Volume-Synchronized Probability of Informed Trading (VPIN)
        
        High VPIN = toxic flow (informed traders present)
        Low VPIN = non-toxic flow (retail, random)
        
        Args:
            recent_trades: Recent trade list
            time_window: Time window for analysis
            
        Returns:
            Toxicity score 0-1 (higher = more toxic)
        """
        if len(recent_trades) < 10:
            return 0.0
        
        # Separate buy and sell volume
        now = datetime.now()
        recent = [t for t in recent_trades if now - t.timestamp < time_window]
        
        buy_volume = sum(t.size for t in recent if t.side == Side.BUY)
        sell_volume = sum(t.size for t in recent if t.side == Side.SELL)
        
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return 0.0
        
        # VPIN = |buy_volume - sell_volume| / total_volume
        vpin = abs(buy_volume - sell_volume) / total_volume
        
        return vpin
    
    def should_cancel_quotes(self,
                            toxicity: float,
                            price_momentum: float,
                            inventory: int) -> bool:
        """
        Determine if should cancel quotes to avoid adverse selection
        
        Cancel when:
        1. Toxicity high (informed flow)
        2. Strong price momentum
        3. Inventory at limits (can't absorb more)
        
        Args:
            toxicity: Order flow toxicity (0-1)
            price_momentum: Recent price momentum
            inventory: Current inventory
            
        Returns:
            True if should cancel quotes
        """
        # High toxicity threshold
        if toxicity > self.cancel_threshold:
            return True
        
        # Strong momentum (may be start of trend)
        if abs(price_momentum) > 0.02:  # 2% move
            return True
        
        # Inventory at extremes
        if abs(inventory) > 800:  # 80% of max (1000)
            return True
        
        return False
    
    def estimate_adverse_selection_cost(self,
                                       trades: List[Trade],
                                       current_price: float) -> float:
        """
        Estimate cost of adverse selection
        
        Compare execution price to price shortly after (e.g., 1 second)
        If we bought and price is now lower, we were adversely selected
        
        Args:
            trades: Trade history
            current_price: Current market price
            
        Returns:
            Estimated adverse selection cost
        """
        total_cost = 0
        
        for trade in trades[-100:]:  # Last 100 trades
            time_since = (datetime.now() - trade.timestamp).total_seconds()
            
            if time_since > 1:  # After 1 second
                if trade.side == Side.BUY:
                    # We bought, if price dropped, we were adversely selected
                    cost = max(0, trade.price - current_price) * trade.size
                else:
                    # We sold, if price rose, we were adversely selected
                    cost = max(0, current_price - trade.price) * trade.size
                
                total_cost += cost
        
        return total_cost
    
    def implement_queue_position_monitoring(self,
                                           our_order_id: str,
                                           orderbook: Dict) -> bool:
        """
        Monitor queue position to detect informed flow
        
        If we're pushed back in queue (many orders ahead of us),
        it signals informed flow building
        
        Args:
            our_order_id: Our order ID
            orderbook: Current orderbook
            
        Returns:
            True if queue position deteriorated (should cancel)
        """
        # Find our order in book
        bid_queue = orderbook.get('bids', [])
        
        our_position = None
        for i, order in enumerate(bid_queue):
            if order['id'] == our_order_id:
                our_position = i
                break
        
        if our_position is None:
            return False
        
        # If pushed beyond position 10, cancel (too much flow ahead)
        if our_position > 10:
            return True
        
        return False

class PredictiveModel:
    """
    Machine learning model to predict adverse selection
    
    Features:
    - Order flow imbalance
    - Trade size
    - Time of day
    - Recent volatility
    - Queue position changes
    """
    
    def __init__(self):
        self.model = None  # Placeholder for ML model
        
    def extract_features(self, market_data: Dict, recent_trades: List[Trade]) -> np.ndarray:
        """
        Extract features for prediction
        
        Args:
            market_data: Current market data
            recent_trades: Recent trade history
            
        Returns:
            Feature vector
        """
        features = []
        
        # Order flow imbalance
        if len(recent_trades) > 0:
            buys = sum(1 for t in recent_trades[-50:] if t.side == Side.BUY)
            sells = sum(1 for t in recent_trades[-50:] if t.side == Side.SELL)
            imbalance = (buys - sells) / (buys + sells) if (buys + sells) > 0 else 0
            features.append(imbalance)
        else:
            features.append(0)
        
        # Average trade size (large trades = informed)
        if len(recent_trades) > 0:
            avg_size = np.mean([t.size for t in recent_trades[-50:]])
            features.append(avg_size / 100)  # Normalize
        else:
            features.append(0)
        
        # Volatility
        features.append(market_data.get('volatility', 0))
        
        # Spread
        spread_bps = market_data.get('spread_bps', 5)
        features.append(spread_bps)
        
        # Time of day (more informed flow at open/close)
        hour = datetime.now().hour
        features.append(hour / 24)
        
        return np.array(features)
    
    def predict_informed_probability(self, features: np.ndarray) -> float:
        """
        Predict probability that next trade is informed
        
        Args:
            features: Feature vector
            
        Returns:
            Probability 0-1
        """
        # Placeholder: In production, use trained ML model
        # For now, simple heuristic
        
        # High imbalance = informed
        imbalance_score = abs(features[0]) * 0.4
        
        # Large trades = informed
        size_score = min(features[1], 1.0) * 0.3
        
        # High volatility = informed
        vol_score = min(features[2] / 0.05, 1.0) * 0.3
        
        probability = imbalance_score + size_score + vol_score
        
        return min(probability, 1.0)
\`\`\`

---

## Inventory Risk Management

### The Inventory Problem

Market makers must balance:
1. **Provide liquidity** (quote both sides, accumulate inventory)
2. **Manage risk** (don't get stuck with large inventory in adverse move)

**Inventory Risk Scenarios:**

**Scenario 1: Bull Run**
- You keep selling at ask (providing liquidity)
- Accumulate short inventory (-1000 shares)
- Stock rallies 5%
- Loss: -1000 × $150 × 5% = -$7,500

**Scenario 2: Flash Crash**
- You keep buying at bid (providing liquidity)
- Accumulate long inventory (+1000 shares)
- Stock crashes 10%
- Loss: +1000 × $150 × 10% = -$15,000

\`\`\`python
class InventoryRiskManager:
    """
    Advanced inventory risk management
    
    Strategies:
    1. Quote skewing (shift prices to push inventory toward zero)
    2. Dynamic hedging (hedge with futures/ETFs)
    3. Position limits (hard stops)
    4. Emergency liquidation (flatten at any cost)
    """
    
    def __init__(self,
                 max_inventory: int = 1000,
                 target_inventory: int = 0,
                 max_inventory_value: float = 200_000):
        self.max_inventory = max_inventory
        self.target_inventory = target_inventory
        self.max_inventory_value = max_inventory_value
        
    def calculate_inventory_risk(self,
                                inventory: int,
                                current_price: float,
                                volatility: float) -> Dict[str, float]:
        """
        Calculate inventory risk metrics
        
        Args:
            inventory: Current inventory
            current_price: Current price
            volatility: Daily volatility
            
        Returns:
            Risk metrics
        """
        # Position value
        position_value = abs(inventory * current_price)
        
        # Value at Risk (95% confidence, 1-day horizon)
        # VaR = Position × Price × Volatility × Z-score
        z_score_95 = 1.65  # 95% confidence
        var_1day = position_value * volatility * z_score_95
        
        # Maximum potential loss (5% move)
        max_loss_5pct = position_value * 0.05
        
        # Inventory as % of limit
        inventory_pct = abs(inventory) / self.max_inventory
        
        return {
            'position_value': position_value,
            'var_1day_95': var_1day,
            'max_loss_5pct': max_loss_5pct,
            'inventory_pct': inventory_pct,
            'risk_level': 'HIGH' if inventory_pct > 0.7 else 'MEDIUM' if inventory_pct > 0.4 else 'LOW'
        }
    
    def calculate_optimal_hedge_ratio(self,
                                     inventory: int,
                                     stock_beta: float = 1.0) -> float:
        """
        Calculate optimal hedge using futures/ETF
        
        Args:
            inventory: Current inventory
            stock_beta: Stock beta to market
            
        Returns:
            Hedge ratio (0-1, what % to hedge)
        """
        # Hedge more aggressively as inventory grows
        inventory_pct = abs(inventory) / self.max_inventory
        
        if inventory_pct < 0.3:
            hedge_ratio = 0.0  # Don't hedge, normal operations
        elif inventory_pct < 0.5:
            hedge_ratio = 0.25  # Light hedge
        elif inventory_pct < 0.7:
            hedge_ratio = 0.50  # Medium hedge
        elif inventory_pct < 0.9:
            hedge_ratio = 0.75  # Heavy hedge
        else:
            hedge_ratio = 1.0  # Full hedge
        
        # Adjust for beta
        hedge_ratio *= stock_beta
        
        return hedge_ratio
    
    def execute_hedge(self,
                     inventory: int,
                     hedge_ratio: float,
                     futures_price: float) -> int:
        """
        Execute hedge trade
        
        Args:
            inventory: Current stock inventory
            hedge_ratio: How much to hedge (0-1)
            futures_price: Current futures price
            
        Returns:
            Futures contracts to trade
        """
        # Calculate hedge size
        shares_to_hedge = int(abs(inventory) * hedge_ratio)
        
        # Convert to futures contracts (typically 100 shares per contract)
        shares_per_contract = 100
        contracts = shares_to_hedge // shares_per_contract
        
        # Direction: opposite of stock position
        # If long stock, short futures
        if inventory > 0:
            contracts = -contracts
        
        return contracts
    
    def emergency_liquidation_check(self,
                                   inventory: int,
                                   current_price: float,
                                   unrealized_pnl: float,
                                   max_loss_threshold: float = -10_000) -> bool:
        """
        Determine if emergency liquidation required
        
        Triggers:
        1. Unrealized loss exceeds threshold
        2. Inventory at absolute max
        3. Market conditions extreme
        
        Args:
            inventory: Current inventory
            current_price: Current price
            unrealized_pnl: Unrealized P&L
            max_loss_threshold: Maximum acceptable loss
            
        Returns:
            True if should liquidate immediately
        """
        # Check unrealized loss
        if unrealized_pnl < max_loss_threshold:
            return True
        
        # Check absolute inventory limit
        if abs(inventory) >= self.max_inventory:
            return True
        
        # Check position value
        position_value = abs(inventory * current_price)
        if position_value >= self.max_inventory_value:
            return True
        
        return False
    
    def calculate_liquidation_cost(self,
                                  inventory: int,
                                  current_bid: float,
                                  current_ask: float,
                                  spread_bps: float) -> float:
        """
        Estimate cost to liquidate inventory
        
        Must cross spread + market impact
        
        Args:
            inventory: Inventory to liquidate
            current_bid: Current bid
            current_ask: Current ask
            spread_bps: Current spread in bps
            
        Returns:
            Estimated liquidation cost
        """
        mid_price = (current_bid + current_ask) / 2
        
        # Cost 1: Spread crossing
        spread_cost = abs(inventory) * mid_price * (spread_bps / 10000) / 2
        
        # Cost 2: Market impact (Kyle's lambda model)
        # Impact ∝ sqrt(size)
        impact_bps = 5 * np.sqrt(abs(inventory) / 100)  # Empirical
        impact_cost = abs(inventory) * mid_price * (impact_bps / 10000)
        
        total_cost = spread_cost + impact_cost
        
        return total_cost
\`\`\`

---

## Real-World Examples

### Citadel Securities

**Scale**: 27% of US equity volume ($3+ trillion daily)

**Technology:**
- Co-location at all major exchanges
- Sub-millisecond latency
- Proprietary matching engines
- Machine learning for pricing

**Revenue Model:**
- Payment for order flow from retail brokers
- Spread capture on institutional orders
- Rebates from exchanges
- Securities lending

**Key Success Factors:**
1. **Scale**: Volume creates data, data improves models
2. **Technology**: Fastest infrastructure wins
3. **Risk Management**: Survived 2008, Flash Crash, COVID
4. **Diversification**: Equities, options, futures, FX

### Virtu Financial

**Performance**: Profitable 1,485 out of 1,486 trading days (99.9%)

**Strategy:**
- Pure market making (no directional bets)
- Global (equities, FX, commodities, crypto)
- ~400 markets worldwide
- High turnover (holds < 15 seconds average)

**Why 99.9% Win Rate:**
- Law of large numbers (millions of trades)
- Tight risk management (small positions)
- No overnight risk (flat by close)
- Diversification across markets

---

## Summary and Key Takeaways

**Market Making Works When:**
- High volume (more opportunities to capture spread)
- Reasonable volatility (not too high or low)
- Liquid markets (can hedge/exit easily)
- Fast infrastructure (minimize adverse selection)

**Market Making Fails When:**
- Runaway inventory in adverse move
- Extreme volatility (spreads too wide, no volume)
- Flash crashes (no time to react)
- Toxic order flow (constantly adversely selected)

**Critical Success Factors:**
1. **Technology**: Sub-millisecond latency required
2. **Risk Management**: Inventory limits sacred
3. **Adverse Selection**: Detect and avoid informed flow
4. **Scale**: Need volume for profitability
5. **Capital**: Minimum $10M to be competitive

**Performance Expectations:**
- Sharpe Ratio: 2-5
- Win Rate: 80-95%
- Return on Capital: 20-50% annual
- Maximum Drawdown: -10 to -20%

**Next Section:** Execution Algorithms (VWAP, TWAP, POV)
`,
};
