export const commoditiesMarketsQuiz = [
  {
    id: 'fm-1-5-q-1',
    question:
      'Crude oil futures are in contango: 1-month at $80, 12-month at $90. Explain: (1) What contango means, (2) Why it occurs, (3) Cost of holding long futures positions, (4) How ETFs like USO are affected by roll costs.',
    sampleAnswer: `**Contango Definition:**
Forward curve slopes upward: near-term contracts cheaper than distant contracts.
- 1-month: $80/barrel
- 12-month: $90/barrel
- Difference: $10 = 12.5% annualized

**Why Contango Occurs:**

1. **Storage costs:** Holding physical oil costs money (tanks, insurance)
2. **Financing costs:** Money tied up in inventory
3. **Insurance:** Protect against fire, theft, spills
4. **Total cost of carry:** $10/barrel over 12 months

Formula: F = S × e^((r + storage - convenience) × T)
- F = futures price ($90)
- S = spot price ($80)
- r = interest rate
- storage = storage cost
- convenience = convenience yield (benefit of having physical oil)

**In contango:** storage cost > convenience yield

**Cost of Holding Long Futures:**

\`\`\`python
class CommodityFuturesRoll:
    """Calculate roll costs in contango/backwardation"""
    
    def roll_cost_contango(self, near_month_price, next_month_price, days_between=30):
        """
        When rolling futures in contango, you sell low and buy high
        """
        roll_loss_per_contract = next_month_price - near_month_price
        roll_loss_pct = (roll_loss_per_contract / near_month_price) * 100
        annualized_loss_pct = roll_loss_pct * (365 / days_between)
        
        return {
            'roll_loss': roll_loss_per_contract,
            'roll_loss_pct': roll_loss_pct,
            'annualized_loss_pct': annualized_loss_pct,
            'explanation': f'Sell {near_month_price}, buy {next_month_price} = ${roll_loss_per_contract} loss per barrel'
        }
    
    def calculate_yearly_erosion(self, contango_pct=12.5):
        """
        Estimate yearly performance drag from rolling in contango
        """
        # Assuming 12 monthly rolls
        monthly_loss_pct = contango_pct / 12
        
        # Compounding monthly losses
        remaining_value = 100
        for month in range(12):
            remaining_value *= (1 - monthly_loss_pct / 100)
        
        total_loss_pct = 100 - remaining_value
        
        return {
            'monthly_loss_pct': monthly_loss_pct,
            'annual_loss_pct': total_loss_pct,
            'interpretation': f'Even if oil price unchanged, lose {total_loss_pct:.1f}% from roll costs'
        }

# Example
roller = CommodityFuturesRoll()

# Monthly roll cost
roll = roller.roll_cost_contango(near_month_price=80, next_month_price=82, days_between=30)
print(f"Roll cost: \${roll['roll_loss']}/barrel ({roll['roll_loss_pct']:.2f}%)")
print(f"Annualized: {roll['annualized_loss_pct']:.1f}%\\n")

# Yearly erosion
erosion = roller.calculate_yearly_erosion(contango_pct=12.5)
print(f"Annual erosion from rolling: {erosion['annual_loss_pct']:.1f}%")
print(f"{erosion['interpretation']}")
\`\`\`

**Impact on ETFs (USO Example):**

USO (United States Oil Fund) holds front-month oil futures.

**Problem in Contango:**
- Every month: Sell expiring contract (cheap), buy next month (expensive)
- Roll cost: ~1-2% per month in normal contango, 5-10% in extreme contango
- Annual drag: 12-25% even if oil prices unchanged

**2020 Example (Extreme Contango):**
- April 2020: Front month at $20, next month at $30 (50% contango!)
- USO had to roll: Sell $20, buy $30 = 33% loss on roll
- Even though oil prices recovered, USO severely underperformed spot oil

**Visualization:**
\`\`\`
Month 1: Oil at $80, USO holds 1-month futures at $80
Month 2: Roll to next month at $82 (2.5% loss)
Month 3: Roll again at $84 (2.4% loss)
...
After 12 months: Oil still at $80, but USO down ~12% from roll costs
\`\`\`

**Alternatives to Avoid Roll Costs:**
1. **Physical ETFs:** Hold oil in tanks (BCOM, but has storage costs)
2. **Equity proxies:** Oil company stocks (XLE)
3. **Longer-dated futures:** Less frequent rolls, but less liquid
4. **Spot proxy instruments:** Synthetic products tracking spot

**Key Insight:**
Contango = upward-sloping curve. Holding long futures in contango has negative roll yield (sell low, buy high every month). ETFs like USO can dramatically underperform oil prices in sustained contango.`,
    keyPoints: [
      'Contango: Near-term cheaper than distant (1M: $80, 12M: $90)',
      'Caused by: Storage costs + financing > convenience yield',
      'Roll cost: Sell expiring cheap contract, buy next expensive one (negative yield)',
      'USO in 2020: Lost 33% on single roll due to extreme contango',
      'Avoid: Use equity proxies, physical ETFs, or longer-dated futures',
    ],
  },
  {
    id: 'fm-1-5-q-2',
    question:
      'Backwardation vs contango: Gold is in backwardation (near > distant), oil in contango (near < distant). Explain: (1) Economic meaning of each, (2) Which is normal for each commodity, (3) Trading strategies for each scenario.',
    sampleAnswer: `**Definitions:**

**Backwardation:** Near-term contracts trade ABOVE distant contracts
- Gold: 1-month at $2000, 12-month at $1980
- Downward-sloping forward curve

**Contango:** Near-term contracts trade BELOW distant contracts
- Oil: 1-month at $80, 12-month at $90
- Upward-sloping forward curve

**Economic Meaning:**

**Contango:**
- Cost of carry > convenience yield
- Supply plentiful → easy to store → futures price = spot + storage/financing
- Market expects stable/higher future prices
- Example: Oil in normal times (plenty of storage)

**Backwardation:**
- Convenience yield > cost of carry
- Supply tight → buyers pay premium for immediate delivery
- Market expects falling prices OR high current demand
- Example: Gold during crisis (everyone wants physical NOW)

**What's Normal for Each:**

\`\`\`python
# Typical term structures

commodities_normal_state = {
    'Crude Oil': {
        'normal': 'contango',
        'reason': 'Storage costs + financing dominate',
        'typical_carry': '5-10% annual'
    },
    'Natural Gas': {
        'normal': 'contango (winter) or backwardation (summer)',
        'reason': 'Seasonal demand patterns',
        'typical_carry': 'Varies by season'
    },
    'Gold': {
        'normal': 'slight contango',
        'reason': 'Low storage cost, but financing cost exists',
        'typical_carry': '1-2% annual (~ interest rate)'
    },
    'Agricultural': {
        'normal': 'contango after harvest, backwardation before',
        'reason': 'Seasonal production cycles',
        'typical_carry': 'Highly seasonal'
    },
    'Copper': {
        'normal': 'contango',
        'reason': 'Industrial metal, storage costs',
        'typical_carry': '3-5% annual'
    }
}
\`\`\`

**Gold typically in contango:**
- Financing cost (borrow money to buy gold): ~5%
- Storage cost: ~0.5%
- Total carry: ~5.5% annual
- BUT: During crises, flips to backwardation (everyone wants physical)

**Oil typically in contango:**
- Storage: ~10% annual
- Financing: ~5%
- BUT: During supply disruptions, flips to backwardation (premium for immediate delivery)

**Trading Strategies:**

**Strategy 1: Contango → Long Spot, Short Futures (Cash-and-Carry Arbitrage)**
\`\`\`python
def cash_and_carry_arbitrage():
    """
    Buy physical commodity, sell futures, capture spread
    """
    spot_oil = 80
    futures_12m = 90
    storage_cost = 6  # $6/barrel for 12 months
    financing_cost = 4  # 5% on $80
    
    # Buy spot
    cash_outflow = spot_oil
    
    # Sell futures (locked in $90 selling price)
    locked_in_price = futures_12m
    
    # After 12 months
    proceeds = locked_in_price - storage_cost - financing_cost
    profit = proceeds - cash_outflow
    return_pct = (profit / cash_outflow) * 100
    
    print(f"Cash-and-Carry:")
    print(f"  Buy spot: \${cash_outflow}")
    print(f"  Sell 12M futures: \${locked_in_price}")
    print(f"  Storage + financing: \${storage_cost + financing_cost}")
    print(f"  Profit: \${profit} ({return_pct:.1f}% return)")
    print(f"  Riskless if storage/financing costs are fixed!")
    
    return profit

profit = cash_and_carry_arbitrage()
# Output: $0 profit if market is efficient (contango = carry costs)
# Positive profit if contango > carry costs (arbitrage opportunity)
\`\`\`

**Strategy 2: Backwardation → Calendar Spreads (Long Near, Short Far)**
\`\`\`python
def backwardation_calendar_spread():
    """
    Exploit backwardation by longing near, shorting far
    """
    near_month = 2000  # Gold 1-month
    far_month = 1980   # Gold 12-month
    spread = near_month - far_month  # $20 backwardation
    
    # Position: Long 1-month, short 12-month
    print(f"Calendar Spread in Backwardation:")
    print(f"  Long near: \${near_month}")
    print(f"  Short far: \${far_month}")
    print(f"  Initial spread: \${spread}")
    print(f"\\nScenario 1: Backwardation persists")
    print(f"  Roll near month forward each month")
    print(f"  Capture positive roll yield (~$1.67/month)")
    print(f"  Annualized: ~10% on capital")
    
    print(f"\\nScenario 2: Curve normalizes (contango)")
    print(f"  Spread narrows → profit as near falls, far rises")
    print(f"  Close spread for profit")
\`\`\`

**Strategy 3: Contango → Short Futures ETFs**
- Short USO (oil ETF) in sustained contango
- ETF bleeds 1-2%/month from roll costs
- Profit from decay even if oil prices stable

**Strategy 4: Backwardation → Long Futures, Roll Continuously**
- Long near-month futures in backwardation
- Positive roll yield: Sell high (expiring), buy low (next month)
- Capture +1-2%/month roll yield

**Risk Management:**
- Contango can persist for years (oil 2015-2019)
- Backwardation can flip to contango overnight (supply shock resolved)
- Arbitrage requires physical infrastructure (storage, transport)

**Bottom Line:**
Contango = normal for most commodities (carry costs). Trade: cash-and-carry arb if profitable.
Backwardation = tight supply. Trade: long near/short far calendar spreads, capture positive roll yield.`,
    keyPoints: [
      'Contango: Near < distant (cost of carry). Normal for oil, metals.',
      'Backwardation: Near > distant (tight supply). Abnormal, signals shortage.',
      'Contango strategy: Buy physical, sell futures (cash-and-carry arbitrage)',
      'Backwardation strategy: Long near, short far (positive roll yield)',
      'ETF impact: Contango hurts long ETFs, backwardation helps',
    ],
  },
  {
    id: 'fm-1-5-q-3',
    question:
      'Design a commodity futures trading system that monitors term structures, detects contango/backwardation, calculates roll costs, identifies arbitrage opportunities, and provides trading recommendations. Include curve analysis, cost calculations, and signal generation.',
    sampleAnswer: `**Commodity Trading System Architecture:**

\`\`\`python
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime, timedelta

@dataclass
class FuturesContract:
    commodity: str
    expiry_date: datetime
    price: float
    volume: int
    open_interest: int

class TermStructureAnalyzer:
    """
    Analyze futures term structure
    Detect contango, backwardation, shape changes
    """
    
    def __init__(self):
        self.history = []  # Historical term structures
    
    def analyze_curve(self, contracts: List[FuturesContract]) -> Dict:
        """
        Analyze current term structure
        """
        # Sort by expiry
        sorted_contracts = sorted(contracts, key=lambda c: c.expiry_date)
        
        if len(sorted_contracts) < 2:
            return {'error': 'Need at least 2 contracts'}
        
        # Get front month and 12-month contracts
        front = sorted_contracts[0]
        try:
            twelve_month = [c for c in sorted_contracts if (c.expiry_date - front.expiry_date).days >= 330][0]
        except IndexError:
            twelve_month = sorted_contracts[-1]
        
        # Calculate contango/backwardation
        spread = twelve_month.price - front.price
        spread_pct = (spread / front.price) * 100
        
        days_between = (twelve_month.expiry_date - front.expiry_date).days
        annualized_pct = spread_pct * (365 / days_between)
        
        # Determine market state
        if spread > 0:
            state = 'contango'
            roll_impact = 'negative'  # Sell low, buy high when rolling
        else:
            state = 'backwardation'
            roll_impact = 'positive'  # Sell high, buy low when rolling
        
        # Calculate slope
        prices = [c.price for c in sorted_contracts]
        slope = np.polyfit(range(len(prices)), prices, 1)[0]
        
        return {
            'state': state,
            'front_price': front.price,
            'back_price': twelve_month.price,
            'spread': spread,
            'spread_pct': spread_pct,
            'annualized_pct': annualized_pct,
            'roll_impact': roll_impact,
            'slope': slope,
            'curve_shape': self._describe_shape(sorted_contracts)
        }
    
    def _describe_shape(self, contracts: List[FuturesContract]) -> str:
        """Describe curve shape beyond simple contango/backwardation"""
        prices = [c.price for c in contracts]
        
        # Check if curve is relatively flat
        price_range = max(prices) - min(prices)
        if price_range / prices[0] < 0.02:
            return 'flat'
        
        # Check if curve is steepening or flattening
        first_half_slope = (prices[len(prices)//2] - prices[0]) / (len(prices)//2)
        second_half_slope = (prices[-1] - prices[len(prices)//2]) / (len(prices) - len(prices)//2)
        
        if abs(second_half_slope) > abs(first_half_slope) * 1.5:
            return 'steepening'
        elif abs(second_half_slope) < abs(first_half_slope) * 0.67:
            return 'flattening'
        
        return 'linear'
    
    def calculate_roll_cost(self, front_price: float, next_price: float, days_to_roll: int) -> Dict:
        """
        Calculate cost of rolling from front month to next month
        """
        roll_cost = next_price - front_price
        roll_cost_pct = (roll_cost / front_price) * 100
        
        # Annualize assuming monthly rolls
        monthly_roll_pct = roll_cost_pct if days_to_roll <= 30 else roll_cost_pct * (30 / days_to_roll)
        annual_drag_pct = monthly_roll_pct * 12
        
        return {
            'roll_cost': roll_cost,
            'roll_cost_pct': roll_cost_pct,
            'monthly_roll_pct': monthly_roll_pct,
            'annual_drag_pct': annual_drag_pct,
            'interpretation': f'{"Negative" if roll_cost > 0 else "Positive"} roll yield: {monthly_roll_pct:.2f}%/month'
        }

class ArbitrageDetector:
    """
    Detect arbitrage opportunities in commodity markets
    """
    
    def detect_cash_and_carry(self, spot_price: float, futures_price: float, 
                              days_to_expiry: int, storage_cost_annual: float,
                              financing_rate: float) -> Dict:
        """
        Detect cash-and-carry arbitrage opportunity
        
        Buy physical, sell futures, hold to expiry
        Profit if: futures_price > spot + carry_costs
        """
        # Calculate total carry costs
        time_fraction = days_to_expiry / 365
        storage_cost = spot_price * (storage_cost_annual / 100) * time_fraction
        financing_cost = spot_price * (financing_rate / 100) * time_fraction
        total_carry = storage_cost + financing_cost
        
        # Fair futures price = spot + carry
        fair_futures_price = spot_price + total_carry
        
        # Arbitrage profit
        arbitrage_profit = futures_price - fair_futures_price
        arbitrage_profit_pct = (arbitrage_profit / spot_price) * 100
        
        # Annualized return
        annualized_return = arbitrage_profit_pct * (365 / days_to_expiry)
        
        opportunity = arbitrage_profit > 0
        
        return {
            'opportunity': opportunity,
            'spot_price': spot_price,
            'futures_price': futures_price,
            'fair_futures_price': fair_futures_price,
            'total_carry_cost': total_carry,
            'arbitrage_profit': arbitrage_profit,
            'arbitrage_profit_pct': arbitrage_profit_pct,
            'annualized_return': annualized_return,
            'trade': {
                'action': 'BUY spot, SELL futures' if opportunity else 'No arbitrage',
                'profit_per_unit': arbitrage_profit if opportunity else 0
            }
        }
    
    def detect_calendar_spread_opportunity(self, near_price: float, far_price: float, 
                                           historical_spread_mean: float,
                                           historical_spread_std: float) -> Dict:
        """
        Detect calendar spread trading opportunities
        
        Trade when spread deviates significantly from historical norm
        """
        current_spread = near_price - far_price
        z_score = (current_spread - historical_spread_mean) / historical_spread_std
        
        # Signal if spread is > 2 std devs from mean
        if z_score > 2:
            signal = 'LONG spread (long near, short far)'
            rationale = 'Spread too wide, expect convergence'
        elif z_score < -2:
            signal = 'SHORT spread (short near, long far)'
            rationale = 'Spread too narrow, expect widening'
        else:
            signal = 'No trade'
            rationale = 'Spread within normal range'
        
        return {
            'near_price': near_price,
            'far_price': far_price,
            'spread': current_spread,
            'historical_mean': historical_spread_mean,
            'z_score': z_score,
            'signal': signal,
            'rationale': rationale
        }

class TradingSignalGenerator:
    """
    Generate trading signals based on term structure analysis
    """
    
    def __init__(self, analyzer: TermStructureAnalyzer, arb_detector: ArbitrageDetector):
        self.analyzer = analyzer
        self.arb_detector = arb_detector
    
    def generate_signals(self, contracts: List[FuturesContract], 
                        spot_price: float, storage_cost: float,
                        financing_rate: float) -> List[Dict]:
        """
        Generate all trading signals
        """
        signals = []
        
        # Analyze term structure
        curve = self.analyzer.analyze_curve(contracts)
        
        # Signal 1: Contango/Backwardation position
        if curve['state'] == 'contango' and abs(curve['annualized_pct']) > 10:
            signals.append({
                'type': 'Term Structure',
                'signal': 'AVOID long futures ETFs',
                'rationale': f'Heavy contango ({curve["annualized_pct"]:.1f}% annual drag)',
                'priority': 'HIGH'
            })
        elif curve['state'] == 'backwardation' and abs(curve['annualized_pct']) > 10:
            signals.append({
                'type': 'Term Structure',
                'signal': 'LONG near-month futures',
                'rationale': f'Strong backwardation (+{abs(curve["annualized_pct"]):.1f}% roll yield)',
                'priority': 'HIGH'
            })
        
        # Signal 2: Arbitrage opportunities
        if len(contracts) >= 2:
            front = sorted(contracts, key=lambda c: c.expiry_date)[0]
            days_to_expiry = (front.expiry_date - datetime.now()).days
            
            arb = self.arb_detector.detect_cash_and_carry(
                spot_price, front.price, days_to_expiry,
                storage_cost, financing_rate
            )
            
            if arb['opportunity'] and arb['annualized_return'] > 5:
                signals.append({
                    'type': 'Arbitrage',
                    'signal': arb['trade']['action'],
                    'rationale': f'Cash-and-carry: {arb["annualized_return"]:.1f}% annualized',
                    'priority': 'CRITICAL'
                })
        
        # Signal 3: Curve shape changes
        if curve['curve_shape'] == 'steepening' and curve['state'] == 'contango':
            signals.append({
                'type': 'Curve Dynamics',
                'signal': 'Consider SHORT calendar spreads',
                'rationale': 'Curve steepening in contango (contango increasing)',
                'priority': 'MEDIUM'
            })
        
        return signals

# Example Usage
print("=== Commodity Futures Trading System ===\\n")

# Create sample contracts
contracts = [
    FuturesContract('CL', datetime.now() + timedelta(days=30), 80.0, 100000, 500000),
    FuturesContract('CL', datetime.now() + timedelta(days=60), 82.0, 80000, 400000),
    FuturesContract('CL', datetime.now() + timedelta(days=90), 84.0, 60000, 300000),
    FuturesContract('CL', datetime.now() + timedelta(days=180), 88.0, 40000, 200000),
    FuturesContract('CL', datetime.now() + timedelta(days=365), 90.0, 30000, 150000),
]

# Analyze
analyzer = TermStructureAnalyzer()
curve = analyzer.analyze_curve(contracts)

print(f"Term Structure Analysis:")
print(f"  State: {curve['state'].upper()}")
print(f"  Front month: \${curve['front_price']: .2f
    }")
print(f"  12-month: \${curve['back_price']:.2f}")
print(f"  Spread: \${curve['spread']:.2f} ({curve['spread_pct']:.1f}%)")
print(f"  Annualized: {curve['annualized_pct']:.1f}%")
print(f"  Roll impact: {curve['roll_impact']}")
print(f"  Curve shape: {curve['curve_shape']}\\n")

# Roll cost
roll = analyzer.calculate_roll_cost(80.0, 82.0, 30)
print(f"Roll Cost Analysis:")
print(f"  {roll['interpretation']}")
print(f"  Annual drag: {roll['annual_drag_pct']:.1f}%\\n")

# Arbitrage detection
arb_detector = ArbitrageDetector()
arb = arb_detector.detect_cash_and_carry(
        spot_price = 79.0,
        futures_price = 90.0,
        days_to_expiry = 365,
        storage_cost_annual = 8.0,
        financing_rate = 5.0
    )

print(f"Arbitrage Detection:")
print(f"  Opportunity: {arb['opportunity']}")
print(f"  Trade: {arb['trade']['action']}")
print(f"  Profit: \${arb['arbitrage_profit']:.2f}/barrel ({arb['annualized_return']:.1f}% annual)\\n")

# Generate signals
signal_gen = TradingSignalGenerator(analyzer, arb_detector)
signals = signal_gen.generate_signals(contracts, 79.0, 8.0, 5.0)

print(f"Trading Signals:")
for signal in signals:
    print(f"  [{signal['priority']}] {signal['type']}: {signal['signal']}")
print(f"    {signal['rationale']}")
\`\`\`

**System Components:**
1. Term structure analyzer (contango/backwardation detection)
2. Roll cost calculator
3. Arbitrage detector (cash-and-carry, calendar spreads)
4. Signal generator (trading recommendations)
5. Historical tracking (curve shape evolution)

**Production Enhancements:**
- Real-time data feeds (Bloomberg, Refinitiv)
- Machine learning for curve prediction
- Execution algorithms (minimize market impact)
- Risk management (position limits, stop losses)
- Backtesting framework`,
    keyPoints: [
      'Term structure: Analyze contango/backwardation, calculate annualized impact',
      'Roll cost: -1 to -2% monthly in contango hurts long positions',
      'Arbitrage: Buy spot + sell futures if futures > spot + carry costs',
      'Calendar spreads: Trade spread deviations from historical mean',
      'Signals: Avoid long ETFs in heavy contango, capture roll yield in backwardation',
    ],
  },
];
