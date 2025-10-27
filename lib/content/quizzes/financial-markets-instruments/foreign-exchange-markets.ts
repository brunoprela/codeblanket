export const foreignExchangeMarketsQuiz = [
  {
    id: 'fm-1-4-q-1',
    question:
      'EUR/USD is quoted at 1.1000/1.1005 (bid/ask). You need to: (1) Buy €1M, (2) Sell €1M. Calculate USD amounts for each. Then explain the spread, why it exists, and how high-frequency traders profit from it.',
    sampleAnswer: `**FX Quote Convention:**
EUR/USD = 1.1000/1.1005
- Bid (1.1000): Bank buys EUR, you sell EUR → receive $1.1000 per €
- Ask (1.1005): Bank sells EUR, you buy EUR → pay $1.1005 per €

**Your Transactions:**

(1) Buy €1M (you want euros):
- You pay the ask: €1M × 1.1005 = $1,100,500

(2) Sell €1M (you want dollars):
- You receive the bid: €1M × 1.1000 = $1,100,000

**The Spread:**
- Spread = 1.1005 - 1.1000 = 0.0005 = 5 pips
- In dollars: $1,100,500 - $1,100,000 = $500 on €1M

**Why Spread Exists:**1. **Inventory risk:** Market maker holds inventory (might move against them)
2. **Adverse selection:** Informed traders (banks, HFTs) might know something
3. **Operating costs:** Technology, personnel, capital
4. **Profit margin:** Compensation for providing liquidity

**High-Frequency Trading Profit:**

\`\`\`python
class FXMarketMaker:
    """HFT market maker strategy"""
    
    def __init__(self):
        self.inventory = 0  # EUR position
        self.pnl = 0  # USD P&L
    
    def quote (self, fair_value, spread_half=0.00025):
        """
        Post bid/ask around fair value
        """
        return {
            'bid': fair_value - spread_half,
            'ask': fair_value + spread_half,
            'spread_pips': spread_half * 2 * 10000
        }
    
    def trade (self, side, amount_eur, price):
        """Execute trade and update position"""
        if side == 'buy':  # We buy EUR (client sells)
            self.inventory += amount_eur
            self.pnl -= amount_eur * price
        else:  # We sell EUR (client buys)
            self.inventory -= amount_eur
            self.pnl += amount_eur * price
    
    def profit_from_spread (self):
        """
        Example: Client buys €1M, then sells €1M
        We capture full spread
        """
        fair_value = 1.10025
        quotes = self.quote (fair_value)
        
        # Client buys €1M (we sell at ask)
        self.trade('sell', 1_000_000, quotes['ask'])
        print(f"Sold €1M at {quotes['ask']} → {self.inventory:,} EUR, \${self.pnl:,.0f} USD")
        
        # Client sells €1M(we buy at bid)
self.trade('buy', 1_000_000, quotes['bid'])
print(f"Bought €1M at {quotes['bid']} → {self.inventory:,} EUR, \${self.pnl:,.0f} USD")

return self.pnl

# Example
mm = FXMarketMaker()
profit = mm.profit_from_spread()
print(f"\\nProfit from round-trip: \${profit:,.0f}")
print("Captured full 5-pip spread with zero inventory risk")

# At scale
daily_volume_eur = 100_000_000  # €100M daily
spread_pips = 5
profit_per_eur = 5 / 10000  # 5 pips
daily_profit = daily_volume_eur * profit_per_eur
print(f"\\nIf €100M daily volume: \${daily_profit:,.0f}/day")
print(f"Annual: \${daily_profit * 250:,.0f}")
\`\`\`

**HFT Edge:**1. **Speed:** Quote faster than competitors, capture more flow
2. **Smart routing:** Send orders to multiple venues simultaneously
3. **Inventory management:** Flatten positions quickly to minimize risk
4. **Adverse selection mitigation:** Detect informed flow, widen spreads

**Bottom Line:**
Bid-ask spread compensates market makers for risk and costs. HFTs profit by capturing spreads at massive scale with minimal risk (quick inventory turnover). On €100M daily volume with 5-pip spread = $50K/day profit.`,
    keyPoints: [
      'Bid/ask spread: Buy at 1.1005, sell at 1.1000 = 5 pips or $500 on €1M',
      'Spread exists for: inventory risk, adverse selection, costs, profit',
      'HFTs profit by capturing spreads at scale with fast inventory turnover',
      'Market maker quotes around fair value, captures spread both sides',
      'At €100M daily volume, 5-pip spread = ~$50K/day revenue',
    ],
  },
  {
    id: 'fm-1-4-q-2',
    question:
      "The carry trade: Borrow in JPY (0.1% rate), invest in AUD (4.0% rate). Explain: (1) Profit calculation, (2) What risk you're taking, (3) Why it unwound violently in 2008, (4) When to enter/exit carry trades.",
    sampleAnswer: `**Carry Trade Mechanics:**

Borrow low-yield currency (JPY), invest in high-yield currency (AUD), pocket the interest rate differential.

**Profit Calculation:**

\`\`\`python
def calculate_carry_trade_return(
    notional_jpy,
    borrow_rate_jpy,
    invest_rate_aud,
    spot_initial,
    spot_final,
    time_years
):
    """
    Calculate carry trade P&L
    
    spot: JPY/AUD (yen per aussie dollar)
    """
    # Convert JPY to AUD at initial spot
    notional_aud = notional_jpy / spot_initial
    
    # Interest earned on AUD
    interest_aud = notional_aud * invest_rate_aud * time_years
    
    # Total AUD after 1 year
    total_aud = notional_aud + interest_aud
    
    # Convert back to JPY at final spot
    final_jpy = total_aud * spot_final
    
    # Interest paid on JPY borrowing
    interest_paid_jpy = notional_jpy * borrow_rate_jpy * time_years
    
    # Total JPY owed
    total_owed_jpy = notional_jpy + interest_paid_jpy
    
    # P&L
    pnl_jpy = final_jpy - total_owed_jpy
    pnl_pct = (pnl_jpy / notional_jpy) * 100
    
    # Breakdown
    interest_differential = (invest_rate_aud - borrow_rate_jpy) * 100
    fx_impact = ((spot_final - spot_initial) / spot_initial) * 100
    
    return {
        'pnl_jpy': pnl_jpy,
        'pnl_pct': pnl_pct,
        'interest_differential': interest_differential,
        'fx_impact': fx_impact,
        'breakdown': f'{interest_differential:.1f}% interest + {fx_impact:.1f}% FX = {pnl_pct:.1f}% total'
    }

# Scenario 1: Stable FX
result = calculate_carry_trade_return(
    notional_jpy=100_000_000,  # ¥100M
    borrow_rate_jpy=0.001,     # 0.1%
    invest_rate_aud=0.040,     # 4.0%
    spot_initial=85.0,         # ¥85/AUD
    spot_final=85.0,           # Unchanged
    time_years=1.0
)
print(f"Stable FX: {result['breakdown']}")
# Result: +3.9% (pure carry)

# Scenario 2: AUD strengthens
result = calculate_carry_trade_return(
    notional_jpy=100_000_000,
    borrow_rate_jpy=0.001,
    invest_rate_aud=0.040,
    spot_initial=85.0,
    spot_final=90.0,          # AUD stronger (more yen per AUD)
    time_years=1.0
)
print(f"AUD strengthens: {result['breakdown']}")
# Result: +9.8% (3.9% carry + 5.9% FX gain)

# Scenario 3: AUD weakens (2008 crisis)
result = calculate_carry_trade_return(
    notional_jpy=100_000_000,
    borrow_rate_jpy=0.001,
    invest_rate_aud=0.040,
    spot_initial=85.0,
    spot_final=60.0,          # AUD crashes -29%
    time_years=1.0
)
print(f"2008 crisis: {result['breakdown']}")
# Result: -25.5% (3.9% carry - 29.4% FX loss)
\`\`\`

**Risk: Currency Depreciation:**
You're LONG the high-yield currency (AUD). If it weakens, losses can dwarf interest earned.

**2008 Crisis Unwind:**1. **Build-up:** Years of carry trades → massive long AUD, short JPY positions globally
2. **Trigger:** Lehman bankruptcy → risk-off → everyone exits at once
3. **Cascade:** AUD crashes, JPY surges (safe haven) → huge losses on carry
4. **Forced liquidation:** Leveraged traders hit margin calls → sell more AUD → vicious cycle
5. **Result:** AUD/JPY fell 40% in months, wiping out years of carry gains

**Why Violent:**
- Crowded trade (everyone on same side)
- Leverage (traders borrowing 10x)
- Risk-off sentiment (flight to safe JPY)
- Forced selling (margin calls create self-reinforcing spiral)

**When to Enter/Exit:**

ENTER when:
- Low volatility environment (carry works when markets calm)
- Strong fundamentals in high-yield country (commodity boom for AUD)
- Carry differential widening (rate hikes in high-yield currency)
- Risk-on sentiment (investors seeking yield)

EXIT when:
- Volatility spikes (VIX > 20)
- Carry differential narrowing (rate cuts coming)
- Weak fundamentals (recession in high-yield country)
- Risk-off signals (credit spreads widening, equity selloff)

**Bottom Line:**
Carry trade = "picking up pennies in front of a steamroller". Works great in calm markets (3-4% annual), but occasional blow-ups (20-40% loss) can wipe out years of gains. Risk management critical.`,
    keyPoints: [
      'Carry: Borrow JPY 0.1%, invest AUD 4.0% = 3.9% profit if FX stable',
      'Risk: AUD depreciation can dwarf carry (2008: -29% FX loss vs +3.9% carry)',
      '2008 unwind: Crowded trade + leverage + risk-off → 40% AUD/JPY crash',
      'Enter: Low vol, strong fundamentals, widening carry, risk-on',
      'Exit: Vol spike, narrowing carry, weak fundamentals, risk-off',
    ],
  },
  {
    id: 'fm-1-4-q-3',
    question:
      'Design an FX trading system with real-time pricing from multiple liquidity providers, smart order routing to get best execution, P&L tracking in base currency, and hedging recommendations. Include pricing aggregation, execution logic, and risk management.',
    sampleAnswer: `**FX Trading System Architecture:**

\`\`\`python
from dataclasses import dataclass
from typing import List, Dict
import time

@dataclass
class FXQuote:
    provider: str
    currency_pair: str
    bid: float
    ask: float
    timestamp: float
    liquidity: int  # Size available at this price

class FXPriceAggregator:
    """
    Aggregate quotes from multiple liquidity providers
    Find best bid (highest) and best ask (lowest)
    """
    def __init__(self):
        self.quotes = {}  # currency_pair -> List[FXQuote]
    
    def update_quote (self, quote: FXQuote):
        """Receive real-time quote from provider"""
        pair = quote.currency_pair
        if pair not in self.quotes:
            self.quotes[pair] = []
        
        # Remove old quote from this provider
        self.quotes[pair] = [q for q in self.quotes[pair] if q.provider != quote.provider]
        
        # Add new quote
        self.quotes[pair].append (quote)
        
        # Remove stale quotes (> 1 second old)
        now = time.time()
        self.quotes[pair] = [q for q in self.quotes[pair] if now - q.timestamp < 1.0]
    
    def get_best_quotes (self, currency_pair: str) -> Dict:
        """Find best bid/ask across all providers"""
        if currency_pair not in self.quotes or not self.quotes[currency_pair]:
            return None
        
        quotes = self.quotes[currency_pair]
        
        # Best bid = highest (best to sell at)
        best_bid_quote = max (quotes, key=lambda q: q.bid)
        
        # Best ask = lowest (best to buy at)
        best_ask_quote = min (quotes, key=lambda q: q.ask)
        
        return {
            'bid': best_bid_quote.bid,
            'ask': best_ask_quote.ask,
            'bid_provider': best_bid_quote.provider,
            'ask_provider': best_ask_quote.provider,
            'spread_pips': (best_ask_quote.ask - best_bid_quote.bid) * 10000,
            'all_quotes': quotes
        }

class SmartOrderRouter:
    """
    Route orders to best venue for execution
    """
    def __init__(self, aggregator: FXPriceAggregator):
        self.aggregator = aggregator
    
    def execute_order (self, currency_pair: str, side: str, amount: float):
        """
        Execute order using smart routing
        
        side: 'buy' or 'sell'
        amount: notional in base currency
        """
        best = self.aggregator.get_best_quotes (currency_pair)
        
        if not best:
            return {'error': 'No quotes available'}
        
        # Buy = take ask, Sell = hit bid
        if side == 'buy':
            price = best['ask']
            provider = best['ask_provider']
        else:
            price = best['bid']
            provider = best['bid_provider']
        
        # Check if sufficient liquidity
        relevant_quotes = [q for q in best['all_quotes'] if q.provider == provider]
        if not relevant_quotes:
            return {'error': 'Provider quote stale'}
        
        provider_quote = relevant_quotes[0]
        
        if amount > provider_quote.liquidity:
            # Need to split order across multiple providers
            return self._execute_split_order (currency_pair, side, amount, best['all_quotes'])
        
        # Execute
        return {
            'status': 'filled',
            'provider': provider,
            'price': price,
            'amount': amount,
            'notional': amount * price if '/' in currency_pair else amount,
            'timestamp': time.time()
        }
    
    def _execute_split_order (self, pair, side, total_amount, quotes):
        """Split large order across multiple providers"""
        fills = []
        remaining = total_amount
        
        # Sort quotes by price (best first)
        if side == 'buy':
            sorted_quotes = sorted (quotes, key=lambda q: q.ask)
            price_key = 'ask'
        else:
            sorted_quotes = sorted (quotes, key=lambda q: q.bid, reverse=True)
            price_key = 'bid'
        
        for quote in sorted_quotes:
            if remaining <= 0:
                break
            
            fill_amount = min (remaining, quote.liquidity)
            fills.append({
                'provider': quote.provider,
                'price': getattr (quote, price_key),
                'amount': fill_amount
            })
            remaining -= fill_amount
        
        # Calculate average price
        total_notional = sum (f['amount'] * f['price'] for f in fills)
        avg_price = total_notional / total_amount
        
        return {
            'status': 'filled' if remaining == 0 else 'partial',
            'fills': fills,
            'average_price': avg_price,
            'total_amount': total_amount - remaining
        }

class FXPortfolio:
    """Track positions and P&L"""
    
    def __init__(self, base_currency='USD'):
        self.base = base_currency
        self.positions = {}  # currency -> amount
        self.trades = []
        self.positions[base_currency] = 1_000_000  # Start with $1M
    
    def execute_trade (self, trade_result):
        """Update positions after trade"""
        if trade_result.get('status') != 'filled':
            return
        
        # Parse currency pair (e.g., EUR/USD)
        pair = trade_result.get('currency_pair', '')
        base_ccy, quote_ccy = pair.split('/')
        
        amount = trade_result['amount']
        price = trade_result['price']
        
        # Update positions
        if 'buy' in str (trade_result.get('side', '')):
            self.positions[base_ccy] = self.positions.get (base_ccy, 0) + amount
            self.positions[quote_ccy] = self.positions.get (quote_ccy, 0) - amount * price
        else:
            self.positions[base_ccy] = self.positions.get (base_ccy, 0) - amount
            self.positions[quote_ccy] = self.positions.get (quote_ccy, 0) + amount * price
        
        self.trades.append (trade_result)
    
    def calculate_pnl (self, current_rates: Dict[str, float]):
        """
        Calculate P&L in base currency
        
        current_rates: {currency: rate_to_base}
        Example: {'EUR': 1.10, 'GBP': 1.25} means 1 EUR = 1.10 USD
        """
        total_base = 0
        
        for currency, amount in self.positions.items():
            if currency == self.base:
                total_base += amount
            else:
                rate = current_rates.get (currency, 1.0)
                total_base += amount * rate
        
        initial_capital = 1_000_000
        pnl = total_base - initial_capital
        pnl_pct = (pnl / initial_capital) * 100
        
        return {
            'total_value_base': total_base,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'positions': self.positions
        }
    
    def hedging_recommendations (self, target_currency=None):
        """
        Recommend hedges to reduce FX risk
        """
        recommendations = []
        
        for currency, amount in self.positions.items():
            if currency == self.base:
                continue
            
            if abs (amount) > 100_000:  # Material exposure
                hedge_direction = 'sell' if amount > 0 else 'buy'
                recommendations.append({
                    'currency': currency,
                    'exposure': amount,
                    'recommendation': f'{hedge_direction} {abs (amount):,.0f} {currency} forward',
                    'rationale': f'Hedge {abs (amount):,.0f} {currency} exposure'
                })
        
        return recommendations

# Example Usage
print("=== FX Trading System ===\\n")

# Set up system
aggregator = FXPriceAggregator()
router = SmartOrderRouter (aggregator)
portfolio = FXPortfolio (base_currency='USD')

# Simulate quotes from multiple providers
aggregator.update_quote(FXQuote('Citibank', 'EUR/USD', 1.1000, 1.1005, time.time(), 1_000_000))
aggregator.update_quote(FXQuote('JPMorgan', 'EUR/USD', 1.1001, 1.1004, time.time(), 2_000_000))
aggregator.update_quote(FXQuote('Deutsche', 'EUR/USD', 1.0999, 1.1006, time.time(), 500_000))

# Get best quotes
best = aggregator.get_best_quotes('EUR/USD')
print(f"Best EUR/USD: {best['bid']}/{best['ask']}")
print(f"Spread: {best['spread_pips']:.1f} pips")
print(f"Best bid from {best['bid_provider']}, ask from {best['ask_provider']}\\n")

# Execute trade
trade = router.execute_order('EUR/USD', 'buy', 1_000_000)
print(f"Executed: Buy €1M at {trade['price']} via {trade['provider']}")

# Calculate P&L
pnl = portfolio.calculate_pnl({'EUR': 1.1050})  # EUR strengthened
print(f"\\nP&L: \${pnl['pnl']:,.0f} ({ pnl['pnl_pct']: .2f } %)")

# Hedging
hedges = portfolio.hedging_recommendations()
for hedge in hedges:
    print(f"Hedge: {hedge['recommendation']}")
\`\`\`

**System Components:**1. Real-time price aggregation from multiple providers
2. Best bid/ask detection
3. Smart order routing (best execution)
4. Position tracking
5. P&L calculation in base currency
6. Hedging recommendations

**Production Features:**
- Latency optimization (<1ms)
- Failover (provider down → route to backup)
- Transaction cost analysis
- Slippage monitoring
- Regulatory reporting (MiFID II)`,
    keyPoints: [
      'Price aggregation: Find best bid (highest) and best ask (lowest) across providers',
      'Smart routing: Route buy to best ask provider, sell to best bid',
      'Split orders: Large orders split across providers for best average price',
      'P&L tracking: Convert all positions to base currency for total P&L',
      'Hedging: Recommend forwards/futures for material FX exposures',
    ],
  },
];
