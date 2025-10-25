export const fixedIncomeMarketsQuiz = [
  {
    id: 'fm-1-2-q-1',
    question:
      'A 10-year Treasury bond has a 3% coupon, trading at par ($100). If interest rates rise to 4%, explain: (1) Why the bond price falls, (2) Calculate the approximate new price using duration, (3) How this differs from a 2-year bond with the same coupon, (4) Design a hedging strategy for a bond portfolio manager.',
    sampleAnswer: `**Why Bond Prices Fall When Rates Rise:**

Inverse relationship: When market rates rise, existing bonds become less attractive because new bonds offer higher yields. To compete, existing bonds must trade at a discount.

**Mathematical Proof:**
Bond price = PV of future cash flows discounted at market rate.

At 3% rates: PV = $3/(1.03) + $3/(1.03)² + ... + $103/(1.03)¹⁰ = $100 (par)
At 4% rates: PV = $3/(1.04) + $3/(1.04)² + ... + $103/(1.04)¹⁰ = $91.89

**Using Duration:**
Duration ≈ 8.5 years for 10-year 3% bond
Approximate price change = -Duration × ΔYield × Price
= -8.5 × (0.04 - 0.03) × $100 = -8.5% = -$8.50
New price ≈ $91.50 (close to exact $91.89)

**2-Year Bond Comparison:**
Duration ≈ 1.9 years for 2-year bond
Price change = -1.9 × 1% × $100 = -1.9%
New price ≈ $98.10

**Key Insight:** Longer maturity = higher duration = more price sensitivity

**Hedging Strategy:**

\`\`\`python
class BondPortfolioHedge:
    def calculate_duration_hedge (self, portfolio_value, portfolio_duration, hedge_instrument_duration):
        """
        Duration-neutral hedge
        """
        # Dollar duration of portfolio
        portfolio_dollar_duration = portfolio_value * portfolio_duration
        
        # Hedge ratio
        hedge_ratio = portfolio_dollar_duration / hedge_instrument_duration
        
        return {
            'hedge_instrument': 'Treasury futures (short position)',
            'notional_to_short': hedge_ratio,
            'result': 'Portfolio neutral to parallel yield curve shifts'
        }

hedge = BondPortfolioHedge()
result = hedge.calculate_duration_hedge(
    portfolio_value=100_000_000,  # $100M portfolio
    portfolio_duration=8.5,  # 10-year bonds
    hedge_instrument_duration=10.0  # Treasury futures
)
# Short $85M notional of futures to hedge $100M portfolio
\`\`\`

**Complete Hedging Strategy:**
1. **Duration hedge:** Short Treasury futures to offset rate risk
2. **Convexity:** Buy/sell options for non-linear moves
3. **Spread risk:** Can't hedge with Treasuries (need credit derivatives)
4. **Rebalance:** Duration changes as time passes (need dynamic hedging)`,
    keyPoints: [
      'Bond prices and yields move inversely (rates up → prices down)',
      'Duration measures price sensitivity: 10-year bond -8.5%, 2-year bond -1.9% for 1% rate rise',
      'Hedge with Treasury futures: short duration × value / futures duration',
      'Longer bonds = higher duration = more rate risk',
      'Duration hedging protects against parallel yield curve shifts only',
    ],
  },
  {
    id: 'fm-1-2-q-2',
    question:
      'The yield curve shows 2-year at 4%, 10-year at 3.5% (inverted). Historically, inversions predict recessions. Explain: (1) Why inversion occurs, (2) The economic signal it sends, (3) Trading strategies to exploit this, (4) Why the signal sometimes fails.',
    sampleAnswer: `**Why Yield Curve Inversion Occurs:**

**Normal Curve:** Long-term yields > Short-term (positive slope)
- Reason: Investors demand premium for locking up money longer
- Compensation for inflation risk and uncertainty

**Inverted Curve:** Short-term > Long-term (negative slope)
- Reason 1: Fed raising short-term rates to fight inflation
- Reason 2: Market expects future rate cuts (recession coming)
- Reason 3: Flight to safety → demand for long-term Treasuries

**Economic Signal:**

When 2-year > 10-year:
1. **Fed tightening**: Short rates high to cool economy
2. **Market expects slowdown**: Demand for long bonds (safe haven)
3. **Credit tightening**: Banks borrow short, lend long → negative spread → less lending
4. **Self-fulfilling**: Reduced lending → economic slowdown → recession

**Historical Accuracy:**
- 7 of last 7 recessions preceded by inversion
- Average lag: 12-18 months
- But: Many false positives (inversions without recession)

**Trading Strategies:**

\`\`\`python
class YieldCurveTrading:
    def curve_steepener_trade (self):
        """
        Bet on curve normalizing (steepening)
        """
        return {
            'position': 'Long 10-year, Short 2-year',
            'rationale': 'Eventually Fed cuts → short rates fall → curve steepens',
            'risk': 'Curve stays inverted longer than you stay solvent',
            'sizing': 'Duration-neutral (offset rate risk, pure curve bet)'
        }
    
    def recession_trade (self):
        """
        Position for recession
        """
        return {
            'bonds': 'Long duration (rates will drop in recession)',
            'equities': 'Short stocks (recession = earnings collapse)',
            'credit': 'Sell corporate bonds, buy Treasuries (flight to safety)',
            'commodities': 'Short cyclicals (demand drops)'
        }
    
    def carry_trade (self):
        """
        Exploit inversion
        """
        return {
            'strategy': 'Borrow long-term (cheap), lend short-term (expensive)',
            'example': 'Issue 10-year bonds at 3.5%, buy 2-year at 4% = 50bps carry',
            'risk': 'Curve steepens → losses on long bond issuance'
        }
\`\`\`

**Why Signal Sometimes Fails:**

1. **Fed intervention:** QE flattens curve artificially
2. **Foreign demand:** Foreign central banks buy long Treasuries (pushes yields down)
3. **Technical factors:** Market structure (not economic)
4. **Timing:** Inversion → recession lag = 12-24 months (hard to time)
5. **Different this time:** 2019 inversion → pandemic (not typical recession)

**Bottom Line:**
Yield curve inversion = reliable recession indicator (7/7) but timing uncertain and false positives exist. Trade: steepener (curve normalizes), recession basket (defensive), or carry (exploit inversion).`,
    keyPoints: [
      'Inversion (short > long rates) signals Fed tightening + market expects cuts',
      'Historical: 7/7 recessions preceded by inversion, 12-18 month lag',
      'Trading: Steepener (long 10Y, short 2Y), recession basket, carry trade',
      'Fails due to: Fed QE, foreign demand, technical factors, timing uncertainty',
      'Self-fulfilling: inversion → tight credit → slower economy → recession',
    ],
  },
  {
    id: 'fm-1-2-q-3',
    question:
      'Design a corporate bond trading system that prices bonds using the Z-spread, calculates duration and convexity, monitors credit spreads in real-time, and alerts when spreads widen beyond 2 standard deviations. Include pricing model, risk metrics, and alert logic.',
    sampleAnswer: `**Corporate Bond Trading System Architecture:**

\`\`\`python
import numpy as np
from scipy.optimize import minimize_scalar
from typing import List, Dict

class CorporateBond:
    def __init__(self, face_value, coupon_rate, maturity_years, credit_rating):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.maturity = maturity_years
        self.rating = credit_rating
    
    def cash_flows (self) -> List[tuple]:
        """Generate (time, cash_flow) tuples"""
        flows = []
        # Coupons
        for t in range(1, self.maturity + 1):
            flows.append((t, self.face_value * self.coupon_rate))
        # Principal at maturity
        flows[-1] = (self.maturity, flows[-1][1] + self.face_value)
        return flows
    
    def price_with_zspread (self, treasury_curve, z_spread):
        """
        Price bond using Z-spread
        
        Z-spread = constant spread over entire Treasury curve
        that makes PV of bond = market price
        """
        pv = 0
        for t, cf in self.cash_flows():
            # Treasury rate at time t
            treasury_rate = treasury_curve.get_rate (t)
            
            # Discount at Treasury rate + Z-spread
            discount_rate = treasury_rate + z_spread
            pv += cf / ((1 + discount_rate) ** t)
        
        return pv
    
    def calculate_zspread (self, market_price, treasury_curve):
        """
        Solve for Z-spread that makes PV = market price
        """
        def objective (z):
            return abs (self.price_with_zspread (treasury_curve, z) - market_price)
        
        result = minimize_scalar (objective, bounds=(0, 0.10), method='bounded')
        return result.x
    
    def calculate_duration (self, market_price, treasury_curve, z_spread):
        """
        Modified duration: % price change for 1% yield change
        """
        # Calculate weighted average time to cash flows
        pv_weighted_time = 0
        for t, cf in self.cash_flows():
            rate = treasury_curve.get_rate (t) + z_spread
            pv = cf / ((1 + rate) ** t)
            pv_weighted_time += t * pv
        
        macaulay_duration = pv_weighted_time / market_price
        
        # Modified duration = Macaulay / (1 + yield)
        ytm = treasury_curve.get_rate (self.maturity) + z_spread
        modified_duration = macaulay_duration / (1 + ytm)
        
        return {
            'macaulay_duration': macaulay_duration,
            'modified_duration': modified_duration,
            'interpretation': f'{modified_duration:.2f}% price change per 1% yield change'
        }
    
    def calculate_convexity (self, market_price, treasury_curve, z_spread):
        """
        Convexity: How duration changes as yields change
        Captures non-linear price/yield relationship
        """
        convexity = 0
        for t, cf in self.cash_flows():
            rate = treasury_curve.get_rate (t) + z_spread
            pv = cf / ((1 + rate) ** t)
            convexity += (t * (t + 1) * pv) / ((1 + rate) ** 2)
        
        convexity /= market_price
        
        return convexity

class CreditSpreadMonitor:
    """
    Monitor credit spreads for anomalies
    """
    def __init__(self):
        self.historical_spreads = {}  # symbol -> List[spread]
    
    def update_spread (self, symbol, spread):
        """Add new spread observation"""
        if symbol not in self.historical_spreads:
            self.historical_spreads[symbol] = []
        
        self.historical_spreads[symbol].append (spread)
        
        # Keep rolling window (90 days)
        if len (self.historical_spreads[symbol]) > 90:
            self.historical_spreads[symbol] = self.historical_spreads[symbol][-90:]
    
    def detect_anomaly (self, symbol, current_spread):
        """
        Alert if spread > 2 standard deviations from mean
        """
        if symbol not in self.historical_spreads:
            return None
        
        history = np.array (self.historical_spreads[symbol])
        
        if len (history) < 30:  # Need minimum history
            return None
        
        mean = np.mean (history)
        std = np.std (history)
        
        z_score = (current_spread - mean) / std if std > 0 else 0
        
        if abs (z_score) > 2:
            return {
                'alert': True,
                'symbol': symbol,
                'current_spread_bps': current_spread * 10000,
                'mean_spread_bps': mean * 10000,
                'std_bps': std * 10000,
                'z_score': z_score,
                'severity': 'HIGH' if abs (z_score) > 3 else 'MEDIUM',
                'interpretation': 'Credit deteriorating' if z_score > 0 else 'Tightening unusually'
            }
        
        return None

class TreasuryCurve:
    """Simple Treasury curve interpolation"""
    def __init__(self, rates_by_maturity):
        self.rates = rates_by_maturity  # {1: 0.03, 2: 0.035, ...}
    
    def get_rate (self, maturity):
        """Linear interpolation"""
        if maturity in self.rates:
            return self.rates[maturity]
        
        # Find surrounding maturities
        lower = max([m for m in self.rates.keys() if m < maturity], default=None)
        upper = min([m for m in self.rates.keys() if m > maturity], default=None)
        
        if lower is None or upper is None:
            return self.rates[min (self.rates.keys(), key=lambda k: abs (k - maturity))]
        
        # Linear interpolation
        t = (maturity - lower) / (upper - lower)
        return self.rates[lower] + t * (self.rates[upper] - self.rates[lower])

# Usage Example
treasury_curve = TreasuryCurve({
    1: 0.040,
    2: 0.038,
    5: 0.035,
    10: 0.036,
    30: 0.040
})

bond = CorporateBond(
    face_value=1000,
    coupon_rate=0.05,  # 5% coupon
    maturity_years=10,
    credit_rating='BBB'
)

market_price = 980  # Trading below par

# Calculate Z-spread
z_spread = bond.calculate_zspread (market_price, treasury_curve)
print(f"Z-Spread: {z_spread * 10000:.0f} bps")

# Calculate duration
duration = bond.calculate_duration (market_price, treasury_curve, z_spread)
print(f"Modified Duration: {duration['modified_duration']:.2f} years")
print(f"{duration['interpretation']}")

# Calculate convexity
convexity = bond.calculate_convexity (market_price, treasury_curve, z_spread)
print(f"Convexity: {convexity:.2f}")

# Monitor spreads
monitor = CreditSpreadMonitor()
for i in range(90):
    # Simulate historical spreads
    monitor.update_spread('CORP_XYZ', 0.015 + np.random.normal(0, 0.002))

# Check for anomaly
anomaly = monitor.detect_anomaly('CORP_XYZ', 0.025)  # Spread widened to 250 bps
if anomaly:
    print(f"\\nALERT: {anomaly['symbol']}")
    print(f"  Spread: {anomaly['current_spread_bps']:.0f} bps")
    print(f"  Mean: {anomaly['mean_spread_bps']:.0f} bps")
    print(f"  Z-Score: {anomaly['z_score']:.2f}")
    print(f"  {anomaly['interpretation']}")
\`\`\`

**System Components:**

1. **Pricing Engine:** Z-spread calculation
2. **Risk Analytics:** Duration, convexity
3. **Monitoring:** Real-time spread tracking
4. **Alerting:** Statistical anomaly detection
5. **Data Storage:** TimescaleDB for time-series

**Production Considerations:**
- Use OAS (Option-Adjusted Spread) for callable bonds
- Multi-factor duration (key rate duration)
- Credit default swap spreads for hedging
- Real-time market data (Bloomberg, Refinitiv)`,
    keyPoints: [
      'Z-spread: constant spread over Treasury curve making PV = market price',
      'Duration: sensitivity to rate changes (10-year ≈ 8 years)',
      'Convexity: captures non-linear price/yield relationship',
      'Anomaly detection: Z-score > 2 indicates unusual spread widening',
      'Alert triggers: Credit deterioration, liquidity stress, market dislocation',
    ],
  },
];
