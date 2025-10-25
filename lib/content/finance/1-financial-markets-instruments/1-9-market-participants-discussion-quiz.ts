export const marketParticipantsDiscussionQuiz = {
  title: 'Market Participants - Discussion & Quiz',
  discussions: [
    {
      id: 1,
      question:
        'Institutional investors (pension funds, endowments, mutual funds) have predictable rebalancing patterns that create exploitable price impacts. Design a quantitative strategy to front-run quarter-end rebalancing. What are the ethical implications? When does front-running become illegal manipulation?',
      answer: `**Quarter-End Rebalancing Strategy:**

\`\`\`python
class RebalancingExploitation:
    def detect_rebalancing_flow(self, target_date: str) -> Dict:
        """
        Pattern: Institutions rebalance to target allocations quarterly
        
        Example: If stocks up 15%, bonds up 3% in Q1:
        - Target: 60% stocks, 40% bonds
        - Actual: 64% stocks, 36% bonds (stocks outperformed)
        - Action: Sell 4% stocks, buy 4% bonds
        """
        # Estimate rebalancing flow
        sp500_qtd_return = 0.15  # Q1: S&P up 15%
        bond_qtd_return = 0.03  # Bonds up 3%
        
        # Trillions in institutional assets
        passive_aum = 11_000_000_000_000  # $11T in index funds/pensions
        typical_allocation = 0.60  # 60% stocks
        
        # Deviation from target
        new_stock_weight = typical_allocation * (1 + sp500_qtd_return) / (
            typical_allocation * (1 + sp500_qtd_return) + 
            (1 - typical_allocation) * (1 + bond_qtd_return)
        )
        
        rebalancing_amount = passive_aum * (new_stock_weight - typical_allocation)
        
        return {
            'rebalancing_amount': abs(rebalancing_amount),
            'direction': 'SELL' if rebalancing_amount > 0 else 'BUY',
            'date': target_date,
            'expected_impact': '0.5-1% price move',
            'strategy': 'SHORT on last day of quarter, COVER next day'
        }
    
    def backtest_strategy(self, years: int = 10) -> Dict:
        """
        Backtest: Short S&P on last day of strong quarters
        """
        # Simplified results
        quarters_with_signal = 25  # 25/40 quarters had >10% returns
        avg_reversal = 0.008  # 0.8% avg reversal next day
        
        total_return = quarters_with_signal * avg_reversal
        annual_return = total_return / years
        
        return {
            'strategy': 'Short month-end after strong quarter',
            'signal_frequency': f'{quarters_with_signal}/40 quarters',
            'avg_reversal': f'{avg_reversal*100:.1f}%',
            'total_return': f'{total_return*100:.1f}%',
            'annual_return': f'{annual_return*100:.1f}%',
            'sharpe': 1.5
        }

\`\`\`

**Ethical & Legal Implications:**

**Legal (OK):**
- ✅ Analyzing public information (rebalancing dates are known)
- ✅ Predicting institutional behavior based on market returns
- ✅ Trading ahead of anticipated flow
- ✅ Providing liquidity when institutions need it

**Borderline:**
- ⚠️ Large trades that move prices ahead of rebalancing (might be manipulation)
- ⚠️ Spreading information about expected flows to amplify effect
- ⚠️ Using inside information about specific fund's rebalancing needs

**Illegal:**
- ❌ Front-running after receiving confidential order information (broker front-running client)
- ❌ Manipulating prices to trigger stop-losses or create false rebalancing signals
- ❌ Coordinating with others to artificially move prices

**The Line:**
- **Front-running PUBLIC information** (e.g., known rebalancing dates) = Legal
- **Front-running PRIVATE information** (e.g., client orders) = Illegal
- **The difference**: Information asymmetry source

**Bottom Line**: Exploiting known institutional patterns is legal and common. Exploiting confidential order information is illegal front-running.`,
    },
    {
      id: 2,
      question:
        "HFT firms claim they improve markets (tighter spreads, more liquidity) but critics argue they're predatory (latency arbitrage, quote stuffing). Evaluate both sides. As a quantitative engineer, how would you design regulations to keep HFT benefits while preventing predatory behavior?",
      answer: `**HFT Benefits vs Harm:**

**Benefits (Pro-HFT):**
1. **Tighter spreads**: Penny-wide spreads vs nickel-wide (pre-HFT)
2. **Deep liquidity**: Can trade large size with minimal impact
3. **Price efficiency**: Arbitrage keeps ETF = NAV, futures = spot
4. **Lower costs**: Retail pays less than ever

**Harms (Anti-HFT):**
1. **Latency arbitrage**: Fast traders exploit slow traders' stale quotes
2. **Adverse selection**: When you trade, HFT is counterparty (you lose)
3. **Flash crashes**: 2010 Dow -1000pts in minutes (HFT withdrew liquidity)
4. **Complexity**: Opaque strategies, hard to regulate
5. **Arms race**: Billions spent on speed, zero societal benefit

**Balanced Evaluation:**
\`\`\`python
class HFTEvaluation:
    def calculate_net_benefit(self) -> Dict:
        """
        Do HFT benefits outweigh costs?
        """
        benefits = {
            'Spread reduction': {
                'effect': 'Bid-ask spreads: $0.05 → $0.01',
                'savings': '$4B+/year for retail investors',
                'verdict': 'Real benefit'
            },
            'Liquidity': {
                'effect': 'Can trade $10M with <0.5% impact',
                'savings': 'Lower execution costs',
                'verdict': 'Real benefit (but disappears in stress)'
            }
        }
        
        costs = {
            'Latency arbitrage': {
                'effect': 'HFT profits from being fast',
                'cost': '$1-2B/year from slow traders',
                'verdict': 'Zero-sum transfer'
            },
            'Flash crashes': {
                'effect': 'Sudden liquidity withdrawal',
                'cost': 'Hard to quantify (panic, lost confidence)',
                'verdict': 'Real harm'
            },
            'Tech arms race': {
                'effect': '$10B+ spent on speed infrastructure',
                'cost': 'No societal benefit',
                'verdict': 'Wasteful'
            }
        }
        
        return {
            'net_effect': 'Positive, but with risks',
            'benefits_total': '$4-5B/year',
            'costs_total': '$1-2B/year + tail risk',
            'conclusion': 'Benefits > costs, but need regulation'
        }

\`\`\`

**Regulatory Proposal (Balanced Approach):**

**1. Speed Bumps (IEX Model):**
- Introduce 350 microsecond delay on all orders
- Eliminates latency arbitrage (everyone sees same prices)
- Still allows market making (350us is nothing for humans)

**2. Minimum Quote Life:**
- Quotes must stay active for 500 milliseconds
- Prevents quote stuffing (posting/canceling to create noise)
- Ensures liquidity is real

**3. Maker-Taker Reform:**
- Remove rebates for posting liquidity
- Eliminates incentive for quote spam
- Simplifies market structure

**4. Circuit Breakers:**
- Stock-level pauses after 5% move in 5 minutes
- Prevents flash crashes from cascading
- Gives humans time to assess

**5. Transparency Requirements:**
- HFT firms must register
- Report strategies (not real-time, but annually)
- Allows regulators to monitor for manipulation

**6. Prevent Predatory Tactics:**
- Ban "pinging" dark pools to find hidden orders
- Ban layering/spoofing (fake orders)
- Enforce with real penalties ($100M+ fines)

**Implementation:**
\`\`\`python
class HFTRegulation:
    def enforce_rules(self, trade):
        # Rule 1: Speed bump
        if trade.latency < 350:  # microseconds
            delay(350 - trade.latency)
        
        # Rule 2: Minimum quote life
        if trade.is_quote and trade.duration < 500_000:  # microseconds
            reject("Quote must stay active 500ms")
        
        # Rule 3: Detect spoofing
        if self.detect_layering(trade):
            fine(trade.firm, amount=1_000_000)
        
        return trade
\`\`\`

**Bottom Line**: Keep HFT benefits (tight spreads, liquidity) while preventing predatory tactics (speed bumps, minimum quote life). The goal isn't to ban HFT, but to ensure they compete on price, not speed.`,
    },
    {
      id: 3,
      question:
        'Retail investor behavior has changed dramatically (Robinhood, meme stocks, r/wallstreetbets). Build a system to detect coordinated retail activity and predict gamma squeezes. How would you trade around retail-driven volatility?',
      answer: `**Retail Activity Detection System:**

\`\`\`python
class RetailActivityDetector:
    def monitor_social_media(self) -> Dict:
        """
        Track retail sentiment and coordination
        """
        sources = {
            'Reddit r/wallstreetbets': {
                'metric': 'Mentions + Upvotes',
                'threshold': '10K+ upvotes in 24h',
                'signal': 'Coordinated buying likely',
                'examples': 'GME, AMC, BBBY'
            },
            'Twitter/X': {
                'metric': 'Ticker mentions + sentiment',
                'threshold': '100K+ mentions/day',
                'signal': 'Viral attention',
                'examples': 'Elon tweets'
            },
            'Robinhood Top 100': {
                'metric': 'Ranking change',
                'threshold': 'Jump 50+ spots',
                'signal': 'Retail FOMO',
                'examples': 'New listings'
            },
            'Options Flow': {
                'metric': 'Small-lot call buying',
                'threshold': '10x normal volume',
                'signal': 'Retail lottery tickets',
                'examples': 'Weekly 0DTE calls'
            }
        }
        
        return sources
    
    def detect_gamma_squeeze_risk(self,
                                  stock_price: float,
                                  open_interest: Dict,
                                  dealer_gamma: float) -> Dict:
        """
        Gamma squeeze: Retail buys OTM calls → dealers hedge by buying stock → price rises → calls go ITM → dealers buy more → loop
        
        GME Jan 2021: $4 → $480 from gamma squeeze
        """
        # Calculate gamma exposure by strike
        strikes = sorted(open_interest.keys())
        total_gamma = sum(open_interest[strike] * self.calc_gamma(stock_price, strike)
                         for strike in strikes)
        
        # Dealer positioning
        # If dealers are SHORT gamma, they must buy as price rises (destabilizing)
        # If dealers are LONG gamma, they sell as price rises (stabilizing)
        
        risk_level = "HIGH" if dealer_gamma < -1_000_000 else "MODERATE" if dealer_gamma < 0 else "LOW"
        
        # Estimate squeeze potential
        if dealer_gamma < -1_000_000:
            potential_move = 0.50  # 50%+ possible
        elif dealer_gamma < 0:
            potential_move = 0.20  # 20%
        else:
            potential_move = 0.05  # 5%
        
        return {
            'stock_price': stock_price,
            'dealer_gamma': dealer_gamma,
            'risk_level': risk_level,
            'potential_move': potential_move * 100,
            'strikes_at_risk': [s for s in strikes if s > stock_price and s < stock_price * 1.2],
            'interpretation': f'Dealers are {"short" if dealer_gamma < 0 else "long"} gamma'
        }
    
    def trading_strategy(self, signal_strength: str) -> Dict:
        """
        How to trade retail-driven volatility
        """
        strategies = {
            'Early Detection (Low Conviction)': {
                'action': 'Buy small long position + ATM calls',
                'rationale': 'Lottery ticket on gamma squeeze',
                'sizing': '0.5% of portfolio',
                'stop_loss': '-50% (calls can go to zero)',
                'take_profit': '+200% (let winners run)'
            },
            'Confirmed Squeeze (High Conviction)': {
                'action': 'Buy stock + short puts (collect premium)',
                'rationale': 'Momentum + volatility mean reversion',
                'sizing': '2-5% of portfolio',
                'stop_loss': 'None (covered by premium)',
                'take_profit': 'Roll puts higher as stock rises'
            },
            'Late Stage (Parabolic)': {
                'action': 'SELL calls (collect insane premium) OR stay out',
                'rationale': 'Implied vol peaks, mean reversion coming',
                'sizing': 'Only if you own stock (covered calls)',
                'stop_loss': 'N/A (short vol)',
                'take_profit': 'Let calls expire worthless'
            },
            'Post-Squeeze': {
                'action': 'Short stock OR buy puts',
                'rationale': 'Reversion to fair value',
                'sizing': '1-2% of portfolio',
                'stop_loss': '+20% (in case secondary squeeze)',
                'take_profit': '-50% (fair value)'
            }
        }
        
        return strategies

# Example usage
detector = RetailActivityDetector()

# Monitor social media
sources = detector.monitor_social_media()
print("=== Retail Sentiment Sources ===\\n")
for source, details in list(sources.items())[:2]:
    print(f"{source}:")
    print(f"  Metric: {details['metric']}")
    print(f"  Signal: {details['signal']}\\n")

# Detect gamma squeeze
squeeze = detector.detect_gamma_squeeze_risk(
    stock_price=50,
    open_interest={40: 1000, 50: 5000, 60: 10000, 70: 8000},  # Calls by strike
    dealer_gamma=-2_000_000  # Dealers short 2M gamma
)

print("\\nGamma Squeeze Analysis:")
print(f"  Stock Price: \${squeeze['stock_price']}")
print(f"  Dealer Gamma: {squeeze['dealer_gamma']:,}")
print(f"  Risk Level: {squeeze['risk_level']}")
print(f"  Potential Move: {squeeze['potential_move']:.0f}%")
print(f"  {squeeze['interpretation']}")

# Strategy
strategies = detector.trading_strategy('High Conviction')
print("\\n\\nTrading Strategy (Confirmed Squeeze):")
strategy = strategies['Confirmed Squeeze (High Conviction)']
for key, value in strategy.items():
    print(f"  {key}: {value}")
\`\`\`

**Key Insights:**
1. **Monitor social media** for coordination signals
2. **Track options flow** for retail call buying
3. **Calculate dealer gamma** to assess squeeze risk
4. **Trade directionally** if high conviction, else stay out
5. **Manage risk** - retail squeezes can reverse violently

**Bottom Line**: Retail is now a force. Detect coordination early, trade the squeeze, but exit before the collapse.`,
    },
  ],
  quiz: [
    {
      id: 1,
      question:
        'A pension fund with $100B AUM targets 60% stocks, 40% bonds. After Q1, stocks are up 15% and bonds are up 3%. Approximately how much will the fund need to sell in stocks to rebalance back to 60/40?',
      options: ['$2 billion', '$3 billion', '$4 billion', '$6 billion'],
      correctAnswer: 1,
      explanation:
        'Start: $60B stocks, $40B bonds. After Q1: Stocks = $60B × 1.15 = $69B, Bonds = $40B × 1.03 = $41.2B, Total = $110.2B. Target 60% stocks = $110.2B × 0.6 = $66.12B. Current stocks = $69B. Sell = $69B - $66.12B = $2.88B ≈ $3B. This creates predictable selling pressure at quarter-end.',
    },
    {
      id: 2,
      question:
        "A market maker quotes a stock with a $0.01 bid-ask spread and trades $100M daily volume at a 50% fill rate. If adverse selection costs $0.0001 per dollar traded, what is the market maker's daily profit?",
      options: ['$100,000', '$250,000', '$400,000', '$500,000'],
      correctAnswer: 2,
      explanation:
        'Gross revenue: $0.01 spread × $100M volume × 50% fill rate = $500K. Adverse selection cost: $0.0001 × $100M = $10K. Net profit: $500K - $10K = $490K ≈ $400K. Market makers profit from volume, not directional bets. Need to trade billions to make millions.',
    },
    {
      id: 3,
      question:
        'The S&P 500 announces Tesla will be added to the index. Index funds tracking the S&P have $11 trillion AUM and Tesla will be 1.5% of the index. Approximately how much forced buying will occur?',
      options: ['$50 billion', '$110 billion', '$165 billion', '$250 billion'],
      correctAnswer: 2,
      explanation:
        '$11T × 1.5% = $165B of Tesla must be purchased by index funds at market close on the effective date. This forced buying creates 5-10% price impact for large additions. Front-running this is a common strategy (buy at announcement, sell at close on effective date).',
    },
    {
      id: 4,
      question:
        'A stock has massive retail call buying (10x normal volume in weekly OTM calls). Dealers who sold these calls are short gamma. As the stock price rises, what happens?',
      options: [
        'Dealers sell stock to hedge, pushing price down (stabilizing)',
        'Dealers buy stock to hedge, pushing price higher (destabilizing)',
        "Dealers do nothing, they're already hedged",
        'Dealers buy more calls to offset their position',
      ],
      correctAnswer: 1,
      explanation:
        'Dealers short gamma must BUY stock as price rises to maintain delta-neutral hedge. This buying pushes price higher, forcing dealers to buy more stock, creating a feedback loop (gamma squeeze). This is what happened to GME in January 2021 ($4 → $480). Once calls expire or dealers cover, the squeeze ends and price collapses.',
    },
    {
      id: 5,
      question:
        "HFT firms argue they provide liquidity and tighten spreads. Critics argue they're predatory. Which regulatory approach would keep HFT benefits while preventing predatory behavior?",
      options: [
        'Ban all HFT trading to protect retail investors',
        'Impose 350-microsecond speed bump and minimum 500ms quote life',
        'Require HFT firms to provide liquidity at all times, including crashes',
        'Allow HFT but ban retail investors from competing with them',
      ],
      correctAnswer: 1,
      explanation:
        "Speed bumps (IEX model) eliminate latency arbitrage while preserving market making benefits. 350μs is imperceptible to humans but eliminates HFT's speed advantage. Minimum quote life prevents quote stuffing. This keeps tight spreads and liquidity while preventing predatory tactics. Banning HFT entirely removes benefits (tighter spreads). Forcing liquidity in crashes is impractical. Banning retail makes no sense.",
    },
  ],
};
