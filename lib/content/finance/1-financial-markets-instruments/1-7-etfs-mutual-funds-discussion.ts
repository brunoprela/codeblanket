export const etfsMutualFundsDiscussion = {
  title: 'ETFs & Mutual Funds - Discussion Questions',
  questions: [
    {
      id: 1,
      question:
        'The data shows that 90%+ of active mutual fund managers underperform passive index funds over 10+ years, yet the active management industry still manages trillions of dollars. Analyze the reasons for this paradox. What behavioral biases and structural factors keep investors in underperforming active funds? For a quantitative engineer, how would you design a system to identify the rare active managers who DO consistently outperform?',
      answer: `**Why Active Funds Persist Despite Underperformance:**

**1. Behavioral Biases:**

**Recency Bias:**
- Investors chase last year's winners
- "This fund returned 35% last year!" (ignoring 10-year underperformance)
- Mean reversion ensures yesterday's winners become tomorrow's losers

**Overconfidence:**
- "I can pick the best managers" (most can't)
- "Active management makes sense for MY portfolio" (it doesn't)
- Illusion of control feels better than passive acceptance

**Narrative Fallacy:**
- Good stories beat boring statistics
- "Star manager with proprietary model" > "Cheap index fund"
- Financial media promotes active management (more exciting)

**Loss Aversion:**
- Fear of "missing out" on star manager's gains
- Forget that underperformance is the more likely outcome
- 10% chance of 2% outperformance feels worth 90% chance of 2% underperformance

**2. Structural Factors:**

**Fees and Incentives:**
- Financial advisors earn higher commissions on active funds (0.75%+ expense ratio)
- Passive funds (0.03%) don't generate enough fees
- Conflict of interest: What\'s best for advisor ≠ what's best for client

**Marketing and Distribution:**
- Active funds spend millions on marketing
- Vanguard doesn't advertise much (doesn't need to)
- Retail investors see ads for "top-performing funds"

**Complexity Bias:**
- "Something this important should be complex" (it shouldn't)
- Passive investing feels "too simple"
- Active management employs smart people with advanced degrees (must be better!)

**Survivorship Bias:**
- Bad active funds close/merge (disappear from statistics)
- Only winners remain visible
- Makes active management look better than it is

**3. When Active Might Make Sense:**

**Inefficient Markets:**
- Small-cap stocks (less analyst coverage)
- International emerging markets
- Corporate bonds (less liquid)
- Not S&P 500 (most efficient market on earth)

**Tax-Loss Harvesting:**
- Active management of individual stocks for tax optimization
- Different from stock-picking active funds

**Designing a System to Identify Rare Outperformers:**

\`\`\`python
class ActiveManagerEvaluationSystem:
    """
    Identify managers who ACTUALLY add value
    
    Spoiler: Very few pass these tests
    """
    
    def evaluate_manager (self, 
                        manager_returns: np.array,
                        benchmark_returns: np.array,
                        expense_ratio: float,
                        years: int) -> Dict:
        """
        Rigorous evaluation framework
        """
        # 1. Alpha (risk-adjusted outperformance)
        alpha = self.calculate_alpha (manager_returns, benchmark_returns)
        
        # 2. Information Ratio (alpha per unit of tracking error)
        ir = self.calculate_information_ratio (manager_returns, benchmark_returns)
        
        # 3. Statistical Significance (is alpha due to skill or luck?)
        t_stat = self.t_statistic (alpha, tracking_error, years)
        p_value = self.p_value (t_stat)
        
        # 4. Consistency (outperform in multiple periods)
        consistency = self.rolling_outperformance_pct (manager_returns, benchmark_returns)
        
        # 5. Downside Protection (outperform in bad markets)
        downside_ratio = self.downside_performance (manager_returns, benchmark_returns)
        
        # 6. Style Drift (does strategy stay consistent)
        style_drift = self.style_consistency (manager_returns)
        
        # 7. Fees (after fees, still outperform?)
        net_alpha = alpha - expense_ratio
        
        # Pass criteria (ALL must be true)
        passes = {
            'positive_alpha': alpha > 0,
            'high_information_ratio': ir > 0.5,  # IR > 0.5 is good
            'statistically_significant': p_value < 0.05,  # 95% confidence
            'consistent': consistency > 0.65,  # Outperform 65%+ of periods
            'downside_protection': downside_ratio > 1.0,  # Better in down markets
            'stable_strategy': style_drift < 0.3,  # Low drift
            'positive_net_alpha': net_alpha > 0  # After fees
        }
        
        all_pass = all (passes.values())
        
        return {
            'alpha': alpha * 100,
            'information_ratio': ir,
            't_statistic': t_stat,
            'p_value': p_value,
            'consistency': consistency * 100,
            'downside_ratio': downside_ratio,
            'style_drift': style_drift,
            'net_alpha': net_alpha * 100,
            'passes_all_criteria': all_pass,
            'recommendation': 'CONSIDER' if all_pass else 'REJECT',
            'failures': [k for k, v in passes.items() if not v]
        }
    
    def survival_bias_adjustment (self, fund_universe: List) -> List:
        """
        Include dead/merged funds (don't just look at survivors)
        """
        # Most databases only show surviving funds
        # Need to include funds that closed (usually underperformers)
        
        alive_funds = [f for f in fund_universe if f.status == 'active']
        dead_funds = [f for f in fund_universe if f.status == 'closed']
        
        # Dead funds usually have worse performance
        # Including them drops average active return by 1-2% annually
        
        return alive_funds + dead_funds
    
    def monte_carlo_luck_test (self,
                              observed_alpha: float,
                              num_simulations: int = 10000) -> float:
        """
        How many managers would beat benchmark by luck alone?
        
        If 1000 managers flip coins, ~50 get heads 10 times in a row
        Is observed alpha skill or luck?
        """
        # Simulate random managers
        lucky_alphas = []
        for _ in range (num_simulations):
            # Random returns (no skill)
            random_returns = np.random.normal(0, 0.10, 120)  # 10 years
            random_alpha = np.mean (random_returns)
            lucky_alphas.append (random_alpha)
        
        # How many random managers beat observed alpha?
        better_by_luck = sum(1 for a in lucky_alphas if a >= observed_alpha)
        probability = better_by_luck / num_simulations
        
        return probability  # If > 0.05, likely luck not skill

# Example evaluation
evaluator = ActiveManagerEvaluationSystem()

# Simulate a "good" active manager
np.random.seed(42)
years = 10
benchmark = np.random.normal(0.10/252, 0.01, 252*years)  # Index: 10% return, 1% daily vol
manager = benchmark + np.random.normal(0.0002/252, 0.005, 252*years)  # Slight outperformance

evaluation = evaluator.evaluate_manager(
    manager_returns=manager,
    benchmark_returns=benchmark,
    expense_ratio=0.0075,  # 0.75% fee
    years=years
)

print("=== Active Manager Evaluation ===\\n")
print(f"Alpha: {evaluation['alpha']:.2f}% annually")
print(f"Information Ratio: {evaluation['information_ratio']:.2f}")
print(f"P-Value: {evaluation['p_value']:.3f} ({'Significant' if evaluation['p_value'] < 0.05 else 'Not Significant'})")
print(f"Consistency: {evaluation['consistency']:.0f}% of periods outperform")
print(f"Net Alpha (after fees): {evaluation['net_alpha']:.2f}%")
print(f"\\nRecommendation: {evaluation['recommendation']}")
if not evaluation['passes_all_criteria']:
    print(f"Failed criteria: {', '.join (evaluation['failures'])}")
\`\`\`

**Key Metrics for Identifying Skill:**

1. **Long track record**: 10+ years (short-term is luck)
2. **Statistical significance**: p < 0.05 (95% confidence alpha is real)
3. **Consistency**: Outperform in 65%+ of rolling 3-year periods
4. **Style stability**: Don't change strategy based on what's working
5. **Capacity**: Does strategy still work with $10B+ AUM? (Most don't scale)
6. **Downside protection**: Outperform in bear markets (not just bull markets)
7. **Fees**: After-fee alpha must be positive

**Reality Check:**
- After these tests, maybe 1-5% of active managers pass
- Even then, past performance ≠ future results
- For most investors: Just buy the index

**Bottom Line**: Active management persists due to behavioral biases, conflicts of interest, and survivorship bias. For engineers building evaluation systems, focus on statistical significance, consistency, and after-fee performance. But honestly? Just buy VOO.`,
    },
    {
      id: 2,
      question:
        'Explain the ETF creation/redemption mechanism and why it makes ETFs more tax-efficient than mutual funds. Design a system that monitors for ETF premium/discount arbitrage opportunities. Under what market conditions does this mechanism break down (hint: look at March 2020 COVID crash)?',
      answer: `**ETF Creation/Redemption Mechanism:**

**How It Works:**

**Creation Process (ETF trading at premium):**
1. **Authorized Participant (AP)** notices ETF trading above NAV (e.g., SPY at $500.20 vs NAV of $500.00)
2. **AP buys underlying stocks** (all 500 S&P 500 stocks in correct weights)
3. **AP delivers stocks to ETF issuer** (in-kind transfer)
4. **ETF issues new shares** to AP (e.g., 50,000 share "creation unit")
5. **AP sells ETF shares** on market at premium ($500.20)
6. **Profit**: $500.20 - $500.00 = $0.20 per share × 50,000 = $10,000

**Redemption Process (ETF trading at discount):**
1. **AP notices ETF trading below NAV** (e.g., SPY at $499.80 vs NAV of $500.00)
2. **AP buys ETF shares** on market at discount
3. **AP delivers ETF shares to issuer**
4. **ETF redeems for underlying stocks** (in-kind transfer)
5. **AP sells stocks** for full value ($500.00)
6. **Profit**: $500.00 - $499.80 = $0.20 per share

**Why This Makes ETFs Tax-Efficient:**

**Mutual Funds (Cash Redemptions):**
\`\`\`
Investor redeems $1M from mutual fund
→ Fund must sell $1M of stocks
→ Selling stocks realizes capital gains
→ Capital gains distributed to ALL remaining shareholders
→ You pay taxes even though YOU didn't sell!
\`\`\`

**ETFs (In-Kind Redemptions):**
\`\`\`
AP redeems ETF shares
→ ETF delivers stocks (not cash)
→ No sale occurs = no capital gains realized
→ ETF can deliver appreciated shares, removing gains from fund
→ Remaining shareholders pay no taxes!
\`\`\`

**Tax Efficiency Example:**

\`\`\`python
class TaxEfficiencyComparison:
    """
    Compare ETF vs Mutual Fund tax efficiency
    """
    
    def mutual_fund_redemption (self,
                               redemption_amount: float,
                               portfolio_value: float,
                               unrealized_gains: float,
                               capital_gains_rate: float = 0.20) -> Dict:
        """
        Mutual fund must sell stocks (cash redemption)
        """
        # % of portfolio being redeemed
        redemption_pct = redemption_amount / portfolio_value
        
        # Gains realized
        gains_realized = unrealized_gains * redemption_pct
        
        # Distributed to all shareholders
        # (even those who didn't redeem!)
        tax_bill = gains_realized * capital_gains_rate
        
        return {
            'redemption': redemption_amount,
            'gains_realized': gains_realized,
            'tax_bill': tax_bill,
            'tax_rate': capital_gains_rate * 100,
            'structure': 'CASH redemption'
        }
    
    def etf_redemption (self,
                      redemption_amount: float,
                      portfolio_value: float,
                      unrealized_gains: float) -> Dict:
        """
        ETF delivers stocks in-kind (no sale, no taxes)
        """
        return {
            'redemption': redemption_amount,
            'gains_realized': 0,  # No gains!
            'tax_bill': 0,  # No taxes!
            'structure': 'IN-KIND redemption',
            'benefit': 'No capital gains distributed'
        }

# Example
tax_comp = TaxEfficiencyComparison()

# $10M redemption from $100M portfolio with $20M unrealized gains

mutual_fund = tax_comp.mutual_fund_redemption(
    redemption_amount=10_000_000,
    portfolio_value=100_000_000,
    unrealized_gains=20_000_000,
    capital_gains_rate=0.20
)

etf = tax_comp.etf_redemption(
    redemption_amount=10_000_000,
    portfolio_value=100_000_000,
    unrealized_gains=20_000_000
)

print("=== Tax Efficiency Comparison ===\\n")
print(f"Mutual Fund (10% redemption):")
print(f"  Gains Realized: \${mutual_fund['gains_realized']:,.0f}")
print(f"  Tax Bill (to remaining shareholders): \${mutual_fund['tax_bill']:,.0f}\\n")

print(f"ETF (10% redemption):")
print(f"  Gains Realized: \${etf['gains_realized']:,.0f}")
print(f"  Tax Bill: \${etf['tax_bill']:,.0f}")
print(f"  Benefit: {etf['benefit']}")
\`\`\`

**Arbitrage Monitoring System:**

\`\`\`python
class ETFArbitrageMonitor:
    """
    Monitor premium/discount for arbitrage opportunities
    """
    
    def __init__(self):
        self.etfs_to_monitor = ['SPY', 'QQQ', 'IWM', 'XLF', 'XLE']
        self.threshold = 0.005  # 0.5% deviation
        self.creation_unit_size = 50000
    
    def calculate_premium_discount (self, 
                                   etf_ticker: str,
                                   market_price: float,
                                   nav: float) -> Dict:
        """
        Calculate premium/discount
        """
        premium_discount = (market_price - nav) / nav
        
        # Arbitrage opportunity?
        if abs (premium_discount) > self.threshold:
            if premium_discount > 0:
                action = "CREATE"
                trade = "Buy stocks → Create ETF → Sell ETF"
                profit_per_share = market_price - nav
            else:
                action = "REDEEM"
                trade = "Buy ETF → Redeem for stocks → Sell stocks"
                profit_per_share = nav - market_price
            
            profit_per_unit = profit_per_share * self.creation_unit_size
            
            # Account for transaction costs
            transaction_costs = self.estimate_costs (etf_ticker)
            net_profit = profit_per_unit - transaction_costs
            
            return {
                'ticker': etf_ticker,
                'market_price': market_price,
                'nav': nav,
                'premium_discount_pct': premium_discount * 100,
                'arbitrage_opportunity': True,
                'action': action,
                'trade': trade,
                'gross_profit': profit_per_unit,
                'transaction_costs': transaction_costs,
                'net_profit': net_profit,
                'attractive': net_profit > 0
            }
        
        return {
            'ticker': etf_ticker,
            'market_price': market_price,
            'nav': nav,
            'premium_discount_pct': premium_discount * 100,
            'arbitrage_opportunity': False
        }
    
    def estimate_costs (self, etf_ticker: str) -> float:
        """
        Transaction costs for creation/redemption
        
        - Creation fee (~$500-2000)
        - Brokerage commissions
        - Bid-ask spreads on underlying stocks
        - Market impact
        """
        if etf_ticker in ['SPY', 'QQQ']:  # Highly liquid
            return 2000  # Low costs
        else:
            return 5000  # Higher costs for less liquid ETFs
    
    def monitor_realtime (self):
        """
        Real-time monitoring
        """
        for etf in self.etfs_to_monitor:
            market_price = self.get_market_price (etf)
            nav = self.get_intraday_nav (etf)  # iNAV updated every 15 seconds
            
            opportunity = self.calculate_premium_discount (etf, market_price, nav)
            
            if opportunity.get('attractive'):
                self.alert_traders (opportunity)

# Example
monitor = ETFArbitrageMonitor()

# SPY trading at small premium
spy_arb = monitor.calculate_premium_discount(
    etf_ticker='SPY',
    market_price=500.30,
    nav=500.00
)

print("\\n=== ETF Arbitrage Opportunity ===\\n")
if spy_arb['arbitrage_opportunity']:
    print(f"Ticker: {spy_arb['ticker']}")
    print(f"Premium: {spy_arb['premium_discount_pct']:.2f}%")
    print(f"Action: {spy_arb['action']}")
    print(f"Trade: {spy_arb['trade']}")
    print(f"Gross Profit per Unit: \${spy_arb['gross_profit']:,.0f}")
    print(f"Transaction Costs: \${spy_arb['transaction_costs']:,.0f}")
    print(f"Net Profit: \${spy_arb['net_profit']:,.0f}")
    print(f"Attractive: {'YES' if spy_arb['attractive'] else 'NO'}")
\`\`\`

**When the Mechanism Breaks Down:**

**March 2020 COVID Crash:**

**What Happened:**
1. **Extreme volatility**: S&P 500 dropped 34% in 23 days
2. **Liquidity crisis**: Everyone selling at once
3. **Bond ETFs hit -10% discounts**: LQD (corporate bond ETF) traded 5-10% below NAV
4. **Arbitrage couldn't keep up**: Too much selling pressure
5. **Underlying bonds stopped trading**: Can't determine fair NAV if bonds don't trade

**Why It Broke:**

\`\`\`python
class MechanismBreakdown:
    """
    Model when arbitrage breaks down
    """
    
    @staticmethod
    def why_arbitrage_fails() -> Dict:
        return {
            'Liquidity Crisis': {
                'problem': 'Underlying stocks/bonds not trading',
                'example': 'March 2020: Bond trading froze',
                'result': 'NAV calculation unreliable',
                'impact': 'APs can\\'t price risk, won\\'t arbitrage'
            },
            'Risk Capacity': {
                'problem': 'APs hit risk limits',
                'example': 'Already holding max inventory',
                'result': 'Can\\'t take on more positions',
                'impact': 'Arbitrage stops even with 5%+ discounts'
            },
            'Funding Stress': {
                'problem': 'APs can\\'t borrow to fund arbitrage',
                'example': 'March 2020: Credit markets frozen',
                'result': 'No capital for trades',
                'impact': 'Profitable arbitrage left on table'
            },
            'Operational Capacity': {
                'problem': 'Settlement delays (T+2)',
                'example': 'Can\\'t create/redeem fast enough',
                'result': 'Arbitrage too slow',
                'impact': 'Discount persists'
            },
            'Extreme Volatility': {
                'problem': 'Prices moving too fast',
                'example': 'VIX hit 80+ (normal is 15)',
                'result': 'Risk of adverse moves during settlement',
                'impact': 'APs demand wider spreads'
            }
        }

breakdown = MechanismBreakdown()
reasons = breakdown.why_arbitrage_fails()

print("\\n=== When ETF Arbitrage Breaks (March 2020) ===\\n")
for reason, details in reasons.items():
    print(f"{reason}:")
    print(f"  Problem: {details['problem']}")
    print(f"  Example: {details['example']}")
    print(f"  Impact: {details['impact']}\\n")
\`\`\`

**Key Lessons:**

1. **ETFs are not perfect**: Can trade at significant discounts in crises
2. **Bond ETFs more vulnerable**: Less liquid than stock ETFs
3. **Liquidity is not constant**: Works in normal times, fails in stress
4. **NAV can be misleading**: Based on stale prices during illiquidity

**For Trading Systems:**
- Monitor premium/discount during normal times (easy arbitrage)
- In crises, wide discounts may persist (liquidity risk)
- Bond ETFs especially vulnerable
- Have cash ready to buy discounts (if you can stomach the risk)

**Bottom Line**: ETF creation/redemption is brilliant in normal markets (tax-efficient, tight spreads). But in March 2020, it broke temporarily. Still better than mutual funds!`,
    },
    {
      id: 3,
      question:
        'Factor investing (value, momentum, quality, size, low volatility) has theoretical foundations and historical evidence of outperformance. However, many factor ETFs have underperformed since 2010. Analyze why. Design a quantitative system that combines multiple factors and dynamically adjusts factor weights based on market regimes. How would you prevent overfitting?',
      answer: `**Why Factor ETFs Have Underperformed (2010-2020):**

**1. Crowding and Arbitrage:**

**The Paradox:**
- Once factors are discovered and published (Fama-French 1993), everyone exploits them
- Capital flows into value stocks → drives up prices → reduces value premium
- "The more people know about it, the less it works"

**Example: Value Factor**
\`\`\`python
# 1970-2000: Value beat growth by ~3% annually
# 2010-2020: Value LAGGED growth by ~5% annually
# Why? Everyone knew about value → bought value stocks → no longer cheap
\`\`\`

**2. Market Regime Changes:**

**Growth/Tech Dominance (2010-2020):**
- Low interest rates → growth stocks valued higher
- Tech disruption → winner-take-all dynamics
- FAANG stocks (growth) crushed everything
- Value stocks (banks, energy, retail) disrupted

**Factor Performance is Cyclical:**
- Value works in some decades, momentum in others
- 2010s were uniquely bad for value
- Doesn't mean factor is dead (but long drought)

**3. Factor Definition and Implementation:**

**Academic vs Practical:**
- Academic factors: Use full universe, long-short, rebalance monthly
- ETF factors: Long-only, limited universe, rebalance quarterly
- Implementation gap reduces returns

**ETF Costs:**
- Expense ratios (0.15-0.30%)
- Trading costs (rebalancing)
- Market impact (everyone rebalances same day)
- Tax drag

**Designing a Multi-Factor Dynamic System:**

\`\`\`python
import numpy as np
from scipy.optimize import minimize

class DynamicMultiFactorSystem:
    """
    Combine factors and adjust weights by market regime
    """
    
    def __init__(self):
        self.factors = ['Value', 'Momentum', 'Quality', 'Size', 'LowVol']
        self.lookback_period = 252 * 3  # 3 years
        self.regime_indicators = ['VIX', 'YIELD_CURVE', 'GDP_GROWTH']
    
    def calculate_factor_returns (self, 
                                factor: str,
                                period: int) -> np.array:
        """
        Get historical factor returns
        
        In production: Fetch from factor data provider
        (Kenneth French library, AQR, etc.)
        """
        # Simplified - use real data in production
        if factor == 'Value':
            # Low P/E, P/B stocks vs high P/E, P/B
            return np.random.normal(0.08/252, 0.15/np.sqrt(252), period)
        elif factor == 'Momentum':
            # Past 12-month winners vs losers
            return np.random.normal(0.10/252, 0.18/np.sqrt(252), period)
        elif factor == 'Quality':
            # High ROE, low debt vs low ROE, high debt
            return np.random.normal(0.07/252, 0.12/np.sqrt(252), period)
        # ... etc
    
    def detect_market_regime (self) -> str:
        """
        Identify current market regime
        
        Different factors work in different regimes:
        - Bull market → Momentum, Quality
        - Bear market → Low Vol, Quality
        - High inflation → Value, Commodities
        - Low rates → Growth (anti-value)
        """
        vix = self.get_vix()
        yield_curve = self.get_yield_curve_slope()
        gdp_growth = self.get_gdp_growth()
        
        # Regime detection rules
        if vix > 30:
            return "CRISIS"  # Low vol, quality
        elif yield_curve < 0:
            return "RECESSION"  # Defensive, quality
        elif gdp_growth > 3:
            return "EXPANSION"  # Momentum, small cap
        else:
            return "NORMAL"  # Balanced
    
    def regime_based_weights (self, regime: str) -> Dict[str, float]:
        """
        Adjust factor weights based on regime
        """
        weights = {
            "CRISIS": {
                'Value': 0.10,
                'Momentum': 0.05,  # Momentum crashes in reversals
                'Quality': 0.40,   # Flight to quality
                'Size': 0.05,      # Small caps suffer
                'LowVol': 0.40     # Defense
            },
            "RECESSION": {
                'Value': 0.15,
                'Momentum': 0.15,
                'Quality': 0.35,
                'Size': 0.10,
                'LowVol': 0.25
            },
            "EXPANSION": {
                'Value': 0.20,
                'Momentum': 0.30,  # Trends persist
                'Quality': 0.20,
                'Size': 0.20,      # Small caps outperform
                'LowVol': 0.10
            },
            "NORMAL": {
                'Value': 0.20,
                'Momentum': 0.20,
                'Quality': 0.20,
                'Size': 0.20,
                'LowVol': 0.20
            }
        }
        
        return weights.get (regime, weights["NORMAL"])
    
    def markowitz_optimization (self,
                              factor_returns: Dict[str, np.array],
                              target_return: float = 0.10) -> Dict[str, float]:
        """
        Optimize factor weights using Markowitz
        
        Minimize portfolio variance for target return
        """
        # Convert to matrix
        returns_matrix = np.column_stack([factor_returns[f] for f in self.factors])
        
        # Calculate mean returns and covariance
        mean_returns = np.mean (returns_matrix, axis=0)
        cov_matrix = np.cov (returns_matrix.T)
        
        # Optimization
        num_factors = len (self.factors)
        
        def portfolio_variance (weights):
            return weights.T @ cov_matrix @ weights
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum (w) - 1},  # Weights sum to 1
            {'type': 'eq', 'fun': lambda w: np.dot (w, mean_returns) - target_return}  # Target return
        ]
        
        bounds = [(0, 1) for _ in range (num_factors)]  # Long only
        
        initial_weights = np.array([1/num_factors] * num_factors)
        
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = {self.factors[i]: result.x[i] for i in range (num_factors)}
        
        return optimal_weights
    
    def combine_regime_and_optimization (self, 
                                       regime_weights: Dict[str, float],
                                       optimal_weights: Dict[str, float],
                                       regime_confidence: float = 0.6) -> Dict[str, float]:
        """
        Blend regime-based and optimization-based weights
        
        regime_confidence: How much to trust regime vs optimization
        """
        combined = {}
        for factor in self.factors:
            combined[factor] = (
                regime_confidence * regime_weights[factor] +
                (1 - regime_confidence) * optimal_weights[factor]
            )
        
        # Normalize
        total = sum (combined.values())
        combined = {k: v/total for k, v in combined.items()}
        
        return combined
    
    def prevent_overfitting (self) -> Dict:
        """
        Techniques to prevent overfitting
        """
        return {
            'Out-of-Sample Testing': {
                'method': 'Train on 2000-2015, test on 2016-2024',
                'why': 'If only works in-sample, it\\'s overfit'
            },
            'Cross-Validation': {
                'method': 'K-fold CV on rolling windows',
                'why': 'Test on multiple time periods'
            },
            'Simple Rules': {
                'method': 'Limit to 3-5 factors, avoid 50+ parameters',
                'why': 'Complex models overfit'
            },
            'Economic Rationale': {
                'method': 'Only use factors with clear economic story',
                'why': 'Value makes sense (mean reversion), "Stocks up on Tuesdays" doesn't'
            },
            'Transaction Costs': {
                'method': 'Include realistic costs (0.2%+ per rebalance)',
                'why': 'Backtests without costs are fantasy'
            },
            'Regime Stability': {
                'method': 'Test if regime rules work across decades',
                'why': 'If only works 2010-2020, probably overfit'
            },
            'Ensemble Methods': {
                'method': 'Average multiple models',
                'why': 'Reduces overfitting risk'
            },
            'Walk-Forward Analysis': {
                'method': 'Reoptimize periodically with out-of-sample validation',
                'why': 'Adaptivity without hindsight bias'
            }
        }

# Usage
system = DynamicMultiFactorSystem()

# Detect regime
current_regime = system.detect_market_regime()
print(f"=== Dynamic Multi-Factor System ===\\n")
print(f"Current Regime: {current_regime}\\n")

# Get regime-based weights
regime_weights = system.regime_based_weights (current_regime)

print("Regime-Based Weights:")
for factor, weight in regime_weights.items():
    print(f"  {factor}: {weight*100:.0f}%")

# Get optimization-based weights (simplified)
factor_returns = {f: system.calculate_factor_returns (f, 252*10) for f in system.factors}
optimal_weights = system.markowitz_optimization (factor_returns, target_return=0.10/252)

print("\\nOptimization-Based Weights:")
for factor, weight in optimal_weights.items():
    print(f"  {factor}: {weight*100:.0f}%")

# Combine
combined_weights = system.combine_regime_and_optimization(
    regime_weights, optimal_weights, regime_confidence=0.6
)

print("\\nCombined Weights (60% regime, 40% optimization):")
for factor, weight in combined_weights.items():
    print(f"  {factor}: {weight*100:.0f}%")

# Overfitting prevention
print("\\n\\nOverfitting Prevention Techniques:")
prevention = system.prevent_overfitting()
for technique, details in list (prevention.items())[:3]:
    print(f"\\n{technique}:")
    print(f"  Method: {details['method']}")
    print(f"  Why: {details['why']}")
\`\`\`

**Implementation Considerations:**

**1. Data Quality:**
- Use long history (30+ years)
- Include survivorship-bias-free data
- Account for transaction costs (0.2%+ per trade)

**2. Rebalancing:**
- Too frequent = high costs
- Too infrequent = stale factors
- Sweet spot: Quarterly or semi-annual

**3. Risk Management:**
- Max allocation per factor: 40%
- Minimum: 10% (diversification)
- Monitor factor correlations (spike to 1 in crashes)

**4. Backtesting Reality Checks:**
- Sharpe > 1.0 → Probably overfit
- Drawdown < 15% → Definitely overfit
- If too good to be true, it is

**Key Insight**: 
Factors work, but:
1. Performance is cyclical (long droughts)
2. Crowding reduces returns over time
3. Implementation matters (costs, timing, universe)
4. Regime-awareness helps
5. Don't overfit!

**Bottom Line**: Multi-factor investing with regime adjustment can work, but keep it simple, use economic rationale, test out-of-sample, and include realistic costs. And accept that even good strategies have bad decades (value 2010-2020).`,
    },
  ],
};
