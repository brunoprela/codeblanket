export const alternativeInvestmentsQuiz = [
  {
    id: 'fm-1-8-q-1',
    question:
      "Hedge funds charge '2-and-20' (2% management fee, 20% performance fee) while most have underperformed the S&P 500 over the past 10 years. Analyze why institutional investors still allocate billions to hedge funds. Design a quantitative framework to evaluate whether a hedge fund's alpha justifies its fees.",
    sampleAnswer: `**Why Institutions Still Invest Despite Underperformance:**

**1. Diversification:** Low correlation to stocks (2008: S&P -37%, hedge funds -15%)
**2. Absolute Return:** Target 8-12% in all environments (pensions need this)
**3. Downside Protection:** Limit drawdowns to 10-15% vs 50%+ for stocks
**4. Strategy Access:** Long-short, merger arb, distressed debt unavailable elsewhere
**5. Behavioral:** FOMO, consultant fees, board pressure

**Alpha Evaluation Framework:**

\`\`\`python
class HedgeFundAlphaEvaluation:
    def evaluate_net_alpha(self, hf_returns, benchmark_returns):
        # Gross alpha
        gross_alpha = mean(hf_returns) - mean(benchmark_returns)
        
        # Fees: 2% + 20% of returns above risk-free
        total_fee = 0.02 + (max(0, mean(hf_returns) - 0.03) * 0.20)
        
        # Net alpha
        net_alpha = gross_alpha - total_fee
        
        # Statistical significance
        t_stat = gross_alpha / std(hf_returns - benchmark_returns) * sqrt(n)
        p_value = t_test(t_stat)
        
        # Downside protection
        down_capture = mean(hf_returns[benchmark < 0]) / mean(benchmark_returns[benchmark < 0])
        
        # Fees justified if:
        justified = (
            net_alpha > 0.02 and  # 2%+ net alpha
            p_value < 0.05 and  # Statistically significant
            down_capture < 0.80  # Protects downside
        )
        
        return justified
\`\`\`

**Bottom Line:** Fees justified only if: Net alpha > 2%, statistically significant, protects downside. Most hedge funds fail.`,
    keyPoints: [
      'Institutions invest for: diversification (low correlation), absolute return (8-12%), downside protection',
      'Fees: 2% mgmt + 20% performance = ~3-4% annual drag',
      'Evaluation: Net alpha > 2%, p < 0.05, downside capture < 80%',
      'Reality: Most hedge funds underperform after fees',
      'Justified only for top-tier funds with proven alpha and risk management',
    ],
  },
  {
    id: 'fm-1-8-q-2',
    question:
      'Private equity funds deliver strong returns (15%+ IRR) but are illiquid (5-7 year lockups) and charge high fees. Build a system to model whether the illiquidity premium justifies the lockup. Under what market conditions would you increase/decrease PE allocation?',
    sampleAnswer: `**Illiquidity Premium Framework:**

Required premium: 3% per year of lockup
- Stocks: 10% return, liquid
- PE: Need 10% + (3% × 5 years) = 25% return for 5-year lockup

**PE Return Sources:**
1. Multiple expansion (30-40%): Buy 10x EBITDA, sell 12x
2. Operational improvement (30-40%): Grow revenue, improve margins
3. Leverage (20-30%): 60% debt amplifies equity returns
4. Fee drag (-10 to -20%): 2% + 20% carry

Net: ~15-18% IRR historically

**Decision Model:**

\`\`\`python
class PEAllocationDecision:
    def should_invest(self, pe_return, public_return, lockup_years, liquidity_need):
        required_premium = 0.03 * lockup_years
        min_acceptable = public_return + required_premium
        
        acceptable = (
            pe_return > min_acceptable and
            liquidity_need < 0.30  # Have enough liquid assets
        )
        
        max_allocation = 0.20 if liquidity_need < 0.10 else 0.10
        
        return acceptable, max_allocation

# Example: 18% PE vs 10% stocks, 5-year lockup, 15% liquidity need
decision = should_invest(0.18, 0.10, 5, 0.15)
# Returns: (True, 0.15) → Invest up to 15%
\`\`\`

**Increase PE When:**
1. Early expansion (PE thrives in growth)
2. Low rates (cheap debt)
3. High public valuations (PE offers value)
4. Long time horizon

**Decrease PE When:**
1. Late cycle (LBO risk)
2. Rising rates (expensive debt)
3. Need liquidity soon
4. Poor GP track record

**Bottom Line:** PE justified if premium > 3% per year lockup, have liquidity buffer, and favorable macro.`,
    keyPoints: [
      'Illiquidity premium: Need 3% per year locked up (5-year lockup → 15% extra return)',
      'PE returns: 15-18% IRR from multiple expansion, operations, leverage, minus fees',
      'Max allocation: 20% if low liquidity need, 10% if higher need',
      'Increase: Early cycle, low rates, high public valuations',
      'Decrease: Late cycle, rising rates, upcoming liquidity needs',
    ],
  },
  {
    id: 'fm-1-8-q-3',
    question:
      'Due diligence is critical for alternatives (Madoff, Theranos, FTX all passed initial checks). Design a comprehensive quantitative + qualitative DD system. What statistical red flags would detect Madoff-style smoothing? How would you verify operational independence?',
    sampleAnswer: `**Fraud Detection Framework:**

**1. Statistical Red Flags:**

\`\`\`python
class FraudDetection:
    def detect_return_smoothing(self, monthly_returns):
        """Madoff's returns were too smooth"""
        vol = std(monthly_returns)
        sharpe = mean(monthly_returns) / vol
        
        # Red flags:
        red_flags = []
        if vol < 0.02:  # <2% monthly vol
            red_flags.append("Too smooth (Madoff: 0.5% vol)")
        if sharpe > 2.0:  # Sharpe > 2 very rare
            red_flags.append("Suspiciously high risk-adjusted return")
        if max_drawdown(monthly_returns) < 0.05:
            red_flags.append("No meaningful drawdown (unrealistic)")
        
        return red_flags
    
    def detect_style_drift(self, returns, market_returns):
        """Check if returns match stated strategy"""
        correlation = corr(returns, market_returns)
        
        # Madoff claimed market-neutral but had 0.7 correlation to S&P
        if abs(correlation) > 0.3 and strategy == "market_neutral":
            return "Style drift: Not actually market neutral"
    
    def check_consistency(self, monthly_returns):
        """Too many positive months"""
        pct_positive = sum(returns > 0) / len(returns)
        
        if pct_positive > 0.85:  # >85% positive months
            return "Unrealistic: Even best managers have 65-70% win rate"
\`\`\`

**Madoff Red Flags Missed:**
- 0.5% monthly vol with 12% annual return (Sharpe ~7, impossible)
- 95%+ positive months (too consistent)
- Perfect smoothness (no jumps, no gaps)
- Zero correlation to any strategy (claimed split-strike conversion but didn't match)

**2. Operational Due Diligence:**

**Verify Independence:**
- ✅ **Separate administrator:** Not fund's own back office
- ✅ **Big-4 auditor:** PwC, Deloitte, not 3-person firm
- ✅ **Prime broker verification:** Call PB directly, verify positions
- ✅ **Custodian independence:** Who holds assets? (Madoff was own custodian!)
- ✅ **Reg visits:** Visit office, meet full team
- ✅ **Reference checks:** Talk to investors, employees, counterparties

**Madoff Failures:**
- Brother was auditor (not independent)
- Small accounting firm in strip mall
- Self-custodian (held own assets)
- SEC visited but didn't verify trades

**3. Comprehensive DD Checklist:**

\`\`\`python
class DueDiligenceChecklist:
    def evaluate(self, fund):
        checks = {
            'Statistical': {
                'Return smoothness': check_smoothness(),
                'Sharpe ratio': check_sharpe(),
                'Style consistency': check_style(),
                'Correlation analysis': check_correlations()
            },
            'Operational': {
                'Independent admin': verify_admin(),
                'Big-4 auditor': verify_auditor(),
                'Prime broker': verify_pb(),
                'Custodian': verify_custodian(),
                'Key person risk': check_team_depth()
            },
            'Legal': {
                'Regulatory history': check_finra(),
                'Litigation': search_court_records(),
                'Related parties': check_conflicts()
            },
            'Qualitative': {
                'Office visit': in_person_meeting(),
                'References': call_investors(),
                'Strategy explanation': stress_test_story()
            }
        }
        
        # ALL must pass
        passed = all(all(category.values()) for category in checks.values())
        
        return passed
\`\`\`

**Bottom Line:**
- **Statistical:** Too smooth, too consistent, too good = fraud
- **Operational:** Self-custody, no independent admin, small auditor = red flags
- **Rule:** If you can't understand it, don't invest in it

**Madoff Lesson:** "Trust but verify" means actually verify. Call prime broker. Visit auditor. Check trade confirms. One layer deeper prevents most fraud.`,
    keyPoints: [
      'Statistical red flags: Vol < 2%, Sharpe > 2, no drawdowns, >85% positive months',
      'Madoff: 0.5% vol with 12% return (Sharpe ~7, impossible), 95%+ positive months',
      'Operational DD: Independent admin, Big-4 auditor, verify prime broker, separate custodian',
      'Madoff failures: Self-custodian, brother as auditor, small accounting firm',
      'Rule: If too good to be true (or too smooth), it is. Verify everything independently.',
    ],
  },
];
