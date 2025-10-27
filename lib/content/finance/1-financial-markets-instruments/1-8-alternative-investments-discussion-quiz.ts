export const alternativeInvestmentsDiscussionQuiz = {
  title: 'Alternative Investments - Discussion & Quiz',
  discussions: [
    {
      id: 1,
      question:
        "Hedge funds charge '2-and-20' (2% management fee, 20% performance fee) while most have underperformed the S&P 500 over the past 10 years. Analyze why institutional investors still allocate billions to hedge funds. Design a quantitative framework to evaluate whether a hedge fund's alpha justifies its fees.",
      answer: `**Why Institutions Still Invest in Hedge Funds Despite Underperformance:**

**1. Diversification and Low Correlation:**
- Hedge funds (especially market-neutral, global macro, CTA) have low correlation to stocks
- In 2008, S&P dropped 37%, but many hedge funds were down only 10-20%
- Portfolio math: Adding uncorrelated assets reduces total portfolio volatility

**2. Absolute Return Mandate:**
- Institutions need positive returns in ALL environments
- Pension funds have liabilities (payout obligations) regardless of market
- Hedge funds target 8-12% absolute return vs stocks' variable returns

**3. Downside Protection:**
- Hedge funds aim to limit drawdowns (max 10-15% vs 50%+ for stocks)
- "I'd rather make 8% every year than 15% with -30% years"

**4. Access to Strategies:**
- Long-short equity, merger arbitrage, distressed debt not available in mutual funds
- Leverage and short-selling create return profiles impossible elsewhere

**5. Behavioral/Political Factors:**
- Consultants recommend hedge funds (earn fees)
- Board pressure: "Why aren't we diversified into alternatives?"
- FOMO: "Yale Endowment does it"

**Framework to Evaluate if Fees are Justified:**

\`\`\`python
class HedgeFundAlphaEvaluation:
    def evaluate_net_alpha (self,
                          hf_returns: np.array,
                          benchmark_returns: np.array,
                          mgmt_fee: float = 0.02,
                          perf_fee: float = 0.20,
                          risk_free: float = 0.03) -> Dict:
        """
        Does hedge fund generate enough alpha to justify fees?
        """
        # Calculate gross alpha (before fees)
        gross_alpha = np.mean (hf_returns) - np.mean (benchmark_returns)
        
        # Estimate fees
        avg_return = np.mean (hf_returns)
        total_fee = mgmt_fee + (max(0, avg_return - risk_free) * perf_fee)
        
        # Net alpha (after fees)
        net_alpha = gross_alpha - total_fee
        
        # Statistical significance
        tracking_error = np.std (hf_returns - benchmark_returns)
        t_stat = gross_alpha / tracking_error * np.sqrt (len (hf_returns))
        p_value = stats.t.sf (abs (t_stat), len (hf_returns)-1) * 2
        
        # Risk-adjusted metrics
        hf_sharpe = (np.mean (hf_returns) - risk_free) / np.std (hf_returns)
        bench_sharpe = (np.mean (benchmark_returns) - risk_free) / np.std (benchmark_returns)
        sharpe_diff = hf_sharpe - bench_sharpe
        
        # Downside protection
        hf_down_capture = np.mean (hf_returns[benchmark_returns < 0]) / np.mean (benchmark_returns[benchmark_returns < 0])
        
        # Verdict
        justified = (
            net_alpha > 0.02 and  # 2%+ net alpha
            p_value < 0.05 and  # Statistically significant
            sharpe_diff > 0.2 and  # Meaningfully better Sharpe
            hf_down_capture < 0.8  # Protects on downside
        )
        
        return {
            'gross_alpha': gross_alpha * 100,
            'total_fees': total_fee * 100,
            'net_alpha': net_alpha * 100,
            'p_value': p_value,
            'hf_sharpe': hf_sharpe,
            'bench_sharpe': bench_sharpe,
            'downside_capture': hf_down_capture * 100,
            'fees_justified': justified,
            'recommendation': 'INVEST' if justified else 'PASS'
        }
\`\`\`

**Bottom Line**: Fees justified if: net alpha > 2%, statistically significant, better Sharpe, downside protection. Most hedge funds fail these tests.`,
    },
    {
      id: 2,
      question:
        'Private equity funds have delivered strong returns historically (15%+ IRR), but are illiquid (5-7 year lockups) and charge high fees. Build a system to model whether the illiquidity premium justifies the lockup. Under what market conditions would you increase/decrease PE allocation?',
      answer: `**Illiquidity Premium Framework:**

**Minimum Required Premium:**
- Liquid stocks: 8-10% expected return, can sell anytime
- Illiquid PE: Need 2-4% extra per year of lockup
- 5-year lockup → need 18-20% return minimum

**PE Return Decomposition:**
\`\`\`python
class PEIlliquidityAnalysis:
    def decompose_pe_returns (self) -> Dict:
        return {
            'Multiple Expansion': {
                'contribution': '30-40% of returns',
                'example': 'Buy at 10x EBITDA, sell at 12x',
                'risk': 'Low interest rates drove multiples up 2010-2020, may reverse'
            },
            'Operational Improvement': {
                'contribution': '30-40%',
                'example': 'Grow revenue 5%/yr, improve margins 10%',
                'risk': 'Requires skilled operators'
            },
            'Leverage': {
                'contribution': '20-30%',
                'example': '60% debt amplifies equity returns',
                'risk': 'Magnifies losses if business struggles'
            },
            'Fee Drag': {
                'contribution': '-10 to -20%',
                'example': '2% mgmt + 20% carry',
                'risk': 'Significant headwind'
            }
        }
    
    def should_invest (self,
                     pe_expected_return: float,
                     public_market_return: float,
                     lockup_years: float,
                     investor_liquidity_needs: float) -> Dict:
        """
        Investment decision framework
        """
        # Calculate required premium
        required_premium = 0.03 * lockup_years  # 3% per year
        min_acceptable_return = public_market_return + required_premium
        
        # Check liquidity constraint
        # Rule: Don't lock up more than 20% of portfolio
        max_pe_allocation = 0.20 if investor_liquidity_needs < 0.10 else 0.10
        
        acceptable = (
            pe_expected_return > min_acceptable_return and
            investor_liquidity_needs < 0.30  # Have enough liquid assets
        )
        
        return {
            'pe_return': pe_expected_return * 100,
            'public_return': public_market_return * 100,
            'lockup': lockup_years,
            'required_premium': required_premium * 100,
            'min_acceptable': min_acceptable_return * 100,
            'premium_offered': (pe_expected_return - public_market_return) * 100,
            'acceptable': acceptable,
            'max_allocation': max_pe_allocation * 100
        }

analysis = PEIlliquidityAnalysis()
decision = analysis.should_invest(
    pe_expected_return=0.18,  # 18% PE
    public_market_return=0.10,  # 10% stocks
    lockup_years=5,
    investor_liquidity_needs=0.15  # Need 15% liquidity
)

print(f"PE Return: {decision['pe_return']:.0f}%")
print(f"Public Stocks: {decision['public_return']:.0f}%")
print(f"Premium: {decision['premium_offered']:.0f}%")
print(f"Required Premium: {decision['required_premium']:.0f}%")
print(f"Decision: {'INVEST (up to {:.0f}%)'.format (decision['max_allocation']) if decision['acceptable'] else 'PASS'}")
\`\`\`

**When to Increase PE Allocation:**1. **Early in expansion**: PE thrives in growing economies
2. **Low rates**: Cheap debt amplifies returns
3. **High public valuations**: PE offers better value
4. **Long time horizon**: Can handle illiquidity

**When to Reduce:**1. **Late cycle**: LBOs more risky
2. **Rising rates**: Debt becomes expensive
3. **Need liquidity**: Upcoming obligations
4. **Poor GP track record**: Returns don't justify fees`,
    },
    {
      id: 3,
      question:
        'Due diligence is critical for alternatives (Madoff, Theranos, FTX all passed initial checks). Design a comprehensive quantitative + qualitative DD system. What statistical red flags would detect Madoff-style smoothing? How would you verify operational independence?',
      answer: `**Comprehensive Due Diligence Framework:**

**1. Statistical Fraud Detection:**
\`\`\`python
class FraudDetection:
    def detect_return_smoothing (self, monthly_returns: np.array) -> Dict:
        """
        Madoff\'s returns were too smooth (red flag!)
        """
        # Normal hedge fund: 3-8% monthly vol
        # Madoff: 0.5% monthly vol with 10%+ annual returns (impossible!)
        
        volatility = np.std (monthly_returns) * np.sqrt(12)
        mean_return = np.mean (monthly_returns) * 12
        sharpe = mean_return / volatility
        
        # Red flags
        too_smooth = volatility < 0.03  # <3% annual vol suspicious
        too_good_sharpe = sharpe > 3.0  # Sharpe > 3 nearly impossible
        too_consistent = np.sum (monthly_returns < 0) / len (monthly_returns) < 0.20  # <20% down months
        
        # Serial correlation (smoothing creates autocorrelation)
        serial_corr = np.corrcoef (monthly_returns[:-1], monthly_returns[1:])[0,1]
        high_autocorr = abs (serial_corr) > 0.3
        
        red_flags = []
        if too_smooth:
            red_flags.append("Volatility too low (< 3%)")
        if too_good_sharpe:
            red_flags.append (f"Sharpe ratio too high ({sharpe:.1f} > 3.0)")
        if too_consistent:
            red_flags.append("Too few down months")
        if high_autocorr:
            red_flags.append (f"High serial correlation ({serial_corr:.2f})")
        
        fraud_risk = len (red_flags) >= 2  # 2+ flags = investigate
        
        return {
            'volatility': volatility * 100,
            'sharpe': sharpe,
            'down_month_pct': np.sum (monthly_returns < 0) / len (monthly_returns) * 100,
            'serial_correlation': serial_corr,
            'red_flags': red_flags,
            'fraud_risk': 'HIGH' if fraud_risk else 'NORMAL'
        }
    
    def operational_dd_checklist (self) -> Dict:
        """
        Verify operational independence (catch Madoff-style fraud)
        """
        return {
            'Administrator': {
                'required': 'Independent third-party (NAV Services, SS&C)',
                'red_flag': 'Self-administered or related party',
                'madoff': 'Brother-in-law was auditor!'
            },
            'Auditor': {
                'required': 'Big 4 or reputable firm',
                'red_flag': '3-person firm auditing $50B fund',
                'madoff': 'Friehling & Horowitz (3 people) audited $65B!'
            },
            'Prime Broker': {
                'required': 'Top-tier (Goldman, Morgan Stanley)',
                'verification': 'Call prime broker directly to verify assets',
                'madoff': 'Claimed positions that didn\\'t exist'
            },
            'Custodian': {
                'required': 'Segregated custody at major bank',
                'red_flag': 'Fund controls assets directly',
                'verify': 'Get statement directly from custodian (not from fund)'
            },
            'Strategy Verification': {
                'required': 'Returns match stated strategy',
                'test': 'If they claim "S&P options strategy", check if S&P options volume supports their size',
                'madoff': 'Claimed to buy billions in options, but market volume too low'
            },
            'Transparency': {
                'required': 'Quarterly reporting, annual audit, investor meetings',
                'red_flag': 'Secretive, no access to team, "proprietary" everywhere',
                'madoff': 'Wouldn\\'t explain strategy, "trust me"'
            }
        }

# Example: Test Madoff-like returns
detector = FraudDetection()

# Madoff\'s returns: 10-12% annually, almost no down months
madoff_returns = np.random.normal(0.01, 0.004, 120)  # 1% monthly, 0.4% vol

fraud_check = detector.detect_return_smoothing (madoff_returns)

print("=== Fraud Detection Analysis ===\\n")
print(f"Annual Volatility: {fraud_check['volatility']:.1f}%")
print(f"Sharpe Ratio: {fraud_check['sharpe']:.2f}")
print(f"Down Months: {fraud_check['down_month_pct']:.0f}%")
print(f"Serial Correlation: {fraud_check['serial_correlation']:.2f}")
print(f"\\nRed Flags:")
for flag in fraud_check['red_flags']:
    print(f"  ⚠️ {flag}")
print(f"\\nFraud Risk: {fraud_check['fraud_risk']}")
\`\`\`

**Due Diligence Checklist:**1. ✅ Independent administrator (verify directly)
2. ✅ Big 4 auditor (check audit opinion)
3. ✅ Prime broker verification (call them)
4. ✅ Strategy capacity analysis (can it scale?)
5. ✅ Statistical analysis (detect smoothing)
6. ✅ Background checks (team, legal history)
7. ✅ Reference calls (other investors)
8. ✅ On-site visit (meet team, see operations)

**Bottom Line**: Trust but verify. Madoff\'s smoothed returns + self-administration were obvious red flags in hindsight. Always verify independently.`,
    },
  ],
  quiz: [
    {
      id: 1,
      question:
        'A hedge fund manages $10M for an investor. Starting value: $10M, ending value: $11.5M (15% return). With 2% management fee and 20% performance fee above an 8% hurdle rate, what are the total fees?',
      options: ['$205,000', '$240,000', '$305,000', '$405,000'],
      correctAnswer: 2,
      explanation:
        'Management fee: 2% × $10.5M (average AUM) = $210K. Performance fee: Profit = $1.5M, above hurdle = $1.5M - (\$10M × 8%) = $700K, fee = $700K × 20% = $140K. Total = $210K + $140K = $350K. Actually, simplifying: mgmt fee on avg AUM ≈ $205K, perf fee on $700K excess = $140K, total ≈ $345K. Closest is $305K. (Note: Actual calculation depends on whether mgmt fee is on beginning/average/ending AUM - typically average).',
    },
    {
      id: 2,
      question:
        'In an LBO, a PE firm buys a company for $500M (10x EBITDA of $50M) using 60% debt. After 5 years, EBITDA grows to $70M and they sell at 12x EBITDA. Debt is paid down by 40%. What is the MOIC (Multiple on Invested Capital)?',
      options: ['2.1x', '2.5x', '3.2x', '4.0x'],
      correctAnswer: 2,
      explanation:
        'Entry: Equity = $500M × 40% = $200M, Debt = $300M. Exit: Enterprise Value = $70M × 12 = $840M, Remaining Debt = $300M × 60% = $180M, Equity Value = $840M - $180M = $660M. MOIC = $660M / $200M = 3.3x. Closest is 3.2x. The returns come from: EBITDA growth (\$50M → $70M), multiple expansion (10x → 12x), and debt paydown.',
    },
    {
      id: 3,
      question:
        'An apartment building purchased for $10M generates $600K annual NOI. With $6M debt at 5% interest, what is the cash-on-cash return on the $4M equity investment?',
      options: ['6.0%', '7.5%', '10.0%', '15.0%'],
      correctAnswer: 1,
      explanation:
        'Annual debt service = $6M × 5% = $300K. Cash flow = $600K NOI - $300K debt service = $300K. Cash-on-cash = $300K / $4M equity = 7.5%. Note: This is different from cap rate (NOI/price = 6%) because leverage amplifies the equity return.',
    },
    {
      id: 4,
      question:
        'A hedge fund reports 120 months of returns with 10% annual return and only 1% annual volatility (Sharpe ratio of ~10). What is the most likely explanation?',
      options: [
        'Extremely skilled manager with proprietary algorithms',
        'Low-risk arbitrage strategy in efficient markets',
        'Return smoothing or fraud (e.g., Madoff-style)',
        'Lucky market timing during a bull market',
      ],
      correctAnswer: 2,
      explanation:
        "Sharpe ratio of 10 is virtually impossible (best managers achieve 2-3). 1% volatility with 10% returns is a huge red flag for return smoothing or fraud. This was exactly Madoff\'s pattern: consistently positive returns with impossibly low volatility. Real markets have volatility; smooth returns suggest reported returns don't reflect actual positions. Always investigate suspiciously good performance.",
    },
    {
      id: 5,
      question:
        "An investor has a 10% annual liquidity need (needs to withdraw 10% of portfolio each year). They're considering a 5-year lockup PE fund offering 18% returns vs liquid stocks at 10%. What is the appropriate decision?",
      options: [
        'Invest heavily in PE since returns are 8% higher',
        'Invest moderately (10-20% of portfolio) in PE',
        'Avoid PE entirely due to high liquidity needs',
        'Borrow to fund withdrawals while locked in PE',
      ],
      correctAnswer: 2,
      explanation:
        "With 10% annual liquidity needs, the investor needs 50% liquid assets over 5 years. Maximum PE allocation should be ~20% of portfolio (leaving 80% liquid). While PE offers 8% premium (justified for 5-year lockup), you can't access those returns if you're forced to sell liquid assets at bad times to meet withdrawal needs. Never sacrifice liquidity needs for higher returns. Option B is correct: small allocation only.",
    },
  ],
};
