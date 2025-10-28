export default {
  id: 'fin-m15-s1-discussion',
  title: 'Risk Management Fundamentals - Discussion Questions',
  questions: [
    {
      question:
        'Explain the three lines of defense model in risk management and why the independence of the risk function is critical. Provide examples of what happens when these lines blur, referencing real-world failures.',
      answer: `The Three Lines of Defense is the foundational framework for enterprise risk management. Understanding why each line must be independent is crucial:

**The Three Lines:**

**First Line: Business Operations**
- **Who**: Trading desks, portfolio managers, business units
- **Role**: Owns and manages risk daily
- **Responsibilities**:
  - Execute trades within limits
  - Monitor positions in real-time
  - Implement risk controls
  - First-level risk checks

Example activities:
\`\`\`python
# First line: Trader checks position before trading
def check_position_limit(symbol, quantity, limit):
    current_position = get_current_position(symbol)
    new_position = current_position + quantity
    
    if abs(new_position) > limit:
        return False  # Reject trade
    return True
\`\`\`

**Second Line: Risk Management**
- **Who**: Independent risk management function
- **Role**: Oversees and challenges first line
- **Responsibilities**:
  - Set risk limits
  - Monitor portfolio risk (VaR, stress tests)
  - Challenge business decisions
  - Report to senior management
  - Independent verification

Example activities:
\`\`\`python
# Second line: Independent risk calculation
class RiskManagement:
    def calculate_portfolio_var(self, positions):
        """Independent VaR calculation"""
        # Cannot be overridden by trading desk
        var = self._monte_carlo_var(positions)
        
        if var > self.var_limit:
            self.alert_management()
            self.escalate_limit_breach()
        
        return var
\`\`\`

**Third Line: Internal Audit**
- **Who**: Internal audit function
- **Role**: Independent assurance
- **Responsibilities**:
  - Audit risk processes
  - Verify controls work
  - Test compliance
  - Report to board/audit committee

**Why Independence Is Critical:**

**1. Prevents Conflicts of Interest**

Without independence:
\`\`\`
Trader: "I want to exceed my limit on this trade"
Risk Manager (reporting to trader): "Okay, since you're my boss..."
Result: Risk limit has no meaning
\`\`\`

With independence:
\`\`\`
Trader: "I want to exceed my limit"
Risk Manager (independent): "Rejected. Escalated to CRO."
Result: Limit is enforced
\`\`\`

**2. Unbiased Risk Assessment**

Problem: If risk reports to business:
- Risk metrics get "adjusted" to look better
- Stress tests use favorable scenarios
- VaR models changed to reduce reported risk
- Bad news is suppressed

**Real-World Failure #1: JPMorgan London Whale (2012)**

**What Happened:**
- Chief Investment Office (CIO) took massive synthetic credit positions
- Positions grew to $157B notional
- Lost $6.2B

**Independence Failures:**

1. **Risk Reporting Structure:**
   - CIO risk function reported to CIO management
   - NOT independent of business
   - Risk managers pressured to approve trades

2. **VaR Model Change:**
   - CIO changed VaR model to report lower risk
   - New model had errors (used wrong correlation approach)
   - Reduced VaR from $132M to $67M
   - Made positions look safer than they were

3. **Limit Breaches Ignored:**
   - Position breached CS01 limits (credit risk)
   - Risk function raised concerns
   - Business overruled risk objections
   - No independent escalation

4. **Missing Third Line:**
   - Model Risk group didn't independently validate new VaR model
   - Internal audit didn't catch problems until after losses

**Lessons:**
\`\`\`
❌ Risk function reported to CIO (business)
✅ Should report to CRO (independent)

❌ Business changed risk models without validation
✅ Models must be independently validated

❌ Limit breaches ignored by management
✅ Hard limits cannot be overridden

❌ No independent audit of risk process
✅ Regular third-line reviews required
\`\`\`

**Real-World Failure #2: Long-Term Capital Management (1998)**

**What Happened:**
- Hedge fund with Nobel Prize winners
- Lost $4.6B in 4 months
- Required Fed bailout

**Independence Failures:**

1. **No Independent Risk Function:**
   - Partners managed their own risk
   - No second line of defense
   - Risk models created by same people making trades

2. **Groupthink:**
   - Everyone believed their models were perfect
   - No independent challenge
   - Correlation assumptions proved wrong in crisis

3. **Lack of Oversight:**
   - Lenders didn't independently assess risk
   - Trusted firm's own risk metrics
   - No independent stress testing

**Real-World Failure #3: AIG Financial Products (2008)**

**What Happened:**
- Sold $500B+ of credit default swaps
- Lost $99B, required $182B government bailout

**Independence Failures:**

1. **Risk Function Weakness:**
   - AIG FP risk management was weak
   - Reported to FP management (business)
   - Concerns ignored

2. **Corporate Risk Didn't Intervene:**
   - AIG corporate risk knew about exposures
   - Didn't stop FP from taking more risk
   - Lacked authority to enforce limits

3. **Ratings Agencies Failed:**
   - Should have been independent third line
   - Didn't understand tail risk
   - Maintained AAA rating too long

**Proper Independence Structure:**

\`\`\`
Board of Directors
    ↓
Chief Risk Officer (CRO)
    ↓
Risk Management Function
    ↓
(Reports to CRO, NOT to business)

Reporting lines:
✅ CRO reports to CEO and Board Risk Committee
✅ Risk function reports to CRO
✅ Internal Audit reports to Board Audit Committee
❌ Risk NEVER reports to trading/business heads
\`\`\`

**Implementation in Code:**

\`\`\`python
class RiskManagementSystem:
    """
    Properly independent risk system
    """
    def __init__(self, cro_approval_required=True):
        self.cro_approval_required = cro_approval_required
        self.limit_overrides = []  # Audit trail
        
    def check_trade(self, trade, trader_id):
        """
        Independent risk check
        Cannot be overridden by trader
        """
        # First line: Trader's own check
        if not self.first_line_check(trade):
            return {'approved': False, 'reason': 'First line rejection'}
        
        # Second line: Independent risk check
        risk_result = self.second_line_risk_check(trade)
        
        if not risk_result['approved']:
            # Cannot be overridden without CRO approval
            if self.cro_approval_required:
                # Escalate to CRO
                self.escalate_to_cro(trade, trader_id, risk_result)
                return {'approved': False, 'reason': 'Requires CRO approval'}
            
        return risk_result
    
    def escalate_to_cro(self, trade, trader_id, risk_result):
        """
        Escalation path when limits breached
        """
        escalation = {
            'timestamp': datetime.now(),
            'trader_id': trader_id,
            'trade': trade,
            'risk_assessment': risk_result,
            'escalated_to': 'CRO',
            'auto_escalation': True  # Cannot be suppressed
        }
        
        # Send to CRO
        self.send_alert(escalation)
        
        # Log for audit trail (third line)
        self.log_escalation(escalation)
        
        print("🚨 ESCALATED TO CRO: Trade exceeds risk limits")
        print(f"   Trader: {trader_id}")
        print(f"   Reason: {risk_result['reason']}")
\`\`\`

**What Happens When Lines Blur:**

**Scenario 1: Risk reports to business**
\`\`\`
Business: "Change the VaR model, our risk looks too high"
Risk (reporting to business): "Okay, boss"
Result: London Whale
\`\`\`

**Scenario 2: No independent risk function**
\`\`\`
Trader: "These positions are fine based on my analysis"
No one to challenge: Risk grows unchecked
Result: LTCM
\`\`\`

**Scenario 3: Audit doesn't verify**
\`\`\`
Business: "We're within all our limits"
Audit doesn't check: Limits are actually breached
Result: Fraud goes undetected
\`\`\`

**Best Practices for Independence:**

1. **Reporting Structure:**
   \`\`\`
   ✅ CRO reports to CEO and Board
   ✅ Risk reports to CRO
   ✅ Compensation tied to risk metrics, not P&L
   ❌ Risk reports to business heads
   ❌ Risk bonus tied to trading profits
   \`\`\`

2. **Systems Architecture:**
   \`\`\`python
   # Independent risk calculation
   class IndependentRiskEngine:
       """Cannot be modified by trading systems"""
       def __init__(self):
           self.read_only_for_traders = True
           
       def calculate_var(self, positions):
           # Trading desk cannot change this calculation
           # Uses independent market data feed
           # Models validated by Model Risk group
           return self._calc()
   \`\`\`

3. **Model Governance:**
   \`\`\`
   ✅ Model Risk group validates all models
   ✅ Changes require independent approval
   ✅ Regular backtesting
   ❌ Business changes models to reduce risk
   ❌ Self-validation
   \`\`\`

4. **Limit Framework:**
   \`\`\`
   ✅ Hard limits enforced by independent system
   ✅ Overrides require CRO approval
   ✅ All overrides logged and audited
   ❌ Traders can override limits
   ❌ Soft limits with no enforcement
   \`\`\`

**Summary - Why Independence Matters:**

Without independent risk management:
- Business pressure distorts risk metrics
- Limits become suggestions, not rules
- Bad news gets suppressed
- Risk grows unchecked until catastrophe

With proper independence:
- Unbiased risk assessment
- Limits are enforced
- Early warning of problems
- Management gets accurate information

**The pattern is clear:** Every major risk management failure involves some breakdown in the independence of risk function. London Whale, LTCM, AIG - all had risk functions that were either weak, conflicted, or ignored.

**Bottom Line:** Independence isn't bureaucracy - it's the safety net that prevents catastrophic losses. The three lines of defense only work when each line is truly independent.`,
    },
    {
      question:
        'Compare and contrast Value at Risk (VaR), Conditional Value at Risk (CVaR), and stress testing as risk measurement tools. What are the strengths and weaknesses of each, and in what situations would you use one over another?',
      answer: `Each risk metric serves a different purpose and has different blind spots. Understanding when to use each is critical:

**Value at Risk (VaR)**

**Definition:** Maximum loss at a given confidence level over a time horizon.

Example: "99% 1-day VaR of $5M" means:
- 99% confidence we won't lose more than $5M tomorrow
- 1% chance of losing more than $5M

**Calculation Example:**
\`\`\`python
import numpy as np

def calculate_historical_var(returns, confidence=0.99):
    """
    Historical VaR calculation
    """
    # Sort returns (losses are negative)
    sorted_returns = np.sort(returns)
    
    # Find percentile
    index = int((1 - confidence) * len(returns))
    var = -sorted_returns[index]  # Convert to loss
    
    return var

# Example: Portfolio returns
returns = np.random.normal(0.001, 0.02, 1000)  # Mean 0.1%, std 2%
var_99 = calculate_historical_var(returns, 0.99)

print(f"99% VaR: {var_99*100:.2f}%")
# Interpretation: 99% confident won't lose more than this % tomorrow
\`\`\`

**Strengths of VaR:**

✅ **Single Number:**
\`\`\`
CEO: "What's our risk?"
CRO: "99% VaR is $5M"
CEO: "Got it."
\`\`\`
Simple to communicate to senior management.

✅ **Widely Understood:**
- Industry standard
- Regulatory requirement (Basel)
- Comparable across firms

✅ **Easy to Aggregate:**
\`\`\`python
# Portfolio VaR considers correlations
portfolio_var = calculate_portfolio_var(positions)
# Can compare: desk A var, desk B var, total var
\`\`\`

✅ **Multiple Calculation Methods:**
- Historical
- Parametric (fast)
- Monte Carlo (flexible)

**Weaknesses of VaR:**

❌ **Doesn't Measure Tail Risk:**

\`\`\`python
# Two portfolios:

# Portfolio A: 99% VaR = $5M
# In 1% case: loses $6M

# Portfolio B: 99% VaR = $5M  
# In 1% case: loses $100M

# VaR is same, but B is much riskier!
\`\`\`

VaR only tells you the cutoff, not how bad it gets beyond that.

❌ **Not Sub-Additive:**

\`\`\`python
# Mathematically possible:
var_portfolio_A = 10
var_portfolio_B = 10
var_combined = 25  # Greater than A + B!

# Violates diversification intuition
\`\`\`

This means VaR can increase when you diversify (theoretical problem).

❌ **Gives False Sense of Security:**

\`\`\`
"99% VaR is $5M, so we're safe"
↓
2008 happens (multi-sigma event)
↓
Lose $500M
↓
"But that was only 1% probability!"
\`\`\`

The 1% is where catastrophes hide.

❌ **Model Risk:**

\`\`\`python
# Historical VaR: Assumes future like past
# 2007: Historical VaR looked great
# 2008: Historical data didn't include financial crisis

# Parametric VaR: Assumes normal distribution
# Real returns have fat tails (more extreme events)
\`\`\`

**Conditional Value at Risk (CVaR / Expected Shortfall)**

**Definition:** Average loss in the worst (1-confidence)% of cases.

Example: "99% CVaR of $8M" means:
- In the worst 1% of days
- Average loss is $8M

**Calculation Example:**
\`\`\`python
def calculate_cvar(returns, confidence=0.99):
    """
    CVaR = average loss beyond VaR
    """
    # Sort returns
    sorted_returns = np.sort(returns)
    
    # Find VaR cutoff
    var_index = int((1 - confidence) * len(returns))
    
    # CVaR = average of losses beyond VaR
    tail_losses = sorted_returns[:var_index]
    cvar = -np.mean(tail_losses)
    
    return cvar

# Example
returns = np.random.normal(0.001, 0.02, 1000)
var_99 = calculate_historical_var(returns, 0.99)
cvar_99 = calculate_cvar(returns, 0.99)

print(f"99% VaR: {var_99*100:.2f}%")
print(f"99% CVaR: {cvar_99*100:.2f}%")
# CVaR is always >= VaR
\`\`\`

**Strengths of CVaR:**

✅ **Measures Tail Risk:**
\`\`\`python
# Portfolio A: VaR = $5M, CVaR = $6M (mild tail)
# Portfolio B: VaR = $5M, CVaR = $20M (severe tail)

# CVaR distinguishes them!
\`\`\`

✅ **Sub-Additive:**
\`\`\`
CVaR(A + B) ≤ CVaR(A) + CVaR(B)
\`\`\`
Satisfies mathematical coherence properties.

✅ **Better for Optimization:**
\`\`\`python
# CVaR optimization produces smoother, more stable portfolios
def optimize_cvar(returns, target_return):
    # Minimize CVaR subject to return target
    # Better behaved than VaR optimization
    pass
\`\`\`

✅ **Captures Severity:**
"Not just how often we lose, but how much we lose when we do"

**Weaknesses of CVaR:**

❌ **Less Intuitive:**
\`\`\`
CFO: "What does 99% CVaR of $8M mean?"
CRO: "Well, if we condition on the worst 1% of outcomes..."
CFO: "Just tell me the maximum loss!"
\`\`\`

Harder to explain than VaR.

❌ **More Computation:**
\`\`\`python
# VaR: Find one percentile
var = np.percentile(losses, 99)

# CVaR: Find percentile, then average beyond it
cvar = losses[losses > var].mean()
# Requires more data/computation
\`\`\`

❌ **Still Model-Dependent:**
\`\`\`python
# If your model is wrong, CVaR is wrong
# Assumes you've captured the tail distribution
# 2008: Models didn't have fat enough tails
\`\`\`

❌ **Regulatory Acceptance:**
Basel uses VaR, not CVaR (though this is changing).

**Stress Testing**

**Definition:** Estimate losses under specific scenarios.

Example: "What if 2008 repeats?"

**Calculation Example:**
\`\`\`python
def stress_test_2008_crisis(portfolio):
    """
    Apply 2008 crisis scenario to portfolio
    """
    # Historical scenario
    scenarios = {
        'S&P 500': -0.45,      # Down 45%
        'Investment Grade': -0.15,
        'High Yield': -0.30,
        'Mortgage Backed': -0.50,
        'VIX': 3.0,            # Triple
        'Spreads': 2.5         # 2.5x wider
    }
    
    # Apply to each position
    stressed_value = 0
    for position in portfolio:
        shock = scenarios.get(position.asset_class, 0)
        stressed_value += position.value * (1 + shock)
    
    loss = portfolio.total_value - stressed_value
    
    return {
        'scenario': '2008 Financial Crisis',
        'loss': loss,
        'loss_pct': loss / portfolio.total_value
    }

# Example
result = stress_test_2008_crisis(my_portfolio)
print(f"2008 Scenario Loss: \${result['loss']/1e6:.1f}M ({result['loss_pct']*100:.1f}%)")
\`\`\`

**Strengths of Stress Testing:**

✅ **Captures Tail Events:**
\`\`\`python
# VaR might say: 1% chance of losing $5M
# Stress test says: If 2008 repeats, lose $50M

# Stress test captures the "impossible" events
\`\`\`

✅ **Specific Scenarios:**
\`\`\`python
scenarios = {
    'COVID Crash': test_covid(),
    'Dot-com Bubble': test_dotcom(),
    'Flash Crash': test_flash_crash(),
    'Rate Spike': test_rate_shock()
}

# Management can understand "What if X happens?"
\`\`\`

✅ **Tests Correlations:**
\`\`\`python
# In stress, correlations go to 1
# Everything falls together
# Stress tests capture this
\`\`\`

✅ **Regulatory Requirement:**
\`\`\`
# CCAR: Comprehensive Capital Analysis and Review
# Fed mandates stress tests for big banks
# Must prove can survive severe recession
\`\`\`

✅ **Forward-Looking:**
\`\`\`python
# Can test scenarios that haven't happened yet
def test_cyber_attack():
    # Model operational disruption
    # Revenue loss
    # Legal costs
    pass
\`\`\`

**Weaknesses of Stress Testing:**

❌ **No Probability:**
\`\`\`
"2008 scenario: Lose $50M"
"But how likely is 2008 to repeat?"
"We don't know..."
\`\`\`

Stress tests don't give probabilities.

❌ **Scenario Selection Bias:**
\`\`\`python
# Only test scenarios you think of
# 2019: No one tested "global pandemic"
# 2020: COVID happens
\`\`\`

Next crisis will be something you didn't test.

❌ **Complex to Aggregate:**
\`\`\`
Desk A 2008 loss: $10M
Desk B 2008 loss: $15M
Total 2008 loss: ???

# Can't just add (correlations, interactions)
\`\`\`

❌ **Can Be Gamed:**
\`\`\`python
# Business: "Let's test mild scenarios"
# Shows low stress losses
# Management: "We're safe!"
# Reality: Severe scenarios still possible
\`\`\`

**When to Use Each:**

**Use VaR for:**

1. **Daily Risk Reporting:**
\`\`\`python
# Daily risk report
print(f"Today's 99% VaR: \${var/1e6:.1f}M")
# Simple, tracks trend
\`\`\`

2. **Limit Setting:**
\`\`\`python
# Desk limits
desk_limit = 5000000  # $5M VaR
if desk_var > desk_limit:
    reject_trade()
\`\`\`

3. **Regulatory Capital:**
\`\`\`python
# Basel III
market_risk_capital = 3 * var_60_day
# VaR is regulatory standard
\`\`\`

4. **When You Need Quick Answer:**
VaR is fast to calculate and communicate.

**Use CVaR for:**

1. **Portfolio Optimization:**
\`\`\`python
# Minimize CVaR instead of VaR
# Produces better diversified portfolios
optimized_weights = minimize_cvar(returns)
\`\`\`

2. **Tail Risk Focus:**
\`\`\`python
# When tail losses matter more
# Hedge funds, prop trading
if cvar > cvar_limit:
    reduce_tail_risk()
\`\`\`

3. **Risk-Adjusted Performance:**
\`\`\`python
# Sharpe uses std dev
# Better: Return / CVaR
risk_adjusted_return = mean_return / cvar
\`\`\`

4. **Academic/Research:**
CVaR is mathematically superior for research.

**Use Stress Testing for:**

1. **Capital Planning:**
\`\`\`python
# CCAR: Can we survive severe recession?
stress_loss = test_severe_recession()
capital_buffer = tier1_capital - stress_loss
# Must stay above minimums
\`\`\`

2. **Understanding Vulnerabilities:**
\`\`\`python
# What if our biggest counterparty defaults?
# What if liquidity dries up?
# What if all hedges fail?
scenarios = test_vulnerabilities()
\`\`\`

3. **Explaining Tail Risk:**
\`\`\`
Board: "What happens in a crisis?"
CRO: "2008 scenario shows $50M loss"
Board: "We need more capital"
\`\`\`

4. **Reverse Stress Testing:**
\`\`\`python
# What scenario makes us insolvent?
def find_breaking_point():
    # Ramp up severity until failure
    # Identifies true vulnerability
    pass
\`\`\`

**Best Practice: Use All Three**

\`\`\`python
class ComprehensiveRiskReport:
    """
    Complete risk picture uses all metrics
    """
    def daily_risk_report(self, portfolio):
        return {
            # VaR: Day-to-day risk
            'var_95': self.calculate_var(portfolio, 0.95),
            'var_99': self.calculate_var(portfolio, 0.99),
            
            # CVaR: Tail risk
            'cvar_99': self.calculate_cvar(portfolio, 0.99),
            
            # Stress: Crisis risk
            'stress_2008': self.stress_test_2008(portfolio),
            'stress_covid': self.stress_test_covid(portfolio),
            'stress_rates': self.stress_test_rate_shock(portfolio)
        }

# Each metric shows different aspect
# VaR: "Normal" risk
# CVaR: Tail severity
# Stress: Specific crisis scenarios
\`\`\`

**Real-World Example: 2008 Crisis**

**VaR Said:** "99% VaR is $10M"
→ Implied: 99% confident won't lose more

**Reality:** Lost $500M (50x VaR)
→ Problem: 2008 was multi-sigma event VaR doesn't capture

**CVaR Might Have Said:** "99% CVaR is $50M"
→ Better: Shows tail losses are severe
→ Still wrong: Tail was fatter than model

**Stress Test Said:** "1987 crash scenario: Lose $300M"
→ Warning: Showed potential for large losses
→ But: Management said "1987 won't repeat"

**Lesson:** Use all three. VaR for day-to-day, CVaR for tail awareness, stress tests for "what if" planning.

**Bottom Line:**

- **VaR**: Daily risk metric, simple but blind to tails
- **CVaR**: Better tail risk measure, harder to explain
- **Stress Testing**: Scenario-based, no probability

Use VaR for daily monitoring, CVaR for optimization and tail risk, stress testing for capital planning and crisis preparedness. No single metric is sufficient - you need all three perspectives.`,
    },
    {
      question:
        'Discuss the key differences between market risk, credit risk, operational risk, and liquidity risk. How do they interact during a crisis, and why is it important to model these interactions rather than treating each risk type in isolation?',
      answer: `Risk types are often modeled separately, but they interact in dangerous ways during crises. Understanding these interactions is critical:

**The Four Major Risk Types:**

**1. Market Risk**

**Definition:** Risk of losses from adverse market movements (prices, rates, volatility).

**Examples:**
\`\`\`python
# Stock price drops
position = 10000  # shares
price_drop = -20  # $20 per share
market_risk_loss = position * price_drop  # -$200,000

# Interest rate spike
bond_duration = 7.5
rate_increase = 0.01  # 100 bp
bond_loss_pct = -duration * rate_increase  # -7.5%

# Volatility spike (for options)
vega = 50000  # $ per vol point
vol_increase = 0.10  # 10 vol points
options_loss = -vega * vol_increase  # -$500,000
\`\`\`

**Characteristics:**
- Measurable with VaR, CVaR
- Can be hedged (options, futures)
- Liquid markets (usually)
- Mark-to-market daily

**2. Credit Risk**

**Definition:** Risk that counterparty fails to pay obligations.

**Examples:**
\`\`\`python
# Corporate bond default
bond_face_value = 1000000
recovery_rate = 0.40  # Recover 40 cents on dollar
credit_loss = bond_face_value * (1 - recovery_rate)  # $600,000

# Derivative counterparty default
derivative_mtm = 5000000  # Mark-to-market value
counterparty_defaults = True
recovery = 0.20
credit_loss = derivative_mtm * (1 - recovery)  # $4,000,000

# Loan default
loan_amount = 10000000
probability_default = 0.02
loss_given_default = 0.60
expected_loss = loan_amount * probability_default * loss_given_default
# $120,000
\`\`\`

**Characteristics:**
- Measured with PD, LGD, EAD
- Credit VaR for portfolio
- Less liquid than market risk
- Can be hedged (CDS, collateral)

**3. Operational Risk**

**Definition:** Risk from failed processes, people, systems, or external events.

**Examples:**
\`\`\`python
# Trading error
intended_order = 1000  # shares
actual_order = 1000000  # Fat finger error!
price = 50
error_cost = (actual_order - intended_order) * price  # $49,950,000

# System failure
downtime_hours = 4
revenue_per_hour = 1000000
opportunity_cost = downtime_hours * revenue_per_hour  # $4,000,000

# Fraud
embezzlement_amount = 5000000
detection_delay_years = 3
total_fraud_loss = embezzlement_amount * detection_delay_years

# Legal/regulatory
fine_amount = 50000000
legal_costs = 10000000
reputation_cost = 100000000  # Lost business
total_cost = fine_amount + legal_costs + reputation_cost
\`\`\`

**Characteristics:**
- Hardest to measure
- Cannot be hedged away
- Measured with scenario analysis
- Basel III: Advanced Measurement Approach

**4. Liquidity Risk**

**Two Types:**

**Funding Liquidity:** Can't meet cash obligations
\`\`\`python
# Need to meet margin call
margin_call = 50000000
available_cash = 30000000
shortfall = margin_call - available_cash  # $20M

# Must sell assets quickly
required_cash = 100000000
illiquid_assets = True
fire_sale_discount = 0.20  # Must sell at 80% of value
liquidity_cost = required_cash * fire_sale_discount  # $20M loss
\`\`\`

**Market Liquidity:** Can't exit position without moving market
\`\`\`python
# Large position in small stock
position_size = 1000000  # shares
daily_volume = 50000  # Average daily volume
days_to_exit = position_size / daily_volume  # 20 days

# Price impact
exit_slippage = 0.05  # 5% worse than mid
price = 100
slippage_cost = position_size * price * exit_slippage  # $5M
\`\`\`

**Characteristics:**
- Measured with LCR, NSFR
- Funding plan crucial
- Cannot be hedged (by definition)
- Most dangerous in crisis

**How They Interact (The Doom Loop):**

**Normal Times (Risks Independent):**
\`\`\`
Market Risk: Managed with VaR limits ✓
Credit Risk: Diversified counterparties ✓
Operational Risk: Controls in place ✓
Liquidity Risk: Ample funding ✓

Everything looks fine!
\`\`\`

**Crisis (Risks Amplify Each Other):**

**Step 1: Market Risk Event**
\`\`\`python
# Market crashes
portfolio_value = 1000000000  # $1B
market_crash = -0.30  # Down 30%
market_loss = portfolio_value * market_crash  # -$300M

# Triggers margin call
\`\`\`

**Step 2: Liquidity Risk Emerges**
\`\`\`python
# Need cash for margin
margin_call = 200000000  # $200M
available_cash = 50000000  # Only $50M

# Must sell assets in falling market
must_sell = margin_call - available_cash  # $150M

# But market is illiquid (everyone selling)
liquidity_cost = must_sell * 0.15  # 15% discount
# Additional $22.5M loss
\`\`\`

**Step 3: Credit Risk Amplifies**
\`\`\`python
# Counterparties see losses
# Start demanding more collateral
# Credit spreads widen

# Derivatives losses
derivative_position = 100000000
spread_widening = 0.05  # 500 bp
credit_mark_down = derivative_position * spread_widening  # $5M

# Counterparty defaults
counterparty_exposure = 50000000
recovery = 0.30
counterparty_loss = counterparty_exposure * (1 - recovery)  # $35M
\`\`\`

**Step 4: Operational Risk**
\`\`\`python
# Systems overwhelmed
# Trading errors increase
# Risk systems crash
# Manual errors

# Key staff quit
# Rushed decisions
# Compliance failures

operational_losses = 20000000  # Errors, fines, etc.
\`\`\`

**Total Loss (Compounding):**
\`\`\`python
market_loss = 300000000
liquidity_cost = 22500000
credit_losses = 40000000
operational_losses = 20000000

total_loss = 382500000  # $382.5M

# But modeling in isolation predicted:
isolated_market_var = 50000000  # 99% VaR
# Only captured $50M, not $382.5M!

# Factor: Actual / Predicted = 7.65x worse
\`\`\`

**Real-World Example: Lehman Brothers (2008)**

**Market Risk:**
- Held $85B in illiquid real estate assets
- Prices crashed 40-60%
- Mark-to-market losses mounting

**Liquidity Risk (The Killer):**
- Needed to rollover $100B+ short-term funding
- Sep 2008: Lenders refused to rollover
- Bank run: Prime brokerage clients withdrew $100B+ in days
- Could not meet obligations

**Credit Risk:**
- Lehman owed $600B+ in derivatives
- Counterparties demanded collateral
- Couldn't post → defaults triggered

**Operational Risk:**
- Chaos as firm collapsed
- Systems failed
- Documentation missing
- Years of legal battles

**Interaction:**
\`\`\`
Market losses
    ↓
Margin calls (liquidity)
    ↓
Can't sell assets (illiquid market)
    ↓
Credit rating downgraded
    ↓
More collateral needed
    ↓
Funding runs away
    ↓
Operational chaos
    ↓
Bankruptcy
\`\`\`

Each risk fed the others in death spiral.

**Why Model Interactions:**

**Problem with Isolated Modeling:**

\`\`\`python
class IsolatedRiskModel:
    """
    WRONG: Treats risks independently
    """
    def total_risk(self):
        market_var = self.market_risk()  # $50M
        credit_var = self.credit_risk()  # $30M
        operational_var = self.op_risk()  # $20M
        
        # Assumes independent!
        total = sqrt(market_var**2 + credit_var**2 + operational_var**2)
        # = $62.4M
        
        # But in crisis, risks correlate
        # Actual loss: $400M+ (6.4x higher!)
        
        return total
\`\`\`

**Correct: Model Interactions:**

\`\`\`python
class IntegratedRiskModel:
    """
    Models risk interactions
    """
    def stress_test_crisis(self, scenario):
        """
        Simulate cascading risks
        """
        # Step 1: Market shock
        market_loss = self.apply_market_shock(scenario)
        
        # Step 2: Liquidity impact
        margin_calls = self.calculate_margin_calls(market_loss)
        liquidity_cost = self.estimate_liquidity_cost(
            margin_calls,
            market_liquidity=scenario['liquidity']  # Low in crisis
        )
        
        # Step 3: Credit impact
        # Market stress increases default probability
        default_prob = self.default_probability(
            market_stress=scenario['market_stress'],
            liquidity_stress=scenario['liquidity_stress']
        )
        credit_losses = self.credit_var(default_prob)
        
        # Step 4: Operational failures
        # Stress increases operational risk
        op_risk_multiplier = 3.0  # Errors increase 3x in crisis
        operational_losses = self.base_op_risk() * op_risk_multiplier
        
        # Step 5: Feedback loop
        # Losses worsen liquidity, credit, etc.
        amplification_factor = self.calculate_amplification(
            total_losses,
            capital_buffer
        )
        
        # Total loss with interactions
        total_loss = (
            market_loss +
            liquidity_cost +
            credit_losses +
            operational_losses
        ) * amplification_factor
        
        return {
            'market_loss': market_loss,
            'liquidity_cost': liquidity_cost,
            'credit_losses': credit_losses,
            'operational_losses': operational_losses,
            'amplification_factor': amplification_factor,
            'total_loss': total_loss
        }
\`\`\`

**Key Interactions to Model:**

**1. Market → Liquidity:**
\`\`\`python
def market_to_liquidity_interaction(market_stress):
    """
    Market stress reduces liquidity
    """
    # Bid-ask spreads widen
    spread_widening = market_stress * 5  # 5x in crisis
    
    # Days to exit position increases
    normal_days_to_exit = 2
    crisis_days_to_exit = normal_days_to_exit * (1 + market_stress * 10)
    
    # Fire sale discounts
    fire_sale_discount = market_stress * 0.30  # Up to 30%
    
    return fire_sale_discount
\`\`\`

**2. Liquidity → Credit:**
\`\`\`python
def liquidity_to_credit_interaction(liquidity_stress):
    """
    Liquidity stress increases credit risk
    """
    # Can't meet obligations → credit event
    if liquidity_stress > 0.8:  # Severe stress
        default_probability *= 10  # Jumps sharply
        
    # Credit rating downgrade
    if liquidity_stress > 0.5:
        rating_downgrade = True
        # Triggers more collateral demands
        
    return default_probability
\`\`\`

**3. Credit → Market:**
\`\`\`python
def credit_to_market_interaction(credit_event):
    """
    Credit events impact markets
    """
    if credit_event == 'MAJOR_DEFAULT':
        # Contagion
        market_impact = -0.15  # 15% market drop
        
        # Correlations go to 1
        correlation_matrix[:] = 0.9  # Everything falls together
        
    return market_impact
\`\`\`

**4. Operational → All:**
\`\`\`python
def operational_in_crisis():
    """
    Operational risk increases in crisis
    """
    stress_level = calculate_stress()
    
    # Trading errors increase
    error_rate = base_error_rate * (1 + stress_level * 5)
    
    # Risk systems may fail
    if stress_level > 0.8:
        system_failure_probability = 0.30
        
    # Key staff may leave
    turnover_rate = base_turnover * (1 + stress_level * 3)
    
    return operational_risk_multiplier
\`\`\`

**Regulatory Recognition:**

**Basel III now requires:**
- Stress testing across risk types
- Liquidity Coverage Ratio (LCR)
- Net Stable Funding Ratio (NSFR)
- Counterparty Credit Risk (CVA)

All recognize that risks interact!

**Practical Implementation:**

\`\`\`python
class ComprehensiveRiskFramework:
    """
    Integrated risk management
    """
    def calculate_firm_wide_risk(self):
        """
        Holistic view of all risks
        """
        # Normal times: Individual metrics
        normal_risk = {
            'market_var': self.market_var(),
            'credit_var': self.credit_var(),
            'op_var': self.op_var(),
            'lcr': self.liquidity_coverage_ratio()
        }
        
        # Stress scenarios: Interactions
        stress_results = {}
        for scenario in self.stress_scenarios:
            stress_results[scenario.name] = self.integrated_stress_test(scenario)
            
        # Reverse stress: What breaks us?
        breaking_scenario = self.reverse_stress_test()
        
        return {
            'normal_metrics': normal_risk,
            'stress_tests': stress_results,
            'breaking_point': breaking_scenario,
            'warning': 'Actual crisis loss likely 3-10x stress test'
        }
\`\`\`

**Bottom Line:**

**Risks in Isolation:**
- Market: $50M VaR
- Credit: $30M VaR
- Operational: $20M
- Total (independent): $62M

**Risks in Crisis (Interacting):**
- Market triggers liquidity
- Liquidity triggers credit
- Credit triggers operational
- Feedback loops amplify
- Total (interacting): $400M+ (6.4x)

**Lesson from 2008:** Firms that modeled risks in isolation failed. Those that understood interactions (and had capital buffers for them) survived.

Always model risk interactions. The crisis will exploit the connections between risks you thought were independent.`,
    },
  ],
} as const;
