export const financialSystemOverviewQuiz = [
  {
    id: 'fso-q-1',
    question:
      "You're building a robo-advisor platform that needs to explain the time value of money to non-technical users. Design an interactive calculator that: (1) demonstrates why $100 today ≠ $100 tomorrow, (2) shows the impact of different discount rates (inflation, opportunity cost, risk), (3) helps users choose between lump sum vs annuity payments, (4) visualizes compound growth over time, (5) incorporates real-world scenarios (lottery winnings, pension choices, inheritance timing). Include the mathematical models, UX considerations, and educational approach.",
    sampleAnswer: `Interactive Time Value of Money Calculator Design:

**1. Core Mathematical Model**
Present Value: PV = FV / (1 + r)^n
Future Value: FV = PV * (1 + r)^n
Annuity PV: PV = PMT * [(1 - (1 + r)^-n) / r]

Implementation:
\`\`\`python
class TVMCalculator:
    def __init__(self, inflation_rate=0.03, opportunity_cost=0.08, risk_premium=0.02):
        self.inflation = inflation_rate
        self.opportunity = opportunity_cost  
        self.risk = risk_premium
        self.total_discount = inflation + opportunity + risk
    
    def present_value (self, future_value, years):
        return future_value / (1 + self.total_discount) ** years
    
    def annuity_pv (self, payment, years):
        r = self.total_discount
        return payment * (1 - (1 + r) ** -years) / r
    
    def compare_options (self, lump_sum, annual_payment, years):
        pv_annuity = self.annuity_pv (annual_payment, years)
        difference = lump_sum - pv_annuity
        better = "lump sum" if lump_sum > pv_annuity else "annuity"
        return {
            "lump_sum": lump_sum,
            "annuity_pv": pv_annuity,
            "difference": difference,
            "recommendation": better
        }
\`\`\`

**2. Discount Rate Breakdown (Educational)**
Display three components separately:
- Inflation (3%): "Your purchasing power erodes 3% yearly"
- Opportunity Cost (8%): "You could invest and earn 8% annually"
- Risk Premium (2%): "Future is uncertain, add 2% for risk"
Total Discount Rate: 13%

Interactive sliders let users adjust each component and see real-time impact on present value calculations.

**3. Real-World Scenarios**

Scenario A: Lottery Winnings
- Lump sum: $10M today
- Annuity: $500K/year for 30 years
- Calculator shows: PV of annuity = $7.69M (at 5% discount)
- Recommendation: Take lump sum (worth $2.31M more)
- Insight: "If you can earn 8%+ on investments, lump sum wins"

Scenario B: Pension Decision
- Lump sum: $500K at retirement
- Annuity: $3K/month for life (assume 25 years)
- PV of annuity = $462K (at 5% discount)
- Recommendation: Depends on longevity and investment skill
- Insight: "Live past 83? Annuity better. Die early? Lump sum better"

Scenario C: Inheritance Timing
- Option 1: $100K today
- Option 2: $150K in 5 years
- PV of option 2 = $93.5K (at 10% discount)
- Recommendation: Take $100K today
- Insight: "8% annual return on $100K = $147K in 5 years"

**4. Visualization Design**

Timeline View:
- Show cash flows on timeline (today vs future payments)
- Shade future dollars lighter (representing discounting)
- Animated bars shrinking as you apply discount rate
- Display "Equivalent Value Today" for all future payments

Compound Growth Chart:
- Interactive chart showing $100 growing over time
- Slider adjusts return rate (0-20%)
- Display future values at 5, 10, 20, 30 years
- Show "Rule of 72" shortcut: 72/rate = years to double
- Example: 8% return → doubles in 9 years

Sensitivity Analysis:
- 2D heatmap: Years (x-axis) vs Discount Rate (y-axis)
- Color intensity shows present value
- Users see how PV changes with assumptions
- Highlight "break-even" discount rate

**5. UX Considerations**

Progressive Disclosure:
Level 1 (Novice): "Would you rather have $100 today or $110 next year?"
Level 2 (Intermediate): Introduce discount rate concept, single input
Level 3 (Advanced): Break down inflation, opportunity cost, risk premium
Level 4 (Expert): Monte Carlo simulation with uncertainty

Gamification:
- "Make 10 decisions" tutorial with instant feedback
- "You chose lump sum. If you invest at 8%, you'll have $X in 10 years"
- Compare user's decisions to "optimal" choices
- Show cumulative impact of good vs bad decisions

Accessibility:
- Avoid jargon: "discount rate" → "your expected return rate"
- Tooltips with examples for every technical term
- "Why does this matter?" sections explaining real-world impact
- Mobile-first design (most users on phones)

**6. Educational Approach**

Start with intuition:
"Would you rather have $100 today or $100 in a year?"
Most people choose today. Why?
- You could spend it now (time preference)
- Prices rise, so $100 buys less next year (inflation)
- You could invest it and have $108 next year (opportunity cost)
- You might not get it next year (risk)

This is the TIME VALUE OF MONEY.

Then formalize:
"We use a 'discount rate' to make today and tomorrow comparable."
Future Value → discount at rate r for n years → Present Value
Formula: PV = FV / (1 + r)^n

Finally, apply:
Real decisions: lottery, pension, inheritance, mortgage, lease vs buy
"Now you have the tools to make financially sound decisions!"

**7. Backend Implementation**

API Endpoints:
- POST /calculate/pv: Calculate present value
- POST /calculate/fv: Calculate future value  
- POST /compare/lumpsum-vs-annuity: Compare payment options
- POST /scenario/lottery: Pre-built scenario with real numbers
- GET /education/tvm: Fetch educational content

Error Handling:
- Negative interest rates? (Possible in Europe, Japan)
- Very long time periods (100+ years)?
- Zero discount rate (no time preference)?
- Handle gracefully with warnings

**8. Advanced Features**

Monte Carlo Mode:
- Uncertain returns: instead of fixed 8%, use distribution (mean=8%, std=5%)
- Run 10,000 simulations
- Show probability distribution of outcomes
- Display: "70% chance lump sum better, 30% chance annuity better"

Tax Considerations:
- Lump sum: taxed immediately at 37% (high bracket)
- Annuity: taxed yearly at lower rates (income spreading)
- Incorporate tax efficiency into calculations

Inflation Adjustment:
- Toggle "real" vs "nominal" dollars
- Show both: "$500K in 30 years = $207K in today's dollars (3% inflation)"

**Key Success Metrics:**
1. Users complete tutorial: >80% completion rate
2. Users change mind: 30%+ change decision after seeing calculation
3. Users share: "This saved me $50K!" testimonials
4. Users return: Calculate multiple scenarios (lottery, pension, etc.)

This calculator transforms abstract financial concepts into concrete, actionable insights that help users make better financial decisions.`,
    keyPoints: [
      'Break discount rate into three components (inflation, opportunity cost, risk) for user understanding',
      'Use real-world scenarios (lottery, pension) with actual numbers for concrete learning',
      'Progressive disclosure: start simple (basic comparison), add complexity gradually (Monte Carlo)',
      'Visualization key: timeline view showing future dollars "shrinking" when discounted to present value',
      'Educational approach: intuition first (would you rather...), formalize second (formula), apply third (real decisions)',
    ],
  },
  {
    id: 'fso-q-2',
    question:
      'Design a risk-adjusted return comparison tool for your portfolio management platform. Requirements: (1) calculate Sharpe ratio, Sortino ratio, and Calmar ratio for multiple strategies, (2) explain why Strategy A with 50% return might be worse than Strategy B with 20% return, (3) build an interactive visualization showing efficient frontier, (4) incorporate downside risk vs total volatility, (5) help users understand "return per unit of risk". Include code implementation, statistical considerations, and user education approach.',
    sampleAnswer: `Answer: Comprehensive Risk-Adjusted Return Comparison System

**1. Risk Metrics Implementation**

\`\`\`python
import numpy as np
import pandas as pd
from typing import Dict, Tuple

class RiskMetrics:
    def __init__(self, returns: pd.Series, risk_free_rate: float = 0.04):
        self.returns = returns
        self.rf_rate = risk_free_rate / 252  # Daily risk-free rate
        self.periods_per_year = 252  # Trading days
    
    def sharpe_ratio (self) -> float:
        """
        Sharpe Ratio = (Return - Risk-Free Rate) / Total Volatility
        Measures return per unit of TOTAL risk
        """
        excess_returns = self.returns - self.rf_rate
        return np.sqrt (self.periods_per_year) * excess_returns.mean() / excess_returns.std()
    
    def sortino_ratio (self) -> float:
        """
        Sortino Ratio = (Return - Risk-Free Rate) / Downside Deviation
        Only penalizes DOWNSIDE volatility (losses)
        Better for asymmetric strategies
        """
        excess_returns = self.returns - self.rf_rate
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.sqrt((downside_returns ** 2).mean())
        return np.sqrt (self.periods_per_year) * excess_returns.mean() / downside_std
    
    def calmar_ratio (self) -> float:
        """
        Calmar Ratio = Annual Return / Maximum Drawdown
        Measures return per unit of DRAWDOWN risk
        Popular in hedge funds (captures tail risk)
        """
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs (drawdown.min())
        annual_return = (cumulative.iloc[-1] ** (self.periods_per_year / len (self.returns))) - 1
        return annual_return / max_drawdown if max_drawdown > 0 else np.inf
    
    def all_metrics (self) -> Dict[str, float]:
        """Calculate all risk-adjusted metrics"""
        cumulative = (1 + self.returns).cumprod()
        total_return = cumulative.iloc[-1] - 1
        annual_return = (cumulative.iloc[-1] ** (self.periods_per_year / len (self.returns))) - 1
        annual_volatility = self.returns.std() * np.sqrt (self.periods_per_year)
        
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "calmar_ratio": self.calmar_ratio(),
            "max_drawdown": self.calculate_max_drawdown(),
        }
    
    def calculate_max_drawdown (self) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs (drawdown.min())
\`\`\`

**2. Why 50% Return Can Be Worse Than 20% Return**

Example:
Strategy A: 50% annual return, 60% volatility, Sharpe = 0.77
Strategy B: 20% annual return, 15% volatility, Sharpe = 1.07

Strategy B is BETTER because:

1. **Risk-Adjusted Return**: B delivers 1.07 units of return per unit of risk, vs 0.77 for A
2. **Consistency**: B has smoother returns (15% vol vs 60% vol)
3. **Drawdown**: A likely has -40% drawdowns, B has -10% drawdowns
4. **Leverage**: Can leverage B to 3x (via margin/futures) → 60% return with 45% vol → Sharpe = 1.24 (better than A!)
5. **Compounding**: Volatility drag hurts A more (50% up, 50% down = -25% net!)

Visual Explanation:
\`\`\`
Strategy A: [+100%] [-50%] [+150%] [-60%] [+80%] = volatile, stressful
Strategy B: [+22%] [+18%] [+21%] [+19%] [+20%] = smooth, predictable

Both end at similar place, but B is MUCH better experience!
\`\`\`

**3. Interactive Visualization: Efficient Frontier**

\`\`\`python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_efficient_frontier (strategies: Dict[str, pd.Series], rf_rate: float = 0.04):
    """
    Plot strategies on risk-return space
    Show efficient frontier
    Highlight Sharpe ratio optimal portfolio
    """
    # Calculate metrics for each strategy
    metrics = {}
    for name, returns in strategies.items():
        rm = RiskMetrics (returns, rf_rate)
        metrics[name] = rm.all_metrics()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Risk-Return Profile', 'Sharpe Ratio Comparison',
                       'Maximum Drawdown', 'Return Distribution'),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "box"}]]
    )
    
    # 1. Risk-Return Scatter (Efficient Frontier)
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    for i, (name, m) in enumerate (metrics.items()):
        fig.add_trace(
            go.Scatter(
                x=[m['annual_volatility'] * 100],
                y=[m['annual_return'] * 100],
                mode='markers+text',
                name=name,
                text=[name],
                textposition='top center',
                marker=dict (size=15, color=colors[i % len (colors)]),
                hovertemplate=f'<b>{name}</b><br>' +
                             f'Return: {m["annual_return"]*100:.1f}%<br>' +
                             f'Volatility: {m["annual_volatility"]*100:.1f}%<br>' +
                             f'Sharpe: {m["sharpe_ratio"]:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Add Capital Market Line (CML) from risk-free rate through best Sharpe
    best_sharpe_name = max (metrics.items(), key=lambda x: x[1]['sharpe_ratio'])[0]
    best_sharpe_metrics = metrics[best_sharpe_name]
    
    # CML: E(R) = Rf + Sharpe * σ
    vol_range = np.linspace(0, max (m['annual_volatility'] for m in metrics.values()) * 1.2, 100)
    cml_returns = rf_rate + best_sharpe_metrics['sharpe_ratio'] * vol_range
    
    fig.add_trace(
        go.Scatter(
            x=vol_range * 100,
            y=cml_returns * 100,
            mode='lines',
            name='Capital Market Line',
            line=dict (dash='dash', color='gray'),
            hovertemplate='CML: Optimal risk-return tradeoff<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.update_xaxes (title_text="Annual Volatility (%)", row=1, col=1)
    fig.update_yaxes (title_text="Annual Return (%)", row=1, col=1)
    
    # 2. Sharpe Ratio Comparison
    sharpe_values = [m['sharpe_ratio'] for m in metrics.values()]
    sharpe_colors = ['green' if s > 1 else 'orange' if s > 0.5 else 'red' for s in sharpe_values]
    
    fig.add_trace(
        go.Bar(
            x=list (metrics.keys()),
            y=sharpe_values,
            marker_color=sharpe_colors,
            text=[f'{s:.2f}' for s in sharpe_values],
            textposition='outside',
            hovertemplate='%{x}<br>Sharpe: %{y:.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Add benchmark lines
    fig.add_hline (y=1.0, line_dash="dash", line_color="green", 
                  annotation_text="Good (>1.0)", row=1, col=2)
    fig.add_hline (y=2.0, line_dash="dash", line_color="darkgreen",
                  annotation_text="Excellent (>2.0)", row=1, col=2)
    
    fig.update_yaxes (title_text="Sharpe Ratio", row=1, col=2)
    
    # 3. Maximum Drawdown
    dd_values = [m['max_drawdown'] * -100 for m in metrics.values()]
    dd_colors = ['green' if abs (d) < 10 else 'orange' if abs (d) < 20 else 'red' for d in dd_values]
    
    fig.add_trace(
        go.Bar(
            x=list (metrics.keys()),
            y=dd_values,
            marker_color=dd_colors,
            text=[f'{d:.1f}%' for d in dd_values],
            textposition='outside',
            hovertemplate='%{x}<br>Max DD: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.update_yaxes (title_text="Maximum Drawdown (%)", row=2, col=1)
    
    # 4. Return Distribution (Box Plots)
    for name, returns in strategies.items():
        fig.add_trace(
            go.Box(
                y=returns * 100,
                name=name,
                boxmean='sd',
                hovertemplate='%{y:.2f}%<extra></extra>'
            ),
            row=2, col=2
        )
    
    fig.update_yaxes (title_text="Daily Return (%)", row=2, col=2)
    
    # Layout
    fig.update_layout(
        title_text="Risk-Adjusted Return Analysis Dashboard",
        showlegend=True,
        height=800,
        template='plotly_white'
    )
    
    return fig
\`\`\`

**4. Downside Risk vs Total Volatility**

Key Insight: Upside volatility is GOOD, downside volatility is BAD.

Sharpe Ratio: Penalizes BOTH upside and downside volatility equally
Sortino Ratio: Only penalizes DOWNSIDE volatility

Example:
Strategy X: Returns: [+30%, -5%, +25%, -3%, +40%]
→ High volatility (includes +40% month!)
→ Low Sharpe ratio (volatility penalized)
→ High Sortino ratio (only -5%, -3% penalized)

Strategy Y: Returns: [+8%, +7%, +9%, +8%, +7%]
→ Low volatility (consistent returns)
→ High Sharpe ratio
→ High Sortino ratio

Sortino is better for:
- Asymmetric strategies (options, tail risk funds)
- Momentum strategies (big wins, small losses)
- Venture capital (most fail, few succeed big)

**5. User Education: "Return Per Unit of Risk"**

Analogy 1: Car Efficiency (MPG)
- Car A: 500 miles, uses 50 gallons → 10 MPG
- Car B: 400 miles, uses 20 gallons → 20 MPG
- Car B is MORE EFFICIENT (more miles per gallon)

Same for investments:
- Strategy A: 50% return, 60% volatility → 0.83 return per risk
- Strategy B: 20% return, 15% volatility → 1.33 return per risk
- Strategy B is MORE EFFICIENT (more return per unit of risk)

Analogy 2: Test Scores (Grade / Hours Studied)
- Student A: 90% grade, 30 hours → 3.0 score per hour
- Student B: 85% grade, 10 hours → 8.5 score per hour
- Student B is MORE EFFICIENT

Visualization:
Create a "Risk Efficiency Score" (Sharpe Ratio renamed):
- Score < 0.5: ⭐ (Poor efficiency)
- Score 0.5-1.0: ⭐⭐ (Fair efficiency)
- Score 1.0-2.0: ⭐⭐⭐ (Good efficiency)
- Score > 2.0: ⭐⭐⭐⭐⭐ (Excellent efficiency)

Interactive Feature:
"Leverage Simulator": Show how efficient strategies can be leveraged
- Strategy B: 20% return, 15% vol, Sharpe 1.07
- Leverage 2x: 40% return, 30% vol, Sharpe 1.20
- Leverage 3x: 60% return, 45% vol, Sharpe 1.24

"You can turn a 20% strategy into a 60% strategy by using leverage, IF it's efficient!"

**Statistical Considerations:**
1. Minimum sample size: Need 3+ years of daily data for reliable Sharpe ratio
2. Non-normal returns: Real returns have fat tails (use Sortino for tail risk)
3. Autocorrelation: Some strategies have serial correlation (adjust standard errors)
4. Regime changes: Bull market Sharpe ≠ Bear market Sharpe (show both)
5. Survivorship bias: Only showing surviving strategies inflates metrics

**Implementation Best Practices:**
- Calculate rolling Sharpe (1-year window) to show stability over time
- Bootstrap confidence intervals for Sharpe ratio (show uncertainty)
- Compare to benchmarks: S&P 500 Sharpe = 0.5-0.7 historically
- Normalize metrics: "Your strategy's Sharpe is 1.5, which beats 80% of mutual funds"

This transforms abstract risk metrics into actionable insights users can understand and act on.`,
    keyPoints: [
      'Sharpe ratio = return per unit of TOTAL risk; Sortino = return per unit of DOWNSIDE risk; Calmar = return per unit of DRAWDOWN risk',
      '50% return with 60% volatility (Sharpe 0.77) is WORSE than 20% return with 15% volatility (Sharpe 1.07) - can leverage efficient strategy',
      'Efficient frontier shows best risk-return tradeoff; strategies below frontier are inefficient (same risk, lower return or same return, higher risk)',
      'Upside volatility is good (big wins), downside volatility is bad (losses) - Sortino ratio accounts for this asymmetry',
      'User education key: analogies (MPG for cars = return per unit of risk for strategies), rename metrics (Risk Efficiency Score instead of Sharpe Ratio)',
    ],
  },
  {
    id: 'fso-q-3',
    question:
      'Analyze the 2008 financial crisis from an engineering perspective. Explain: (1) what went wrong with the risk models (Value at Risk failures), (2) how leverage amplified losses (30:1 ratios), (3) systemic risk and contagion effects, (4) what you would have built differently as an engineer (better risk systems, circuit breakers, stress tests), (5) lessons for building financial systems today. Include mathematical analysis of leverage impact, network effects in contagion, and specific technical solutions.',
    sampleAnswer: `Answer to be completed (placeholder for comprehensive analysis of 2008 crisis from engineering lens, including: VaR model assumptions and failures, leverage mathematics showing 30:1 amplification, network analysis of bank interconnections showing contagion paths, technical solutions like real-time risk monitoring dashboards, automated circuit breakers at leverage thresholds, stress testing frameworks using Monte Carlo simulation, transparency systems for complex derivatives using blockchain-like technology, and cultural changes needed in financial engineering around risk management vs profit optimization).`,
    keyPoints: [
      'VaR models assumed normal distributions (5σ event ≈ 1 in 3.5M days) but real markets have fat tails (crashes happen every 10-20 years)',
      'Leverage amplification: 30:1 ratio means 3.3% portfolio loss = 100% equity loss = bankruptcy (Lehman Brothers)',
      'Systemic risk: network effects where one bank failure cascades to others via interconnected exposures (credit default swaps)',
      'Engineering solutions: real-time risk dashboards, automated deleveraging at thresholds, mandatory stress tests, blockchain for derivative transparency',
      'Cultural lesson: optimize for robustness (survive worst case) not just profit (maximize expected return) - antifragility',
    ],
  },
];
