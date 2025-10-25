export const incomeStatement = {
  title: 'Income Statement Analysis',
  slug: 'income-statement',
  description:
    'Master financial statement analysis and build production systems',
  content: `
# Income Statement Analysis

## Overview

The Income Statement (also called Profit & Loss or P&L) is arguably the **most scrutinized** financial statement. It answers the fundamental question: **Did the company make money?**

For engineers building financial systems, the income statement is the foundation for:
- **Earnings prediction models** (ML-based forecasting)
- **Revenue quality analysis** (detecting unsustainable growth)
- **Margin analysis** (competitive positioning)
- **Earnings surprise detection** (trading signals)
- **Fraud detection** (unusual patterns)

**Real-world**: When a company reports quarterly earnings, the stock can move 10-20% in minutes based on whether EPS beats or misses analyst expectations. Your job is to analyze these statements **faster and deeper** than the market.

---

## Income Statement Structure

### The Cascade

Think of the income statement as a **waterfall**—revenue flowing down through expenses to net income:

\`\`\`
REVENUE (Top Line)
  ↓
- Cost of Goods Sold (COGS)
= GROSS PROFIT
  ↓
- Operating Expenses
  • Selling, General & Administrative (SG&A)
  • Research & Development (R&D)
  • Depreciation & Amortization (D&A)
= OPERATING INCOME (EBIT)
  ↓
- Interest Expense
+/- Other Income/Expense
= INCOME BEFORE TAX (EBT)
  ↓
- Income Tax Expense
= NET INCOME (Bottom Line)
  ↓
÷ Shares Outstanding
= EARNINGS PER SHARE (EPS)
\`\`\`

### Key Terminology

| Term | Also Known As | What It Means |
|------|---------------|---------------|
| **Revenue** | Sales, Top Line | Money earned from customers |
| **COGS** | Cost of Sales | Direct costs to make/deliver product |
| **Gross Profit** | - | Revenue minus COGS |
| **EBITDA** | - | Earnings Before Interest, Tax, Depreciation, Amortization |
| **EBIT** | Operating Income | Earnings Before Interest and Tax |
| **EBT** | Pretax Income | Earnings Before Tax |
| **Net Income** | Net Profit, Bottom Line, Earnings | Final profit after all expenses |
| **EPS** | Earnings Per Share | Net Income ÷ Shares Outstanding |

---

## Revenue Analysis

### Revenue Recognition Principles

Under GAAP (ASC 606) and IFRS (IFRS 15), revenue is recognized when:

1. **Contract exists** with customer
2. **Performance obligations** identified
3. **Transaction price** determined
4. **Price allocated** to obligations
5. **Performance obligation satisfied** (control transferred)

**Key Insight**: Revenue ≠ Cash Received

\`\`\`python
# Example: SaaS company with annual contracts
contract_value = 120_000  # $120K annual subscription
contract_start = "2024-01-01"
payment_received = 120_000  # Customer paid upfront

# Revenue recognition (ratably over 12 months):
monthly_revenue = contract_value / 12  # $10K per month

# Month 1:
revenue_recognized = 10_000  # Only $10K, not $120K!
cash_received = 120_000      # But got $120K cash
deferred_revenue = 110_000   # Liability on balance sheet

# This is why cash flow ≠ net income
\`\`\`

### Revenue Quality Indicators

**High-quality revenue** characteristics:
- ✅ Cash collected promptly
- ✅ Recurring (subscriptions, contracts)
- ✅ Diversified customers (no concentration)
- ✅ Organic growth (not acquisitions)
- ✅ Growing faster than receivables

**Low-quality revenue** red flags:
- ⚠️ Long payment terms (90+ days)
- ⚠️ One-time deals
- ⚠️ Large customer concentration (>10% from one customer)
- ⚠️ Channel stuffing
- ⚠️ Receivables growing faster than revenue

### Analyzing Revenue with Python

\`\`\`python
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List

class RevenueAnalyzer:
    """Analyze revenue quality and trends."""
    
    def __init__(self, financial_data: pd.DataFrame):
        """
        financial_data should have columns:
        - period_end (date)
        - revenue (float)
        - accounts_receivable (float)
        - cash_from_operations (float)
        """
        self.data = financial_data.sort_values('period_end')
    
    def calculate_revenue_quality_score (self) -> pd.DataFrame:
        """Calculate comprehensive revenue quality metrics."""
        
        df = self.data.copy()
        
        # 1. Revenue growth rate
        df['revenue_growth'] = df['revenue'].pct_change()
        
        # 2. Days Sales Outstanding (DSO)
        df['dso'] = (df['accounts_receivable'] / df['revenue']) * 90  # Quarterly
        df['dso_change'] = df['dso'].diff()
        
        # 3. Cash realization ratio
        df['cash_realization'] = df['cash_from_operations'] / df['revenue']
        
        # 4. Receivables growth vs revenue growth
        df['receivables_growth'] = df['accounts_receivable'].pct_change()
        df['rev_quality_flag'] = df['receivables_growth'] > df['revenue_growth'] * 1.3
        
        # 5. Quality score (0-100)
        df['quality_score'] = self._calculate_quality_score (df)
        
        return df
    
    def _calculate_quality_score (self, df: pd.DataFrame) -> pd.Series:
        """Calculate composite quality score."""
        
        score = pd.Series(100, index=df.index)
        
        # Penalize high DSO
        score -= np.clip((df['dso'] - 45) / 2, 0, 20)  # -20 if DSO > 85 days
        
        # Penalize increasing DSO
        score -= np.clip (df['dso_change'] * 2, 0, 15)
        
        # Penalize low cash realization
        score -= np.clip((0.8 - df['cash_realization']) * 50, 0, 20)
        
        # Penalize receivables growing faster than revenue
        score -= df['rev_quality_flag'] * 15
        
        # Reward consistent growth
        growth_volatility = df['revenue_growth'].rolling(4).std()
        score -= np.clip (growth_volatility * 100, 0, 10)
        
        return np.clip (score, 0, 100)
    
    def detect_channel_stuffing (self, threshold: float = 0.3) -> Dict:
        """Detect potential channel stuffing (aggressive sales to distributors)."""
        
        df = self.data.copy()
        
        # Calculate quarter-end spikes
        df['qoq_revenue_growth'] = df['revenue'].pct_change()
        df['qoq_receivables_growth'] = df['accounts_receivable'].pct_change()
        
        # Red flag: Receivables growing much faster than revenue
        df['stuffing_indicator'] = (
            df['qoq_receivables_growth'] - df['qoq_revenue_growth']
        )
        
        alerts = df[df['stuffing_indicator'] > threshold]
        
        return {
            'detected': len (alerts) > 0,
            'periods': alerts['period_end'].tolist(),
            'severity': alerts['stuffing_indicator'].mean() if len (alerts) > 0 else 0
        }
    
    def analyze_revenue_composition (self, revenue_by_segment: Dict[str, List[float]]) -> pd.DataFrame:
        """Analyze revenue diversification and trends by segment."""
        
        df = pd.DataFrame (revenue_by_segment)
        
        # Calculate segment percentages
        total = df.sum (axis=1)
        pct_df = df.div (total, axis=0) * 100
        
        # Calculate concentration (Herfindahl-Hirschman Index)
        hhi = (pct_df ** 2).sum (axis=1)
        
        # Growth rates by segment
        growth_df = df.pct_change()
        
        return pd.DataFrame({
            'total_revenue': total,
            'hhi_concentration': hhi,  # Higher = more concentrated
            'dominant_segment': pct_df.idxmax (axis=1),
            'dominant_segment_pct': pct_df.max (axis=1),
            **{f'{col}_growth': growth_df[col] for col in df.columns}
        })

# Usage Example
# Analyzing Apple\'s revenue quality
apple_data = pd.DataFrame({
    'period_end': pd.date_range('2020-01-01', periods=12, freq='Q'),
    'revenue': [91_800, 58_300, 64_700, 111_400] * 3,  # Quarterly, in millions
    'accounts_receivable': [22_900, 16_100, 18_500, 28_100] * 3,
    'cash_from_operations': [80_700, 50_200, 58_000, 95_000] * 3
})

analyzer = RevenueAnalyzer (apple_data)
quality_metrics = analyzer.calculate_revenue_quality_score()

print("Revenue Quality Analysis:")
print(quality_metrics[['period_end', 'revenue_growth', 'dso', 'quality_score']].tail())

# Check for channel stuffing
stuffing = analyzer.detect_channel_stuffing()
if stuffing['detected']:
    print(f"⚠️ Potential channel stuffing detected in {len (stuffing['periods'])} periods")
\`\`\`

---

## Cost of Goods Sold (COGS)

### What COGS Includes

**Manufacturing Company** (e.g., Tesla):
- Raw materials (steel, batteries)
- Direct labor (assembly workers)
- Factory overhead (utilities, maintenance)
- Manufacturing supplies

**Software Company** (e.g., Microsoft):
- Cloud infrastructure costs (Azure servers)
- Content delivery network (CDN)
- Customer support (direct)
- Third-party licenses

**Retail Company** (e.g., Walmart):
- Purchase cost of inventory
- Shipping from suppliers
- Warehousing costs

### COGS vs Operating Expenses

**Rule**: If tied to **producing/delivering** the product → COGS. If tied to **running the company** → OpEx.

\`\`\`python
# Example: SaaS company expense classification
expenses = {
    'COGS': [
        ('AWS hosting costs', 500_000),
        ('Customer success team', 300_000),  # Directly serving customers
        ('API costs (Stripe, etc)', 100_000)
    ],
    'SG&A': [
        ('Sales team salaries', 1_000_000),
        ('Marketing spend', 800_000),
        ('Office rent', 200_000),
        ('HR and finance', 300_000)
    ],
    'R&D': [
        ('Engineering salaries', 2_000_000),
        ('Product team', 500_000),
        ('Development tools', 100_000)
    ]
}

cogs_total = sum([amt for _, amt in expenses['COGS']])
sga_total = sum([amt for _, amt in expenses['SG&A']])
rd_total = sum([amt for _, amt in expenses['R&D']])

print(f"COGS: \${cogs_total:,}")  # $900,000
print(f"SG&A: \${sga_total:,}")   # $2, 300,000
print(f"R&D: \${rd_total:,}")     # $2, 600,000
\`\`\`

### Gross Margin Analysis

**Gross Margin** = (Revenue - COGS) / Revenue

This is the **single most important metric** for understanding business model quality.

\`\`\`python
def analyze_gross_margin (company_data: pd.DataFrame) -> Dict:
    """Analyze gross margin trends and compare to industry."""
    
    company_data['gross_profit'] = company_data['revenue'] - company_data['cogs']
    company_data['gross_margin'] = (
        company_data['gross_profit'] / company_data['revenue']
    )
    
    # Calculate trends
    current_margin = company_data['gross_margin'].iloc[-1]
    margin_change = company_data['gross_margin'].diff().iloc[-1]
    margin_trend = company_data['gross_margin'].rolling(4).mean()
    
    # Industry benchmarks (approximate)
    industry_benchmarks = {
        'Software (SaaS)': 0.75,      # 75% - Very high
        'Hardware (Apple)': 0.38,      # 38% - Good
        'Retail (Walmart)': 0.25,      # 25% - Low
        'Manufacturing': 0.30,         # 30% - Moderate
        'Banking': 0.65,               # 65% - High (NIM)
    }
    
    return {
        'current_margin': current_margin,
        'margin_change_qoq': margin_change,
        'trend': 'improving' if margin_change > 0 else 'declining',
        'vs_last_year': current_margin - company_data['gross_margin'].iloc[-5],
        'volatility': company_data['gross_margin'].std()
    }

# Real example: Compare tech companies
companies = {
    'Microsoft': {'revenue': 211_900_000_000, 'cogs': 65_900_000_000},  # 69% margin
    'Apple': {'revenue': 394_300_000_000, 'cogs': 223_500_000_000},     # 43% margin
    'Amazon': {'revenue': 514_000_000_000, 'cogs': 304_000_000_000},    # 41% margin
    'Walmart': {'revenue': 611_300_000_000, 'cogs': 463_000_000_000},   # 24% margin
}

for name, data in companies.items():
    gm = (data['revenue'] - data['cogs']) / data['revenue']
    print(f"{name:12} Gross Margin: {gm:.1%}")

# Output:
# Microsoft    Gross Margin: 68.9%
# Apple        Gross Margin: 43.3%
# Amazon       Gross Margin: 40.9%
# Walmart      Gross Margin: 24.2%
\`\`\`

**Key Insights**:
- **Software**: 70-80% margins (low variable costs)
- **Hardware**: 30-40% margins (manufacturing costs)
- **Retail**: 20-30% margins (thin margins, volume business)
- **Services**: 40-60% margins (labor costs)

---

## Operating Expenses

### SG&A (Selling, General & Administrative)

**Selling expenses**:
- Sales team compensation
- Marketing and advertising
- Trade shows and events
- Sales commissions

**General & Administrative**:
- Executive compensation
- Office rent and utilities
- Legal and accounting fees
- HR and finance teams
- IT infrastructure (non-product)

\`\`\`python
def analyze_sga_efficiency (financials: pd.DataFrame) -> pd.DataFrame:
    """Analyze SG&A as percentage of revenue (operating leverage)."""
    
    financials['sga_ratio'] = financials['sga'] / financials['revenue']
    financials['sga_change'] = financials['sga'].pct_change()
    financials['revenue_change'] = financials['revenue'].pct_change()
    
    # Good sign: Revenue growing faster than SG&A (operating leverage)
    financials['operating_leverage'] = (
        financials['revenue_change'] > financials['sga_change']
    )
    
    # Calculate operating leverage ratio
    financials['leverage_ratio'] = (
        financials['revenue_change'] / financials['sga_change']
    )
    
    return financials

# Example: Company achieving operating leverage
data = pd.DataFrame({
    'quarter': ['Q1', 'Q2', 'Q3', 'Q4'],
    'revenue': [100, 115, 132, 152],  # Growing 15% per quarter
    'sga': [50, 55, 58, 61]           # Growing only 6% per quarter
})

result = analyze_sga_efficiency (data)
print("Operating Leverage Analysis:")
print(result[['quarter', 'revenue', 'sga', 'sga_ratio', 'operating_leverage']])

# As company scales, SG&A as % of revenue should decrease
# Q1: 50%, Q2: 48%, Q3: 44%, Q4: 40% - Good trend!
\`\`\`

### R&D (Research & Development)

**What counts as R&D**:
- Engineering salaries
- Product development
- Research projects
- Prototype materials
- Testing and QA

**Industry comparison**:
\`\`\`python
rd_intensity = {
    'Pharmaceutical': 0.15,   # 15% of revenue (drug development)
    'Software': 0.15,         # 15% (constant innovation)
    'Semiconductors': 0.18,   # 18% (R&D intensive)
    'Automotive': 0.05,       # 5% (traditional)
    'Retail': 0.00,          # <1% (not R&D intensive)
}

def evaluate_rd_investment (company: str, revenue: float, rd_spend: float) -> Dict:
    """Evaluate if company is investing appropriately in R&D."""
    
    rd_ratio = rd_spend / revenue
    
    analysis = {
        'rd_ratio': rd_ratio,
        'rd_dollars': rd_spend,
        'assessment': ''
    }
    
    if company in ['Software', 'Pharma', 'Semiconductors']:
        if rd_ratio < 0.10:
            analysis['assessment'] = 'UNDER-INVESTING - May fall behind competitors'
        elif rd_ratio > 0.25:
            analysis['assessment'] = 'OVER-INVESTING - May indicate inefficiency'
        else:
            analysis['assessment'] = 'APPROPRIATE - Competitive investment level'
    
    return analysis

# Example: Analyzing tech companies
tech_companies = {
    'Meta': {'revenue': 117_000_000_000, 'rd': 35_300_000_000},      # 30% - Very high!
    'Alphabet': {'revenue': 283_000_000_000, 'rd': 39_500_000_000},  # 14% - Appropriate
    'Amazon': {'revenue': 514_000_000_000, 'rd': 73_200_000_000},    # 14% - Appropriate
}

for name, data in tech_companies.items():
    eval_result = evaluate_rd_investment('Software', data['revenue'], data['rd'])
    print(f"{name}: {eval_result['rd_ratio']:.1%} R&D intensity - {eval_result['assessment']}")
\`\`\`

---

## EBITDA vs EBIT vs Net Income

Understanding the cascade:

\`\`\`
Revenue                          $1,000
- COGS                            (600)
= Gross Profit                     400
- SG&A                            (150)
- R&D                              (80)
= EBITDA                           170  ← Earnings Before Interest, Tax, Depreciation, Amortization
- Depreciation & Amortization      (30)
= EBIT (Operating Income)          140  ← Earnings Before Interest and Tax
- Interest Expense                 (20)
+ Other Income                       5
= EBT (Pretax Income)              125  ← Earnings Before Tax
- Income Tax (25%)                 (31)
= Net Income                        94  ← Bottom Line
\`\`\`

### When to Use Each Metric

\`\`\`python
class EarningsMetrics:
    """Calculate and interpret different earnings metrics."""
    
    @staticmethod
    def when_to_use_ebitda() -> str:
        return """
        Use EBITDA when:
        1. Comparing companies with different capital structures (debt levels)
        2. Valuing companies for M&A (EV/EBITDA multiple)
        3. Assessing cash generation ability
        4. Comparing across countries (different tax rates)
        
        ⚠️ WARNING: EBITDA ignores:
        - Capital expenditures (CapEx) - critical for capital-intensive businesses
        - Working capital changes
        - Actual cash taxes paid
        
        "EBITDA is bullshit earnings" - Charlie Munger
        """
    
    @staticmethod
    def when_to_use_ebit() -> str:
        return """
        Use EBIT (Operating Income) when:
        1. Comparing operating efficiency
        2. Analyzing core business performance
        3. Excluding financing and tax decisions
        4. Operating margin analysis
        
        Better than EBITDA because includes D&A (real costs).
        """
    
    @staticmethod
    def when_to_use_net_income() -> str:
        return """
        Use Net Income when:
        1. Calculating EPS
        2. Measuring actual profitability to shareholders
        3. Valuing using P/E ratio
        4. ROE calculation
        
        Most comprehensive - includes ALL costs.
        """

# Practical comparison
def compare_earnings_metrics (company_data: Dict) -> pd.DataFrame:
    """Compare different earnings measures."""
    
    data = {
        'Metric': ['Revenue', 'EBITDA', 'EBIT', 'Net Income', 'Free Cash Flow'],
        'Amount': [
            company_data['revenue'],
            company_data['ebitda'],
            company_data['ebit'],
            company_data['net_income'],
            company_data['fcf']
        ]
    }
    
    df = pd.DataFrame (data)
    df['% of Revenue'] = (df['Amount'] / company_data['revenue'] * 100).round(1)
    
    return df

# Example: Capital-intensive vs asset-light business
manufacturing = {
    'revenue': 1000,
    'ebitda': 200,      # 20% EBITDA margin
    'ebit': 100,        # 10% EBIT margin (high D&A)
    'net_income': 60,   # 6% net margin
    'fcf': 20          # 2% FCF margin (high CapEx)
}

software = {
    'revenue': 1000,
    'ebitda': 300,      # 30% EBITDA margin
    'ebit': 280,        # 28% EBIT margin (low D&A)
    'net_income': 200,  # 20% net margin
    'fcf': 250         # 25% FCF margin (low CapEx)
}

print("Manufacturing Company:")
print(compare_earnings_metrics (manufacturing))
print("\\nSoftware Company:")
print(compare_earnings_metrics (software))

# Key insight: Software business has better margins at EVERY level
\`\`\`

---

## Earnings Per Share (EPS)

### Basic vs Diluted EPS

\`\`\`python
def calculate_eps (financial_data: Dict) -> Dict:
    """Calculate both basic and diluted EPS."""
    
    net_income = financial_data['net_income']
    basic_shares = financial_data['shares_outstanding']
    
    # Basic EPS
    basic_eps = net_income / basic_shares
    
    # Diluted EPS accounts for potential dilution from:
    # - Stock options
    # - Convertible bonds
    # - Warrants
    # - RSUs (Restricted Stock Units)
    
    options_dilution = financial_data.get('stock_options_outstanding', 0)
    
    # Treasury stock method for options
    option_strike = financial_data.get('avg_option_strike', 0)
    current_price = financial_data.get('stock_price', 0)
    
    if current_price > option_strike:  # Options are "in the money"
        # Proceed from option exercises
        proceeds = options_dilution * option_strike
        
        # Shares that could be bought back with proceeds
        shares_buyback = proceeds / current_price
        
        # Net dilution
        net_new_shares = options_dilution - shares_buyback
    else:
        net_new_shares = 0  # Options out of money, no dilution
    
    diluted_shares = basic_shares + net_new_shares
    diluted_eps = net_income / diluted_shares
    
    return {
        'basic_eps': basic_eps,
        'diluted_eps': diluted_eps,
        'basic_shares': basic_shares,
        'diluted_shares': diluted_shares,
        'dilution_pct': (diluted_shares - basic_shares) / basic_shares * 100
    }

# Example: Tech company with lots of stock options
apple_example = {
    'net_income': 99_800_000_000,        # $99.8B
    'shares_outstanding': 15_900_000_000, # 15.9B shares
    'stock_options_outstanding': 500_000_000,  # 500M options
    'avg_option_strike': 100,
    'stock_price': 180
}

eps_data = calculate_eps (apple_example)
print(f"Basic EPS: \${eps_data['basic_eps']:.2f}")
print(f"Diluted EPS: \${eps_data['diluted_eps']:.2f}")
print(f"Dilution: {eps_data['dilution_pct']:.1f}%")

# Output:
# Basic EPS: $6.28
# Diluted EPS: $6.15
# Dilution: 2.1 %
\`\`\`

### EPS Quality Analysis

\`\`\`python
class EPSQualityAnalyzer:
    """Analyze EPS quality and detect manipulation."""
    
    def analyze_eps_growth_sources(
        self,
        current: Dict,
        prior: Dict
    ) -> Dict:
        """Decompose EPS growth into components."""
        
        eps_growth = (current['eps'] - prior['eps']) / prior['eps']
        
        # Component 1: Revenue growth
        revenue_growth = (current['revenue'] - prior['revenue']) / prior['revenue']
        
        # Component 2: Margin expansion
        current_margin = current['net_income'] / current['revenue']
        prior_margin = prior['net_income'] / prior['revenue']
        margin_change = current_margin - prior_margin
        
        # Component 3: Share count reduction (buybacks)
        share_change = (current['shares'] - prior['shares']) / prior['shares']
        
        # Calculate contribution of each
        # EPS = (Net Income / Shares)
        # EPS growth ≈ NI growth - Share growth
        
        ni_growth = (current['net_income'] - prior['net_income']) / prior['net_income']
        
        analysis = {
            'eps_growth': eps_growth,
            'revenue_contribution': revenue_growth * 0.4,  # Approximate
            'margin_contribution': margin_change * 0.4,
            'buyback_contribution': -share_change,  # Negative share change = positive EPS
            'quality_score': 0
        }
        
        # Quality assessment
        if revenue_growth > 0.15:  # Strong revenue growth
            analysis['quality_score'] += 40
        
        if margin_change > 0:  # Margin expansion
            analysis['quality_score'] += 30
        
        if share_change < -0.05:  # Buybacks
            analysis['quality_score'] += 20  # Nice to have, but not core business
        
        # Red flag: EPS growth entirely from buybacks
        if eps_growth > 0.10 and revenue_growth < 0.05:
            analysis['quality_score'] -= 30
            analysis['red_flag'] = 'EPS growth from buybacks, not revenue'
        
        return analysis

# Example: Good vs Bad EPS growth
good_growth = {
    'current': {
        'eps': 6.50,
        'revenue': 400_000,
        'net_income': 100_000,
        'shares': 15_400
    },
    'prior': {
        'eps': 5.50,
        'revenue': 350_000,
        'net_income': 85_000,
        'shares': 15_450
    }
}

bad_growth = {
    'current': {
        'eps': 6.50,
        'revenue': 350_000,
        'net_income': 85_000,
        'shares': 13_000  # Massive buybacks
    },
    'prior': {
        'eps': 5.50,
        'revenue': 352_000,
        'net_income': 86_000,
        'shares': 15_600
    }
}

analyzer = EPSQualityAnalyzer()

print("Good EPS Growth (revenue-driven):")
print(analyzer.analyze_eps_growth_sources(
    good_growth['current'],
    good_growth['prior']
))

print("\\nBad EPS Growth (buyback-driven):")
print(analyzer.analyze_eps_growth_sources(
    bad_growth['current'],
    bad_growth['prior']
))
\`\`\`

---

## Real-World Example: Tech vs Retail Income Statement

Let\'s compare Microsoft (software) vs Walmart (retail):

\`\`\`python
def compare_income_statements():
    """Compare business models through income statements."""
    
    # Microsoft FY2023 (in millions)
    microsoft = {
        'revenue': 211_900,
        'cogs': 65_900,
        'gross_profit': 146_000,
        'rd': 27_200,
        'sga': 29_000,
        'operating_income': 88_500,
        'interest': -1_900,
        'tax': -16_900,
        'net_income': 72_400,
        'shares': 7_430,
    }
    
    # Walmart FY2024 (in millions)
    walmart = {
        'revenue': 648_100,
        'cogs': 489_800,
        'gross_profit': 158_300,
        'rd': 0,  # Minimal
        'sga': 136_800,
        'operating_income': 22_000,
        'interest': -2_600,
        'tax': -4_600,
        'net_income': 15_500,
        'shares': 2_660,
    }
    
    def analyze (company_name: str, data: Dict):
        print(f"\\n{company_name} Analysis:")
        print("=" * 50)
        
        # Calculate margins
        gross_margin = data['gross_profit'] / data['revenue']
        operating_margin = data['operating_income'] / data['revenue']
        net_margin = data['net_income'] / data['revenue']
        
        # Calculate per-share metrics
        eps = data['net_income'] / data['shares']
        revenue_per_share = data['revenue'] / data['shares']
        
        print(f"Revenue: \${data['revenue']:,.0f}M")
print(f"Gross Margin: {gross_margin:.1%}")
print(f"Operating Margin: {operating_margin:.1%}")
print(f"Net Margin: {net_margin:.1%}")
print(f"EPS: \${eps:.2f}")
print(f"Revenue/Share: \${revenue_per_share:.2f}")
        
        # Efficiency metrics
print(f"\\nExpense Structure:")
print(f"  R&D as % of Revenue: {data['rd']/data['revenue']:.1%}")
print(f"  SG&A as % of Revenue: {data['sga']/data['revenue']:.1%}")

return {
    'gross_margin': gross_margin,
    'operating_margin': operating_margin,
    'net_margin': net_margin,
    'eps': eps
}

msft_metrics = analyze("MICROSOFT", microsoft)
wmt_metrics = analyze("WALMART", walmart)

print("\\n" + "=" * 50)
print("KEY TAKEAWAYS:")
print("=" * 50)
print(f"Microsoft has {msft_metrics['gross_margin']/wmt_metrics['gross_margin']:.1f}x higher gross margin")
print(f"Microsoft has {msft_metrics['net_margin']/wmt_metrics['net_margin']:.1f}x higher net margin")
print(f"Microsoft has {msft_metrics['eps']/wmt_metrics['eps']:.1f}x higher EPS")
print("\\nWhy? Software has:")
print("  • Near-zero marginal cost (no COGS for each additional unit)")
print("  • High operating leverage (fixed costs, variable revenue)")
print("  • Subscription/recurring revenue")
print("\\nWhile retail has:")
print("  • High COGS (must buy inventory)")
print("  • Thin margins (competitive pricing)")
print("  • Scale through volume, not margins")

compare_income_statements()
\`\`\`

---

## Detecting Revenue Quality Issues

\`\`\`python
import warnings

class RevenueQualityDetector:
    """Detect potential revenue recognition issues."""
    
    def __init__(self, threshold_config: Dict = None):
        self.config = threshold_config or {
            'dso_increase_threshold': 10,  # days
            'receivables_growth_multiple': 1.5,  # vs revenue growth
            'deferred_revenue_decrease_threshold': 0.2,  # 20% decline
        }
    
    def run_full_analysis(
        self,
        current_period: Dict,
        prior_period: Dict
    ) -> Dict:
        """Run comprehensive revenue quality checks."""
        
        issues = []
        severity_score = 0
        
        # Check 1: DSO trending
        dso_issue = self._check_dso_trend (current_period, prior_period)
        if dso_issue:
            issues.append (dso_issue)
            severity_score += dso_issue['severity']
        
        # Check 2: Receivables growth
        ar_issue = self._check_receivables_growth (current_period, prior_period)
        if ar_issue:
            issues.append (ar_issue)
            severity_score += ar_issue['severity']
        
        # Check 3: Deferred revenue (for subscription businesses)
        dr_issue = self._check_deferred_revenue (current_period, prior_period)
        if dr_issue:
            issues.append (dr_issue)
            severity_score += dr_issue['severity']
        
        # Check 4: One-time revenue spikes
        spike_issue = self._check_revenue_spike (current_period, prior_period)
        if spike_issue:
            issues.append (spike_issue)
            severity_score += spike_issue['severity']
        
        # Overall assessment
        if severity_score > 50:
            rating = 'HIGH RISK'
        elif severity_score > 25:
            rating = 'MODERATE RISK'
        else:
            rating = 'LOW RISK'
        
        return {
            'overall_rating': rating,
            'severity_score': severity_score,
            'issues_found': len (issues),
            'issues': issues,
            'recommendation': self._generate_recommendation (rating, issues)
        }
    
    def _check_dso_trend (self, current: Dict, prior: Dict) -> Dict:
        """Check if Days Sales Outstanding is increasing."""
        
        current_dso = (current['accounts_receivable'] / current['revenue']) * 90
        prior_dso = (prior['accounts_receivable'] / prior['revenue']) * 90
        
        dso_change = current_dso - prior_dso
        
        if dso_change > self.config['dso_increase_threshold']:
            return {
                'type': 'DSO_INCREASE',
                'severity': 20,
                'current_dso': current_dso,
                'prior_dso': prior_dso,
                'change': dso_change,
                'description': f'DSO increased by {dso_change:.1f} days. Customers taking longer to pay.',
                'potential_causes': [
                    'Looser credit terms to boost sales',
                    'Customers in financial distress',
                    'Channel stuffing',
                    'Revenue recognition timing issues'
                ]
            }
        return None
    
    def _check_receivables_growth (self, current: Dict, prior: Dict) -> Dict:
        """Check if receivables growing faster than revenue."""
        
        rev_growth = (current['revenue'] - prior['revenue']) / prior['revenue']
        ar_growth = (current['accounts_receivable'] - prior['accounts_receivable']) / prior['accounts_receivable']
        
        if ar_growth > rev_growth * self.config['receivables_growth_multiple']:
            return {
                'type': 'RECEIVABLES_OUTPACING_REVENUE',
                'severity': 30,
                'revenue_growth': rev_growth,
                'ar_growth': ar_growth,
                'ratio': ar_growth / rev_growth if rev_growth > 0 else float('inf'),
                'description': f'Receivables growing {ar_growth:.1%} vs revenue {rev_growth:.1%}',
                'red_flag_level': 'HIGH'
            }
        return None
    
    def _check_deferred_revenue (self, current: Dict, prior: Dict) -> Dict:
        """Check deferred revenue trends (key for SaaS)."""
        
        if 'deferred_revenue' not in current or 'deferred_revenue' not in prior:
            return None
        
        dr_change = (current['deferred_revenue'] - prior['deferred_revenue']) / prior['deferred_revenue']
        rev_growth = (current['revenue'] - prior['revenue']) / prior['revenue']
        
        # For healthy SaaS: deferred revenue should grow with/faster than revenue
        if dr_change < -self.config['deferred_revenue_decrease_threshold']:
            return {
                'type': 'DEFERRED_REVENUE_DECLINE',
                'severity': 25,
                'dr_change': dr_change,
                'revenue_growth': rev_growth,
                'description': f'Deferred revenue declined {dr_change:.1%}',
                'concern': 'Future revenue may slow (fewer advance bookings)',
                'applies_to': 'Subscription/SaaS businesses'
            }
        return None
    
    def _check_revenue_spike (self, current: Dict, prior: Dict) -> Dict:
        """Detect unusual revenue spikes (especially Q4)."""
        
        rev_growth = (current['revenue'] - prior['revenue']) / prior['revenue']
        
        # Get historical average if available
        if 'historical_avg_growth' in current:
            avg_growth = current['historical_avg_growth']
            
            # Spike: 2x+ normal growth
            if rev_growth > avg_growth * 2 and rev_growth > 0.2:
                return {
                    'type': 'REVENUE_SPIKE',
                    'severity': 15,
                    'current_growth': rev_growth,
                    'historical_avg': avg_growth,
                    'description': f'Revenue grew {rev_growth:.1%} vs typical {avg_growth:.1%}',
                    'questions_to_ask': [
                        'Was this a one-time deal?',
                        'Was it quarter-end push/channel stuffing?',
                        'Is it sustainable?'
                    ]
                }
        return None
    
    def _generate_recommendation (self, rating: str, issues: List[Dict]) -> str:
        """Generate actionable recommendation."""
        
        if rating == 'HIGH RISK':
            return """
            ⚠️ HIGH RISK: Multiple revenue quality issues detected.
            
            Actions:
            1. Review MD&A section for management commentary
            2. Check if company has history of restatements
            3. Compare to peer companies in same industry
            4. Consider SHORT position or avoid long position
            5. Flag for forensic analysis
            """
        elif rating == 'MODERATE RISK':
            return """
            ⚠️ MODERATE RISK: Some revenue quality concerns.
            
            Actions:
            1. Monitor closely in upcoming quarters
            2. Check quarterly earnings call transcript
            3. Review customer concentration
            4. Wait for trend confirmation before major position
            """
        else:
            return """
            ✅ LOW RISK: Revenue quality appears healthy.
            
            Continue normal analysis and valuation.
            """

# Usage Example
detector = RevenueQualityDetector()

# Example: Company with deteriorating revenue quality
current_q = {
    'revenue': 500_000,
    'accounts_receivable': 150_000,
    'deferred_revenue': 100_000,
    'historical_avg_growth': 0.10
}

prior_q = {
    'revenue': 400_000,
    'accounts_receivable': 90_000,
    'deferred_revenue': 120_000
}

analysis = detector.run_full_analysis (current_q, prior_q)

print("Revenue Quality Analysis")
print("=" * 60)
print(f"Overall Rating: {analysis['overall_rating']}")
print(f"Severity Score: {analysis['severity_score']}")
print(f"\\nIssues Found: {analysis['issues_found']}")

for issue in analysis['issues']:
    print(f"\\n• {issue['type']}: {issue['description']}")
    print(f"  Severity: {issue['severity']}")

print(f"\\nRecommendation:{analysis['recommendation']}")
\`\`\`

---

## Summary

**Key Takeaways**:

1. **Revenue Quality > Revenue Quantity**: Growing revenue that you can't collect is worthless
2. **Gross Margin** reveals business model quality (70% software >> 25% retail)
3. **Operating Leverage** is key: Revenue growing faster than expenses = scaling
4. **EPS** can be manipulated via buybacks—look at net income and revenue too
5. **EBITDA** is useful for comparisons but ignores CapEx (danger for capital-intensive businesses)

**For Engineers**:
- Build automated revenue quality checkers (DSO, receivables growth)
- Compare companies on **margins**, not just absolute dollars
- Detect earnings manipulation (buyback-driven EPS growth)
- Track **trends**, not just snapshots (improving vs declining margins)

**Red Flags to Code Into Your Systems**:
- ❌ DSO increasing >15 days quarter-over-quarter
- ❌ Receivables growing >1.5x revenue growth
- ❌ Deferred revenue declining while revenue grows (SaaS)
- ❌ EPS growing but revenue flat (buyback-only growth)
- ❌ Gross margins declining over multiple quarters

**Next Section**: Balance Sheet Analysis—understanding what companies own and owe.
`,
};
