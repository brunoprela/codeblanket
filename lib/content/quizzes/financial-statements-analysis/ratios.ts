export const ratiosDiscussionQuestions = [
  {
    id: 1,
    question:
      "Company A has ROE of 25% with debt-to-equity of 2.0. Company B has ROE of 15% with debt-to-equity of 0.3. An investor says 'Company A is better because ROE is higher.' Perform a DuPont analysis to determine which company actually has better operating performance, and explain the role of leverage in creating ROE differences.",
    answer: `**DuPont Analysis reveals Company B has superior operating performance despite lower ROE.**

\`\`\`python
# Company A: High ROE, High Leverage
company_a = {
    'net_income': 50_000_000,
    'revenue': 500_000_000,
    'total_assets': 300_000_000,
    'shareholders_equity': 100_000_000,  # D/E = 2.0 means debt = $200M
    'total_debt': 200_000_000
}

# Company B: Lower ROE, Low Leverage
company_b = {
    'net_income': 45_000_000,
    'revenue': 500_000_000,
    'total_assets': 300_000_000,
    'shareholders_equity': 230_000_000,  # D/E = 0.3 means debt = $70M
    'total_debt': 70_000_000
}

def dupont_analysis(company):
    net_margin = company['net_income'] / company['revenue']
    asset_turnover = company['revenue'] / company['total_assets']
    equity_multiplier = company['total_assets'] / company['shareholders_equity']
    roe = net_margin * asset_turnover * equity_multiplier
    roa = company['net_income'] / company['total_assets']
    
    return {
        'roe': roe,
        'net_margin': net_margin,
        'asset_turnover': asset_turnover,
        'equity_multiplier': equity_multiplier,
        'roa': roa
    }

a_metrics = dupont_analysis(company_a)
b_metrics = dupont_analysis(company_b)

print("DuPont Analysis Comparison:")
print(f"\\nCompany A (High Leverage):")
print(f"  ROE: {a_metrics['roe']:.1%}")
print(f"  = Net Margin ({a_metrics['net_margin']:.1%}) × Asset Turnover ({a_metrics['asset_turnover']:.2f}) × Equity Multiplier ({a_metrics['equity_multiplier']:.2f})")
print(f"  ROA: {a_metrics['roa']:.1%}")

print(f"\\nCompany B (Low Leverage):")
print(f"  ROE: {b_metrics['roe']:.1%}")
print(f"  = Net Margin ({b_metrics['net_margin']:.1%}) × Asset Turnover ({b_metrics['asset_turnover']:.2f}) × Equity Multiplier ({b_metrics['equity_multiplier']:.2f})")
print(f"  ROA: {b_metrics['roa']:.1%}")
\`\`\`

**Key Insight**: Both companies have **identical ROA (16.7%)** - same operating performance. Company A's higher ROE comes **entirely from leverage**, not superior operations. In a downturn, Company A faces higher bankruptcy risk while Company B is safer. **Winner: Company B** for quality, Company A only for risk-adjusted returns.`,
  },

  {
    id: 2,
    question:
      "A retail company has current ratio of 2.0, quick ratio of 0.6, and cash ratio of 0.3. The CFO claims 'Our current ratio is healthy at 2.0x.' Analyze the complete liquidity picture. What does the divergence between these ratios tell you? What specific balance sheet issues exist?",
    answer: `**The company has a severe liquidity problem disguised by inventory.**

\`\`\`python
# Given ratios
current_ratio = 2.0
quick_ratio = 0.6
cash_ratio = 0.3
current_liabilities = 100_000_000  # Assume

# Calculate asset components
current_assets = current_ratio * current_liabilities  # $200M
quick_assets = quick_ratio * current_liabilities  # $60M
cash_and_securities = cash_ratio * current_liabilities  # $30M

# Derive inventory and other
inventory = current_assets - quick_assets  # $140M (70% of current assets!)
receivables = quick_assets - cash_and_securities  # $30M

print("Balance Sheet Analysis:")
print(f"Current Assets:        \${current_assets:, .0f}")
print(f"  Cash & Securities:   \${cash_and_securities:,.0f} (15%)")
print(f"  Receivables:         \${receivables:,.0f} (15%)")
print(f"  Inventory:           \${inventory:,.0f} (70%) ← PROBLEM!")
print(f"Current Liabilities:   \${current_liabilities:,.0f}")
print()
print("Liquidity Assessment:")
print("  Current Ratio: 2.0x ✓ Looks good")
print("  Quick Ratio: 0.6x ✗ Can't pay liabilities without selling inventory")
print("  Cash Ratio: 0.3x ✗ Very little actual cash")
print()
print("RED FLAG: 70% of current assets are ILLIQUID inventory")
print("If inventory doesn't sell quickly → LIQUIDITY CRISIS")
\`\`\`

**The Problem**: Inventory represents 70% of current assets. Quick ratio of 0.6 means company can only cover 60% of obligations with liquid assets. This suggests:
- **Slow-moving inventory** (potential obsolescence)
- **Overstock** (poor inventory management)  
- **Seasonal buildup** (or worse, unsellable goods)

**Action**: Examine inventory turnover, aging, and check for upcoming write-downs. The 2.0 current ratio is **misleading** - real liquidity is poor.`,
  },

  {
    id: 3,
    question:
      "You're comparing two SaaS companies with identical revenue ($200M) and growth (40% YoY). Company A has LTV/CAC of 5.0x, magic number of 1.2, and net dollar retention of 130%. Company B has LTV/CAC of 2.5x, magic number of 0.6, and NDR of 105%. Both are valued at $2B. Which is more attractive and why? Calculate which should command a premium valuation.",
    answer: `**Company A is significantly more attractive and deserves a much higher valuation.**

\`\`\`python
import pandas as pd

companies = {
    'Metric': ['LTV/CAC', 'Magic Number', 'Net Dollar Retention', 'Valuation', 'Revenue Multiple'],
    'Company A': ['5.0x', '1.2', '130%', '$2.0B', '10x'],
    'Company B': ['2.5x', '0.6', '105%', '$2.0B', '10x'],
    'Benchmark': ['>3.0x', '>0.75', '>110%', '-', '-']
}

df = pd.DataFrame(companies)
print(df.to_string(index=False))

print("\\n=== Unit Economics Analysis ===")
print("\\nCompany A:")
print("  • LTV/CAC 5.0x: Each $1 spent on acquisition returns $5")
print("  • Magic Number 1.2: Generates $1.20 ARR per $1 S&M spend (excellent!)")
print("  • NDR 130%: Existing customers expand 30% annually")
print("  → EXCEPTIONAL unit economics, highly efficient GTM")
print()
print("Company B:")
print("  • LTV/CAC 2.5x: Only $2.50 return per $1 (below 3x benchmark)")
print("  • Magic Number 0.6: Only $0.60 ARR per $1 S&M (inefficient)")
print("  • NDR 105%: Minimal expansion (only 5%)")
print("  → WEAK unit economics, burning cash to grow")

print("\\n=== Implications ===")
print("Company A:")
print("  • Can grow profitably (strong payback)")
print("  • Compounding growth from expansion revenue")
print("  • Can scale marketing efficiently")
print("  • FAIR VALUE: $2.8B+ (14x revenue)")
print()
print("Company B:")
print("  • Unsustainable unit economics")
print("  • Will struggle to reach profitability")
print("  • High customer acquisition costs")
print("  • FAIR VALUE: $1.2B (6x revenue)")
\`\`\`

**Conclusion**: At identical $2B valuations, **Company A is undervalued** and **Company B is overvalued**. Company A's superior unit economics mean it will generate far more free cash flow over time. Company B's weak metrics suggest it may never reach profitability without significant improvement. **Recommend: Buy A, Sell B.**`,
  },
];
