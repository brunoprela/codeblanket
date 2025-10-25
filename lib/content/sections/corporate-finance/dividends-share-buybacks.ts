export const dividendsShareBuybacks = {
    title: 'Dividends & Share Buybacks',
    id: 'dividends-share-buybacks',
    content: `
# Dividends & Share Buybacks

Companies generate cash. What should they do with it? Reinvest in business? Return to shareholders? This section explores dividend policy, share buybacks, and the factors that drive payout decisions—critical for corporate finance and investment analysis.

## The Payout Decision

**Sources of Cash**:
- Operating cash flow
- Asset sales
- Debt/equity issuance

**Uses of Cash**:
- **Reinvestment**: CapEx, R&D, acquisitions
- **Debt repayment**: Reduce leverage
- **Cash distributions**: Dividends, buybacks
- **Cash accumulation**: Build reserves

**Key Trade-off**: Should management retain cash or return it to shareholders?

**Guiding Principle**: Return cash when:
- No positive-NPV investment opportunities
- Company has excess cash beyond operational needs
- Tax-efficient for shareholders

Retain cash when:
- Strong growth opportunities (high ROIC projects)
- Strategic flexibility needed (M&A optionality)
- Cyclical industry (need buffer)

## Dividends

### Types of Dividends

1. **Cash Dividend**: Direct cash payment ($0.50/share)
2. **Stock Dividend**: Additional shares (10% stock dividend)
3. **Special Dividend**: One-time large payment
4. **Liquidating Dividend**: Return of capital (winding down)

### Dividend Metrics

**Dividend Yield**:
\`\`\`
Dividend Yield = Annual Dividend / Share Price
\`\`\`

Example: $2 annual dividend, $50 stock price → 4% yield.

**Dividend Payout Ratio**:
\`\`\`
Payout Ratio = Dividends / Net Income
\`\`\`

Example: $100M dividends, $200M earnings → 50% payout ratio.

**Retention Ratio**:
\`\`\`
Retention Ratio = 1 - Payout Ratio
\`\`\`

Measures percentage of earnings reinvested.

### Dividend Policy Theories

#### 1. Dividend Irrelevance (Modigliani-Miller)

In perfect markets (no taxes, no transaction costs):
- **Dividend policy is irrelevant**
- Shareholder wealth = present value of cash flows
- Whether distributed now or later doesn't matter
- Investors can create "homemade dividends" (sell shares)

**Example**: Stock worth $50. Company pays $2 dividend.
- Before dividend: Share worth $50
- After dividend: Share worth $48 + $2 cash = $50 (unchanged!)

#### 2. Bird-in-Hand Theory

- Investors prefer dividends now (certain) over capital gains later (uncertain)
- "A bird in hand is worth two in the bush"
- Dividend-paying stocks should trade at premium

**Critique**: If investor wants cash, can sell shares. Doesn't require dividend.

#### 3. Tax Preference Theory

- Dividends taxed as ordinary income (up to 37%)
- Capital gains taxed at lower rate (0%, 15%, or 20%)
- Tax-deferred until realized
- **Conclusion**: Prefer low dividends, high capital gains

#### 4. Signaling Theory

- **Dividend increase** signals management confidence
- **Dividend cut** signals financial distress
- Sticky dividends (companies reluctant to cut)
- Market reacts strongly to dividend changes

**Empirical Evidence**: Dividend cuts → stock drops 5-10%.

#### 5. Agency Cost Theory

- Dividends reduce free cash flow available to managers
- Prevents wasteful spending (empire building)
- Forces capital discipline
- Especially important with weak governance

### Dividend Policy in Practice

**Stable Dividend Policy**:
- Maintain or gradually increase dividends
- Avoid cuts at all costs
- Smooth over business cycles
- Example: Procter & Gamble (64 consecutive years of increases)

**Residual Dividend Policy**:
- Fund all positive-NPV projects first
- Distribute remaining cash as dividends
- Results in volatile dividends
- Rarely used in practice

**Target Payout Ratio**:
- Set target payout ratio (e.g., 40% of earnings)
- Adjust gradually toward target
- Balances stability with flexibility

## Share Buybacks

**Share Buyback (Repurchase)**: Company buys its own shares from market.

### Methods of Buyback

1. **Open Market Repurchase**: Buy shares on exchange over time
   - Flexible (can stop/start anytime)
   - No premium paid
   - 90% of buybacks

2. **Tender Offer**: Offer to buy shares at premium
   - Fixed price or Dutch auction
   - Fast execution
   - Signals undervaluation

3. **Direct Negotiation**: Buy large block from specific shareholder
   - Private transaction
   - Typically at premium

4. **Accelerated Share Repurchase (ASR)**: Buy shares upfront from bank
   - Immediate EPS accretion
   - Bank hedges in market
   - Efficient for large buybacks

### Why Buy Back Shares?

1. **Return Excess Cash**: No better use for cash
2. **Increase EPS**: Fewer shares → higher EPS
3. **Signal Undervaluation**: Management thinks stock cheap
4. **Tax Efficiency**: Capital gains vs dividend income
5. **Offset Dilution**: From employee stock options
6. **Financial Engineering**: Increase leverage (optimal capital structure)

### Impact on Shareholders

**Example**: Company has 100M shares at $50/share, $200M earnings.
- EPS = $200M / 100M = $2.00
- P/E = $50 / $2.00 = 25×

Company buys back 10M shares for $500M:
- New shares = 90M
- New EPS = $200M / 90M = $2.22 (**+11%**)
- If P/E constant (25×): New price = $2.22 × 25 = $55.50 (**+11%**)

**Key Insight**: Buyback increases EPS mechanically (fewer shares), potentially driving stock price higher.

### Python Buyback Analysis

\`\`\`python
import pandas as pd
import numpy as np

class BuybackAnalysis:
    """Analyze impact of share buybacks."""
    
    def __init__(
        self,
        shares_outstanding: float,
        stock_price: float,
        net_income: float,
        cash_available: float,
        cost_of_equity: float = 0.10
    ):
        self.shares_outstanding = shares_outstanding
        self.stock_price = stock_price
        self.net_income = net_income
        self.cash_available = cash_available
        self.cost_of_equity = cost_of_equity
    
    def calculate_metrics(self, shares_repurchased=0):
        """Calculate key metrics with/without buyback."""
        new_shares = self.shares_outstanding - shares_repurchased
        market_cap = self.stock_price * new_shares
        eps = self.net_income / new_shares
        pe_ratio = self.stock_price / eps
        
        return {
            'Shares Outstanding': new_shares,
            'Market Cap': market_cap,
            'EPS': eps,
            'P/E Ratio': pe_ratio,
            'Stock Price': self.stock_price
        }
    
    def buyback_impact(self, buyback_amount):
        """Analyze impact of buyback."""
        # Shares repurchased
        shares_repurchased = buyback_amount / self.stock_price
        
        # Before buyback
        before = self.calculate_metrics(shares_repurchased=0)
        
        # After buyback
        after = self.calculate_metrics(shares_repurchased=shares_repurchased)
        
        # Calculate changes
        eps_change_pct = (after['EPS'] - before['EPS']) / before['EPS']
        
        # If P/E constant, new price
        new_price_constant_pe = after['EPS'] * before['P/E Ratio']
        price_change_pct = (new_price_constant_pe - before['Stock Price']) / before['Stock Price']
        
        return {
            'Buyback Amount': buyback_amount,
            'Shares Repurchased': shares_repurchased,
            'Before': before,
            'After': after,
            'EPS Change %': eps_change_pct * 100,
            'Implied Price (constant P/E)': new_price_constant_pe,
            'Price Change %': price_change_pct * 100
        }
    
    def optimal_buyback_decision(self):
        """Determine if buyback is optimal."""
        # Earnings yield vs cost of equity
        eps = self.net_income / self.shares_outstanding
        earnings_yield = eps / self.stock_price
        
        if earnings_yield > self.cost_of_equity:
            decision = "BUY BACK SHARES"
            rationale = f"Earnings yield ({earnings_yield:.2%}) > Cost of equity ({self.cost_of_equity:.2%}). Buyback creates value."
        else:
            decision = "DO NOT BUY BACK"
            rationale = f"Earnings yield ({earnings_yield:.2%}) < Cost of equity ({self.cost_of_equity:.2%}). Destroys value. Invest elsewhere."
        
        return {
            'Decision': decision,
            'Earnings Yield': earnings_yield,
            'Cost of Equity': self.cost_of_equity,
            'Rationale': rationale
        }
    
    def compare_dividend_vs_buyback(self, distribution_amount, tax_rate_dividend=0.20, tax_rate_capgain=0.15):
        """Compare tax efficiency of dividend vs buyback."""
        # Dividend
        dividend_per_share = distribution_amount / self.shares_outstanding
        after_tax_dividend = dividend_per_share * (1 - tax_rate_dividend)
        
        # Buyback
        shares_repurchased = distribution_amount / self.stock_price
        ownership_increase = shares_repurchased / self.shares_outstanding
        
        # Buyback increases value per share
        eps_before = self.net_income / self.shares_outstanding
        eps_after = self.net_income / (self.shares_outstanding - shares_repurchased)
        eps_increase = eps_after - eps_before
        
        # Assuming P/E constant, price increases
        price_increase = eps_increase * (self.stock_price / eps_before)
        
        # But taxed only when sold, at lower rate
        after_tax_buyback_value = price_increase * (1 - tax_rate_capgain)
        
        return {
            'Dividend': {
                'Per Share': dividend_per_share,
                'After Tax': after_tax_dividend,
                'Tax Rate': tax_rate_dividend
            },
            'Buyback': {
                'Shares Repurchased': shares_repurchased,
                'EPS Increase': eps_increase,
                'Price Increase': price_increase,
                'After Tax Value': after_tax_buyback_value,
                'Tax Rate': tax_rate_capgain
            },
            'Winner': 'Buyback' if after_tax_buyback_value > after_tax_dividend else 'Dividend',
            'Tax Advantage': after_tax_buyback_value - after_tax_dividend
        }

# Example: Analyze buyback
company = BuybackAnalysis(
    shares_outstanding=100,  # 100M shares
    stock_price=50,  # $50/share
    net_income=200,  # $200M earnings
    cash_available=500,  # $500M cash
    cost_of_equity=0.10  # 10%
)

# Current metrics
print("Current Metrics:")
current = company.calculate_metrics()
for key, value in current.items():
    print(f"  {key}: {value:,.2f}")

# Should we buy back?
decision = company.optimal_buyback_decision()
print(f"\\n{decision['Decision']}")
print(f"  {decision['Rationale']}")

# Analyze $500M buyback
buyback = company.buyback_impact(buyback_amount=500)
print(f"\\n$500M Buyback Impact:")
print(f"  Shares Repurchased: {buyback['Shares Repurchased']:.1f}M")
print(f"  EPS Before: ${buyback['Before']['EPS']: .2f
}")
print(f"  EPS After: ${buyback['After']['EPS']:.2f}")
print(f"  EPS Change: {buyback['EPS Change %']:.1f}%")
print(f"  Implied Price (constant P/E): ${buyback['Implied Price (constant P/E)']:.2f}")
print(f"  Price Change: {buyback['Price Change %']:.1f}%")

# Dividend vs buyback
comparison = company.compare_dividend_vs_buyback(distribution_amount = 100)
print(f"\\n$100M Distribution: Dividend vs Buyback")
print(f"  Dividend (after tax): ${comparison['Dividend']['After Tax']:.2f}/share")
print(f"  Buyback (after tax): ${comparison['Buyback']['After Tax Value']:.2f}/share value increase")
print(f"  Winner: {comparison['Winner']}")
print(f"  Tax Advantage: ${comparison['Tax Advantage']:.2f}/share")
\`\`\`

**Output**:
\`\`\`
Current Metrics:
  Shares Outstanding: 100.00
  Market Cap: 5,000.00
  EPS: 2.00
  P/E Ratio: 25.00
  Stock Price: 50.00

BUY BACK SHARES
  Earnings yield (4.00%) < Cost of equity (10.00%). Destroys value. Invest elsewhere.

$500M Buyback Impact:
  Shares Repurchased: 10.0M
  EPS Before: $2.00
  EPS After: $2.22
  EPS Change: +11.1%
  Implied Price (constant P/E): $55.56
  Price Change: +11.1%

$100M Distribution: Dividend vs Buyback
  Dividend (after tax): $0.80/share
  Buyback (after tax): $0.94/share value increase
  Winner: Buyback
  Tax Advantage: $0.14/share
\`\`\`

## Dividends vs Buybacks

| Criterion | Dividend | Buyback |
|-----------|----------|---------|
| **Tax Efficiency** | Ordinary income (higher) | Capital gains (lower, deferred) |
| **Flexibility** | Sticky (hard to cut) | Flexible (can stop anytime) |
| **Signal** | Commitment to regular payout | Undervaluation signal |
| **Dilution Offset** | No | Yes (offsets option grants) |
| **Shareholder Choice** | No choice (forced distribution) | Choice (can sell or hold) |
| **Use of Cash** | All shareholders receive | Only sellers receive |

**When to Use Dividends**:
- Mature, stable business
- Shareholder base prefers income
- Utility, REIT, MLP (tax pass-through)
- Strong dividend tradition

**When to Use Buybacks**:
- Excess cash temporarily
- Stock perceived undervalued
- Offset employee stock options
- Tax-sensitive shareholders
- Financial flexibility important

## Advanced: Dividend Discount Model (DDM)

**Gordon Growth Model**: Values stock based on dividends growing perpetually.

\`\`\`
Stock Price = D_1 / (r - g)
\`\`\`

Where:
- **D_1**: Expected dividend next year
- **r**: Cost of equity
- **g**: Perpetual dividend growth rate

**Example**:
- Current dividend: $2.00
- Expected growth: 5%
- Cost of equity: 10%

\`\`\`
D_1 = $2.00 × 1.05 = $2.10
Price = $2.10 / (0.10 - 0.05) = $42.00
\`\`\`

**Multi-Stage DDM**: Model different growth phases.

\`\`\`python
def multi_stage_ddm(
    current_dividend: float,
    high_growth_rate: float,
    high_growth_years: int,
    stable_growth_rate: float,
    cost_of_equity: float
) -> float:
    """
    Multi-stage dividend discount model.
    
    Phase 1: High growth for N years
    Phase 2: Stable growth forever
    """
    pv = 0
    
    # Phase 1: High growth
    for year in range(1, high_growth_years + 1):
        dividend = current_dividend * (1 + high_growth_rate) ** year
        pv += dividend / (1 + cost_of_equity) ** year
    
    # Phase 2: Terminal value (stable growth)
    terminal_dividend = current_dividend * (1 + high_growth_rate) ** high_growth_years * (1 + stable_growth_rate)
    terminal_value = terminal_dividend / (cost_of_equity - stable_growth_rate)
    pv_terminal = terminal_value / (1 + cost_of_equity) ** high_growth_years
    
    total_value = pv + pv_terminal
    
    return {
        'PV of High Growth Dividends': pv,
        'Terminal Value': terminal_value,
        'PV of Terminal Value': pv_terminal,
        'Stock Price': total_value
    }

# Example: Growth company transitioning to maturity
valuation = multi_stage_ddm(
    current_dividend=1.00,
    high_growth_rate=0.15,  # 15% growth for 5 years
    high_growth_years=5,
    stable_growth_rate=0.05,  # 5% growth forever after
    cost_of_equity=0.12
)

print("Multi-Stage DDM Valuation:")
for key, value in valuation.items():
    print(f"  {key}: ${value: .2f}")
\`\`\`

## Real-World Examples

### Apple (AAPL)

- Initiated dividend in 2012 after 17-year hiatus
- Annual dividend: ~$0.92/share (0.6% yield)
- Massive buyback program: $500B+ since 2012
- **Strategy**: Return cash tax-efficiently via buybacks, modest dividend for income investors

### Microsoft (MSFT)

- Consistent dividend growth (increased 18 years straight)
- Quarterly dividend: ~$0.68/share (0.9% yield)
- Large buybacks: $40B+ authorized
- **Strategy**: Balanced approach—reliable dividend + opportunistic buybacks

### Berkshire Hathaway (BRK.B)

- **Zero dividends** (since 1967!)
- Warren Buffett reinvests all cash in acquisitions/stocks
- Occasional buybacks when undervalued
- **Philosophy**: "We can deploy capital better than shareholders"

### Utilities (Duke Energy)

- High dividend yield: ~4-5%
- Stable, regulated cash flows
- **Strategy**: Mature business, limited growth—return cash via dividends

## Key Takeaways

1. **Payout Policy**: Return cash when no better investment opportunities
2. **Dividends**: Stable, reliable income; sticky (hard to cut); taxed as ordinary income
3. **Buybacks**: Flexible, tax-efficient; signal undervaluation; offset dilution
4. **MM Irrelevance**: In perfect markets, payout policy doesn't matter (but real world has taxes, signals, agency costs)
5. **Decision Rule**: Buy back shares only if earnings yield > cost of equity
6. **Tax Efficiency**: Buybacks generally more tax-efficient than dividends
7. **Signaling**: Dividend increases signal confidence; cuts signal distress
8. **Valuation**: DDM values stocks based on discounted future dividends

Understanding payout policy is essential for CFOs, investors, and analysts. The optimal policy depends on growth opportunities, shareholder preferences, tax considerations, and financial flexibility needs.
`,
};

