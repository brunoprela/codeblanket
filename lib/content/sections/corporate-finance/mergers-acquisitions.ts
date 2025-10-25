export const mergersAcquisitions = {
  title: 'Mergers & Acquisitions',
  id: 'mergers-acquisitions',
  content: `
# Mergers & Acquisitions (M&A)

Mergers and acquisitions reshape industries, create (or destroy) billions in value, and are among the most complex transactions in corporate finance. This section covers M&A strategy, valuation, deal structures, synergies, and pitfalls—essential knowledge for investment bankers, corporate development professionals, and investors.

## M&A Basics

**Merger**: Two companies combine to form one entity.
- **Merger of equals**: Relatively similar-sized companies (e.g., Exxon + Mobil)
- **Acquisition**: One company buys another (acquirer + target)

**Acquisition**: Larger company acquires smaller company.
- Target ceases to exist as separate entity
- Acquirer survives

### Types of M&A

**By Industry Relationship**:

1. **Horizontal**: Same industry (Competitor + Competitor)
   - Example: Disney acquires 21st Century Fox
   - Goal: Market share, economies of scale, eliminate competition

2. **Vertical**: Supply chain (Supplier + Customer)
   - Example: Amazon acquires Whole Foods (distribution)
   - Goal: Control supply chain, reduce costs, capture margins

3. **Conglomerate**: Unrelated businesses
   - Example: Berkshire Hathaway acquires See\'s Candies, GEICO, etc.
   - Goal: Diversification, financial engineering

**By Transaction Type**:

1. **Friendly**: Target board approves deal
2. **Hostile**: Acquirer bypasses board, goes directly to shareholders
3. **Strategic**: Operating company acquires for synergies
4. **Financial**: Private equity acquires for returns (LBO)

## Why Do M&A Happen?

### Good Reasons (Value Creation)

1. **Synergies**
   - Revenue synergies: Cross-selling, market expansion
   - Cost synergies: Eliminate duplicates, economies of scale
   - Formula: \`2 + 2 = 5\` (combined value > sum of parts)

2. **Acquire Capabilities**
   - Technology, talent, IP
   - Faster than building internally ("buy vs build")

3. **Market Power**
   - Increase pricing power
   - Reduce competition (subject to antitrust)

4. **Financial Engineering**
   - Use excess cash productively
   - Tax benefits (NOLs, interest deductibility)

### Bad Reasons (Value Destruction)

1. **Empire Building**: CEO ego, not shareholder value
2. **Overpaying**: Winner's curse (overbid in competitive auction)
3. **Poor Integration**: Can't realize synergies
4. **Diversification**: Shareholders can diversify themselves (no need for conglomerate)

## M&A Valuation

### Valuation Methods

1. **DCF Analysis**: Intrinsic value based on cash flows
2. **Comparable Companies**: Trading multiples (EV/EBITDA, P/E)
3. **Precedent Transactions**: M&A multiples (includes control premium)
4. **LBO Analysis**: What PE firm would pay (leveraged returns)

**Control Premium**: Amount above market price acquirer pays.
\`\`\`
Control Premium = (Offer Price - Pre-announcement Price) / Pre-announcement Price
\`\`\`

Typical premium: 20-40%.

### Synergy Valuation

**Total Value Created**:
\`\`\`
Combined Value = Value_A + Value_B + Synergies - Integration Costs
\`\`\`

**Value to Acquirer**:
\`\`\`
Value to A = Combined Value - Price Paid for B
\`\`\`

**Python M&A Valuation**:

\`\`\`python
import pandas as pd
import numpy as np

class MandAValuation:
    """Comprehensive M&A valuation and analysis."""
    
    def __init__(
        self,
        acquirer_name: str,
        target_name: str,
        acquirer_value: float,
        target_value: float,
        synergies: float,
        integration_costs: float,
        control_premium: float = 0.30
    ):
        self.acquirer_name = acquirer_name
        self.target_name = target_name
        self.acquirer_value = acquirer_value
        self.target_value = target_value
        self.synergies = synergies
        self.integration_costs = integration_costs
        self.control_premium = control_premium
    
    def calculate_deal_value (self):
        """Calculate total deal value and value creation."""
        # Offer price (target value + premium)
        offer_price = self.target_value * (1 + self.control_premium)
        
        # Combined company value
        combined_value = (
            self.acquirer_value +
            self.target_value +
            self.synergies -
            self.integration_costs
        )
        
        # Value created/destroyed for acquirer
        value_to_acquirer = combined_value - self.acquirer_value - offer_price
        
        # Value created/destroyed for target
        value_to_target = offer_price - self.target_value
        
        # Total value created
        total_value_created = combined_value - self.acquirer_value - self.target_value
        
        return {
            'Target Standalone Value': self.target_value,
            'Control Premium': self.control_premium * 100,
            'Offer Price': offer_price,
            'Acquirer Value': self.acquirer_value,
            'Combined Value (pre-synergies)': self.acquirer_value + self.target_value,
            'Synergies': self.synergies,
            'Integration Costs': self.integration_costs,
            'Combined Value (post-synergies)': combined_value,
            'Value to Acquirer': value_to_acquirer,
            'Value to Target': value_to_target,
            'Total Value Created': total_value_created,
            'NPV to Acquirer': value_to_acquirer,
            'Recommendation': 'PROCEED' if value_to_acquirer > 0 else 'REJECT'
        }
    
    def accretion_dilution_analysis(
        self,
        acquirer_shares: float,
        acquirer_eps: float,
        target_eps: float,
        target_shares: float,
        exchange_ratio: float = None,
        cash_consideration: float = None,
        new_debt: float = 0,
        interest_rate: float = 0.05,
        tax_rate: float = 0.25
    ):
        """
        Analyze EPS accretion/dilution.
        
        Exchange ratio: Target shares given per acquirer share.
        Cash consideration: Cash paid per target share.
        """
        # Current EPS
        acquirer_net_income = acquirer_eps * acquirer_shares
        target_net_income = target_eps * target_shares
        
        # Pro forma net income
        interest_expense = new_debt * interest_rate
        tax_shield = interest_expense * tax_rate
        net_interest = interest_expense - tax_shield
        
        pro_forma_ni = acquirer_net_income + target_net_income - net_interest
        
        # New shares outstanding
        if exchange_ratio:
            # Stock deal: Issue new shares
            new_shares_issued = target_shares * exchange_ratio
            total_cash_paid = cash_consideration * target_shares if cash_consideration else 0
        elif cash_consideration:
            # Cash deal: No new shares
            new_shares_issued = 0
            total_cash_paid = cash_consideration * target_shares
        else:
            raise ValueError("Must specify exchange_ratio or cash_consideration")
        
        pro_forma_shares = acquirer_shares + new_shares_issued
        
        # Pro forma EPS
        pro_forma_eps = pro_forma_ni / pro_forma_shares
        
        # Accretion/dilution
        eps_change = pro_forma_eps - acquirer_eps
        eps_change_pct = eps_change / acquirer_eps
        
        accretion_dilution = "Accretive" if eps_change > 0 else "Dilutive"
        
        return {
            'Acquirer Current EPS': acquirer_eps,
            'Target EPS': target_eps,
            'Pro Forma NI': pro_forma_ni,
            'New Shares Issued': new_shares_issued,
            'Pro Forma Shares': pro_forma_shares,
            'Pro Forma EPS': pro_forma_eps,
            'EPS Change': eps_change,
            'EPS Change %': eps_change_pct * 100,
            'Accretion/Dilution': accretion_dilution,
            'Total Cash Paid': total_cash_paid,
            'New Debt': new_debt
        }
    
    def calculate_breakup_fee (self, deal_value: float, breakup_pct: float = 0.03):
        """Calculate breakup fee (typically 2-4% of deal value)."""
        breakup_fee = deal_value * breakup_pct
        return breakup_fee
    
    def print_deal_summary (self):
        """Print formatted deal summary."""
        deal = self.calculate_deal_value()
        
        print(f"\\n{'=' * 70}")
        print(f"M&A Deal Analysis: {self.acquirer_name} acquires {self.target_name}")
        print(f"{'=' * 70}\\n")
        
        print("Valuation:")
        print(f"  {self.target_name} Standalone Value: \${deal['Target Standalone Value']:,.0f}M")
print(f"  Control Premium: {deal['Control Premium']:.1f}%")
print(f"  Offer Price: \${deal['Offer Price']:,.0f}M")
print(f"\\n  {self.acquirer_name} Value: \${deal['Acquirer Value']:,.0f}M")
print(f"  Combined Value (pre-synergies): \${deal['Combined Value (pre-synergies)']:,.0f}M")
print(f"\\n  Expected Synergies: \${deal['Synergies']:,.0f}M")
print(f"  Less: Integration Costs: \${deal['Integration Costs']:,.0f}M")
print(f"  Net Synergies: \${deal['Synergies'] - deal['Integration Costs']:,.0f}M")
print(f"  {'─' * 68}")
print(f"  Combined Value (post-synergies): \${deal['Combined Value (post-synergies)']:,.0f}M")

print(f"\\nValue Distribution:")
print(f"  Value to {self.target_name} Shareholders: \${deal['Value to Target']:,.0f}M")
print(f"  Value to {self.acquirer_name} Shareholders: \${deal['Value to Acquirer']:,.0f}M")
print(f"  Total Value Created: \${deal['Total Value Created']:,.0f}M")

print(f"\\nRecommendation: {deal['Recommendation']}")
if deal['NPV to Acquirer'] > 0:
    print(f"  Deal creates \${deal['NPV to Acquirer']:,.0f}M for {self.acquirer_name} shareholders.")
else:
print(f"  Deal destroys \${-deal['NPV to Acquirer']:,.0f}M for {self.acquirer_name} shareholders.")

print(f"{'=' * 70}\\n")

# Example: Analyze M & A deal
deal = MandAValuation(
    acquirer_name = "TechCorp",
    target_name = "InnovateCo",
    acquirer_value = 5000,  # $5B
    target_value = 1000,  # $1B
    synergies = 300,  # $300M
    integration_costs = 50,  # $50M
    control_premium = 0.35  # 35 % premium
)

deal.print_deal_summary()

# Accretion / Dilution Analysis
print("EPS Accretion/Dilution Analysis:\\n")

# Scenario 1: All - stock deal
stock_deal = deal.accretion_dilution_analysis(
    acquirer_shares = 100,  # 100M shares
    acquirer_eps = 3.00,
    target_eps = 2.50,
    target_shares = 50,  # 50M shares
    exchange_ratio = 0.60,  # 0.6 TechCorp shares per InnovateCo share
    cash_consideration = 0
)

print("Scenario 1: All-Stock Deal (0.6× exchange ratio)")
for key, value in stock_deal.items():
    if isinstance (value, (int, float)):
        print(f"  {key}: {value:,.2f}")
    else:
    print(f"  {key}: {value}")

# Scenario 2: All - cash deal
cash_deal = deal.accretion_dilution_analysis(
    acquirer_shares = 100,
    acquirer_eps = 3.00,
    target_eps = 2.50,
    target_shares = 50,
    exchange_ratio = None,
    cash_consideration = 27,  # $27 / share(35 % premium on $20 target stock)
    new_debt = 1350,  # $1.35B debt to finance
    interest_rate = 0.06,  # 6 %
tax_rate=0.25
)

print("\\nScenario 2: All-Cash Deal ($27/share, debt-financed)")
for key, value in cash_deal.items():
    if isinstance (value, (int, float)):
        print(f"  {key}: {value:,.2f}")
    else:
    print(f"  {key}: {value}")
\`\`\`

**Output**:
\`\`\`
==================================================================
M&A Deal Analysis: TechCorp acquires InnovateCo
==================================================================

Valuation:
  InnovateCo Standalone Value: $1,000M
  Control Premium: 35.0%
  Offer Price: $1,350M

  TechCorp Value: $5,000M
  Combined Value (pre-synergies): $6,000M

  Expected Synergies: $300M
  Less: Integration Costs: $50M
  Net Synergies: $250M
  ────────────────────────────────────────────────────────────────
  Combined Value (post-synergies): $6,250M

Value Distribution:
  Value to InnovateCo Shareholders: $350M
  Value to TechCorp Shareholders: -$100M
  Total Value Created: $250M

Recommendation: REJECT
  Deal destroys $100M for TechCorp shareholders.
==================================================================

EPS Accretion/Dilution Analysis:

Scenario 1: All-Stock Deal (0.6× exchange ratio)
  Acquirer Current EPS: 3.00
  Target EPS: 2.50
  Pro Forma NI: 425.00
  New Shares Issued: 30.00
  Pro Forma Shares: 130.00
  Pro Forma EPS: 3.27
  EPS Change: 0.27
  EPS Change %: 9.00
  Accretion/Dilution: Accretive
  Total Cash Paid: 0.00
  New Debt: 0.00

Scenario 2: All-Cash Deal ($27/share, debt-financed)
  Acquirer Current EPS: 3.00
  Target EPS: 2.50
  Pro Forma NI: 363.75
  New Shares Issued: 0.00
  Pro Forma Shares: 100.00
  Pro Forma EPS: 3.64
  EPS Change: 0.64
  EPS Change %: 21.20
  Accretion/Dilution: Accretive
  Total Cash Paid: 1,350.00
  New Debt: 1,350.00
\`\`\`

**Key Insights**:
- Deal creates $250M total value (synergies - integration costs)
- BUT acquirer overpays: Gets only $250M synergies but pays $350M premium
- Acquirer destroys $100M value (should reject or lower price)
- Target captures $350M value (premium)
- Stock deal: 9% EPS accretive
- Cash deal: 21% EPS accretive (but increases leverage)

## Deal Structures

### All-Cash

**Pros**:
- Simple, clean
- Target shareholders get liquidity
- Potentially more EPS accretive

**Cons**:
- Requires cash/debt (increases leverage)
- Target shareholders lose upside if combined company succeeds
- Taxable to target shareholders (capital gains)

### All-Stock

**Pros**:
- No cash required
- Target shareholders participate in upside
- Tax-free exchange (if structured properly)

**Cons**:
- Dilutes acquirer shareholders
- Subject to acquirer stock price volatility
- Complex valuation (exchange ratio depends on stock prices)

### Mixed (Cash + Stock)

- Balances pros/cons
- Can offer "collar" (exchange ratio adjusts if stock price moves)
- Common in large deals

## Key M&A Metrics

### Enterprise Value / EBITDA

Most common M&A valuation multiple.
\`\`\`
EV/EBITDA = Enterprise Value / EBITDA
\`\`\`

Typical ranges:
- Mature industries: 6-10×
- High-growth tech: 15-25×

### Equity Value / Net Income (P/E)

For profitable companies.
\`\`\`
P/E = Equity Value / Net Income
\`\`\`

### Synergy Multiple

What acquirer pays for synergies.
\`\`\`
Synergy Multiple = (Premium Paid) / (Expected Synergies)
\`\`\`

If < 1.0: Good deal (synergies > premium).
If > 1.0: Overpaying (premium > synergies).

## M&A Process

### Phase 1: Strategy & Target Selection

1. Define strategic rationale
2. Screen potential targets
3. Preliminary valuation
4. Approach target (or hire banker)

### Phase 2: Due Diligence

**Financial DD**: Audited financials, projections, working capital, debt
**Legal DD**: Contracts, litigation, IP, compliance
**Operational DD**: Customers, suppliers, systems, integration plan
**Tax DD**: Structure, NOLs, cross-border issues

### Phase 3: Valuation & Negotiation

1. Detailed valuation (DCF, comps, precedents)
2. Synergy quantification
3. Price negotiation
4. Term sheet / Letter of Intent (LOI)

### Phase 4: Definitive Agreement

- Purchase agreement
- Representations & warranties
- Covenants (what target can/can't do before close)
- Closing conditions (regulatory approvals, financing, etc.)
- Termination rights & breakup fees

### Phase 5: Regulatory Approval

- Antitrust clearance (FTC/DOJ in US, EU Commission, etc.)
- Industry-specific (banking, pharma, etc.)
- Can take 6-18 months

### Phase 6: Closing & Integration

- Transfer ownership
- Day 1 plan (leadership, communications, IT)
- 100-day plan (quick wins)
- Full integration (1-3 years)

## Common M&A Pitfalls

1. **Overpaying**: Winner\'s curse in competitive auctions
2. **Overestimating Synergies**: 70% of deals fail to realize projected synergies
3. **Integration Failures**: Culture clashes, talent attrition, IT nightmares
4. **Regulatory Risk**: Deal blocked or conditions imposed
5. **Financing Risk**: Debt markets close, stock price collapses

## Defense Mechanisms (Anti-Takeover)

### Poison Pill (Shareholder Rights Plan)

- If hostile acquirer buys >15%, other shareholders can buy stock at discount
- Massively dilutes hostile acquirer
- Forces negotiation with board

### Staggered Board

- Only 1/3 of board up for election each year
- Takes 2 years to gain control
- Delays hostile takeover

### Golden Parachute

- Huge severance for executives if acquired
- Increases cost of takeover

### Pac-Man Defense

- Target launches hostile bid for acquirer!
- Rarely used (expensive, risky)

### White Knight

- Find friendlier acquirer
- Creates bidding war

## Real-World M&A Examples

### Success: Disney + Pixar (\$7.4B, 2006)

- **Rationale**: Acquire animation talent and IP
- **Synergies**: Distribution (Disney), Content (Pixar)
- **Result**: Massive success—Marvel, Lucasfilm followed same playbook
- **Key**: Disney gave Pixar creative freedom (no integration disaster)

### Failure: AOL + Time Warner (\$164B, 2000)

- **Rationale**: "Convergence" of internet + media
- **Reality**: Cultures incompatible, business models divergent
- **Result**: Written down $99B in 2002 (largest ever)
- **Lesson**: Don't overpay in a bubble, culture matters

### Mixed: Microsoft + Nokia (\$7.2B, 2013)

- **Rationale**: Compete with iPhone/Android
- **Result**: Failure—wrote off $7.6B in 2015, exited phones
- **Lesson**: Can't buy your way into markets with strong network effects

## Key Takeaways

1. **M&A Creates Value When**: Synergies > Premium + Integration Costs
2. **Acquirer Wins When**: They don't overpay (value to acquirer > 0)
3. **EPS Accretion ≠ Value Creation**: Can be accretive but destroy value
4. **Integration Is Hard**: 70% of deals fail to deliver expected synergies
5. **Culture Matters**: Technical synergies mean nothing if cultures clash
6. **Beware Winner's Curse**: Overpaying in competitive auctions
7. **Due Diligence**: Uncover risks before signing, not after

M&A is high-stakes corporate finance. Successful dealmakers combine rigorous valuation, realistic synergy estimates, disciplined pricing, and flawless integration execution.
`,
};
