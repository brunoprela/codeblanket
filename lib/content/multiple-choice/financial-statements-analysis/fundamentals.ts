export const fundamentalsMultipleChoiceQuestions = [
  {
    id: 1,
    question:
      'A company reports Net Income of $100M for the year. Its Retained Earnings increased by only $60M during the same period. Which of the following is the MOST likely explanation?',
    options: [
      'The company issued $40M in new shares',
      'The company paid $40M in dividends to shareholders',
      'The company made an accounting error',
      'The company acquired another company for $40M',
      'The company recorded $40M in depreciation expense',
    ],
    correctAnswer: 1,
    explanation: `The correct answer is B: The company paid $40M in dividends.

**Retained Earnings Reconciliation**:
\`\`\`
Beginning Retained Earnings
+ Net Income                   $100M
- Dividends Paid              ($40M)
= Ending Retained Earnings     
\`\`\`

The change in Retained Earnings = $100M - $40M = $60M

**Why other options are incorrect**:

A) **Issuing new shares** affects Common Stock and Additional Paid-in Capital, NOT Retained Earnings. The accounting entry is:
\`\`\`
Dr. Cash                    $40M
   Cr. Common Stock              $XM
   Cr. Additional Paid-in        $(40-X)M
\`\`\`

C) **Accounting error** is possible but not the "MOST likely" explanation. The $40M difference is precisely explainable by dividends.

D) **Acquiring another company** would reduce cash and increase assets (or goodwill), but doesn't directly reduce Retained Earnings. The entry would be:
\`\`\`
Dr. Assets/Goodwill        $40M
   Cr. Cash                    $40M
\`\`\`

E) **Depreciation** is already included in Net Income calculation. Depreciation reduces net income *before* it flows to Retained Earnings, so it wouldn't create a separate $40M gap.

**Key Concept**: Only two things normally change Retained Earnings:
1. **Net Income** (increases it)
2. **Dividends** (decreases it)

**Python validation**:
\`\`\`python
def validate_retained_earnings(beginning_re, net_income, dividends, ending_re):
    expected_re = beginning_re + net_income - dividends
    assert abs(expected_re - ending_re) < 1000, "RE reconciliation failed"
    return True

# This case:
# beginning_re + 100 - dividends = beginning_re + 60
# Therefore: dividends = 40
\`\`\``,
  },

  {
    id: 2,
    question:
      "You're parsing SEC filings using the XBRL API and encounter a company that reported Revenue of $500M in their 10-K but later filed an 10-K/A (amendment) showing Revenue of $480M for the same period. How should your automated system handle this?",
    options: [
      'Always use the first filing (10-K) as amendments often contain errors',
      'Use the average of both figures ($490M) to be conservative',
      'Use the amended figure ($480M) and flag the 4% restatement as significant',
      'Reject both filings and wait for a third confirmation',
      'Use the lower figure to be conservative',
    ],
    correctAnswer: 2,
    explanation: `The correct answer is C: Use the amended figure ($480M) and flag the 4% restatement as significant.

**Regulatory Framework**:

Under SEC rules, when a company files an amendment (10-K/A), it supersedes the original filing. The amended version is the **official** and **correct** version.

**Key Reasons**:

1. **Legal Requirement**: The 10-K/A explicitly states it replaces the original filing
2. **Material Restatement**: A 4% revenue reduction is material and likely indicates:
   - Revenue recognition error
   - Improper timing of revenue
   - Customer contract adjustments
   - Fraud correction (in severe cases)

**Implementation**:

\`\`\`python
class FilingTracker:
    """Track and handle amended filings."""
    
    def process_filing(self, filing):
        key = (filing['cik'], filing['form_type'].replace('/A', ''), filing['period_end'])
        
        if '/A' in filing['form_type']:
            # This is an amendment
            original = self.get_original_filing(key)
            
            if original:
                changes = self.calculate_changes(original, filing)
                
                # Flag material restatements (>2% for revenue)
                if abs(changes['revenue_pct']) > 0.02:
                    self.alert_material_restatement({
                        'cik': filing['cik'],
                        'ticker': filing['ticker'],
                        'item': 'Revenue',
                        'original': original['revenue'],
                        'amended': filing['revenue'],
                        'change_pct': changes['revenue_pct'],
                        'severity': 'HIGH' if abs(changes['revenue_pct']) > 0.05 else 'MEDIUM'
                    })
                
                # Replace original with amended version
                self.database.replace_filing(key, filing)
            
            return filing  # Use amended version
        
        return filing  # Use original version
    
    def calculate_changes(self, original, amended):
        """Calculate changes between filings."""
        return {
            'revenue_pct': (amended['revenue'] - original['revenue']) / original['revenue'],
            'net_income_pct': (amended['net_income'] - original['net_income']) / original['net_income'],
            'total_assets_pct': (amended['total_assets'] - original['total_assets']) / original['total_assets']
        }
\`\`\`

**Why other options are incorrect**:

A) **Using the first filing** ignores the legal reality that amendments supersede originals. This would use **wrong data**.

B) **Averaging** is statistically nonsensical when one value is incorrect and one is correct. Only the amended value is valid.

D) **Rejecting both** leaves you with no data. Better to use the corrected (amended) data with appropriate flags.

E) **Using the lower figure** isn't based on which is correct, just conservatism. What if the amendment *increased* revenue (fixing an understatement)?

**Real-World Example**: 

In 2018, Tesla filed an amended 10-Q after discovering errors in warranty reserves. Automated systems that didn't catch the amendment used incorrect data for months.

**Trading Implications**:

A 4% revenue miss typically causes:
- 5-10% stock price decline (depending on margin impact)
- Analyst downgrades
- Questions about internal controls
- Potential SEC investigation if pattern emerges

**Best Practice**:
\`\`\`python
# Always check for amendments before using data
def get_latest_filing(cik, form_type, period_end):
    """Get most recent version (original or amended)."""
    
    filings = query_filings(cik, form_type, period_end)
    
    # Sort by filed_date (latest first)
    filings.sort(key=lambda x: x['filed_date'], reverse=True)
    
    # Return most recent (could be 10-K/A)
    return filings[0]
\`\`\`

This is why using the SEC's XBRL API is better than web scraping—the API always provides the most current data.`,
  },

  {
    id: 3,
    question:
      'When building an integrated 3-statement financial model, which of the following statements is TRUE about the relationship between the statements?',
    options: [
      'Cash Flow from Operations on the Cash Flow Statement should always equal Net Income on the Income Statement',
      'Capital Expenditures flows from the Cash Flow Statement to increase PP&E on the Balance Sheet by the same amount',
      "Depreciation expense on the Income Statement is added back in the Cash Flow Statement because it's a non-cash expense",
      'Total Liabilities on the Balance Sheet must increase by the same amount as Interest Expense on the Income Statement',
      'Dividends paid appear as an expense on the Income Statement before flowing to the Cash Flow Statement',
    ],
    correctAnswer: 2,
    explanation: `The correct answer is C: Depreciation expense on the Income Statement is added back in the Cash Flow Statement because it's a non-cash expense.

**Why this is correct**:

Depreciation is an accounting expense that reduces Net Income but doesn't involve any cash payment. Therefore:

1. **Income Statement**: Depreciation reduces Net Income
2. **Cash Flow Statement**: Depreciation is added back to Net Income to get Operating Cash Flow

\`\`\`python
# Income Statement
revenue = 1_000_000
cash_expenses = 600_000
depreciation = 100_000  # Non-cash expense
net_income = revenue - cash_expenses - depreciation  # $300,000

# Cash Flow Statement (Indirect Method)
operating_cf = (
    net_income +           # $300,000
    depreciation          # $100,000 (add back non-cash charge)
)  # = $400,000

# Actual cash generated is $400K, not $300K
\`\`\`

**Why other options are incorrect**:

**A) OCF should always equal Net Income - FALSE**

OCF almost never equals Net Income due to:
- Non-cash expenses (depreciation, amortization, stock-based comp)
- Working capital changes
- Accrual vs cash timing differences

\`\`\`python
# Example showing difference:
net_income = 100_000
depreciation = 20_000
increase_in_receivables = 30_000  # Cash not yet collected

operating_cf = net_income + depreciation - increase_in_receivables
# = 100,000 + 20,000 - 30,000 = 90,000

# OCF ($90K) ≠ Net Income ($100K)
\`\`\`

**B) CapEx increases PP&E by the same amount - FALSE**

CapEx (from Cash Flow Statement) increases **gross PP&E**, but **net PP&E** (on Balance Sheet) also decreases due to depreciation:

\`\`\`python
# Balance Sheet PP&E movement:
beginning_ppe_net = 500_000
capex = 100_000              # From Cash Flow Statement
depreciation = 80_000        # From Income Statement

ending_ppe_net = beginning_ppe_net + capex - depreciation
# = 500,000 + 100,000 - 80,000 = 520,000

# Net PP&E only increased by $20K, not $100K!
\`\`\`

**D) Liabilities increase by Interest Expense - FALSE**

Interest Expense on the Income Statement represents interest **paid** (or accrued) during the period. It:
- Reduces Net Income
- May increase Interest Payable (if accrued but not paid)
- But Total Liabilities are affected by many other factors

\`\`\`python
# Interest Expense does not dictate total liability change
interest_expense = 10_000  # From Income Statement

# Meanwhile, company could:
# - Repay $1M in debt (reduces liabilities)
# - Issue $2M in new bonds (increases liabilities)
# - Accrue $50K in payables (increases liabilities)

# Total liability change ≠ Interest Expense
\`\`\`

**E) Dividends appear on Income Statement - FALSE**

Dividends are a **distribution of earnings**, not an expense. They:
- Do NOT appear on Income Statement
- Do appear on Cash Flow Statement (Financing Activities)
- Do reduce Retained Earnings on Balance Sheet

\`\`\`python
# Correct treatment of dividends:

# Income Statement - NOT HERE

# Cash Flow Statement:
financing_cf = -dividends_paid  # Negative cash flow

# Balance Sheet:
retained_earnings = (
    beginning_retained_earnings +
    net_income -
    dividends_paid  # Reduces equity
)
\`\`\`

**Complete Example of Statement Integration**:

\`\`\`python
class IntegratedFinancialModel:
    def link_statements(self):
        # Income Statement
        self.net_income = self.revenue - self.total_expenses
        
        # Cash Flow Statement
        self.operating_cf = (
            self.net_income +
            self.depreciation +  # Add back non-cash expense (KEY CONCEPT)
            self.change_in_working_capital
        )
        
        self.investing_cf = -self.capex  # Negative = cash outflow
        
        self.financing_cf = (
            self.debt_issued -
            self.debt_repaid -
            self.dividends_paid  # Not an expense!
        )
        
        self.net_change_cash = (
            self.operating_cf +
            self.investing_cf +
            self.financing_cf
        )
        
        # Balance Sheet
        self.ending_cash = self.beginning_cash + self.net_change_cash
        
        self.ending_ppe = (
            self.beginning_ppe +
            self.capex -
            self.depreciation  # CapEx adds, depreciation reduces
        )
        
        self.ending_retained_earnings = (
            self.beginning_retained_earnings +
            self.net_income -
            self.dividends_paid  # Reduces equity, not expense
        )
        
        # Validate balance sheet balances
        assert self.total_assets == (
            self.total_liabilities +
            self.shareholders_equity
        )
\`\`\`

**Key Takeaway**: Depreciation's add-back in the Cash Flow Statement is one of the most fundamental linkages in financial modeling and is essential for understanding the difference between accounting earnings and actual cash generation.`,
  },

  {
    id: 4,
    question:
      "You're analyzing a European company that reports under IFRS and a U.S. company that reports under GAAP. The European company shows PP&E of $1B after a revaluation (mark-to-market) adjustment that increased it by $200M from historical cost. To compare the two companies fairly, you should:",
    options: [
      "Add $200M to the U.S. company's PP&E to put them on equal footing",
      "Subtract $200M from the European company's PP&E to use historical cost for both",
      'Make no adjustment since both are following their respective accounting standards correctly',
      "Add $200M to the European company's Revenue to reflect the economic gain",
      'Use PP&E/Sales ratio instead since the revaluation affects both companies equally',
    ],
    correctAnswer: 1,
    explanation: `The correct answer is B: Subtract $200M from the European company's PP&E to use historical cost for both.

**The Core Issue**:

- **IFRS**: Allows companies to revalue PP&E to fair market value (upward or downward)
- **U.S. GAAP**: Requires historical cost (original purchase price) minus accumulated depreciation
- **Problem**: This makes balance sheets non-comparable

**Why Historical Cost Normalization is Correct**:

\`\`\`python
def normalize_ppe_for_comparison(company_a_ifrs, company_b_gaap):
    """Normalize PP&E to historical cost for fair comparison."""
    
    # European company (IFRS)
    ppe_reported_a = 1_000_000_000  # $1B after revaluation
    revaluation_surplus = 200_000_000  # $200M gain
    ppe_historical_cost_a = ppe_reported_a - revaluation_surplus
    # = $800M (comparable to GAAP)
    
    # U.S. company (GAAP)
    ppe_b = 800_000_000  # Already at historical cost
    
    # Now comparable:
    # Company A: $800M (adjusted)
    # Company B: $800M
    # Can now calculate meaningful ratios!
    
    return {
        'company_a_normalized_ppe': ppe_historical_cost_a,
        'company_b_ppe': ppe_b,
        'comparable': True
    }
\`\`\`

**Impact on Financial Ratios**:

Before adjustment:
\`\`\`python
# Company A (IFRS, revalued)
ppe_a = 1_000_000_000
total_assets_a = 2_000_000_000
sales_a = 1_500_000_000

asset_turnover_a = sales_a / total_assets_a  # 0.75x

# Company B (GAAP, historical cost)
ppe_b = 800_000_000
total_assets_b = 1_800_000_000
sales_b = 1_500_000_000

asset_turnover_b = sales_b / total_assets_b  # 0.83x

# Company B looks more efficient, but is it real or accounting?
\`\`\`

After adjustment:
\`\`\`python
# Adjust Company A to historical cost
adjusted_ppe_a = 800_000_000  # Remove $200M revaluation
adjusted_assets_a = 1_800_000_000

asset_turnover_a_adjusted = sales_a / adjusted_assets_a  # 0.83x

# Now directly comparable!
# Both companies have same efficiency
\`\`\`

**Why Other Options are Incorrect**:

**A) Add $200M to U.S. company's PP&E - WRONG**

This would artificially inflate the U.S. company's assets to match an accounting choice, not economic reality. You'd be adding value that doesn't exist in their accounting system.

**C) Make no adjustment - WRONG**

While both are following their standards "correctly," that doesn't make them comparable. The whole point of normalization is to adjust for accounting differences.

\`\`\`python
# Example of why "both following rules" isn't enough:
company_a_roe = net_income / equity_with_revaluation  # Lower ROE
company_b_roe = net_income / equity_without_revaluation  # Higher ROE

# Same economic performance, different accounting → not comparable!
\`\`\`

**D) Add $200M to European company's Revenue - WRONG**

Revaluation gains:
- Do NOT go through the income statement (under IFRS)
- Go to "Revaluation Surplus" in equity (part of Other Comprehensive Income)
- Are not revenue and shouldn't be treated as such

\`\`\`python
# Correct accounting for revaluation:
# Dr. PP&E                   $200M
#    Cr. Revaluation Surplus     $200M (in Equity, not Revenue!)
\`\`\`

**E) Use PP&E/Sales ratio instead - WRONG**

The revaluation affects PP&E, not sales, so the ratio would still be distorted:

\`\`\`python
# Company A (IFRS)
ppe_sales_ratio_a = 1_000_000_000 / 1_500_000_000  # 0.67

# Company B (GAAP)
ppe_sales_ratio_b = 800_000_000 / 1_500_000_000  # 0.53

# Company A looks more capital-intensive, but it's just accounting!
\`\`\`

**Production Implementation**:

\`\`\`python
class IFRStoGAAPNormalizer:
    """Normalize IFRS statements to GAAP basis for comparison."""
    
    def normalize_ppe(self, statements: dict) -> dict:
        """Remove revaluation effects."""
        
        if statements['accounting_standard'] != 'IFRS':
            return statements
        
        # Find revaluation surplus in equity
        revaluation_surplus = statements['equity']['revaluation_surplus']
        
        if revaluation_surplus == 0:
            return statements  # No revaluation
        
        # Adjust PP&E to historical cost
        statements['balance_sheet']['ppe'] -= revaluation_surplus
        statements['balance_sheet']['total_assets'] -= revaluation_surplus
        
        # Adjust equity
        statements['balance_sheet']['equity'] -= revaluation_surplus
        
        # Mark as normalized
        statements['_metadata']['ppe_normalized'] = True
        statements['_metadata']['revaluation_removed'] = revaluation_surplus
        
        # Recalculate affected ratios
        statements['ratios'] = self.recalculate_ratios(statements)
        
        return statements
    
    def recalculate_ratios(self, statements: dict) -> dict:
        """Recalculate ratios with normalized figures."""
        bs = statements['balance_sheet']
        inc = statements['income_statement']
        
        return {
            'asset_turnover': inc['revenue'] / bs['total_assets'],
            'roa': inc['net_income'] / bs['total_assets'],
            'roe': inc['net_income'] / bs['equity'],
            'debt_to_equity': bs['total_debt'] / bs['equity']
        }
\`\`\`

**Real-World Example**:

Many European real estate companies use fair value accounting for properties, leading to:
- Higher asset values than U.S. peers
- Lower apparent ROAs and ROEs
- Valuation differences (P/B ratios)

\`\`\`python
# European REIT (IFRS, fair value)
property_value = 5_000_000_000  # Fair value
net_income = 300_000_000
roe = 300_000_000 / 5_000_000_000  # 6%

# U.S. REIT (GAAP, historical cost)
property_value = 3_000_000_000  # Historical cost (bought years ago)
net_income = 300_000_000
roe = 300_000_000 / 3_000_000_000  # 10%

# Adjust European REIT to historical cost basis to compare fairly
\`\`\`

**Key Principle**: When comparing across accounting standards, normalize to the **most conservative basis** (usually historical cost) to ensure you're comparing like with like.`,
  },

  {
    id: 5,
    question:
      "Your automated 10-K parser successfully extracts Apple's Revenue as $394.3B from their FY2022 filing. However, your validation system flags an error because the Balance Sheet totals don't match (Assets = $352.8B vs Liabilities + Equity = $352.5B, a $300M difference). What is the MOST likely cause and best approach?",
    options: [
      'Apple made a material accounting error and you should short the stock immediately',
      'Your parser has a bug and is missing some line items from the Balance Sheet',
      'The difference is due to rounding across thousands of line items and is immaterial (~0.08%)',
      'You should re-download the filing as the file was corrupted during transmission',
      "The SEC's XBRL data is incorrect and you should report it to the agency",
    ],
    correctAnswer: 2,
    explanation: `The correct answer is C: The difference is due to rounding across thousands of line items and is immaterial (~0.08%).

**Why This Is The Right Answer**:

\`\`\`python
# Apple's actual figures (FY2022):
assets = 352_755_000_000  # $352.8B (rounded to millions)
liabilities = 302_083_000_000  # $302.1B
equity = 50_672_000_000  # $50.7B

# Check if balanced:
liabilities_plus_equity = liabilities + equity
# = 352,755,000,000

# Perfect balance! The $300M discrepancy described in the question
# is due to rounding in reporting or parsing.

# Materiality check:
discrepancy = 300_000_000  # $300M
percentage = discrepancy / assets
# = 0.085% = 0.00085

# This is WELL below materiality threshold (typically 1-5%)
\`\`\`

**Understanding Materiality**:

\`\`\`python
def assess_materiality(discrepancy: float, total: float) -> dict:
    """Determine if a discrepancy is material."""
    
    percentage = abs(discrepancy / total)
    
    if percentage < 0.001:  # 0.1%
        severity = 'IGNORE'
        action = 'Accept - likely rounding'
    elif percentage < 0.01:  # 1%
        severity = 'LOW'
        action = 'Flag for review but likely OK'
    elif percentage < 0.05:  # 5%
        severity = 'MEDIUM'
        action = 'Investigate - possible parser error'
    else:
        severity = 'HIGH'
        action = 'Critical - likely parser bug or data error'
    
    return {
        'percentage': percentage,
        'severity': severity,
        'action': action
    }

# Apply to Apple case:
result = assess_materiality(300_000_000, 352_755_000_000)
# {'percentage': 0.00085, 'severity': 'IGNORE', 'action': 'Accept - likely rounding'}
\`\`\`

**Sources of Rounding Differences**:

1. **Presentation Rounding**: Financial statements shown in millions or billions
2. **Thousands of Line Items**: Each rounded individually
3. **Consolidation Adjustments**: Intercompany eliminations, FX translation
4. **Parsing Precision**: Different fields may have different decimal places

\`\`\`python
# Example of how rounding accumulates:
line_items = [
    1_234_567_890,  # $1,234,567,890 actual
    5_678_123_456,
    9_876_543_210
]

# Rounded to millions for presentation:
line_items_millions = [
    1_235,  # $1,235M (rounded)
    5_678,
    9_877
]

# Sum of rounded:
sum_rounded = sum(line_items_millions) * 1_000_000
# = 16,790,000,000

# Actual sum:
sum_actual = sum(line_items)
# = 16,789,234,556

# Difference: $765,444 from just 3 line items!
# With 100s of line items, $300M difference is reasonable
\`\`\`

**Why Other Options Are Incorrect**:

**A) Apple made an error / short the stock - WRONG**

0.08% is far too small to indicate an accounting error. Public companies have:
- External auditors (Deloitte, PwC, etc.)
- Internal controls
- SEC review

Material errors are typically 5%+ and trigger restatements.

\`\`\`python
# Real accounting errors look like:
revenue_as_reported = 10_000_000_000
revenue_restated = 8_000_000_000
error = 20%  # THIS is material

# Not:
assets = 352_755_000_000
error = 0.08%  # This is rounding
\`\`\`

**B) Parser has a bug and is missing line items - POSSIBLE BUT UNLIKELY**

If you were missing entire line items, you'd see:
- Much larger discrepancy (>1%)
- Obvious missing categories
- Consistent patterns across companies

\`\`\`python
def validate_parser_completeness(statements: dict) -> bool:
    """Check if major line items are present."""
    
    required_items = [
        'cash',
        'receivables',
        'inventory',
        'ppe',
        'intangibles',
        'current_liabilities',
        'long_term_debt',
        'common_stock',
        'retained_earnings'
    ]
    
    missing = [item for item in required_items if item not in statements]
    
    if missing:
        print(f"Parser BUG: Missing {missing}")
        return False
    
    return True

# If parser validation passes but still see small discrepancy → rounding
\`\`\`

**D) Re-download due to corruption - WRONG**

File corruption would cause:
- Parse failures (malformed XML/HTML)
- Missing entire sections
- Gibberish text

Not a precisely calculated $300M difference (0.08%).

**E) SEC's XBRL data is incorrect - EXTREMELY UNLIKELY**

The SEC's XBRL data is the official record, validated by:
- Company's CFO/CEO certification
- External auditors
- SEC staff review

If there *were* an error, it would be corrected via amendment (10-K/A).

**Production Implementation**:

\`\`\`python
class BalanceSheetValidator:
    """Validate balance sheet with materiality thresholds."""
    
    ROUNDING_TOLERANCE = 0.001  # 0.1% - accept as rounding
    INVESTIGATION_THRESHOLD = 0.01  # 1% - investigate
    ERROR_THRESHOLD = 0.05  # 5% - likely error
    
    def validate(self, balance_sheet: dict) -> dict:
        """Validate balance sheet equation with materiality analysis."""
        
        assets = balance_sheet['total_assets']
        liabilities = balance_sheet['total_liabilities']
        equity = balance_sheet['shareholders_equity']
        
        expected = liabilities + equity
        difference = assets - expected
        percentage = abs(difference / assets)
        
        result = {
            'balanced': False,
            'difference': difference,
            'difference_pct': percentage,
            'severity': None,
            'action': None
        }
        
        if percentage < self.ROUNDING_TOLERANCE:
            result['balanced'] = True
            result['severity'] = 'NONE'
            result['action'] = 'Accept - immaterial rounding'
        elif percentage < self.INVESTIGATION_THRESHOLD:
            result['balanced'] = False
            result['severity'] = 'LOW'
            result['action'] = 'Flag for review'
        elif percentage < self.ERROR_THRESHOLD:
            result['balanced'] = False
            result['severity'] = 'MEDIUM'
            result['action'] = 'Investigate parser and data'
        else:
            result['balanced'] = False
            result['severity'] = 'HIGH'
            result['action'] = 'Critical error - halt processing'
        
        return result

# Usage:
validator = BalanceSheetValidator()
result = validator.validate({
    'total_assets': 352_755_000_000,
    'total_liabilities': 302_083_000_000,
    'shareholders_equity': 50_372_000_000  # $300M less than perfect
})

print(result)
# {'balanced': True, 'difference': 300000000, 'difference_pct': 0.00085,
#  'severity': 'NONE', 'action': 'Accept - immaterial rounding'}
\`\`\`

**Real-World Example**:

Even Warren Buffett's Berkshire Hathaway has tiny rounding differences in their filings. It's universal and expected.

\`\`\`python
# Typical tolerances used by institutional investors:
TOLERANCES = {
    'balance_sheet_equation': 0.1%,  # Assets = Liabilities + Equity
    'retained_earnings_rollforward': 0.5%,  # RE movement
    'cash_reconciliation': 0.1%,  # Cash flow to balance sheet
    'eps_calculation': 0.01,  # $0.01 per share
}
\`\`\`

**Key Principle**: Always build materiality thresholds into your validation logic. Perfect precision is impossible with rounded figures across thousands of accounts. Focus on material discrepancies (>1%) that indicate real problems.`,
  },
];
