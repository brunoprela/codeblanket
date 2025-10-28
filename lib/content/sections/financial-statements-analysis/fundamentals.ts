export const fundamentals = {
  title: 'Financial Statements Fundamentals',
  slug: 'fundamentals',
  description:
    'Master reading and analyzing the three core financial statements programmatically',
  content: `
# Financial Statements Fundamentals

## Overview

Financial statements are the **language of business**—the primary way companies communicate their financial performance and position to investors, creditors, regulators, and other stakeholders. For engineers entering finance, mastering financial statements is non-negotiable. You'll parse them automatically, build models from them, detect fraud in them, and use them to drive trading decisions.

This section covers the three core financial statements, how they interconnect, and how to read SEC filings programmatically.

---

## Why This Matters for Engineers

Unlike traditional finance students who learn to read statements manually, **you'll build systems that analyze thousands of companies automatically**:

- **Automated screening**: Filter 5,000+ public companies by financial health
- **Real-time alerts**: Detect material changes in filings
- **Fraud detection**: Identify accounting irregularities at scale
- **Trading signals**: Generate buy/sell signals from financial metrics
- **Risk assessment**: Evaluate credit risk across portfolios

**Real-world example**: Renaissance Technologies uses automated financial statement analysis to identify mispriced securities across global markets.

---

## The Three Financial Statements

### 1. **Income Statement (P&L - Profit & Loss)**

Shows **profitability** over a period (quarter or year).

**Key question**: How much money did the company make or lose?

**Structure**:
\`\`\`
Revenue (Sales)
- Cost of Goods Sold (COGS)
= Gross Profit
- Operating Expenses (SG&A, R&D)
= Operating Income (EBIT)
- Interest Expense
- Taxes
= Net Income
\`\`\`

**Analogy**: Like your personal income statement—salary minus expenses equals what you save (or lose).

### 2. **Balance Sheet**

Shows **financial position** at a point in time (snapshot).

**Key question**: What does the company own and owe?

**Structure**:
\`\`\`
ASSETS = LIABILITIES + SHAREHOLDERS' EQUITY

Assets (what company owns):
  - Current Assets (cash, receivables, inventory)
  - Long-term Assets (PP&E, intangibles)

Liabilities (what company owes):
  - Current Liabilities (payables, short-term debt)
  - Long-term Liabilities (bonds, long-term debt)

Shareholders' Equity (residual ownership):
  - Common stock
  - Retained earnings
\`\`\`

**Analogy**: Like your personal balance sheet—home and car (assets) minus mortgage and loans (liabilities) equals your net worth (equity).

### 3. **Cash Flow Statement**

Shows **cash movements** over a period.

**Key question**: Where did cash come from and where did it go?

**Structure**:
\`\`\`
Operating Cash Flow (from business operations)
+ Investing Cash Flow (CapEx, acquisitions)
+ Financing Cash Flow (debt, dividends, buybacks)
= Change in Cash
\`\`\`

**Analogy**: Like your bank account—money in from salary, money out for rent and investments.

---

## How the Statements Connect

This is **critical** for building integrated financial models:

### Connection 1: Net Income → Retained Earnings
\`\`\`
Net Income (from Income Statement)
→ flows to Retained Earnings (on Balance Sheet)
\`\`\`

**Python representation**:
\`\`\`python
# End of period
balance_sheet['retained_earnings_end'] = (
    balance_sheet['retained_earnings_start'] +
    income_statement['net_income'] -
    cash_flow['dividends_paid']
)
\`\`\`

### Connection 2: Cash Flow → Cash Balance
\`\`\`
Change in Cash (from Cash Flow Statement)
→ changes Cash (on Balance Sheet)
\`\`\`

**Python representation**:
\`\`\`python
balance_sheet['cash_end'] = (
    balance_sheet['cash_start'] +
    cash_flow['total_cash_flow']
)
\`\`\`

### Connection 3: CapEx → PP&E
\`\`\`
Capital Expenditures (from Cash Flow Statement)
- Depreciation (from Income Statement)
→ changes PP&E (on Balance Sheet)
\`\`\`

**Python representation**:
\`\`\`python
balance_sheet['ppe_end'] = (
    balance_sheet['ppe_start'] +
    cash_flow['capex'] -
    income_statement['depreciation']
)
\`\`\`

### The Full Picture

\`\`\`python
class FinancialStatements:
    """Integrated financial statements with automatic linking."""
    
    def __init__(self, period: str):
        self.period = period
        self.income_statement = {}
        self.balance_sheet = {}
        self.cash_flow = {}
    
    def link_statements (self):
        """Ensure statements are properly linked."""
        # Net income flows to retained earnings
        net_income = self.income_statement['net_income']
        dividends = self.cash_flow['dividends_paid']
        
        self.balance_sheet['retained_earnings'] += (net_income - dividends)
        
        # Cash flow reconciles to cash
        total_cf = (
            self.cash_flow['operating_cf'] +
            self.cash_flow['investing_cf'] +
            self.cash_flow['financing_cf']
        )
        self.balance_sheet['cash'] += total_cf
        
        # Balance sheet must balance
        assert self.balance_sheet['total_assets'] == (
            self.balance_sheet['total_liabilities'] +
            self.balance_sheet['shareholders_equity']
        ), "Balance sheet doesn't balance!"
        
        return True
\`\`\`

---

## Accrual vs Cash Accounting

**Critical concept**: The income statement uses **accrual accounting**, while the cash flow statement shows **actual cash**.

### Accrual Accounting

Revenue recognized when **earned**, not when cash received:
- Sell product on credit → Record revenue now, receive cash later
- Customer prepays → Record revenue later (when delivered), receive cash now

Expenses recognized when **incurred**, not when cash paid:
- Use electricity in December → Record expense in December, pay bill in January
- Pay insurance for year → Allocate expense monthly

### Cash Accounting

Records only when **cash changes hands**.

### Why This Matters

**A profitable company can go bankrupt** if it runs out of cash!

\`\`\`python
# Company showing profit but burning cash
income_statement = {
    'revenue': 1_000_000,        # Recorded sales (some on credit)
    'expenses': 800_000,
    'net_income': 200_000        # Profitable!
}

cash_flow = {
    'operating_cf': -100_000     # But customers haven't paid yet!
}

# Company is profitable but has negative cash flow
# This is a red flag - potentially unsustainable
\`\`\`

**Real-world example**: WeWork reported positive EBITDA but massive negative cash flow before its failed IPO.

---

## GAAP vs IFRS

Two major accounting standards:

### GAAP (Generally Accepted Accounting Principles)
- Used in **United States**
- Rules-based (very specific)
- SEC requires for public companies

### IFRS (International Financial Reporting Standards)
- Used in **120+ countries** (Europe, Asia, etc.)
- Principles-based (more flexibility)

### Key Differences for Engineers

\`\`\`python
def adjust_for_accounting_standard (statement: dict, standard: str):
    """Adjust financials for different accounting standards."""
    
    if standard == 'IFRS':
        # IFRS: LIFO inventory not allowed
        if statement.get('inventory_method') == 'LIFO':
            statement['inventory'] = convert_lifo_to_fifo (statement)
    
    elif standard == 'GAAP':
        # GAAP: Development costs usually expensed
        if statement.get('development_costs_capitalized'):
            statement['intangible_assets'] -= statement['development_costs']
            statement['r&d_expense'] += statement['development_costs']
    
    return statement
\`\`\`

**For automated analysis**: Always check which standard a company uses before comparing financials!

---

## Reading SEC Filings

### SEC EDGAR System

The **Electronic Data Gathering, Analysis, and Retrieval** system—where all public companies file reports.

**Access**: Free at [https://www.sec.gov/edgar](https://www.sec.gov/edgar)

### Key Filing Types

| Filing | Description | Frequency | Key Content |
|--------|-------------|-----------|-------------|
| **10-K** | Annual report | Yearly | Complete financials, full detail |
| **10-Q** | Quarterly report | Quarterly | Quarterly financials, less detail |
| **8-K** | Current events | As needed | Material events (M&A, CEO change) |
| **DEF 14A** | Proxy statement | Yearly | Executive compensation, voting |
| **S-1** | IPO registration | One-time | Pre-IPO financials |

### 10-K Structure

\`\`\`
PART I
  Item 1: Business (what company does)
  Item 1A: Risk Factors (what could go wrong)
  Item 3: Legal Proceedings

PART II
  Item 6: Selected Financial Data
  Item 7: MD&A (Management Discussion & Analysis)
  Item 8: Financial Statements & Notes ← **Most important**
  Item 9A: Controls and Procedures

PART III
  Item 10: Directors and Officers
  Item 11: Executive Compensation

PART IV
  Item 15: Exhibits & Schedules
\`\`\`

---

## Parsing SEC Filings with Python

### Method 1: Using sec-edgar-downloader

\`\`\`python
from sec_edgar_downloader import Downloader

# Initialize downloader
dl = Downloader("MyCompany", "my.email@company.com")

# Download all 10-K filings for Apple
dl.get("10-K", "AAPL", after="2020-01-01", before="2024-01-01")

# Files saved to: sec-edgar-filings/AAPL/10-K/
\`\`\`

### Method 2: Direct API Access

\`\`\`python
import requests
from bs4 import BeautifulSoup
import pandas as pd

class SECFilingParser:
    """Parse SEC filings automatically."""
    
    BASE_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
    
    def __init__(self, user_agent: str):
        self.headers = {'User-Agent': user_agent}
    
    def get_company_cik (self, ticker: str) -> str:
        """Get CIK number from ticker."""
        url = f"https://www.sec.gov/cgi-bin/browse-edgar"
        params = {
            'action': 'getcompany',
            'company': ticker,
            'type': '10-K',
            'count': '1',
            'output': 'xml'
        }
        
        response = requests.get (url, params=params, headers=self.headers)
        soup = BeautifulSoup (response.content, 'xml')
        
        return soup.find('CIK').text.zfill(10)
    
    def get_latest_10k (self, ticker: str) -> dict:
        """Download and parse latest 10-K."""
        cik = self.get_company_cik (ticker)
        
        # Get filing list
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        response = requests.get (url, headers=self.headers)
        data = response.json()
        
        # Find most recent 10-K
        filings = data['filings']['recent']
        for i, form in enumerate (filings['form']):
            if form == '10-K':
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{filings['accessionNumber'][i].replace('-', '')}/{filings['primaryDocument'][i]}"
                break
        
        # Parse filing
        return self.parse_10k (filing_url)
    
    def parse_10k (self, url: str) -> dict:
        """Extract financial data from 10-K HTML."""
        response = requests.get (url, headers=self.headers)
        soup = BeautifulSoup (response.content, 'html.parser')
        
        # Find financial statement tables
        tables = soup.find_all('table')
        
        financials = {
            'income_statement': None,
            'balance_sheet': None,
            'cash_flow': None
        }
        
        # Parse each table
        for table in tables:
            # Identify statement type by headers
            text = table.get_text().lower()
            
            if 'income' in text or 'operations' in text:
                financials['income_statement'] = self._parse_table (table)
            elif 'balance sheet' in text or 'financial position' in text:
                financials['balance_sheet'] = self._parse_table (table)
            elif 'cash flow' in text:
                financials['cash_flow'] = self._parse_table (table)
        
        return financials
    
    def _parse_table (self, table) -> pd.DataFrame:
        """Convert HTML table to pandas DataFrame."""
        rows = []
        for tr in table.find_all('tr'):
            cells = [cell.get_text (strip=True) for cell in tr.find_all(['td', 'th'])]
            if cells:
                rows.append (cells)
        
        # Create DataFrame
        if len (rows) > 1:
            df = pd.DataFrame (rows[1:], columns=rows[0])
            return df
        return None

# Usage
parser = SECFilingParser("MyCompany myemail@company.com")
apple_financials = parser.get_latest_10k("AAPL")
print(apple_financials['income_statement'].head())
\`\`\`

### Method 3: XBRL Parsing (Structured Data)

\`\`\`python
import json
import requests

def get_xbrl_facts (cik: str, user_agent: str) -> dict:
    """
    Get structured financial data via SEC's XBRL API.
    Much more reliable than HTML parsing!
    """
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"
    headers = {'User-Agent': user_agent}
    
    response = requests.get (url, headers=headers)
    return response.json()

def extract_financial_metrics (facts: dict) -> dict:
    """Extract key metrics from XBRL data."""
    
    us_gaap = facts['facts'].get('us-gaap', {})
    
    metrics = {}
    
    # Revenue
    if 'Revenues' in us_gaap:
        metrics['revenue'] = us_gaap['Revenues']['units']['USD']
    elif 'RevenueFromContractWithCustomerExcludingAssessedTax' in us_gaap:
        metrics['revenue'] = us_gaap['RevenueFromContractWithCustomerExcludingAssessedTax']['units']['USD']
    
    # Net Income
    if 'NetIncomeLoss' in us_gaap:
        metrics['net_income'] = us_gaap['NetIncomeLoss']['units']['USD']
    
    # Total Assets
    if 'Assets' in us_gaap:
        metrics['total_assets'] = us_gaap['Assets']['units']['USD']
    
    # Total Equity
    if 'StockholdersEquity' in us_gaap:
        metrics['equity'] = us_gaap['StockholdersEquity']['units']['USD']
    
    return metrics

# Usage
cik = "0000320193"  # Apple
facts = get_xbrl_facts (cik, "myemail@company.com")
metrics = extract_financial_metrics (facts)

# Get latest annual figures
for metric, values in metrics.items():
    # Filter for annual reports (10-K)
    annual = [v for v in values if v.get('form') == '10-K']
    if annual:
        latest = annual[-1]
        print(f"{metric}: \\$\{latest['val']:,.0f}(as of { latest['end'] }) ")
\`\`\`

**Output example**:
\`\`\`
revenue: $394,328,000,000 (as of 2022-09-24)
net_income: $99,803,000,000 (as of 2022-09-24)
total_assets: $352,755,000,000 (as of 2022-09-24)
equity: $50,672,000,000 (as of 2022-09-24)
\`\`\`

---

## Real-World Example: Apple\'s 10-K Walkthrough

Let's analyze Apple's actual 10-K (fiscal 2022):

### Step 1: Download the Filing

\`\`\`python
from sec_edgar_downloader import Downloader

dl = Downloader("MyCompany", "my.email@company.com")
dl.get("10-K", "AAPL", amount=1)  # Most recent only
\`\`\`

### Step 2: Extract Key Financials

\`\`\`python
def analyze_apple_10k():
    """Extract and analyze Apple\'s key financials."""
    
    # Actual numbers from Apple's FY2022 10-K
    apple_financials = {
        'income_statement': {
            'revenue': 394_328_000_000,
            'cost_of_revenue': 223_546_000_000,
            'gross_profit': 170_782_000_000,
            'operating_expenses': 51_345_000_000,
            'operating_income': 119_437_000_000,
            'interest_expense': 2_931_000_000,
            'income_tax': 19_300_000_000,
            'net_income': 99_803_000_000,
        },
        'balance_sheet': {
            'cash': 23_646_000_000,
            'receivables': 60_932_000_000,
            'inventory': 4_946_000_000,
            'current_assets': 135_405_000_000,
            'total_assets': 352_755_000_000,
            'current_liabilities': 153_982_000_000,
            'total_liabilities': 302_083_000_000,
            'shareholders_equity': 50_672_000_000,
        },
        'cash_flow': {
            'operating_cf': 122_151_000_000,
            'capex': -10_708_000_000,
            'investing_cf': -22_354_000_000,
            'financing_cf': -110_749_000_000,
            'free_cash_flow': 111_443_000_000,
        }
    }
    
    # Quick analysis
    analysis = {
        'gross_margin': (
            apple_financials['income_statement']['gross_profit'] /
            apple_financials['income_statement']['revenue']
        ),
        'operating_margin': (
            apple_financials['income_statement']['operating_income'] /
            apple_financials['income_statement']['revenue']
        ),
        'net_margin': (
            apple_financials['income_statement']['net_income'] /
            apple_financials['income_statement']['revenue']
        ),
        'roe': (
            apple_financials['income_statement']['net_income'] /
            apple_financials['balance_sheet']['shareholders_equity']
        ),
        'debt_to_equity': (
            apple_financials['balance_sheet']['total_liabilities'] /
            apple_financials['balance_sheet']['shareholders_equity']
        ),
    }
    
    print("Apple Inc. Financial Analysis (FY2022)")
    print("=" * 50)
    print(f"Revenue: \\$\{apple_financials['income_statement']['revenue']:,.0f}")
print(f"Net Income: \\$\{apple_financials['income_statement']['net_income']:,.0f}")
print(f"\\nMargins:")
print(f"  Gross Margin: {analysis['gross_margin']:.1%}")
print(f"  Operating Margin: {analysis['operating_margin']:.1%}")
print(f"  Net Margin: {analysis['net_margin']:.1%}")
print(f"\\nROE: {analysis['roe']:.1%}")
print(f"Debt-to-Equity: {analysis['debt_to_equity']:.2f}x")

return apple_financials, analysis

# Run analysis
financials, analysis = analyze_apple_10k()
\`\`\`

**Output**:
\`\`\`
Apple Inc. Financial Analysis (FY2022)
==================================================
Revenue: $394,328,000,000
Net Income: $99,803,000,000

Margins:
  Gross Margin: 43.3%
  Operating Margin: 30.3%
  Net Margin: 25.3%

ROE: 196.9%
Debt-to-Equity: 5.96x
\`\`\`

### Key Observations

1. **High margins**: 25% net margin is exceptional (compare to Walmart at ~2%)
2. **Massive FCF**: $111B free cash flow funds buybacks and dividends
3. **High leverage**: Debt-to-equity of 6x (but manageable with strong cash flow)
4. **Incredible ROE**: 197% return on equity (though inflated by buybacks reducing equity)

---

## Building Your First Parser

Let\'s build a production-ready financial statement parser:

\`\`\`python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import pandas as pd

@dataclass
class IncomeStatement:
    """Income statement data structure."""
    period_end: datetime
    revenue: float
    cogs: float
    gross_profit: float
    operating_expenses: float
    operating_income: float
    interest_expense: float
    tax_expense: float
    net_income: float
    eps_basic: float
    eps_diluted: float
    shares_outstanding: int

@dataclass
class BalanceSheet:
    """Balance sheet data structure."""
    period_end: datetime
    cash: float
    receivables: float
    inventory: float
    current_assets: float
    ppe_net: float
    intangible_assets: float
    total_assets: float
    current_liabilities: float
    long_term_debt: float
    total_liabilities: float
    common_stock: float
    retained_earnings: float
    shareholders_equity: float

@dataclass
class CashFlowStatement:
    """Cash flow statement data structure."""
    period_end: datetime
    operating_cf: float
    capex: float
    investing_cf: float
    debt_issued: float
    dividends_paid: float
    buybacks: float
    financing_cf: float
    net_change_cash: float

class FinancialStatementParser:
    """Production-grade financial statement parser."""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.statements = {
            'income_statement': [],
            'balance_sheet': [],
            'cash_flow': []
        }
    
    def parse_from_sec (self, cik: str, user_agent: str):
        """Parse statements from SEC XBRL data."""
        facts = get_xbrl_facts (cik, user_agent)
        
        # Extract all periods
        us_gaap = facts['facts'].get('us-gaap', {})
        
        # Parse income statements for all periods
        revenue_data = us_gaap.get('Revenues', {}).get('units', {}).get('USD', [])
        
        for item in revenue_data:
            if item.get('form') == '10-K':  # Annual only
                period = datetime.fromisoformat (item['end'])
                
                income_stmt = self._extract_income_statement (us_gaap, period)
                if income_stmt:
                    self.statements['income_statement'].append (income_stmt)
        
        # Similarly for balance sheet and cash flow...
        
        return self.statements
    
    def _extract_income_statement(
        self,
        us_gaap: dict,
        period: datetime
    ) -> Optional[IncomeStatement]:
        """Extract income statement for specific period."""
        
        def get_value (tag: str) -> Optional[float]:
            """Helper to extract value for specific period."""
            if tag not in us_gaap:
                return None
            
            values = us_gaap[tag]['units']['USD']
            for v in values:
                if datetime.fromisoformat (v['end']) == period:
                    return v['val']
            return None
        
        revenue = get_value('Revenues')
        if not revenue:
            return None
        
        return IncomeStatement(
            period_end=period,
            revenue=revenue,
            cogs=get_value('CostOfRevenue') or 0,
            gross_profit=get_value('GrossProfit') or 0,
            operating_expenses=get_value('OperatingExpenses') or 0,
            operating_income=get_value('OperatingIncomeLoss') or 0,
            interest_expense=get_value('InterestExpense') or 0,
            tax_expense=get_value('IncomeTaxExpenseBenefit') or 0,
            net_income=get_value('NetIncomeLoss') or 0,
            eps_basic=get_value('EarningsPerShareBasic') or 0,
            eps_diluted=get_value('EarningsPerShareDiluted') or 0,
            shares_outstanding=get_value('WeightedAverageNumberOfSharesOutstandingBasic') or 0,
        )
    
    def to_dataframe (self) -> dict:
        """Convert to pandas DataFrames for analysis."""
        return {
            'income_statement': pd.DataFrame([
                vars (stmt) for stmt in self.statements['income_statement']
            ]),
            'balance_sheet': pd.DataFrame([
                vars (stmt) for stmt in self.statements['balance_sheet']
            ]),
            'cash_flow': pd.DataFrame([
                vars (stmt) for stmt in self.statements['cash_flow']
            ]),
        }

# Usage
parser = FinancialStatementParser("AAPL")
statements = parser.parse_from_sec("0000320193", "myemail@company.com")
dfs = parser.to_dataframe()

# Now you have clean, structured financial data ready for analysis
print(dfs['income_statement'][['period_end', 'revenue', 'net_income']])
\`\`\`

---

## Common Pitfalls

### 1. Not Handling Restated Financials

Companies sometimes restate prior periods:

\`\`\`python
def handle_restatements (statements: pd.DataFrame) -> pd.DataFrame:
    """Keep only most recent version of each period."""
    
    # Group by period, keep last filed
    return statements.sort_values('filed_date').groupby('period_end').last()
\`\`\`

### 2. Mixing Quarterly and Annual

Always specify which you're using:

\`\`\`python
def filter_annual_only (statements: list) -> list:
    """Keep only 10-K (annual) filings."""
    return [s for s in statements if s.form == '10-K']
\`\`\`

### 3. Currency Confusion

International companies may report in different currencies:

\`\`\`python
def normalize_currency (value: float, from_currency: str, to_currency: str = 'USD') -> float:
    """Convert financial values to consistent currency."""
    if from_currency == to_currency:
        return value
    
    # Use exchange rate API
    rate = get_exchange_rate (from_currency, to_currency)
    return value * rate
\`\`\`

### 4. Different Fiscal Year Ends

Not all companies use calendar year:

\`\`\`python
# Apple\'s fiscal year ends in September
apple_fy2022 = "2022-09-24"  # Fiscal 2022 ≠ calendar 2022

# Microsoft's fiscal year ends in June
msft_fy2022 = "2022-06-30"

# When comparing companies, align by fiscal period, not calendar!
\`\`\`

---

## Production Checklist

Before deploying your financial statement parser:

- [ ] Handle missing data gracefully
- [ ] Support both GAAP and IFRS
- [ ] Parse all historical periods (5-10 years)
- [ ] Validate statement linkages
- [ ] Handle currency conversion
- [ ] Check for restatements
- [ ] Add error logging
- [ ] Implement retry logic for SEC API
- [ ] Cache parsed data to avoid re-downloading
- [ ] Unit tests for each statement type

---

## Next Steps

Now that you understand financial statement fundamentals:

1. **Section 2**: Deep dive into income statement analysis
2. **Section 3**: Master balance sheet analysis
3. **Section 4**: Cash flow statement mastery
4. **Build**: Automated SEC filing parser for your favorite companies

---

## Summary

**Key Takeaways**:

1. Three financial statements tell complete story: profitability, position, and cash
2. Statements are interconnected—models must respect these links
3. Accrual accounting ≠ cash accounting (both matter!)
4. SEC EDGAR provides free, structured data via XBRL
5. Production parsers must handle edge cases (currencies, restatements, fiscal years)

**For Engineers**:
- Build automated parsers, not manual spreadsheets
- Use XBRL API for structured data (avoid HTML parsing)
- Validate data quality obsessively
- Think at scale (thousands of companies, not one)

**Next Module**: We'll analyze each statement in depth, calculate key ratios, and build fraud detection systems.
`,
};
