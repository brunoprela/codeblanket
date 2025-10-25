export const section1Discussion = {
    title: "Financial Statements Fundamentals - Discussion Questions",
    questions: [
        {
            id: 1,
            question: "Design a system to automatically detect when a company's financial statements don't properly link (e.g., net income doesn't flow correctly to retained earnings, or the balance sheet doesn't balance). What checks would you implement, and how would you handle edge cases like stock-based compensation, foreign currency translation adjustments, and spin-offs?",
            sample_answer: `A production-grade financial statement validation system would include multiple layers of checks:

**1. Core Linkage Validations**

\`\`\`python
class FinancialStatementValidator:
    """Validate statement linkages and consistency."""
    
    TOLERANCE = 1_000  # $1K tolerance for rounding
    
    def validate_balance_sheet(self, bs: dict) -> list:
        """Validate balance sheet equation."""
        errors = []
        
        assets = bs['total_assets']
        liabilities = bs['total_liabilities']
        equity = bs['shareholders_equity']
        
        if abs(assets - (liabilities + equity)) > self.TOLERANCE:
            errors.append({
                'type': 'BALANCE_SHEET_NOT_BALANCED',
                'severity': 'CRITICAL',
                'expected': assets,
                'actual': liabilities + equity,
                'difference': assets - (liabilities + equity)
            })
        
        return errors
    
    def validate_retained_earnings(
        self,
        bs_start: dict,
        bs_end: dict,
        income: dict,
        cf: dict
    ) -> list:
        """Validate retained earnings rollforward."""
        errors = []
        
        expected_re = (
            bs_start['retained_earnings'] +
            income['net_income'] -
            cf['dividends_paid'] -
            cf.get('buybacks_treasury_method', 0)
        )
        
        actual_re = bs_end['retained_earnings']
        
        if abs(expected_re - actual_re) > self.TOLERANCE:
            # Check for adjustments
            if 'other_comprehensive_income' in income:
                expected_re += income['other_comprehensive_income']
            
            if abs(expected_re - actual_re) > self.TOLERANCE:
                errors.append({
                    'type': 'RETAINED_EARNINGS_MISMATCH',
                    'severity': 'HIGH',
                    'expected': expected_re,
                    'actual': actual_re,
                    'possible_causes': [
                        'Missing OCI',
                        'Restatement',
                        'Spin-off',
                        'Accounting change'
                    ]
                })
        
        return errors
\`\`\`

**2. Edge Case Handlers**

Stock-Based Compensation:
- Increases equity (additional paid-in capital)
- Non-cash expense on income statement
- Add back in operating cash flow

\`\`\`python
def handle_stock_comp(statements: dict) -> dict:
    """Adjust for stock-based compensation."""
    sbc = statements['cash_flow']['stock_based_comp']
    
    # Verify: SBC added back in operating CF
    assert statements['cash_flow']['sbc_addback'] == sbc
    
    # Verify: SBC increases APIC
    statements['balance_sheet']['apic'] += sbc
    
    return statements
\`\`\`

Foreign Currency Translation:
- Goes to OCI, not retained earnings
- Part of accumulated OCI in equity section

\`\`\`python
def handle_forex_translation(statements: dict) -> dict:
    """Handle foreign currency translation adjustments."""
    fx_adjustment = statements['equity_changes']['fx_translation']
    
    # Should be in OCI, not net income
    statements['balance_sheet']['aoci'] += fx_adjustment
    
    # Does NOT affect retained earnings
    return statements
\`\`\`

Spin-offs and Divestitures:
- Direct reduction in equity
- Not through income statement

\`\`\`python
def detect_spinoffs(statements: dict) -> bool:
    """Detect if spin-off occurred."""
    
    # Large unexplained equity reduction
    equity_change = (
        statements['balance_sheet_end']['equity'] -
        statements['balance_sheet_start']['equity']
    )
    
    net_income = statements['income']['net_income']
    dividends = statements['cash_flow']['dividends']
    
    expected_change = net_income - dividends
    
    if (equity_change - expected_change) < -100_000_000:  # $100M threshold
        return True  # Likely spin-off
    
    return False
\`\`\`

**3. Comprehensive Validation Framework**

\`\`\`python
def full_validation_suite(statements: dict) -> dict:
    """Run all validation checks."""
    
    validator = FinancialStatementValidator()
    
    results = {
        'passed': True,
        'errors': [],
        'warnings': [],
        'metadata': {}
    }
    
    # Critical checks
    errors = []
    errors.extend(validator.validate_balance_sheet(statements['bs']))
    errors.extend(validator.validate_cash_reconciliation(statements))
    errors.extend(validator.validate_retained_earnings(
        statements['bs_start'],
        statements['bs_end'],
        statements['income'],
        statements['cf']
    ))
    
    # Warning-level checks
    warnings = []
    warnings.extend(validator.check_for_unusual_items(statements))
    warnings.extend(validator.verify_statement_dates_align(statements))
    
    if errors:
        results['passed'] = False
        results['errors'] = errors
    
    results['warnings'] = warnings
    
    return results
\`\`\`

**4. Real-World Edge Cases**

The system must handle:
- Treasury stock method for buybacks
- Deferred taxes flowing through balance sheet
- Goodwill from acquisitions
- Pension obligation adjustments
- Discontinued operations
- Non-controlling interests

**Production Implementation**:
At scale (5,000+ companies), you'd need:
- Configurable tolerances by company size
- Historical pattern learning (some companies always have small discrepancies)
- Manual review queue for unresolved errors
- Integration with SEC's amendment detection
- Alert system for material changes

This validation layer is critical before using financial data for automated trading or credit decisions.`
        },

        {
            id: 2,
            question: "You're building an automated trading system that needs to process 10-K filings within seconds of publication to gain an information edge. Describe your architecture for: (1) detecting new filings immediately, (2) parsing them faster than competitors, (3) extracting actionable signals, and (4) generating trades. What are the bottlenecks and how would you optimize each stage?",
            sample_answer: `Building a sub-second 10-K processing pipeline requires careful optimization at every stage:

**Overall Architecture**

\`\`\`
SEC EDGAR → Detection Service → Parser Pool → Signal Extraction → Trading Engine
   (RSS)      (WebSocket)      (Parallel)      (ML Models)        (Orders)
   
Latency breakdown:
- Detection: <100ms
- Download: 200-500ms
- Parsing: 500-1000ms
- Signal extraction: 200-500ms
- Trade generation: <100ms
Total: ~1-2 seconds target
\`\`\`

**Stage 1: Real-Time Filing Detection**

The SEC provides an RSS feed, but it updates every 10 minutes—too slow.

\`\`\`python
import asyncio
import aiohttp
from datetime import datetime

class SECFilingMonitor:
    """Ultra-low latency filing detection."""
    
    RSS_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=10-K&output=atom"
    POLL_INTERVAL = 5  # Poll every 5 seconds
    
    def __init__(self, callback):
        self.callback = callback
        self.seen_filings = set()
        self.session = None
    
    async def monitor(self):
        """Continuously monitor for new 10-Ks."""
        self.session = aiohttp.ClientSession()
        
        while True:
            try:
                new_filings = await self.check_for_new_filings()
                
                for filing in new_filings:
                    if filing['accession'] not in self.seen_filings:
                        self.seen_filings.add(filing['accession'])
                        
                        # Non-blocking callback
                        asyncio.create_task(
                            self.callback(filing)
                        )
                        
                        print(f"[{datetime.now()}] New 10-K: {filing['ticker']}")
                
                await asyncio.sleep(self.POLL_INTERVAL)
                
            except Exception as e:
                print(f"Monitor error: {e}")
                await asyncio.sleep(1)
    
    async def check_for_new_filings(self) -> list:
        """Fetch and parse RSS feed."""
        async with self.session.get(self.RSS_URL) as response:
            xml = await response.text()
            return self.parse_rss(xml)
\`\`\`

**Optimization**: Use multiple strategies:
- SEC RSS feed (primary)
- Twitter bot monitoring (@SEC_Filing_Bot)
- Direct EDGAR page scraping
- Company IR page webhooks

Take the **first** signal you receive.

**Stage 2: Parallel Parsing**

Download and parse multiple sections simultaneously:

\`\`\`python
class ParallelFilingParser:
    """Parse 10-K sections in parallel."""
    
    async def parse_filing(self, url: str) -> dict:
        """Download and parse with parallelization."""
        
        # Download full HTML
        html = await self.download(url)
        
        # Parse sections in parallel
        tasks = [
            self.extract_financials(html),
            self.extract_mda(html),
            self.extract_risk_factors(html),
            self.extract_notes(html)
        ]
        
        results = await asyncio.gather(*tasks)
        
        return {
            'financials': results[0],
            'mda': results[1],
            'risks': results[2],
            'notes': results[3],
            'parsed_at': datetime.now()
        }
    
    async def extract_financials(self, html: str) -> dict:
        """Extract financial statements."""
        # Use XBRL tags instead of HTML parsing (faster)
        xbrl_url = html.replace('.htm', '_htm.xml')
        xbrl_data = await self.download(xbrl_url)
        
        return self.parse_xbrl(xbrl_data)
\`\`\`

**Optimization**: 
- Pre-download XBRL instance documents (more structured)
- Cache company CIK mappings
- Use compiled regex patterns
- Parallel processing across CPU cores

**Stage 3: Signal Extraction**

Pre-trained models for instant analysis:

\`\`\`python
class SignalExtractor:
    """Extract trading signals from 10-K."""
    
    def __init__(self):
        # Pre-load models at startup
        self.sentiment_model = load_model('finbert')
        self.fraud_model = load_model('fraud_detector')
        self.peer_comps = load_peer_database()
    
    async def extract_signals(self, filing: dict, ticker: str) -> dict:
        """Generate trading signals."""
        
        signals = {}
        
        # 1. Earnings surprise (vs consensus)
        signals['eps_surprise'] = self.calculate_eps_surprise(
            filing['financials']['eps'],
            ticker
        )
        
        # 2. Margin trends
        current_margin = filing['financials']['operating_margin']
        prior_margin = self.get_prior_margin(ticker)
        signals['margin_change'] = current_margin - prior_margin
        
        # 3. MD&A sentiment
        mda_text = filing['mda']
        signals['sentiment'] = self.sentiment_model.predict(mda_text)
        
        # 4. Red flags
        signals['fraud_score'] = self.fraud_model.predict(filing['financials'])
        
        # 5. Relative valuation
        signals['valuation_vs_peers'] = self.compare_to_peers(
            filing['financials'],
            ticker
        )
        
        return signals
    
    def generate_trade_decision(self, signals: dict) -> str:
        """Combine signals into trade decision."""
        
        score = 0
        
        # Weight different signals
        score += signals['eps_surprise'] * 0.3
        score += signals['margin_change'] * 100 * 0.2
        score += (signals['sentiment'] - 0.5) * 0.2
        score -= signals['fraud_score'] * 0.2
        score += signals['valuation_vs_peers'] * 0.1
        
        if score > 0.5:
            return 'BUY'
        elif score < -0.5:
            return 'SELL'
        else:
            return 'HOLD'
\`\`\`

**Stage 4: Trade Execution**

\`\`\`python
class TradingEngine:
    """Execute trades based on signals."""
    
    async def execute_filing_trade(
        self,
        ticker: str,
        decision: str,
        signals: dict
    ):
        """Execute trade with risk checks."""
        
        # Risk checks
        if not self.passes_risk_checks(ticker, decision):
            return
        
        # Position sizing
        size = self.calculate_position_size(signals['confidence'])
        
        # Execute
        if decision == 'BUY':
            await self.place_order(ticker, 'BUY', size, 'MARKET')
        elif decision == 'SELL':
            await self.place_order(ticker, 'SELL', size, 'MARKET')
\`\`\`

**Bottlenecks and Optimizations**

1. **Network I/O (biggest bottleneck)**
   - Use async I/O (aiohttp)
   - Multiple connections to SEC
   - Consider AWS instance in us-east-1 (near SEC servers)
   - Pre-fetch likely filings (known earnings dates)

2. **Parsing**
   - XBRL >> HTML parsing (10x faster)
   - Compiled regex patterns
   - Cython for hot paths
   - Pre-build company-specific parsers

3. **Signal Extraction**
   - Pre-load all models into memory
   - Use ONNX runtime (faster inference)
   - GPU for NLP models
   - Cache peer comparisons

4. **Database Queries**
   - Keep hot data in Redis
   - Denormalize for speed
   - Use read replicas

**Real-World Considerations**

The edge is measured in seconds, not minutes:
- At t=0s: Filing published
- At t=10s: Sophisticated firms have traded
- At t=60s: Market has mostly priced in information
- At t=300s: Too late for alpha

**Alternative Strategy**: Instead of being fastest, be **smartest**—extract signals others miss (footnotes, subtle MD&A language, historical pattern matching).

**Production Requirements**:
- Redundant systems (no single point of failure)
- Monitoring and alerting
- Paper trading validation
- Kill switches for anomalies
- Compliance logging (every decision must be auditable)

The best firms combine speed AND depth: quick first-pass analysis, then deeper analysis over hours/days for longer-term positions.`
        },

        {
            id: 3,
            question: "Compare and contrast GAAP vs IFRS from a data engineering perspective. If you're building a global stock screener that analyzes 10,000+ companies across 50 countries, how would you normalize financial statements to enable fair comparisons? What specific adjustments would you make, and how would you handle the fact that some accounting choices (like inventory methods) can't be perfectly reconciled?",
            sample_answer: `Building a global financial statement normalization system requires understanding both accounting differences and practical data engineering trade-offs:

**Major GAAP vs IFRS Differences**

\`\`\`python
class AccountingStandardNormalizer:
    """Normalize financial statements across GAAP and IFRS."""
    
    ADJUSTMENTS = {
        'inventory': {
            'description': 'IFRS prohibits LIFO, GAAP allows it',
            'impact': 'HIGH',
            'normalizable': 'PARTIAL'
        },
        'development_costs': {
            'description': 'IFRS can capitalize, GAAP usually expenses',
            'impact': 'HIGH',
            'normalizable': 'YES'
        },
        'revaluation': {
            'description': 'IFRS allows PP&E revaluation, GAAP does not',
            'impact': 'MEDIUM',
            'normalizable': 'NO'
        },
        'extraordinary_items': {
            'description': 'GAAP eliminated, IFRS different treatment',
            'impact': 'MEDIUM',
            'normalizable': 'YES'
        }
    }
    
    def normalize_to_common_standard(
        self,
        statements: dict,
        from_standard: str
    ) -> dict:
        """Normalize to common standard (IFRS as base)."""
        
        if from_standard == 'GAAP':
            statements = self.gaap_to_ifrs(statements)
        elif from_standard != 'IFRS':
            statements = self.other_to_ifrs(statements, from_standard)
        
        return statements
\`\`\`

**Key Adjustments**

**1. Inventory Methods**

\`\`\`python
def adjust_inventory_method(
    statements: dict,
    company_info: dict
) -> dict:
    """
    Adjust LIFO to FIFO equivalent.
    Problem: GAAP allows LIFO, IFRS doesn't.
    """
    
    if company_info['inventory_method'] != 'LIFO':
        return statements  # No adjustment needed
    
    # Companies using LIFO must disclose LIFO reserve
    lifo_reserve = statements['notes'].get('lifo_reserve', 0)
    
    if lifo_reserve == 0:
        # Can't adjust - must flag as incomparable
        statements['_metadata']['inventory_not_normalized'] = True
        return statements
    
    # Adjust inventory on balance sheet
    statements['balance_sheet']['inventory'] += lifo_reserve
    
    # Adjust COGS on income statement (reverse)
    # Higher inventory = lower COGS (in rising price environment)
    delta_reserve = lifo_reserve - statements['prior_lifo_reserve']
    statements['income_statement']['cogs'] -= delta_reserve
    
    # Flow through to gross profit and net income
    statements['income_statement']['gross_profit'] += delta_reserve
    statements['income_statement']['net_income'] += delta_reserve * (1 - statements['tax_rate'])
    
    # Adjust equity
    statements['balance_sheet']['retained_earnings'] += (lifo_reserve * (1 - statements['tax_rate']))
    
    # Mark as adjusted
    statements['_metadata']['inventory_adjusted'] = True
    
    return statements
\`\`\`

**2. Development Costs**

\`\`\`python
def adjust_development_costs(
    statements: dict,
    standard: str
) -> dict:
    """
    Adjust for development cost capitalization.
    IFRS: Can capitalize development costs (if criteria met)
    GAAP: Usually expense immediately (except software)
    """
    
    if standard == 'IFRS':
        # IFRS company - reverse capitalization to match GAAP approach
        dev_costs_capitalized = statements['balance_sheet']['intangible_assets_dev']
        amortization = statements['income_statement']['amortization_dev']
        
        # Reverse: Add back to R&D expense
        statements['income_statement']['rd_expense'] += (
            dev_costs_capitalized - amortization
        )
        
        # Remove from balance sheet
        statements['balance_sheet']['intangible_assets'] -= dev_costs_capitalized
        
        # Adjust equity
        tax_effect = dev_costs_capitalized * (1 - statements['tax_rate'])
        statements['balance_sheet']['retained_earnings'] -= tax_effect
    
    return statements
\`\`\`

**3. Revenue Recognition**

\`\`\`python
def normalize_revenue_recognition(statements: dict) -> dict:
    """
    Both GAAP and IFRS now use similar standards (ASC 606 / IFRS 15).
    But implementation can vary.
    """
    
    # Check for unusual revenue timing
    revenue_growth = statements['income_statement']['revenue'] / statements['prior_revenue'] - 1
    receivables_growth = statements['balance_sheet']['receivables'] / statements['prior_receivables'] - 1
    
    # Red flag: Receivables growing much faster than revenue
    if receivables_growth > revenue_growth * 1.5:
        statements['_metadata']['revenue_quality_concern'] = True
        statements['_metadata']['days_sales_outstanding'] = calculate_dso(statements)
    
    return statements
\`\`\`

**Comprehensive Normalization Pipeline**

\`\`\`python
class GlobalFinancialNormalizer:
    """Production system for normalizing global financial statements."""
    
    def __init__(self):
        self.standard_converters = {
            'US-GAAP': self.gaap_normalizer,
            'IFRS': self.ifrs_normalizer,
            'Japanese-GAAP': self.jgaap_normalizer,
            'Chinese-ASBE': self.casbe_normalizer
        }
        
        self.currency_converter = CurrencyConverter()
    
    def normalize(self, company: dict) -> dict:
        """Full normalization pipeline."""
        
        statements = company['financial_statements']
        
        # Step 1: Currency normalization (to USD)
        statements = self.normalize_currency(
            statements,
            company['reporting_currency']
        )
        
        # Step 2: Accounting standard normalization
        statements = self.normalize_accounting_standard(
            statements,
            company['accounting_standard']
        )
        
        # Step 3: Segment adjustments
        statements = self.adjust_for_segments(statements, company)
        
        # Step 4: Calculate standardized metrics
        statements['normalized_metrics'] = self.calculate_normalized_metrics(statements)
        
        # Step 5: Quality scores
        statements['data_quality'] = self.assess_quality(statements)
        
        return statements
    
    def normalize_currency(self, statements: dict, currency: str) -> dict:
        """Convert all values to USD."""
        
        if currency == 'USD':
            return statements
        
        # Get average exchange rate for period
        rate = self.currency_converter.get_average_rate(
            from_currency=currency,
            to_currency='USD',
            start_date=statements['period_start'],
            end_date=statements['period_end']
        )
        
        # Convert income statement (use average rate)
        for key, value in statements['income_statement'].items():
            if isinstance(value, (int, float)):
                statements['income_statement'][key] = value * rate
        
        # Convert balance sheet (use period-end rate)
        end_rate = self.currency_converter.get_spot_rate(
            currency,
            'USD',
            statements['period_end']
        )
        
        for key, value in statements['balance_sheet'].items():
            if isinstance(value, (int, float)):
                statements['balance_sheet'][key] = value * end_rate
        
        return statements
    
    def calculate_normalized_metrics(self, statements: dict) -> dict:
        """Calculate metrics that are comparable across standards."""
        
        # These metrics are relatively comparable
        return {
            'revenue_growth': calculate_growth(statements['revenue']),
            'operating_margin': statements['operating_income'] / statements['revenue'],
            'roa': statements['net_income'] / statements['total_assets'],
            'asset_turnover': statements['revenue'] / statements['total_assets'],
            'current_ratio': statements['current_assets'] / statements['current_liabilities'],
            'debt_to_equity': statements['total_debt'] / statements['equity'],
            
            # Adjusted metrics
            'adjusted_roe': self.calculate_adjusted_roe(statements),
            'normalized_earnings': self.calculate_normalized_earnings(statements),
            'fcf_yield': statements['free_cash_flow'] / statements['market_cap']
        }
\`\`\`

**Handling Non-Normalizable Differences**

Some differences can't be reconciled:

\`\`\`python
def assess_comparability(company_a: dict, company_b: dict) -> dict:
    """Assess if two companies are truly comparable."""
    
    issues = []
    
    # Check 1: Different inventory methods without adjustment data
    if (company_a['inventory_method'] != company_b['inventory_method'] and
        not (company_a.get('lifo_reserve') and company_b.get('lifo_reserve'))):
        issues.append({
            'type': 'INVENTORY_METHOD',
            'severity': 'HIGH',
            'impact': 'COGS and margins not directly comparable'
        })
    
    # Check 2: Different depreciation methods
    if company_a['depreciation_method'] != company_b['depreciation_method']:
        issues.append({
            'type': 'DEPRECIATION',
            'severity': 'MEDIUM',
            'impact': 'Earnings timing differs'
        })
    
    # Check 3: Goodwill impairment vs amortization
    # (US GAAP: impairment only, some other standards: systematic amortization)
    if company_a['goodwill_accounting'] != company_b['goodwill_accounting']:
        issues.append({
            'type': 'GOODWILL',
            'severity': 'MEDIUM',
            'impact': 'Earnings quality differs'
        })
    
    return {
        'comparable': len([i for i in issues if i['severity'] == 'HIGH']) == 0,
        'issues': issues,
        'confidence_score': 1.0 - (len(issues) * 0.1)
    }
\`\`\`

**Production Architecture**

For 10,000+ companies:

\`\`\`
Raw Data → Validation → Normalization → Metrics → Database
(XBRL)      (Quality)    (Standards)    (Ratios)   (TimescaleDB)
   ↓            ↓            ↓             ↓           ↓
Parallel    Schema       Currency       Cache     Indexed
Processing  Checks       Convert        Results   Queries
\`\`\`

**Practical Approach**

1. **Tier 1 Normalization** (Always do):
   - Currency conversion
   - Fiscal year alignment
   - Share count adjustments
   - One-time item removal

2. **Tier 2 Normalization** (When possible):
   - Inventory method (if LIFO reserve disclosed)
   - Development costs (if disclosed)
   - Operating lease adjustments

3. **Tier 3** (Flag as non-comparable):
   - PP&E revaluation (can't reverse)
   - Different goodwill treatments
   - Pension accounting differences

**Key Insight**: Perfect normalization is impossible. Instead:
- Normalize what you can
- Flag what you can't
- Use confidence scores
- Compare within similar accounting regimes when possible

This is why quant funds often focus on single markets (US-only) or use factors (like momentum) that don't require perfect fundamental comparisons.`
        }
    ]
};

