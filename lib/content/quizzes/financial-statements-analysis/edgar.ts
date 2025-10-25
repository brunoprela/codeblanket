export const edgarDiscussionQuestions = [
  {
    id: 1,
    question:
      'Design a system that monitors SEC filings in real-time and automatically flags companies with deteriorating financial metrics. What filings would you monitor? What metrics trigger alerts? How would you handle false positives?',
    answer: `**System Design: Real-Time SEC Filing Monitor**

\`\`\`python
class SECFilingMonitor:
    """Real-time monitoring system for SEC filings."""
    
    def __init__(self):
        self.monitored_filings = ['10-Q', '10-K', '8-K']
        self.alert_thresholds = {
            'cfo_ni_ratio': 0.7,  # CFO < 70% of NI
            'current_ratio': 1.0,  # Below 1.0
            'debt_to_equity': 3.0,  # Above 3.0
            'revenue_decline': -0.10,  # 10%+ decline
            'gross_margin_decline': -0.05  # 5%+ decline
        }
    
    def monitor_filing (self, ticker: str, filing_type: str):
        """Process new filing and generate alerts."""
        
        # 1. Extract financials
        current = self.extract_metrics (ticker, filing_type)
        prior = self.get_prior_period (ticker)
        
        # 2. Calculate changes
        alerts = []
        
        # CFO quality check
        if current['cfo'] / current['net_income'] < self.alert_thresholds['cfo_ni_ratio']:
            alerts.append({
                'severity': 'HIGH',
                'metric': 'CFO/NI Ratio',
                'value': current['cfo'] / current['net_income'],
                'threshold': 0.7,
                'message': 'Earnings quality deterioration - CFO < NI'
            })
        
        # Revenue trend
        rev_change = (current['revenue'] - prior['revenue']) / prior['revenue']
        if rev_change < self.alert_thresholds['revenue_decline']:
            alerts.append({
                'severity': 'MEDIUM',
                'metric': 'Revenue Growth',
                'value': rev_change,
                'message': f'Revenue declined {rev_change:.1%}'
            })
        
        return alerts
\`\`\`

**Monitored Filings**:
- **10-K/10-Q**: Financial statements (primary data source)
- **8-K Item 2.02**: Earnings warnings
- **8-K Item 5.02**: Management changes (red flag if CFO/CEO)
- **Form 4**: Heavy insider selling

**Alert Triggers**:
1. CFO/NI ratio < 0.7 for 2 consecutive quarters
2. Current ratio < 1.0 (liquidity crisis)
3. Revenue decline + inventory increase (channel stuffing)
4. Debt/Equity > 3.0 + interest coverage < 2.0 (solvency risk)
5. Multiple C-suite departures within 6 months

**False Positive Handling**:
- Require 2+ quarters of deterioration (not single quarter)
- Context-aware rules (tech vs manufacturing thresholds)
- Exclude restructuring/acquisition impacts
- Severity tiering (high/medium/low)
- Manual review queue for borderline cases`,
  },

  {
    id: 2,
    question:
      "You're building an NLP system to analyze MD&A sections. What specific textual patterns or keywords indicate: (a) management hiding problems, (b) genuine optimism vs forced optimism, (c) earnings manipulation risk? Provide implementation strategy.",
    answer: `**NLP-Based MD&A Analysis System**

\`\`\`python
class MDATextAnalyzer:
    """Detect red flags in Management Discussion & Analysis."""
    
    def __init__(self):
        self.red_flag_patterns = {
            'evasive_language': [
                'substantially',
                'approximately',
                'certain circumstances',
                'various factors',
                'challenging environment'
            ],
            'manipulation_indicators': [
                'one-time',
                'non-recurring',
                'adjusted earnings',
                'normalized',
                'excluding certain items'
            ],
            'aggressive_tone': [
                'confident',
                'strongly believe',
                'unprecedented opportunity',
                'game-changing',
                'revolutionary'
            ]
        }
    
    def analyze_mda (self, mda_text: str, prior_mda: str) -> Dict:
        """Compare current MD&A to prior period."""
        
        current_analysis = {
            'fog_index': self.calculate_readability (mda_text),
            'evasive_language_count': self.count_patterns (mda_text, 'evasive_language'),
            'sentiment_score': self.analyze_sentiment (mda_text),
            'forward_looking_count': self.count_forward_looking (mda_text)
        }
        
        prior_analysis = self.analyze_prior (prior_mda)
        
        # Red flag: Increased complexity (hiding problems)
        if current_analysis['fog_index'] > prior_analysis['fog_index'] + 2:
            return {'alert': 'Increased complexity - potential obfuscation'}
        
        # Red flag: Excessive adjustments mentioned
        if current_analysis['evasive_language_count'] > 10:
            return {'alert': 'Excessive qualifying language'}
        
        return current_analysis
\`\`\`

**(a) Management Hiding Problems**:
- **Increased document length** (padding with jargon)
- **Rising Fog Index** (deliberately complex writing)
- **Passive voice increase** ("mistakes were made")
- **Vague attributions** ("certain market conditions")
- **Future-focused deflection** (avoid discussing current problems)

**(b) Genuine vs Forced Optimism**:
**Genuine**:
- Specific metrics cited ("30% growth in segment X")
- Concrete initiatives ("launching product in Q2")
- Balanced tone (acknowledges risks)

**Forced**:
- Generic platitudes ("well-positioned for growth")
- No specifics, only adjectives
- Overly promotional tone
- Contradicts financial metrics

**(c) Earnings Manipulation Risk**:
- **Frequent "adjusted" mentions** (>5 times)
- **Growing adjustments over time** (excluding more items)
- **Complex revenue recognition discussion** (changed policies)
- **Emphasis on non-GAAP metrics** (de-emphasizing GAAP)
- **Working capital discussion absent** (when it should be addressed)

**Implementation**:
1. Use FinBERT for financial sentiment
2. Calculate readability metrics (Fog, Flesch-Kincaid)
3. Track keyword frequency year-over-year
4. Flag documents with >20% complexity increase
5. Cross-reference text sentiment with actual financial metrics`,
  },

  {
    id: 3,
    question:
      "How would you build a system that detects when companies are 'managing' earnings by manipulating XBRL tags or classifications? What validation checks would you implement?",
    answer: `**XBRL Validation & Manipulation Detection System**

\`\`\`python
class XBRLValidator:
    """Detect XBRL manipulation and classification games."""
    
    def validate_filing (self, facts: Dict, ticker: str) -> List[str]:
        """Run validation checks on XBRL data."""
        
        issues = []
        
        # Check 1: Revenue recognition consistency
        revenue_tags = ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet']
        if self.multiple_revenue_tags_used (facts, revenue_tags):
            issues.append('Multiple revenue tags used - check for classification shift')
        
        # Check 2: Operating expense reclassification
        if self.detect_expense_reclassification (facts):
            issues.append('Operating expenses may have been reclassified')
        
        # Check 3: Unusual custom tags
        custom_tag_pct = self.calculate_custom_tag_percentage (facts)
        if custom_tag_pct > 0.15:  # >15% custom tags
            issues.append (f'High custom tag usage ({custom_tag_pct:.1%}) - reduced comparability')
        
        # Check 4: Balance sheet doesn't balance
        if not self.validate_balance_sheet (facts):
            issues.append('CRITICAL: Balance sheet assets ≠ liabilities + equity')
        
        # Check 5: Cash flow reconciliation
        if not self.validate_cash_flow_reconciliation (facts):
            issues.append('Cash flow statement doesn't reconcile with balance sheet')
        
        return issues
    
    def detect_expense_reclassification (self, facts: Dict) -> bool:
        """Detect if expenses moved between categories."""
        
        # Compare current vs prior period classifications
        # Flag if same $ amount moved from OpEx to COGS (boosts margins)
        pass
    
    def validate_balance_sheet (self, facts: Dict) -> bool:
        """Verify accounting equation: Assets = Liabilities + Equity."""
        
        assets = self.get_value (facts, 'Assets')
        liabilities = self.get_value (facts, 'Liabilities')
        equity = self.get_value (facts, 'StockholdersEquity')
        
        # Allow 1% rounding difference
        return abs (assets - (liabilities + equity)) / assets < 0.01
\`\`\`

**Validation Checks**:

1. **Tag Consistency Check**:
   - Same metric should use same XBRL tag year-over-year
   - Flag if company switches tags (e.g., revenue tag changed)

2. **Classification Gaming**:
   - Detect if COGS reclassified as operating expense (inflates gross margin)
   - Check if R&D capitalized vs expensed (changes over time)

3. **Custom Extension Abuse**:
   - Flag if >15% of tags are company-specific extensions
   - Reduces comparability with peers

4. **Accounting Equation Validation**:
   - Assets must equal Liabilities + Equity
   - Income Statement net income must match cash flow starting point

5. **Cash Flow Reconciliation**:
   - Change in balance sheet cash must equal cash flow statement
   - Working capital changes must tie to balance sheet

6. **Peer Comparison**:
   - Compare tag usage to industry peers
   - Flag if company is outlier in tag selection

**Red Flags**:
- Frequent tag changes (instability or manipulation)
- High custom tag usage (avoid standard metrics)
- Balance sheet doesn't balance (data quality issues)
- Inconsistent classifications year-over-year (managing metrics)

**Implementation**: Build automated validator that runs on every new filing, generates report of issues, and flags companies for manual review if >3 validation failures.`,
  },
];
