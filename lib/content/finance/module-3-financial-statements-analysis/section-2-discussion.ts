export const section2Discussion = {
    title: "Income Statement Analysis - Discussion Questions",
    questions: [
        {
            id: 1,
            question: "You're building a machine learning model to predict quarterly earnings surprises (actual EPS vs analyst estimates). Design a feature engineering pipeline that extracts predictive signals from income statements. What features would you create from the revenue, expense, and margin data? How would you handle seasonality, one-time items, and the fact that companies can manage earnings to meet targets?",
            sample_answer: `A sophisticated earnings prediction model requires careful feature engineering that captures both fundamental trends and potential earnings management:

**1. Core Financial Features**

\`\`\`python
import pandas as pd
import numpy as np
from typing import Dict, List

class IncomeStatementFeatureEngine:
    """Extract predictive features from income statements."""
    
    def __init__(self, lookback_periods: int = 8):
        self.lookback = lookback_periods
    
    def create_features(self, company_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive feature set.
        
        Input DataFrame should have columns:
        - period_end, revenue, cogs, gross_profit, operating_expenses,
          operating_income, net_income, eps, shares_outstanding,
          accounts_receivable, inventory, etc.
        """
        
        df = company_data.sort_values('period_end').copy()
        
        # 1. Growth Features (YoY and QoQ)
        df = self._add_growth_features(df)
        
        # 2. Margin Features and Trends
        df = self._add_margin_features(df)
        
        # 3. Quality Metrics
        df = self._add_quality_features(df)
        
        # 4. Seasonality Features
        df = self._add_seasonality_features(df)
        
        # 5. Earnings Management Indicators
        df = self._add_em_features(df)
        
        # 6. Momentum and Acceleration
        df = self._add_momentum_features(df)
        
        return df
    
    def _add_growth_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Growth rates and acceleration."""
        
        # Quarter-over-quarter growth
        df['revenue_qoq'] = df['revenue'].pct_change()
        df['eps_qoq'] = df['eps'].pct_change()
        df['operating_income_qoq'] = df['operating_income'].pct_change()
        
        # Year-over-year growth (more stable, removes seasonality)
        df['revenue_yoy'] = df['revenue'].pct_change(periods=4)
        df['eps_yoy'] = df['eps'].pct_change(periods=4)
        
        # Growth acceleration (is growth accelerating or decelerating?)
        df['revenue_acceleration'] = df['revenue_qoq'].diff()
        df['eps_acceleration'] = df['eps_qoq'].diff()
        
        # Compound growth rates (smoother)
        df['revenue_cagr_2q'] = (df['revenue'] / df['revenue'].shift(2)) ** (1/2) - 1
        df['revenue_cagr_4q'] = (df['revenue'] / df['revenue'].shift(4)) ** (1/4) - 1
        
        return df
    
    def _add_margin_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Margin calculations and trends."""
        
        # Current margins
        df['gross_margin'] = df['gross_profit'] / df['revenue']
        df['operating_margin'] = df['operating_income'] / df['revenue']
        df['net_margin'] = df['net_income'] / df['revenue']
        
        # Margin changes (QoQ and YoY)
        df['gross_margin_change_qoq'] = df['gross_margin'].diff()
        df['gross_margin_change_yoy'] = df['gross_margin'].diff(periods=4)
        
        df['operating_margin_change_qoq'] = df['operating_margin'].diff()
        df['net_margin_change_qoq'] = df['net_margin'].diff()
        
        # Margin trends (rolling averages)
        df['gross_margin_trend_4q'] = df['gross_margin'].rolling(4).mean()
        df['operating_margin_trend_4q'] = df['operating_margin'].rolling(4).mean()
        
        # Margin volatility (predictor of earnings surprises)
        df['gross_margin_volatility'] = df['gross_margin'].rolling(4).std()
        df['operating_margin_volatility'] = df['operating_margin'].rolling(4).std()
        
        return df
    
    def _add_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Revenue and earnings quality indicators."""
        
        # Days Sales Outstanding
        df['dso'] = (df['accounts_receivable'] / df['revenue']) * 90
        df['dso_change'] = df['dso'].diff()
        
        # Receivables as % of revenue
        df['receivables_ratio'] = df['accounts_receivable'] / df['revenue']
        df['receivables_ratio_change'] = df['receivables_ratio'].diff()
        
        # Inventory turnover (for companies with inventory)
        if 'inventory' in df.columns:
            df['inventory_turnover'] = df['cogs'] / df['inventory']
            df['days_inventory'] = 90 / df['inventory_turnover']
            df['inventory_change'] = df['inventory'].pct_change()
        
        # Accruals (difference between net income and cash flow)
        if 'operating_cash_flow' in df.columns:
            df['accruals'] = df['net_income'] - df['operating_cash_flow']
            df['accruals_ratio'] = df['accruals'] / df['total_assets']
        
        return df
    
    def _add_seasonality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Capture seasonal patterns."""
        
        # Extract quarter
        df['quarter'] = pd.to_datetime(df['period_end']).dt.quarter
        
        # One-hot encode quarters
        for q in [1, 2, 3, 4]:
            df[f'is_q{q}'] = (df['quarter'] == q).astype(int)
        
        # Sequential quarter counter (for trend)
        df['quarter_sequence'] = range(len(df))
        
        # YoY same quarter comparison
        df['revenue_vs_same_quarter_ly'] = df['revenue'] / df['revenue'].shift(4) - 1
        df['eps_vs_same_quarter_ly'] = df['eps'] / df['eps'].shift(4) - 1
        
        return df
    
    def _add_em_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Earnings management detection features."""
        
        # 1. Unusual accruals (earnings management indicator)
        if 'accruals_ratio' in df.columns:
            # High accruals relative to history = potential EM
            df['accruals_percentile'] = df['accruals_ratio'].rolling(8).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1]
            )
        
        # 2. "Just beat" indicator (suspiciously close to estimates)
        # Would need analyst estimates data for this
        
        # 3. Fourth quarter shenanigans
        df['is_q4'] = (df['quarter'] == 4).astype(int)
        
        # Q4 revenue as % of annual (companies may stuff Q4)
        df['q4_revenue_concentration'] = np.where(
            df['is_q4'] == 1,
            df['revenue'] / df['revenue'].rolling(4).sum(),
            np.nan
        )
        
        # 4. Sudden changes in accounting estimates
        if 'depreciation' in df.columns:
            df['depreciation_ratio'] = df['depreciation'] / df['ppe']
            df['depreciation_ratio_change'] = df['depreciation_ratio'].diff()
        
        # 5. Share count changes (buybacks can artificially boost EPS)
        df['shares_change'] = df['shares_outstanding'].pct_change()
        
        # EPS growth vs Net Income growth (buyback effect)
        ni_growth = df['net_income'].pct_change()
        df['eps_ni_growth_diff'] = df['eps_qoq'] - ni_growth
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum and mean reversion features."""
        
        # Revenue momentum (consecutive beats)
        df['revenue_growth_positive'] = (df['revenue_qoq'] > 0).astype(int)
        df['revenue_momentum_streak'] = df['revenue_growth_positive'].groupby(
            (df['revenue_growth_positive'] != df['revenue_growth_positive'].shift()).cumsum()
        ).cumsum()
        
        # Mean reversion features
        df['revenue_growth_vs_avg'] = (
            df['revenue_qoq'] - df['revenue_qoq'].rolling(8).mean()
        )
        
        df['margin_vs_avg'] = (
            df['operating_margin'] - df['operating_margin'].rolling(8).mean()
        )
        
        # Z-scores (how unusual is current performance?)
        df['revenue_growth_zscore'] = (
            (df['revenue_qoq'] - df['revenue_qoq'].rolling(8).mean()) /
            df['revenue_qoq'].rolling(8).std()
        )
        
        return df

# Usage Example
\`\`\`

**2. Handling Seasonality**

\`\`\`python
def adjust_for_seasonality(df: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Remove seasonal component using additive decomposition.
    Critical for retail, travel, holiday-driven businesses.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Decompose into trend + seasonal + residual
    decomposition = seasonal_decompose(
        df[target_col],
        model='additive',
        period=4,  # Quarterly
        extrapolate_trend='freq'
    )
    
    # Return deseasonalized (trend + residual)
    return df[target_col] - decomposition.seasonal

# Feature: Seasonally-adjusted revenue growth
df['revenue_sa'] = adjust_for_seasonality(df, 'revenue')
df['revenue_growth_sa'] = df['revenue_sa'].pct_change()
\`\`\`

**3. Detecting and Handling One-Time Items**

\`\`\`python
class OneTimeItemDetector:
    """Identify and adjust for non-recurring items."""
    
    def detect_one_time_items(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag unusual items that may distort comparisons."""
        
        # 1. Statistical outliers
        for col in ['revenue', 'operating_income', 'net_income']:
            mean = df[col].rolling(8).mean()
            std = df[col].rolling(8).std()
            
            # Flag if >3 standard deviations from rolling mean
            df[f'{col}_outlier'] = (
                np.abs(df[col] - mean) > 3 * std
            ).astype(int)
        
        # 2. Restructuring charges (if disclosed)
        if 'restructuring_charges' in df.columns:
            df['has_restructuring'] = (df['restructuring_charges'] > 0).astype(int)
        
        # 3. Large gains/losses
        if 'other_income' in df.columns:
            # Unusual if >5% of operating income
            df['unusual_other_income'] = (
                np.abs(df['other_income']) > 0.05 * df['operating_income']
            ).astype(int)
        
        # 4. Adjusted earnings (removing one-time items)
        df['adjusted_net_income'] = df['net_income'].copy()
        
        if 'restructuring_charges' in df.columns:
            df['adjusted_net_income'] += df['restructuring_charges']
        
        if 'asset_impairments' in df.columns:
            df['adjusted_net_income'] += df['asset_impairments']
        
        df['adjusted_eps'] = df['adjusted_net_income'] / df['shares_outstanding']
        
        return df
\`\`\`

**4. Modeling Earnings Management**

Companies can "manage" earnings to hit targets. Detect this:

\`\`\`python
class EarningsManagementDetector:
    """Detect potential earnings manipulation."""
    
    def calculate_beneish_m_score(self, current: Dict, prior: Dict) -> float:
        """
        Beneish M-Score: Probability of earnings manipulation.
        M-Score > -2.22 suggests potential manipulation.
        """
        
        # Days Sales in Receivables Index
        dsri = (current['receivables'] / current['revenue']) / \
               (prior['receivables'] / prior['revenue'])
        
        # Gross Margin Index
        gmi = (prior['gross_profit'] / prior['revenue']) / \
              (current['gross_profit'] / current['revenue'])
        
        # Asset Quality Index
        aqi = (1 - (current['current_assets'] + current['ppe']) / current['total_assets']) / \
              (1 - (prior['current_assets'] + prior['ppe']) / prior['total_assets'])
        
        # Sales Growth Index
        sgi = current['revenue'] / prior['revenue']
        
        # Depreciation Index
        depi = (prior['depreciation'] / (prior['ppe'] + prior['depreciation'])) / \
               (current['depreciation'] / (current['ppe'] + current['depreciation']))
        
        # SG&A Index
        sgai = (current['sga'] / current['revenue']) / \
               (prior['sga'] / prior['revenue'])
        
        # Accruals (simplified)
        tata = (current['net_income'] - current['operating_cf']) / current['total_assets']
        
        # M-Score formula
        m_score = (
            -4.84 +
            0.920 * dsri +
            0.528 * gmi +
            0.404 * aqi +
            0.892 * sgi +
            0.115 * depi -
            0.172 * sgai +
            4.679 * tata
        )
        
        return m_score
    
    def assess_em_risk(self, m_score: float) -> str:
        """Interpret M-Score."""
        if m_score > -2.22:
            return "HIGH RISK - Likely manipulator"
        elif m_score > -2.5:
            return "MEDIUM RISK - Monitor closely"
        else:
            return "LOW RISK - Appears legitimate"
\`\`\`

**5. Complete ML Pipeline**

\`\`\`python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

class EarningsSurprisePredictor:
    """Predict earnings surprises using income statement features."""
    
    def __init__(self):
        self.feature_engine = IncomeStatementFeatureEngine()
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            min_samples_split=10
        )
        
        self.feature_cols = [
            'revenue_yoy', 'eps_yoy', 'revenue_acceleration',
            'gross_margin', 'gross_margin_change_yoy',
            'operating_margin', 'operating_margin_change_qoq',
            'dso_change', 'receivables_ratio_change',
            'accruals_ratio', 'accruals_percentile',
            'shares_change', 'eps_ni_growth_diff',
            'revenue_growth_zscore', 'is_q1', 'is_q2', 'is_q3', 'is_q4'
        ]
    
    def train(self, historical_data: pd.DataFrame, analyst_estimates: pd.DataFrame):
        """
        Train model to predict earnings surprises.
        
        Target: (Actual EPS - Consensus Estimate) / Stock Price (%)
        """
        
        # Generate features
        features = self.feature_engine.create_features(historical_data)
        
        # Merge with analyst estimates
        data = features.merge(analyst_estimates, on='period_end')
        
        # Calculate actual surprise
        data['eps_surprise_pct'] = (
            (data['actual_eps'] - data['consensus_eps']) / data['stock_price']
        )
        
        # Prepare train data
        X = data[self.feature_cols].dropna()
        y = data.loc[X.index, 'eps_surprise_pct']
        
        # Time series cross-validation (no lookahead!)
        tscv = TimeSeriesSplit(n_splits=5)
        
        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            self.model.fit(X_train, y_train)
            score = self.model.score(X_test, y_test)
            scores.append(score)
        
        print(f"Cross-validated R²: {np.mean(scores):.3f}")
        
        # Final model on all data
        self.model.fit(X, y)
        
        return self
    
    def predict_surprise(
        self,
        current_financials: pd.DataFrame,
        consensus_estimate: float
    ) -> Dict:
        """Predict if company will beat/miss earnings."""
        
        features = self.feature_engine.create_features(current_financials)
        X = features[self.feature_cols].iloc[-1:].dropna()
        
        predicted_surprise_pct = self.model.predict(X)[0]
        
        # Feature importance
        importance = dict(zip(
            self.feature_cols,
            self.model.feature_importances_
        ))
        
        top_drivers = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'predicted_surprise_pct': predicted_surprise_pct,
            'predicted_direction': 'BEAT' if predicted_surprise_pct > 0 else 'MISS',
            'confidence': abs(predicted_surprise_pct) * 100,
            'top_drivers': top_drivers
        }
\`\`\`

**Key Insights**:

1. **YoY growth** more predictive than QoQ (removes seasonality)
2. **Margin trends** are leading indicators (margins improve before earnings)
3. **Quality metrics** (DSO, accruals) predict sustainability
4. **Earnings management indicators** reveal companies likely to disappoint
5. **Momentum features** capture trends (acceleration/deceleration)

**Production Considerations**:
- Update features in real-time as new filings arrive
- Backtesting with strict time-series splits (no lookahead!)
- Ensemble multiple models (gradient boosting + neural net)
- Monitor feature drift (relationships change over time)
- Combine with alternative data (credit card spending, web traffic)

This approach has been used successfully by quantitative hedge funds to generate alpha from earnings announcements.`
        },

        {
            id: 2,
            question: "Design a system to automatically classify operating expenses (SG&A, R&D, etc.) from unstructured 10-K text when companies don't break them out clearly. You have the full text of the filing but line items are aggregated or described inconsistently. How would you extract and normalize these figures across thousands of companies with different reporting formats?",
            sample_answer: `Building a robust expense classification system requires combining NLP, accounting knowledge, and pattern matching:

**1. System Architecture**

\`\`\`python
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

@dataclass
class ExpenseLineItem:
    """Parsed expense line item."""
    description: str
    amount: float
    category: str
    confidence: float
    source_text: str
    
class ExpenseClassificationSystem:
    """Extract and classify operating expenses from 10-K text."""
    
    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')
        self.amount_pattern = self._compile_amount_patterns()
        self.category_keywords = self._load_category_keywords()
        self.ml_classifier = None  # Trained ML model
        
    def process_filing(self, filing_text: str, cik: str) -> Dict:
        """
        Full pipeline to extract and classify expenses.
        
        Steps:
        1. Locate income statement section
        2. Extract line items with amounts
        3. Classify each line item
        4. Aggregate by category
        5. Validate against totals
        """
        
        # Step 1: Find income statement
        income_stmt_text = self._extract_income_statement(filing_text)
        
        # Step 2: Parse line items
        line_items = self._extract_line_items(income_stmt_text)
        
        # Step 3: Classify each
        classified_items = []
        for item in line_items:
            category = self._classify_expense(item)
            classified_items.append(category)
        
        # Step 4: Aggregate
        aggregated = self._aggregate_by_category(classified_items)
        
        # Step 5: Validate
        validation = self._validate_classification(aggregated, filing_text)
        
        return {
            'cik': cik,
            'line_items': classified_items,
            'aggregated': aggregated,
            'validation': validation
        }
    
    def _compile_amount_patterns(self) -> List[re.Pattern]:
        """Regex patterns to extract dollar amounts."""
        
        return [
            # Parentheses notation: $(1,234) or (1,234)
            re.compile(r'\\$?\\(([\\d,]+)\\)'),
            
            # Dollar sign: $1,234 or $1,234.5
            re.compile(r'\\$([\\d,]+\\.?\\d*)'),
            
            # No dollar sign: 1,234 or 1234
            re.compile(r'([\\d,]{4,})'),
            
            # With millions/thousands notation
            re.compile(r'([\\d,]+\\.?\\d*)\\s*(?:million|thousand)', re.I),
        ]
    
    def _load_category_keywords(self) -> Dict[str, List[str]]:
        """Keyword dictionary for each expense category."""
        
        return {
            'selling': [
                'selling', 'sales and marketing', 'sales force',
                'advertising', 'promotion', 'marketing',
                'sales commission', 'distribution', 'selling expense'
            ],
            'general_admin': [
                'general and administrative', 'g&a', 'administrative',
                'corporate', 'overhead', 'office', 'facilities',
                'legal', 'accounting', 'audit', 'professional fees',
                'executive compensation', 'management'
            ],
            'research_dev': [
                'research and development', 'r&d', 'research',
                'development', 'engineering', 'product development',
                'innovation', 'technology development'
            ],
            'depreciation': [
                'depreciation', 'amortization', 'd&a',
                'depreciation and amortization'
            ],
            'restructuring': [
                'restructuring', 'impairment', 'asset impairment',
                'restructuring charges', 'severance',
                'facility closure'
            ],
            'stock_comp': [
                'stock-based compensation', 'share-based',
                'equity compensation', 'stock options',
                'restricted stock units', 'rsu'
            ]
        }
    
    def _extract_income_statement(self, filing_text: str) -> str:
        """Locate income statement section in 10-K."""
        
        # Common section headers
        headers = [
            r'CONSOLIDATED STATEMENTS? OF (?:INCOME|OPERATIONS|EARNINGS)',
            r'STATEMENT OF (?:INCOME|OPERATIONS|EARNINGS)',
            r'CONDENSED CONSOLIDATED STATEMENTS? OF (?:INCOME|OPERATIONS)',
        ]
        
        for header_pattern in headers:
            match = re.search(header_pattern, filing_text, re.I)
            if match:
                start = match.start()
                
                # Find end (next major section or footer)
                end_patterns = [
                    r'CONSOLIDATED BALANCE SHEETS?',
                    r'STATEMENTS? OF CASH FLOWS?',
                    r'The accompanying notes are an integral part'
                ]
                
                end = len(filing_text)
                for end_pattern in end_patterns:
                    end_match = re.search(
                        end_pattern,
                        filing_text[start:start+50000],
                        re.I
                    )
                    if end_match:
                        end = start + end_match.start()
                        break
                
                return filing_text[start:end]
        
        return ""  # Not found
    
    def _extract_line_items(self, text: str) -> List[Dict]:
        """Parse line items from income statement text."""
        
        lines = text.split('\\n')
        line_items = []
        
        for i, line in enumerate(lines):
            # Skip if line doesn't contain numbers
            if not any(char.isdigit() for char in line):
                continue
            
            # Extract amounts
            amounts = self._parse_amounts(line)
            if not amounts:
                continue
            
            # Extract description (text before amount)
            description = self._extract_description(line, amounts[0])
            
            # Skip if it's a header or total line
            if self._is_header_or_total(description):
                continue
            
            line_items.append({
                'description': description.strip(),
                'amount': amounts[-1],  # Latest period amount
                'raw_line': line,
                'line_number': i
            })
        
        return line_items
    
    def _parse_amounts(self, line: str) -> List[float]:
        """Extract all monetary amounts from a line."""
        
        amounts = []
        
        for pattern in self.amount_pattern:
            matches = pattern.findall(line)
            for match in matches:
                # Clean and convert to float
                clean = match.replace(',', '').replace('$', '')
                try:
                    amount = float(clean)
                    amounts.append(amount)
                except ValueError:
                    continue
        
        return amounts
    
    def _extract_description(self, line: str, first_amount: float) -> str:
        """Extract descriptive text before the first amount."""
        
        # Find position of first amount
        amount_str = f"{first_amount:,.0f}"
        
        # Try different formats
        for fmt in [amount_str, f"${amount_str}", f"({amount_str})"]:
            pos = line.find(fmt)
            if pos > 0:
                return line[:pos]
        
        # Fallback: take first half of line
        return line[:len(line)//2]
    
    def _is_header_or_total(self, text: str) -> bool:
        """Check if line is a header or total line."""
        
        text_lower = text.lower()
        
        # Headers
        if any(h in text_lower for h in ['year ended', 'three months', 'quarter ended']):
            return True
        
        # Totals (but not "total operating expenses")
        total_patterns = [
            r'^total$',
            r'^net income',
            r'^income before',
            r'^operating income$'
        ]
        
        for pattern in total_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _classify_expense(self, line_item: Dict) -> ExpenseLineItem:
        """
        Classify expense line item into category.
        
        Uses hybrid approach:
        1. Keyword matching
        2. ML classification
        3. Heuristics
        """
        
        description = line_item['description'].lower()
        
        # Try keyword matching first (high precision)
        keyword_category = self._keyword_classify(description)
        if keyword_category:
            return ExpenseLineItem(
                description=line_item['description'],
                amount=line_item['amount'],
                category=keyword_category,
                confidence=0.9,
                source_text=line_item['raw_line']
            )
        
        # Try ML classification (broader coverage)
        if self.ml_classifier:
            ml_category, confidence = self._ml_classify(description)
            if confidence > 0.7:
                return ExpenseLineItem(
                    description=line_item['description'],
                    amount=line_item['amount'],
                    category=ml_category,
                    confidence=confidence,
                    source_text=line_item['raw_line']
                )
        
        # Fallback: heuristic rules
        heuristic_category = self._heuristic_classify(description)
        
        return ExpenseLineItem(
            description=line_item['description'],
            amount=line_item['amount'],
            category=heuristic_category,
            confidence=0.5,
            source_text=line_item['raw_line']
        )
    
    def _keyword_classify(self, description: str) -> str:
        """Classify using keyword matching."""
        
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in description:
                    return category
        
        return None
    
    def _ml_classify(self, description: str) -> Tuple[str, float]:
        """Classify using trained ML model."""
        
        # Extract features (TF-IDF, etc.)
        features = self._extract_text_features(description)
        
        # Predict
        proba = self.ml_classifier.predict_proba([features])[0]
        category_idx = proba.argmax()
        confidence = proba[category_idx]
        
        categories = list(self.category_keywords.keys())
        return categories[category_idx], confidence
    
    def _heuristic_classify(self, description: str) -> str:
        """Fallback heuristic classification."""
        
        # Check word patterns
        doc = self.nlp(description)
        
        # If contains "sales" or "marketing" → selling
        if any(token.text.lower() in ['sales', 'marketing'] for token in doc):
            return 'selling'
        
        # If contains "research" or "development" → R&D
        if any(token.text.lower() in ['research', 'development'] for token in doc):
            return 'research_dev'
        
        # Default: general & administrative
        return 'general_admin'
    
    def _aggregate_by_category(
        self,
        classified_items: List[ExpenseLineItem]
    ) -> Dict[str, float]:
        """Aggregate line items by category."""
        
        aggregated = {}
        
        for item in classified_items:
            category = item.category
            if category not in aggregated:
                aggregated[category] = 0
            
            aggregated[category] += item.amount
        
        # Calculate combined SG&A
        aggregated['sga'] = (
            aggregated.get('selling', 0) +
            aggregated.get('general_admin', 0)
        )
        
        aggregated['operating_expenses'] = sum(aggregated.values())
        
        return aggregated
    
    def _validate_classification(
        self,
        aggregated: Dict,
        filing_text: str
    ) -> Dict:
        """Validate extracted amounts against disclosed totals."""
        
        # Try to find disclosed "Total operating expenses"
        total_opex_pattern = r'Total operating expenses.*?\\$?([\\d,]+)'
        match = re.search(total_opex_pattern, filing_text, re.I)
        
        if match:
            disclosed_total = float(match.group(1).replace(',', ''))
            extracted_total = aggregated.get('operating_expenses', 0)
            
            diff_pct = abs(disclosed_total - extracted_total) / disclosed_total
            
            return {
                'validated': diff_pct < 0.05,  # Within 5%
                'disclosed_total': disclosed_total,
                'extracted_total': extracted_total,
                'difference_pct': diff_pct,
                'status': 'PASS' if diff_pct < 0.05 else 'FAIL'
            }
        
        return {'validated': False, 'status': 'NO_BENCHMARK'}
\`\`\`

**2. Training the ML Classifier**

\`\`\`python
def train_expense_classifier(training_data: pd.DataFrame):
    """
    Train ML model on manually labeled examples.
    
    training_data columns:
    - description: Expense line item description
    - category: Manually labeled category
    """
    
    # Vectorize descriptions
    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
        min_df=2
    )
    
    X = vectorizer.fit_transform(training_data['description'])
    y = training_data['category']
    
    # Train classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced'
    )
    
    clf.fit(X, y)
    
    # Evaluate
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"Cross-validated accuracy: {scores.mean():.2%}")
    
    return clf, vectorizer
\`\`\`

**3. Handling Edge Cases**

\`\`\`python
class EdgeCaseHandler:
    """Handle non-standard reporting formats."""
    
    def handle_combined_line_items(self, description: str, amount: float) -> List[ExpenseLineItem]:
        """
        Some companies combine categories:
        "Sales, general, and administrative expenses"
        
        Need to split if possible.
        """
        
        if 'sales' in description.lower() and 'administrative' in description.lower():
            # Combined SG&A - try to split based on industry averages
            # or just classify as 'sga' combined
            return [ExpenseLineItem(
                description=description,
                amount=amount,
                category='sga',
                confidence=0.8,
                source_text=description
            )]
        
        return []
    
    def handle_xbrl_data(self, xbrl_facts: Dict) -> Dict:
        """
        Extract from XBRL when available (more structured).
        XBRL has standard tags.
        """
        
        us_gaap = xbrl_facts.get('us-gaap', {})
        
        expenses = {}
        
        # Standard XBRL tags
        tag_mapping = {
            'SellingAndMarketingExpense': 'selling',
            'GeneralAndAdministrativeExpense': 'general_admin',
            'ResearchAndDevelopmentExpense': 'research_dev',
            'DepreciationAndAmortization': 'depreciation',
        }
        
        for xbrl_tag, category in tag_mapping.items():
            if xbrl_tag in us_gaap:
                values = us_gaap[xbrl_tag]['units']['USD']
                # Get most recent annual value
                annual = [v for v in values if v.get('form') == '10-K']
                if annual:
                    expenses[category] = annual[-1]['val']
        
        return expenses
\`\`\`

**4. Production System Design**

\`\`\`
┌─────────────┐
│  10-K HTML  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ XBRL Available? │────Yes────▶ Direct Extract (high confidence)
└────────┬────────┘
         No
         │
         ▼
┌─────────────────┐
│ Text Extraction │
│  (Income Stmt)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Line Item Parse │
└────────┬────────┘
         │
         ▼
    ┌───┴───┐
    │       │
    ▼       ▼
┌────────┐ ┌────────┐
│Keyword │ │   ML   │
│Matching│ │Classify│
└────┬───┘ └────┬───┘
     │          │
     └────┬─────┘
          │
          ▼
  ┌───────────────┐
  │  Aggregate    │
  │  & Validate   │
  └───────┬───────┘
          │
          ▼
    ┌─────────┐
    │Database │
    └─────────┘
\`\`\`

**Key Success Factors**:

1. **XBRL First**: Always check for structured XBRL data (90% of filings have it)
2. **Hybrid Approach**: Keywords + ML + Heuristics for best coverage
3. **Validation**: Always compare extracted totals to disclosed totals
4. **Confidence Scores**: Flag low-confidence classifications for review
5. **Continuous Learning**: Update training data with new patterns

**Normalization Challenges**:
- Company A: "Sales and marketing" = $100M
- Company B: "Selling expenses" = $80M + "Marketing" = $20M
- Company C: "SG&A" = $150M (combined, not broken out)

Solution: Normalize to common categories but preserve original structure in metadata for transparency.

This system can achieve ~95% accuracy on standard filings and ~75% on non-standard formats.`
        },

        {
            id: 3,
            question: "You're analyzing two companies: Company A (SaaS) has 80% gross margins, Company B (Hardware) has 35% gross margins. Company B's stock trades at a higher P/E ratio. Explain the economic factors that could justify this valuation difference, and build a framework to compare companies with fundamentally different business models based on their income statements.",
            sample_answer: `Comparing companies across different business models requires understanding the **economic drivers** behind margins and translating them into comparable metrics:

**1. Why Gross Margin Differences Don't Tell the Full Story**

\`\`\`python
def analyze_business_economics(company_a_saas: Dict, company_b_hardware: Dict):
    """
    Deep comparison of business economics beyond gross margin.
    """
    
    print("Company A (SaaS) vs Company B (Hardware)")
    print("=" * 60)
    
    # Gross margin (what everyone looks at)
    print(f"\\nGross Margin:")
    print(f"  Company A: {company_a_saas['gross_margin']:.1%}")
    print(f"  Company B: {company_b_hardware['gross_margin']:.1%}")
    print(f"  → Company A looks better, but...")
    
    # Operating margin (more important!)
    print(f"\\nOperating Margin:")
    print(f"  Company A: {company_a_saas['operating_margin']:.1%}")
    print(f"  Company B: {company_b_hardware['operating_margin']:.1%}")
    
    # Return on Invested Capital (ultimate metric)
    roic_a = company_a_saas['nopat'] / company_a_saas['invested_capital']
    roic_b = company_b_hardware['nopat'] / company_b_hardware['invested_capital']
    
    print(f"\\nReturn on Invested Capital:")
    print(f"  Company A: {roic_a:.1%}")
    print(f"  Company B: {roic_b:.1%}")
    
    # Free cash flow conversion
    fcf_margin_a = company_a_saas['fcf'] / company_a_saas['revenue']
    fcf_margin_b = company_b_hardware['fcf'] / company_b_hardware['revenue']
    
    print(f"\\nFree Cash Flow Margin:")
    print(f"  Company A: {fcf_margin_a:.1%}")
    print(f"  Company B: {fcf_margin_b:.1%}")

# Example: Why hardware might deserve higher P/E
saas_company = {
    'revenue': 1000,
    'gross_margin': 0.80,  # 80%
    'operating_margin': 0.15,  # But 60% goes to S&M and R&D!
    'nopat': 120,
    'invested_capital': 500,
    'fcf': 100,
    'growth_rate': 0.30  # 30% growth
}

hardware_company = {
    'revenue': 10000,  # 10x larger
    'gross_margin': 0.35,  # 35%
    'operating_margin': 0.20,  # Lower OpEx as % of revenue!
    'nopat': 1500,
    'invested_capital': 5000,
    'fcf': 1800,  # Higher FCF due to scale
    'growth_rate': 0.10  # 10% growth
}

analyze_business_economics(saas_company, hardware_company)
\`\`\`

**Output**:
\`\`\`
Gross Margin:
  Company A: 80.0%
  Company B: 35.0%
  → Company A looks better, but...

Operating Margin:
  Company A: 15.0%  ← High gross margin, but massive S&M and R&D spend
  Company B: 20.0%  ← Lower gross margin, but efficient operations

ROIC:
  Company A: 24.0%
  Company B: 30.0%  ← Hardware generates MORE return on capital!

FCF Margin:
  Company A: 10.0%
  Company B: 18.0%  ← Hardware generates MORE cash
\`\`\`

**2. Economic Factors Justifying Different Valuations**

\`\`\`python
class BusinessModelComparator:
    """Framework for comparing different business models."""
    
    def __init__(self):
        self.valuation_drivers = {
            'growth_rate': 0.30,
            'roic': 0.25,
            'cash_generation': 0.20,
            'profitability': 0.15,
            'capital_efficiency': 0.10
        }
    
    def calculate_quality_score(self, company: Dict) -> float:
        """
        Comprehensive quality score accounting for business model differences.
        """
        
        scores = {}
        
        # 1. Growth Quality (higher for young companies)
        scores['growth'] = min(company['revenue_growth'] / 0.30, 1.0)
        
        # 2. Return on Invested Capital (ultimate measure)
        roic = company['nopat'] / company['invested_capital']
        scores['roic'] = min(roic / 0.25, 1.0)  # 25% ROIC = perfect score
        
        # 3. Cash Generation
        fcf_margin = company['fcf'] / company['revenue']
        scores['cash_generation'] = min(fcf_margin / 0.20, 1.0)
        
        # 4. Profitability
        scores['profitability'] = company['operating_margin'] / 0.25
        
        # 5. Capital Efficiency (asset-light is better)
        capital_intensity = company['invested_capital'] / company['revenue']
        scores['capital_efficiency'] = max(1 - capital_intensity, 0)
        
        # Weighted score
        total_score = sum(
            scores[metric] * weight
            for metric, weight in self.valuation_drivers.items()
        )
        
        return total_score * 100  # 0-100 scale
    
    def justify_valuation_difference(
        self,
        company_a: Dict,
        company_b: Dict,
        pe_a: float,
        pe_b: float
    ) -> Dict:
        """
        Explain why company B might have higher P/E despite lower margins.
        """
        
        analysis = {}
        
        # Factor 1: Growth Rates
        peg_a = pe_a / (company_a['revenue_growth'] * 100)
        peg_b = pe_b / (company_b['revenue_growth'] * 100)
        
        analysis['peg_ratio'] = {
            'company_a': peg_a,
            'company_b': peg_b,
            'insight': 'Lower PEG = cheaper relative to growth' if peg_b < peg_a else 'Higher PEG = expensive for growth'
        }
        
        # Factor 2: Profitability Stability
        # Hardware: predictable, stable (lower risk = higher multiple)
        # SaaS: volatile growth investments (higher risk = lower multiple)
        
        ebitda_margin_a = company_a['ebitda'] / company_a['revenue']
        ebitda_margin_b = company_b['ebitda'] / company_b['revenue']
        
        analysis['stability'] = {
            'ebitda_margin_a': ebitda_margin_a,
            'ebitda_margin_b': ebitda_margin_b,
            'insight': 'More stable earnings deserve higher multiple'
        }
        
        # Factor 3: Capital Requirements
        # SaaS: needs constant R&D investment
        # Hardware: more mature, lower reinvestment needs
        
        reinvestment_rate_a = (company_a['capex'] + company_a['rd']) / company_a['nopat']
        reinvestment_rate_b = (company_b['capex'] + company_b['rd']) / company_b['nopat']
        
        analysis['capital_intensity'] = {
            'reinvestment_a': reinvestment_rate_a,
            'reinvestment_b': reinvestment_rate_b,
            'insight': 'Lower reinvestment needs = more FCF to shareholders = higher multiple'
        }
        
        # Factor 4: Market Size and TAM
        analysis['market_opportunity'] = {
            'insight': 'If Hardware has larger TAM or better competitive moat, justifies premium'
        }
        
        # Factor 5: Business Model Maturity
        analysis['maturity'] = {
            'saas': 'High growth but burning cash on S&M (land grab phase)',
            'hardware': 'Mature, generating cash, returning to shareholders',
            'insight': 'Mature FCF businesses often trade at higher multiples than growth-stage'
        }
        
        return analysis

# Real example: Nvidia (hardware) vs Snowflake (SaaS)
nvidia = {
    'revenue': 60_000_000_000,
    'gross_margin': 0.65,  # Actually high for hardware!
    'operating_margin': 0.32,
    'nopat': 16_000_000_000,
    'invested_capital': 30_000_000_000,
    'fcf': 20_000_000_000,
    'revenue_growth': 0.60,  # 60% growth (AI boom)
    'ebitda': 22_000_000_000,
    'capex': 1_000_000_000,
    'rd': 7_000_000_000
}

snowflake = {
    'revenue': 2_000_000_000,
    'gross_margin': 0.67,
    'operating_margin': -0.05,  # Negative! (investing heavily)
    'nopat': -100_000_000,
    'invested_capital': 5_000_000_000,
    'fcf': -200_000_000,  # Negative FCF
    'revenue_growth': 0.40,  # 40% growth
    'ebitda': 0,
    'capex': 200_000_000,
    'rd': 500_000_000
}

comparator = BusinessModelComparator()

quality_nvidia = comparator.calculate_quality_score(nvidia)
quality_snowflake = comparator.calculate_quality_score(snowflake)

print(f"Nvidia Quality Score: {quality_nvidia:.1f}/100")
print(f"Snowflake Quality Score: {quality_snowflake:.1f}/100")
\`\`\`

**3. Normalized Comparison Framework**

\`\`\`python
def create_normalized_comparison(companies: List[Dict]) -> pd.DataFrame:
    """
    Create apples-to-apples comparison across business models.
    """
    
    comparison = []
    
    for company in companies:
        # Normalize all metrics to 0-100 scale
        normalized = {
            'Company': company['name'],
            'Business Model': company['model'],
            
            # Profitability (normalized)
            'Gross Margin Score': min(company['gross_margin'] / 0.80, 1.0) * 100,
            'Operating Margin Score': min(company['operating_margin'] / 0.30, 1.0) * 100,
            'FCF Margin Score': min(company['fcf_margin'] / 0.25, 1.0) * 100,
            
            # Growth (normalized)
            'Growth Score': min(company['revenue_growth'] / 0.50, 1.0) * 100,
            
            # Returns (normalized)
            'ROIC Score': min(company['roic'] / 0.35, 1.0) * 100,
            'ROE Score': min(company['roe'] / 0.25, 1.0) * 100,
            
            # Efficiency
            'Capital Efficiency Score': (1 - min(company['capital_intensity'], 1.0)) * 100,
            'Asset Turnover Score': min(company['asset_turnover'] / 2.0, 1.0) * 100,
        }
        
        # Composite score
        normalized['Overall Score'] = np.mean([
            normalized['ROIC Score'] * 0.30,
            normalized['Growth Score'] * 0.25,
            normalized['FCF Margin Score'] * 0.20,
            normalized['Operating Margin Score'] * 0.15,
            normalized['Capital Efficiency Score'] * 0.10
        ])
        
        comparison.append(normalized)
    
    return pd.DataFrame(comparison)

# Usage
companies = [
    {
        'name': 'Microsoft',
        'model': 'Software',
        'gross_margin': 0.69,
        'operating_margin': 0.42,
        'fcf_margin': 0.38,
        'revenue_growth': 0.18,
        'roic': 0.35,
        'roe': 0.40,
        'capital_intensity': 0.30,
        'asset_turnover': 0.60
    },
    {
        'name': 'Apple',
        'model': 'Hardware',
        'gross_margin': 0.43,
        'operating_margin': 0.30,
        'fcf_margin': 0.28,
        'revenue_growth': 0.08,
        'roic': 0.45,
        'roe': 1.50,  # High due to buybacks reducing equity
        'capital_intensity': 0.12,
        'asset_turnover': 1.10
    },
    {
        'name': 'Amazon',
        'model': 'Retail + Cloud',
        'gross_margin': 0.48,
        'operating_margin': 0.08,
        'fcf_margin': 0.09,
        'revenue_growth': 0.12,
        'roic': 0.13,
        'roe': 0.11,
        'capital_intensity': 0.45,
        'asset_turnover': 1.15
    }
]

comparison_df = create_normalized_comparison(companies)
print(comparison_df.to_string())
\`\`\`

**4. Key Insights**

**Why Hardware Might Trade at Higher P/E**:

1. **Cash Generation**: Mature hardware businesses generate more cash relative to earnings (high FCF conversion)

2. **Predictability**: Established products with stable demand (lower risk premium)

3. **Capital Efficiency**: Apple has ROIC of 45% despite lower gross margins (efficiency matters more than margins)

4. **Shareholder Returns**: Mature companies return cash via dividends/buybacks (bird in hand)

5. **Market Position**: Dominant market share with high switching costs (moats)

**When to Prefer SaaS Despite Lower P/E**:

1. **Growth Trajectory**: If SaaS is growing 50%+ while hardware is 5-10%

2. **Operating Leverage**: Margins will expand as SaaS scales (40% to 80% is plausible)

3. **Market Opportunity**: Larger addressable market ahead

4. **Switching Costs**: SaaS often has higher retention than hardware

**The Right Framework**:

Don't compare P/E ratios directly. Instead:
- Compare PEG ratios (P/E relative to growth)
- Compare EV/FCF (enterprise value to free cash flow)
- Compare ROIC-adjusted valuations
- Use DCF models with appropriate growth assumptions

**Production Implementation**:
\`\`\`python
# Automated comparable company analysis accounting for business model differences
def find_true_comparables(target_company: Dict, universe: List[Dict]) -> List[Dict]:
    """Find companies with similar economics, not just similar industries."""
    
    scores = []
    
    for comp in universe:
        similarity_score = 0
        
        # Similar ROIC (±5%)
        if abs(comp['roic'] - target_company['roic']) < 0.05:
            similarity_score += 30
        
        # Similar growth (±10%)
        if abs(comp['growth'] - target_company['growth']) < 0.10:
            similarity_score += 25
        
        # Similar FCF margin (±10%)
        if abs(comp['fcf_margin'] - target_company['fcf_margin']) < 0.10:
            similarity_score += 25
        
        # Similar capital intensity (±20%)
        if abs(comp['capital_intensity'] - target_company['capital_intensity']) < 0.20:
            similarity_score += 20
        
        scores.append((comp, similarity_score))
    
    # Return top 10 most similar
    return sorted(scores, key=lambda x: x[1], reverse=True)[:10]
\`\`\`

**Bottom Line**: P/E ratios are meaningless without context. A hardware company at 30x P/E generating 20% ROIC might be cheaper than a SaaS company at 20x P/E with 15% ROIC and negative FCF. Always look at ROIC, FCF, and growth-adjusted metrics.`
        }
    ]
};

