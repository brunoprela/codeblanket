export const section7 = {
  title: 'Financial Statement Quality & Red Flags',
  content: `
# Financial Statement Quality & Red Flags

Learn to detect earnings manipulation, accounting fraud, and low-quality financials before investing.

## Section 1: Earnings Quality Framework

\`\`\`python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class EarningsQualityAnalyzer:
    """Comprehensive earnings quality assessment."""
    
    net_income: float
    cfo: float
    revenue: float
    total_accruals: float
    discretionary_accruals: float
    accounts_receivable: float
    inventory: float
    
    def cfo_to_ni_ratio (self) -> float:
        """CFO / Net Income - should be >1.0 for high quality."""
        return self.cfo / self.net_income if self.net_income != 0 else 0
    
    def accruals_ratio (self) -> float:
        """Total Accruals / Net Income - lower is better."""
        return abs (self.total_accruals) / abs (self.net_income) if self.net_income != 0 else 0
    
    def quality_score (self) -> Dict:
        """Calculate composite quality score (0-100)."""
        
        score = 100
        flags = []
        
        # Test 1: CFO/NI ratio
        cfo_ni = self.cfo_to_ni_ratio()
        if cfo_ni < 0.8:
            score -= 30
            flags.append("CFO < 80% of NI")
        elif cfo_ni < 1.0:
            score -= 15
            flags.append("CFO < NI")
        
        # Test 2: Accruals ratio
        accr_ratio = self.accruals_ratio()
        if accr_ratio > 0.15:
            score -= 25
            flags.append("High accruals (>15% of NI)")
        
        # Test 3: Working capital quality
        if self.accounts_receivable / self.revenue > 0.20:
            score -= 15
            flags.append("High AR/Revenue ratio")
        
        # Assessment
        if score >= 80:
            assessment = "HIGH QUALITY"
        elif score >= 60:
            assessment = "MODERATE QUALITY"
        elif score >= 40:
            assessment = "LOW QUALITY"
        else:
            assessment = "VERY POOR QUALITY"
        
        return {
            'score': score,
            'assessment': assessment,
            'cfo_ni_ratio': cfo_ni,
            'accruals_ratio': accr_ratio,
            'red_flags': flags
        }

# Example: High quality company
high_quality = EarningsQualityAnalyzer(
    net_income=100_000_000,
    cfo=120_000_000,  # CFO > NI ✓
    revenue=500_000_000,
    total_accruals=20_000_000,  # Low accruals
    discretionary_accruals=5_000_000,
    accounts_receivable=80_000_000,  # 16% of revenue
    inventory=50_000_000
)

print("Earnings Quality Analysis:")
print(high_quality.quality_score())
\`\`\`

## Section 2: Beneish M-Score (Fraud Detection)

\`\`\`python
class BeneishMScore:
    """
    Beneish M-Score: Probability of earnings manipulation.
    M-Score > -2.22 suggests high probability of manipulation.
    """
    
    @staticmethod
    def calculate (current: Dict, prior: Dict) -> Dict:
        """Calculate all 8 Beneish variables."""
        
        # DSRI: Days Sales in Receivables Index
        dsri = (current['ar'] / current['revenue']) / (prior['ar'] / prior['revenue'])
        
        # GMI: Gross Margin Index
        gm_prior = (prior['revenue'] - prior['cogs']) / prior['revenue']
        gm_current = (current['revenue'] - current['cogs']) / current['revenue']
        gmi = gm_prior / gm_current
        
        # AQI: Asset Quality Index
        aqi = (1 - (current['current_assets'] + current['ppe']) / current['total_assets']) / \
              (1 - (prior['current_assets'] + prior['ppe']) / prior['total_assets'])
        
        # SGI: Sales Growth Index
        sgi = current['revenue'] / prior['revenue']
        
        # DEPI: Depreciation Index
        depi_prior = prior['depreciation'] / (prior['ppe'] + prior['depreciation'])
        depi_current = current['depreciation'] / (current['ppe'] + current['depreciation'])
        depi = depi_prior / depi_current
        
        # SGAI: Sales, General & Admin expenses Index
        sgai = (current['sga'] / current['revenue']) / (prior['sga'] / prior['revenue'])
        
        # LVGI: Leverage Index
        lvgi = (current['total_debt'] / current['total_assets']) / \
               (prior['total_debt'] / prior['total_assets'])
        
        # TATA: Total Accruals to Total Assets
        tata = (current['net_income'] - current['cfo']) / current['total_assets']
        
        # M-Score formula
        m_score = (-4.84 + 
                   0.920 * dsri +
                   0.528 * gmi +
                   0.404 * aqi +
                   0.892 * sgi +
                   0.115 * depi -
                   0.172 * sgai +
                   4.679 * tata -
                   0.327 * lvgi)
        
        # Interpret
        if m_score > -2.22:
            risk = "HIGH - Likely manipulator"
        elif m_score > -2.50:
            risk = "MODERATE - Monitor closely"
        else:
            risk = "LOW - Appears clean"
        
        return {
            'm_score': m_score,
            'risk_level': risk,
            'components': {
                'DSRI': dsri,
                'GMI': gmi,
                'AQI': aqi,
                'SGI': sgi,
                'DEPI': depi,
                'SGAI': sgai,
                'LVGI': lvgi,
                'TATA': tata
            }
        }

# Example calculation
current_year = {
    'ar': 150_000_000, 'revenue': 500_000_000, 'cogs': 300_000_000,
    'current_assets': 250_000_000, 'ppe': 200_000_000, 
    'total_assets': 600_000_000, 'depreciation': 20_000_000,
    'sga': 100_000_000, 'total_debt': 200_000_000,
    'net_income': 50_000_000, 'cfo': 60_000_000
}

prior_year = {
    'ar': 100_000_000, 'revenue': 400_000_000, 'cogs': 240_000_000,
    'current_assets': 200_000_000, 'ppe': 180_000_000,
    'total_assets': 500_000_000, 'depreciation': 18_000_000,
    'sga': 80_000_000, 'total_debt': 150_000_000,
    'net_income': 45_000_000, 'cfo': 50_000_000
}

beneish = BeneishMScore.calculate (current_year, prior_year)
print(f"\\nBeneish M-Score: {beneish['m_score']:.2f}")
print(f"Risk Level: {beneish['risk_level']}")
\`\`\`

## Section 3: Altman Z-Score (Bankruptcy Prediction)

\`\`\`python
class AltmanZScore:
    """
    Altman Z-Score: Predicts bankruptcy risk.
    Z > 2.99: Safe zone
    1.81 < Z < 2.99: Grey zone
    Z < 1.81: Distress zone
    """
    
    @staticmethod
    def calculate (balance_sheet: Dict, income_statement: Dict, market_data: Dict) -> Dict:
        """Calculate Z-Score for public manufacturing companies."""
        
        # X1: Working Capital / Total Assets
        wc = balance_sheet['current_assets'] - balance_sheet['current_liabilities']
        x1 = wc / balance_sheet['total_assets']
        
        # X2: Retained Earnings / Total Assets
        x2 = balance_sheet['retained_earnings'] / balance_sheet['total_assets']
        
        # X3: EBIT / Total Assets
        x3 = income_statement['ebit'] / balance_sheet['total_assets']
        
        # X4: Market Value of Equity / Book Value of Total Liabilities
        x4 = market_data['market_cap'] / balance_sheet['total_liabilities']
        
        # X5: Sales / Total Assets
        x5 = income_statement['revenue'] / balance_sheet['total_assets']
        
        # Z-Score formula
        z_score = (1.2 * x1 +
                   1.4 * x2 +
                   3.3 * x3 +
                   0.6 * x4 +
                   1.0 * x5)
        
        # Interpret
        if z_score > 2.99:
            zone = "SAFE - Low bankruptcy risk"
        elif z_score > 1.81:
            zone = "GREY - Monitor closely"
        else:
            zone = "DISTRESS - High bankruptcy risk"
        
        return {
            'z_score': z_score,
            'zone': zone,
            'components': {'X1': x1, 'X2': x2, 'X3': x3, 'X4': x4, 'X5': x5}
        }
\`\`\`

## Section 4: Piotroski F-Score (Value Quality)

\`\`\`python
class PiotroskiFScore:
    """
    Piotroski F-Score: Quality score for value stocks (0-9).
    Score ≥7: High quality
    Score 4-6: Medium quality
    Score ≤3: Low quality
    """
    
    @staticmethod
    def calculate (current: Dict, prior: Dict) -> Dict:
        """Calculate 9-point F-Score."""
        
        score = 0
        details = {}
        
        # Profitability (4 points)
        # 1. Positive net income
        if current['net_income'] > 0:
            score += 1
            details['ROA_positive'] = True
        
        # 2. Positive operating cash flow
        if current['cfo'] > 0:
            score += 1
            details['CFO_positive'] = True
        
        # 3. ROA increased
        roa_current = current['net_income'] / current['total_assets']
        roa_prior = prior['net_income'] / prior['total_assets']
        if roa_current > roa_prior:
            score += 1
            details['ROA_improving'] = True
        
        # 4. Quality of earnings (CFO > NI)
        if current['cfo'] > current['net_income']:
            score += 1
            details['Quality_earnings'] = True
        
        # Leverage, Liquidity, Source of Funds (3 points)
        # 5. Decreased long-term debt
        if current['long_term_debt'] < prior['long_term_debt']:
            score += 1
            details['Leverage_decreasing'] = True
        
        # 6. Increased current ratio
        cr_current = current['current_assets'] / current['current_liabilities']
        cr_prior = prior['current_assets'] / prior['current_liabilities']
        if cr_current > cr_prior:
            score += 1
            details['Liquidity_improving'] = True
        
        # 7. No new shares issued
        if current['shares_outstanding'] <= prior['shares_outstanding']:
            score += 1
            details['No_dilution'] = True
        
        # Operating Efficiency (2 points)
        # 8. Increased gross margin
        gm_current = current['gross_profit'] / current['revenue']
        gm_prior = prior['gross_profit'] / prior['revenue']
        if gm_current > gm_prior:
            score += 1
            details['Margin_improving'] = True
        
        # 9. Increased asset turnover
        at_current = current['revenue'] / current['total_assets']
        at_prior = prior['revenue'] / prior['total_assets']
        if at_current > at_prior:
            score += 1
            details['Efficiency_improving'] = True
        
        # Overall assessment
        if score >= 7:
            quality = "HIGH QUALITY - Strong fundamentals"
        elif score >= 4:
            quality = "MEDIUM QUALITY - Mixed signals"
        else:
            quality = "LOW QUALITY - Weak fundamentals"
        
        return {
            'f_score': score,
            'quality': quality,
            'details': details
        }
\`\`\`

## Section 5: Common Accounting Red Flags

\`\`\`python
class RedFlagDetector:
    """Detect common financial statement manipulation tactics."""
    
    @staticmethod
    def check_channel_stuffing (ar_growth: float, revenue_growth: float, 
                               inventory_growth: float) -> Dict:
        """Detect revenue manipulation via channel stuffing."""
        
        flags = []
        score = 0
        
        # Red flag 1: AR growing faster than revenue
        if ar_growth > revenue_growth * 1.2:
            flags.append (f"AR growth ({ar_growth:.1%}) >> Revenue growth ({revenue_growth:.1%})")
            score += 30
        
        # Red flag 2: Inventory also building
        if inventory_growth > revenue_growth * 1.2:
            flags.append (f"Inventory growth ({inventory_growth:.1%}) >> Revenue growth")
            score += 20
        
        # Red flag 3: Both AR and Inventory accelerating
        if ar_growth > 0.15 and inventory_growth > 0.15 and revenue_growth < 0.10:
            flags.append("AR + Inventory building despite slow revenue growth")
            score += 30
        
        if score > 50:
            assessment = "HIGH RISK - Likely channel stuffing"
        elif score > 25:
            assessment = "MODERATE RISK - Monitor closely"
        else:
            assessment = "LOW RISK"
        
        return {
            'risk_score': score,
            'assessment': assessment,
            'red_flags': flags
        }
    
    @staticmethod
    def check_reserve_manipulation (current_reserves: Dict, 
                                   prior_reserves: Dict,
                                   revenue: float) -> List[str]:
        """Detect cookie jar reserve manipulation."""
        
        flags = []
        
        # Bad debt reserve declining despite growing AR
        if (current_reserves['bad_debt'] / current_reserves['ar'] < 
            prior_reserves['bad_debt'] / prior_reserves['ar']):
            flags.append("Bad debt reserve % declining (potential earnings boost)")
        
        # Warranty reserve declining despite revenue growth
        if (current_reserves['warranty'] < prior_reserves['warranty'] and 
            revenue > prior_reserves.get('prior_revenue', revenue)):
            flags.append("Warranty reserve declining despite revenue growth")
        
        return flags
    
    @staticmethod
    def check_one_time_items (income_statement_history: List[Dict]) -> Dict:
        """Detect if 'one-time' items are actually recurring."""
        
        one_time_count = sum(1 for stmt in income_statement_history 
                             if stmt.get('restructuring_charges', 0) > 0)
        
        if one_time_count >= 3:
            return {
                'alert': f"'One-time' charges in {one_time_count} of last {len (income_statement_history)} years",
                'message': "These are recurring, not one-time!"
            }
        
        return {'alert': None}

detector = RedFlagDetector()
channel_stuffing_risk = detector.check_channel_stuffing(
    ar_growth=0.25,  # 25%
    revenue_growth=0.10,  # 10%
    inventory_growth=0.20  # 20%
)

print("\\nChannel Stuffing Analysis:")
print(f"Risk Score: {channel_stuffing_risk['risk_score']}/100")
print(f"Assessment: {channel_stuffing_risk['assessment']}")
\`\`\`

## Section 6: Comprehensive Quality Checklist

\`\`\`python
class ComprehensiveQualityCheck:
    """Run all quality tests on a company."""
    
    def __init__(self, current_data: Dict, prior_data: Dict):
        self.current = current_data
        self.prior = prior_data
        self.results = {}
    
    def run_all_tests (self) -> pd.DataFrame:
        """Execute full quality assessment."""
        
        tests = [
            ('Earnings Quality', self.test_earnings_quality()),
            ('Beneish M-Score', self.test_beneish()),
            ('Altman Z-Score', self.test_altman()),
            ('Piotroski F-Score', self.test_piotroski()),
            ('Channel Stuffing', self.test_channel_stuffing()),
            ('Reserve Quality', self.test_reserves()),
            ('Cash Flow Quality', self.test_cash_flow())
        ]
        
        results = []
        for test_name, result in tests:
            results.append({
                'Test': test_name,
                'Score': result.get('score', 'N/A'),
                'Result': result.get('result', 'N/A'),
                'Flags': len (result.get('flags', []))
            })
        
        return pd.DataFrame (results)
    
    def test_earnings_quality (self) -> Dict:
        analyzer = EarningsQualityAnalyzer(
            net_income=self.current['net_income'],
            cfo=self.current['cfo'],
            revenue=self.current['revenue'],
            total_accruals=self.current['accruals'],
            discretionary_accruals=0,
            accounts_receivable=self.current['ar'],
            inventory=self.current['inventory']
        )
        return analyzer.quality_score()
    
    # Additional test methods...
    
    def generate_report (self) -> str:
        """Generate summary report."""
        df = self.run_all_tests()
        
        report = f"""
        FINANCIAL STATEMENT QUALITY REPORT
        {'='*60}
        
        {df.to_string (index=False)}
        
        OVERALL ASSESSMENT:
        """
        
        # Calculate overall score
        passing_tests = len (df[df['Result'].str.contains('PASS|HIGH|SAFE')])
        total_tests = len (df)
        
        if passing_tests / total_tests > 0.75:
            report += "\\n✓ HIGH QUALITY - Financials appear clean"
        elif passing_tests / total_tests > 0.50:
            report += "\\n⚠ MODERATE QUALITY - Some concerns"
        else:
            report += "\\n✗ LOW QUALITY - Significant red flags"
        
        return report
\`\`\`

## Key Takeaways

1. **No single metric is perfect** - Use multiple models (Beneish, Altman, Piotroski)
2. **CFO > NI is critical** - Primary earnings quality indicator
3. **Watch accruals** - High accruals often precede restatements
4. **AR/Inventory trends** - Growing faster than revenue = red flag
5. **One-time items** - If recurring for 3+ years, not "one-time"
6. **Reserve manipulation** - Cookie jar accounting inflates earnings
7. **Historical patterns** - Most frauds show multiple red flags

Use this framework to avoid accounting landmines before they detonate!
`,
  discussionQuestions: [],
  multipleChoiceQuestions: [],
};
