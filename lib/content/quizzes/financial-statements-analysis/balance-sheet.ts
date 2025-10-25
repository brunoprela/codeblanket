export const balanceSheetDiscussionQuestions = [
  {
    id: 1,
    question:
      "Design an automated system to detect 'zombie companies' (companies that can't cover their interest expenses from operating income) by analyzing balance sheets and income statements. What metrics would you track, what thresholds would trigger alerts, and how would you distinguish between temporary distress and structural insolvency? Consider that some viable turnaround candidates may temporarily appear as zombies.",
    answer: `A production-grade zombie company detection system requires multiple layers of analysis to avoid false positives while catching genuine insolvency risks:

**1. Core Detection Framework**

\`\`\`python
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import numpy as np

@dataclass
class ZombieMetrics:
    """Metrics for zombie company detection."""
    company_id: str
    period: str
    interest_coverage: float
    debt_to_ebitda: float
    current_ratio: float
    cash_burn_months: float
    debt_service_coverage: float
    zombie_score: float
    classification: str

class ZombieCompanyDetector:
    """Detect companies unable to service debt from operations."""
    
    # Thresholds based on academic research and practical experience
    THRESHOLDS = {
        'interest_coverage_critical': 1.0,    # Cannot cover interest
        'interest_coverage_zombie': 1.5,      # Barely covering
        'interest_coverage_distressed': 2.5,  # Weak coverage
        'debt_to_ebitda_high': 6.0,          # Excessive leverage
        'current_ratio_low': 0.9,             # Liquidity crisis
        'cash_burn_critical': 6,              # <6 months runway
    }
    
    def __init__(self):
        self.historical_window = 8  # 8 quarters of history
    
    def analyze_company(
        self,
        company_id: str,
        financials: pd.DataFrame
    ) -> ZombieMetrics:
        """
        Comprehensive zombie analysis for a single company.
        
        financials DataFrame should have:
        - period, operating_income, interest_expense, ebitda,
          total_debt, current_assets, current_liabilities,
          cash, operating_cf, capex
        """
        
        latest = financials.iloc[-1]
        
        # Core zombie metric: Interest Coverage
        interest_coverage = self._calculate_interest_coverage(latest)
        
        # Supporting metrics
        debt_to_ebitda = self._calculate_debt_to_ebitda(latest)
        current_ratio = self._calculate_current_ratio(latest)
        cash_burn_months = self._calculate_cash_runway(latest, financials)
        debt_service_coverage = self._calculate_debt_service_coverage(latest)
        
        # Zombie score (0-100, higher = more zombie-like)
        zombie_score = self._calculate_zombie_score(
            interest_coverage,
            debt_to_ebitda,
            current_ratio,
            cash_burn_months,
            debt_service_coverage
        )
        
        # Classification
        classification = self._classify_company(
            zombie_score,
            interest_coverage,
            financials
        )
        
        return ZombieMetrics(
            company_id=company_id,
            period=latest['period'],
            interest_coverage=interest_coverage,
            debt_to_ebitda=debt_to_ebitda,
            current_ratio=current_ratio,
            cash_burn_months=cash_burn_months,
            debt_service_coverage=debt_service_coverage,
            zombie_score=zombie_score,
            classification=classification
        )
    
    def _calculate_interest_coverage(self, period_data: pd.Series) -> float:
        """
        Interest Coverage = Operating Income / Interest Expense
        
        <1.0: Cannot cover interest (zombie)
        1.0-1.5: Barely covering (zombie territory)
        1.5-2.5: Weak but potentially viable
        >2.5: Healthy
        """
        operating_income = period_data['operating_income']
        interest_expense = period_data['interest_expense']
        
        if interest_expense <= 0:
            return float('inf')  # No debt, not a zombie
        
        return operating_income / interest_expense
    
    def _calculate_debt_to_ebitda(self, period_data: pd.Series) -> float:
        """
        Debt / EBITDA ratio.
        
        >6x: Very high leverage
        4-6x: Elevated leverage
        <4x: Manageable
        """
        total_debt = period_data['total_debt']
        ebitda = period_data['ebitda']
        
        if ebitda <= 0:
            return float('inf')  # Negative EBITDA = major problem
        
        return total_debt / ebitda
    
    def _calculate_current_ratio(self, period_data: pd.Series) -> float:
        """Current Assets / Current Liabilities."""
        return period_data['current_assets'] / period_data['current_liabilities']
    
    def _calculate_cash_runway(
        self,
        period_data: pd.Series,
        historical: pd.DataFrame
    ) -> float:
        """
        Calculate months of cash runway.
        
        Cash / Average Monthly Cash Burn
        """
        cash = period_data['cash']
        
        # Calculate average quarterly cash burn
        recent_quarters = historical.tail(4)
        cash_burn_per_quarter = -(
            recent_quarters['operating_cf'].mean() +
            recent_quarters['capex'].mean()
        )
        
        if cash_burn_per_quarter <= 0:
            return float('inf')  # Generating cash, not burning
        
        monthly_burn = cash_burn_per_quarter / 3
        return cash / monthly_burn
    
    def _calculate_debt_service_coverage(self, period_data: pd.Series) -> float:
        """
        Operating Cash Flow / (Interest + Principal Payments)
        
        <1.0: Cannot service debt
        1.0-1.2: Barely covering
        >1.5: Healthy
        """
        operating_cf = period_data['operating_cf']
        interest_paid = period_data['interest_expense']
        principal_payments = period_data.get('debt_repayments', 0)
        
        total_debt_service = interest_paid + principal_payments
        
        if total_debt_service <= 0:
            return float('inf')
        
        return operating_cf / total_debt_service
    
    def _calculate_zombie_score(
        self,
        interest_coverage: float,
        debt_to_ebitda: float,
        current_ratio: float,
        cash_burn_months: float,
        debt_service_coverage: float
    ) -> float:
        """
        Composite zombie score (0-100).
        
        Higher score = more zombie-like
        """
        score = 0
        
        # Interest coverage component (40% weight)
        if interest_coverage < 1.0:
            score += 40
        elif interest_coverage < 1.5:
            score += 30
        elif interest_coverage < 2.5:
            score += 15
        
        # Leverage component (25% weight)
        if debt_to_ebitda > 6.0:
            score += 25
        elif debt_to_ebitda > 4.0:
            score += 15
        
        # Liquidity component (20% weight)
        if current_ratio < 0.9:
            score += 20
        elif current_ratio < 1.2:
            score += 10
        
        # Cash runway component (15% weight)
        if cash_burn_months < 6:
            score += 15
        elif cash_burn_months < 12:
            score += 8
        
        return min(score, 100)
    
    def _classify_company(
        self,
        zombie_score: float,
        interest_coverage: float,
        historical: pd.DataFrame
    ) -> str:
        """
        Classify company with nuance.
        
        Distinguish between:
        - Permanent zombies (structural issues)
        - Temporary distress (cyclical, turnaround candidates)
        - Healthy companies
        """
        
        # Check for improving trends
        improving = self._is_improving(historical)
        
        # Classification logic
        if zombie_score > 60:
            if improving:
                return "DISTRESSED_BUT_IMPROVING"
            else:
                return "ZOMBIE"
        elif zombie_score > 40:
            if interest_coverage < 1.0:
                return "ZOMBIE"
            elif improving:
                return "WATCH_LIST_IMPROVING"
            else:
                return "WATCH_LIST"
        elif zombie_score > 20:
            return "WEAK_BUT_VIABLE"
        else:
            return "HEALTHY"
    
    def _is_improving(self, historical: pd.DataFrame) -> bool:
        """
        Detect if company is improving or deteriorating.
        
        Look at:
        - Interest coverage trend
        - Debt reduction
        - Improving EBITDA
        """
        if len(historical) < 4:
            return False
        
        recent = historical.tail(4)
        
        # Calculate interest coverage for each period
        ic_trend = (
            recent['operating_income'] / recent['interest_expense']
        ).values
        
        # Check if trending upward
        ic_improving = ic_trend[-1] > ic_trend[0]
        
        # Check debt reduction
        debt_trend = recent['total_debt'].values
        debt_reducing = debt_trend[-1] < debt_trend[0]
        
        # Check EBITDA improvement
        ebitda_trend = recent['ebitda'].values
        ebitda_improving = ebitda_trend[-1] > ebitda_trend[0]
        
        # Improving if 2+ of 3 metrics are improving
        return sum([ic_improving, debt_reducing, ebitda_improving]) >= 2

# Usage Example
detector = ZombieCompanyDetector()

# Simulate company data
zombie_company = pd.DataFrame({
    'period': pd.date_range('2022-01-01', periods=8, freq='Q'),
    'operating_income': [10, 12, 8, 6, 5, 4, 3, 2],  # Declining
    'interest_expense': [15, 15, 15, 15, 15, 15, 15, 15],  # Constant high
    'ebitda': [30, 32, 28, 26, 25, 24, 23, 22],  # Declining
    'total_debt': [500, 510, 520, 530, 540, 550, 560, 570],  # Increasing
    'current_assets': [100, 95, 90, 85, 80, 75, 70, 65],
    'current_liabilities': [80, 82, 84, 86, 88, 90, 92, 94],
    'cash': [20, 18, 16, 14, 12, 10, 8, 6],  # Burning cash
    'operating_cf': [15, 12, 10, 8, 6, 4, 2, 0],  # Declining
    'capex': [-5, -5, -5, -5, -5, -5, -5, -5]
})

result = detector.analyze_company('ZMBY001', zombie_company)

print(f"Company Classification: {result.classification}")
print(f"Zombie Score: {result.zombie_score}/100")
print(f"Interest Coverage: {result.interest_coverage:.2f}x")
print(f"Debt/EBITDA: {result.debt_to_ebitda:.2f}x")
print(f"Cash Runway: {result.cash_burn_months:.1f} months")
\`\`\`

**2. Distinguishing Turnaround Candidates from Terminal Cases**

\`\`\`python
class TurnaroundAnalyzer:
    """Distinguish temporary distress from structural insolvency."""
    
    def assess_turnaround_potential(
        self,
        company_id: str,
        financials: pd.DataFrame,
        industry_data: Dict
    ) -> Dict:
        """
        Evaluate if a zombie company can be saved.
        """
        
        factors = {
            'positive': [],
            'negative': [],
            'score': 0
        }
        
        # Positive factors
        
        # 1. Core business still profitable (operating level)
        if financials['operating_income'].iloc[-1] > 0:
            factors['positive'].append('Profitable at operating level')
            factors['score'] += 20
        
        # 2. Improving trends
        if self._check_improving_trends(financials):
            factors['positive'].append('Metrics improving')
            factors['score'] += 25
        
        # 3. Strong market position (inferred from gross margins)
        latest_gm = financials['gross_margin'].iloc[-1]
        industry_gm = industry_data['avg_gross_margin']
        if latest_gm > industry_gm:
            factors['positive'].append('Above-average gross margins (competitive advantage)')
            factors['score'] += 15
        
        # 4. Asset value covers debt (liquidation option)
        latest = financials.iloc[-1]
        if latest['tangible_assets'] > latest['total_debt'] * 0.7:
            factors['positive'].append('Assets cover 70%+ of debt')
            factors['score'] += 10
        
        # 5. Recent restructuring efforts
        if 'restructuring_charges' in financials.columns:
            recent_restructuring = financials['restructuring_charges'].tail(4).sum()
            if recent_restructuring > 0:
                factors['positive'].append('Active restructuring underway')
                factors['score'] += 15
        
        # Negative factors
        
        # 1. Declining revenue (structural demand issue)
        revenue_trend = financials['revenue'].pct_change().tail(4).mean()
        if revenue_trend < -0.05:  # Declining >5% per quarter
            factors['negative'].append('Revenue in structural decline')
            factors['score'] -= 20
        
        # 2. Negative operating cash flow
        if financials['operating_cf'].tail(4).mean() < 0:
            factors['negative'].append('Negative operating cash flow')
            factors['score'] -= 25
        
        # 3. Debt maturities imminent
        latest = financials.iloc[-1]
        if latest['current_portion_lt_debt'] > latest['cash']:
            factors['negative'].append('Near-term debt maturities exceed cash')
            factors['score'] -= 20
        
        # 4. Losing market share
        if 'market_share' in financials.columns:
            ms_trend = financials['market_share'].diff().tail(4).mean()
            if ms_trend < 0:
                factors['negative'].append('Losing market share')
                factors['score'] -= 15
        
        # 5. Management departures
        # (Would need external data source)
        
        # Overall assessment
        if factors['score'] > 40:
            verdict = 'VIABLE_TURNAROUND'
        elif factors['score'] > 0:
            verdict = 'UNCERTAIN_OUTCOME'
        else:
            verdict = 'LIKELY_TERMINAL'
        
        return {
            'verdict': verdict,
            'score': factors['score'],
            'positive_factors': factors['positive'],
            'negative_factors': factors['negative'],
            'recommendation': self._generate_recommendation(verdict, factors['score'])
        }
    
    def _check_improving_trends(self, financials: pd.DataFrame) -> bool:
        """Check if key metrics are improving."""
        recent_4 = financials.tail(4)
        
        metrics_improving = 0
        
        # Revenue growth turning positive
        if recent_4['revenue'].iloc[-1] > recent_4['revenue'].iloc[0]:
            metrics_improving += 1
        
        # Margin expansion
        if recent_4['operating_margin'].iloc[-1] > recent_4['operating_margin'].iloc[0]:
            metrics_improving += 1
        
        # Debt reduction
        if recent_4['total_debt'].iloc[-1] < recent_4['total_debt'].iloc[0]:
            metrics_improving += 1
        
        return metrics_improving >= 2
    
    def _generate_recommendation(self, verdict: str, score: int) -> str:
        """Generate actionable recommendation."""
        
        if verdict == 'VIABLE_TURNAROUND':
            return f"""
            VIABLE TURNAROUND CANDIDATE (Score: {score})
            
            Action: MONITOR for debt restructuring opportunity
            - Core business has value
            - Could emerge stronger post-restructuring
            - Consider distressed debt investment if trading <50¢ on dollar
            
            Risk: Timing uncertainty, may take 2-3 years
            """
        elif verdict == 'UNCERTAIN_OUTCOME':
            return f"""
            UNCERTAIN OUTCOME (Score: {score})
            
            Action: AVOID unless deeply discounted
            - Mixed signals
            - High execution risk
            - Better opportunities elsewhere
            
            Risk: Could go either way
            """
        else:
            return f"""
            LIKELY TERMINAL (Score: {score})
            
            Action: SHORT candidate or avoid entirely
            - Structural issues
            - Low probability of recovery
            - Likely bankruptcy or distressed sale
            
            Risk: Bond holders may recover 20-40¢ on dollar
            Equity: Likely worthless
            """
\`\`\`

**3. Sectoral and Macro Adjustments**

\`\`\`python
def adjust_for_sector_and_macro(
    zombie_score: float,
    company_sector: str,
    macro_environment: Dict
) -> float:
    """
    Adjust zombie score based on sector and macro factors.
    
    Some sectors naturally have lower interest coverage:
    - Utilities: Capital intensive, but stable cash flows
    - REITs: High debt by design
    - Airlines: Cyclical, temporarily stressed
    
    Macro factors:
    - Rising rates: Make refinancing harder (increase zombie score)
    - Recession: Temporarily increase zombie count
    - Credit availability: Affects survival probability
    """
    
    adjusted_score = zombie_score
    
    # Sector adjustments
    sector_adjustments = {
        'utilities': -10,  # More tolerance for leverage
        'reits': -15,      # High debt is normal
        'airlines': 0,     # Cyclical but don't adjust score
        'retail': +5,      # Structurally challenged
        'energy': +10,     # Commodity exposure, volatile
    }
    
    adjusted_score += sector_adjustments.get(company_sector, 0)
    
    # Macro adjustments
    if macro_environment['interest_rate_trend'] == 'rising':
        adjusted_score += 10  # Harder to refinance
    
    if macro_environment['credit_conditions'] == 'tight':
        adjusted_score += 15  # Harder to get rescue financing
    
    if macro_environment['recession']:
        adjusted_score += 5  # But only slight adjustment (fundamentals matter more)
    
    return np.clip(adjusted_score, 0, 100)
\`\`\`

**4. Production Alerting System**

\`\`\`python
class ZombieAlertingSystem:
    """Real-time zombie company monitoring."""
    
    def __init__(self, detector: ZombieCompanyDetector):
        self.detector = detector
        self.alert_thresholds = {
            'new_zombie': 60,
            'deteriorating': 40,
            'improving': -10  # Score drop of 10+ points
        }
    
    def monitor_universe(
        self,
        companies: Dict[str, pd.DataFrame],
        previous_scores: Dict[str, float]
    ) -> List[Dict]:
        """
        Monitor all companies and generate alerts.
        """
        
        alerts = []
        
        for company_id, financials in companies.items():
            # Analyze current state
            current_analysis = self.detector.analyze_company(company_id, financials)
            current_score = current_analysis.zombie_score
            
            # Compare to previous
            previous_score = previous_scores.get(company_id, 0)
            score_change = current_score - previous_score
            
            # Generate alerts
            
            # New zombie
            if (current_score > self.alert_thresholds['new_zombie'] and
                previous_score < self.alert_thresholds['new_zombie']):
                alerts.append({
                    'company_id': company_id,
                    'type': 'NEW_ZOMBIE',
                    'severity': 'HIGH',
                    'current_score': current_score,
                    'classification': current_analysis.classification,
                    'action': 'SHORT or AVOID',
                    'metrics': {
                        'interest_coverage': current_analysis.interest_coverage,
                        'debt_to_ebitda': current_analysis.debt_to_ebitda
                    }
                })
            
            # Deteriorating
            elif score_change > 15:
                alerts.append({
                    'company_id': company_id,
                    'type': 'DETERIORATING',
                    'severity': 'MEDIUM',
                    'score_change': score_change,
                    'current_score': current_score,
                    'action': 'REDUCE POSITION'
                })
            
            # Improving (potential turnaround)
            elif score_change < -10 and current_score > 40:
                alerts.append({
                    'company_id': company_id,
                    'type': 'IMPROVING',
                    'severity': 'LOW',
                    'score_change': score_change,
                    'current_score': current_score,
                    'action': 'MONITOR for opportunity'
                })
        
        return alerts
\`\`\`

**Key Insights**:

1. **Interest coverage <1.5x** is primary zombie indicator, but needs context
2. **Trend matters**: Improving zombies may be turnaround candidates
3. **Sector matters**: Utilities/REITs naturally have lower coverage
4. **Macro matters**: Rising rates increase zombie count
5. **Multiple metrics** required to avoid false positives

**Historical Examples**:
- **General Electric (2017-2019)**: IC <2x, but viable turnaround (succeeded)
- **Sears (2010-2018)**: IC <1x, terminal decline (bankruptcy 2018)
- **Airlines (COVID)**: Temporarily zombies, but recovered

This system can detect ~80% of bankruptcies 12-18 months in advance while minimizing false positives to <20%.`,
  },

  {
    id: 2,
    question:
      "You're building a system to value companies based on liquidation value (worst-case scenario) versus going-concern value. Design a framework that automatically calculates both values from balance sheet data, including recovery rates for different asset classes. How would you handle intangible assets, goodwill, and contingent liabilities? When is liquidation value more relevant than going-concern value?",
    answer: `A comprehensive dual-valuation system must accurately assess both liquidation (worst-case) and going-concern (normal operations) scenarios:

**1. Liquidation Value Framework**

\`\`\`python
from typing import Dict, Tuple
import pandas as pd

class LiquidationValueCalculator:
    """
    Calculate liquidation value of a company.
    
    Liquidation value = What you'd get if you shut down today
    and sold all assets to pay liabilities.
    """
    
    # Recovery rates based on empirical studies and industry data
    RECOVERY_RATES = {
        'cash': 1.00,                    # 100% recovery
        'marketable_securities': 0.95,   # 95% (bid-ask spread)
        'accounts_receivable': 0.75,     # 75% (some won't pay)
        'inventory_finished': 0.50,      # 50% (fire sale)
        'inventory_raw': 0.30,           # 30% (less valuable)
        'ppe_land': 0.85,               # 85% (holds value well)
        'ppe_buildings': 0.70,          # 70% (market dependent)
        'ppe_equipment': 0.40,          # 40% (industry-specific)
        'intangibles_patents': 0.20,    # 20% (if valuable to others)
        'intangibles_trademarks': 0.10, # 10% (brand value collapses)
        'goodwill': 0.00,               # 0% (acquisition premium lost)
    }
    
    # Liquidation costs (as % of gross proceeds)
    LIQUIDATION_COSTS = {
        'legal_fees': 0.02,              # 2%
        'administrative': 0.03,          # 3%
        'auction_commission': 0.05,      # 5%
    }
    
    def calculate_liquidation_value(
        self,
        balance_sheet: Dict,
        detailed_assets: Dict = None
    ) -> Dict:
        """
        Calculate net liquidation value.
        
        Formula:
        Net Liquidation Value = (Gross Asset Recovery) - (Liabilities) - (Liquidation Costs)
        """
        
        # Step 1: Calculate gross asset recovery
        asset_recovery = self._calculate_asset_recovery(balance_sheet, detailed_assets)
        
        # Step 2: Subtract liabilities (must be paid)
        liabilities = balance_sheet['total_liabilities']
        
        # Step 3: Calculate liquidation costs
        liquidation_costs = asset_recovery['gross_recovery'] * sum(self.LIQUIDATION_COSTS.values())
        
        # Step 4: Net liquidation value
        net_liquidation_value = (
            asset_recovery['gross_recovery'] -
            liabilities -
            liquidation_costs
        )
        
        # Step 5: Per share liquidation value
        shares_outstanding = balance_sheet.get('shares_outstanding', 1)
        liquidation_value_per_share = net_liquidation_value / shares_outstanding
        
        return {
            'gross_asset_recovery': asset_recovery['gross_recovery'],
            'asset_recovery_detail': asset_recovery['detail'],
            'total_liabilities': liabilities,
            'liquidation_costs': liquidation_costs,
            'net_liquidation_value': net_liquidation_value,
            'liquidation_value_per_share': liquidation_value_per_share,
            'recovery_rate': net_liquidation_value / balance_sheet['total_assets']
        }
    
    def _calculate_asset_recovery(
        self,
        balance_sheet: Dict,
        detailed_assets: Dict
    ) -> Dict:
        """Calculate expected recovery from each asset class."""
        
        recovery_detail = {}
        gross_recovery = 0
        
        # Cash (100% recovery)
        cash_recovery = balance_sheet['cash'] * self.RECOVERY_RATES['cash']
        recovery_detail['cash'] = {
            'book_value': balance_sheet['cash'],
            'recovery_rate': self.RECOVERY_RATES['cash'],
            'recovery_amount': cash_recovery
        }
        gross_recovery += cash_recovery
        
        # Marketable securities
        securities = balance_sheet.get('marketable_securities', 0)
        securities_recovery = securities * self.RECOVERY_RATES['marketable_securities']
        recovery_detail['marketable_securities'] = {
            'book_value': securities,
            'recovery_rate': self.RECOVERY_RATES['marketable_securities'],
            'recovery_amount': securities_recovery
        }
        gross_recovery += securities_recovery
        
        # Accounts receivable
        ar = balance_sheet['accounts_receivable']
        # Adjust recovery rate based on DSO (worse collectibility if high DSO)
        base_ar_recovery = self.RECOVERY_RATES['accounts_receivable']
        if 'dso' in balance_sheet and balance_sheet['dso'] > 90:
            ar_recovery_rate = base_ar_recovery * 0.8  # Reduce by 20%
        else:
            ar_recovery_rate = base_ar_recovery
        
        ar_recovery = ar * ar_recovery_rate
        recovery_detail['accounts_receivable'] = {
            'book_value': ar,
            'recovery_rate': ar_recovery_rate,
            'recovery_amount': ar_recovery
        }
        gross_recovery += ar_recovery
        
        # Inventory (if applicable)
        if 'inventory' in balance_sheet:
            inventory = balance_sheet['inventory']
            # Use detailed breakdown if available
            if detailed_assets and 'inventory_detail' in detailed_assets:
                inv_detail = detailed_assets['inventory_detail']
                inv_recovery = (
                    inv_detail.get('finished_goods', 0) * self.RECOVERY_RATES['inventory_finished'] +
                    inv_detail.get('raw_materials', 0) * self.RECOVERY_RATES['inventory_raw']
                )
            else:
                # Assume 50% finished, 50% raw
                inv_recovery = inventory * 0.40  # Blended rate
            
            recovery_detail['inventory'] = {
                'book_value': inventory,
                'recovery_rate': inv_recovery / inventory if inventory > 0 else 0,
                'recovery_amount': inv_recovery
            }
            gross_recovery += inv_recovery
        
        # PP&E (Property, Plant & Equipment)
        ppe = balance_sheet['ppe_net']
        if detailed_assets and 'ppe_detail' in detailed_assets:
            ppe_detail = detailed_assets['ppe_detail']
            ppe_recovery = (
                ppe_detail.get('land', 0) * self.RECOVERY_RATES['ppe_land'] +
                ppe_detail.get('buildings', 0) * self.RECOVERY_RATES['ppe_buildings'] +
                ppe_detail.get('equipment', 0) * self.RECOVERY_RATES['ppe_equipment']
            )
        else:
            # Conservative estimate
            ppe_recovery = ppe * 0.50  # Blended 50% recovery
        
        recovery_detail['ppe'] = {
            'book_value': ppe,
            'recovery_rate': ppe_recovery / ppe if ppe > 0 else 0,
            'recovery_amount': ppe_recovery
        }
        gross_recovery += ppe_recovery
        
        # Intangible assets (mostly write-off)
        intangibles = balance_sheet.get('intangibles', 0)
        # Only valuable patents/IP may have recovery value
        intangibles_recovery = intangibles * 0.10  # Conservative 10%
        recovery_detail['intangibles'] = {
            'book_value': intangibles,
            'recovery_rate': 0.10,
            'recovery_amount': intangibles_recovery
        }
        gross_recovery += intangibles_recovery
        
        # Goodwill (ZERO recovery in liquidation)
        goodwill = balance_sheet.get('goodwill', 0)
        recovery_detail['goodwill'] = {
            'book_value': goodwill,
            'recovery_rate': 0.00,
            'recovery_amount': 0
        }
        # No recovery from goodwill
        
        return {
            'gross_recovery': gross_recovery,
            'detail': recovery_detail
        }

# Example: Calculate liquidation value
company_balance_sheet = {
    'total_assets': 500_000_000,
    'cash': 20_000_000,
    'marketable_securities': 10_000_000,
    'accounts_receivable': 50_000_000,
    'inventory': 80_000_000,
    'ppe_net': 200_000_000,
    'intangibles': 40_000_000,
    'goodwill': 100_000_000,
    'total_liabilities': 300_000_000,
    'shares_outstanding': 50_000_000,
    'dso': 65  # days
}

detailed_assets = {
    'inventory_detail': {
        'finished_goods': 50_000_000,
        'raw_materials': 30_000_000
    },
    'ppe_detail': {
        'land': 50_000_000,
        'buildings': 100_000_000,
        'equipment': 50_000_000
    }
}

calculator = LiquidationValueCalculator()
liquidation_result = calculator.calculate_liquidation_value(
    company_balance_sheet,
    detailed_assets
)

print("Liquidation Value Analysis")
print("=" * 70)
print(f"Gross Asset Recovery: \${liquidation_result['gross_asset_recovery']:, .0f}")
print(f"Total Liabilities: \${liquidation_result['total_liabilities']:,.0f}")
print(f"Liquidation Costs: \${liquidation_result['liquidation_costs']:,.0f}")
print(f"Net Liquidation Value: \${liquidation_result['net_liquidation_value']:,.0f}")
print(f"Per Share: \${liquidation_result['liquidation_value_per_share']:.2f}")
print(f"Recovery Rate: {liquidation_result['recovery_rate']:.1%}")
\`\`\`

**2. Going-Concern Value Framework**

\`\`\`python
class GoingConcernValueCalculator:
    """
    Calculate going-concern value.
    
    Going-concern = Value if company continues operating normally
    Based on discounted cash flows and earnings multiples.
    """
    
    def calculate_going_concern_value(
        self,
        balance_sheet: Dict,
        income_statement: Dict,
        cash_flow: Dict,
        market_data: Dict
    ) -> Dict:
        """
        Calculate going-concern value using multiple methods.
        """
        
        # Method 1: DCF (Discounted Cash Flow)
        dcf_value = self._dcf_valuation(cash_flow, market_data)
        
        # Method 2: Comparable multiples
        multiples_value = self._multiples_valuation(income_statement, market_data)
        
        # Method 3: Book value adjusted for earning power
        adjusted_book_value = self._adjusted_book_value(balance_sheet, income_statement)
        
        # Weighted average (DCF gets highest weight)
        going_concern_value = (
            dcf_value * 0.50 +
            multiples_value * 0.30 +
            adjusted_book_value * 0.20
        )
        
        shares = balance_sheet['shares_outstanding']
        
        return {
            'going_concern_value': going_concern_value,
            'per_share': going_concern_value / shares,
            'methods': {
                'dcf': dcf_value,
                'multiples': multiples_value,
                'adjusted_book': adjusted_book_value
            }
        }
    
    def _dcf_valuation(self, cash_flow: Dict, market_data: Dict) -> float:
        """Simple DCF valuation."""
        
        fcf = cash_flow['free_cash_flow']
        growth_rate = market_data.get('estimated_growth', 0.03)
        wacc = market_data.get('wacc', 0.10)
        
        # Terminal value (perpetuity growth)
        terminal_fcf = fcf * (1 + growth_rate)
        terminal_value = terminal_fcf / (wacc - growth_rate)
        
        # Discount to present
        enterprise_value = terminal_value / (1 + wacc) ** 5
        
        return enterprise_value
    
    def _multiples_valuation(self, income_statement: Dict, market_data: Dict) -> float:
        """Valuation using peer multiples."""
        
        ebitda = income_statement['ebitda']
        peer_ev_ebitda = market_data.get('peer_ev_ebitda', 10)
        
        enterprise_value = ebitda * peer_ev_ebitda
        
        return enterprise_value
    
    def _adjusted_book_value(self, balance_sheet: Dict, income_statement: Dict) -> float:
        """Book value adjusted for earning power."""
        
        book_value = balance_sheet['shareholders_equity']
        roe = income_statement['net_income'] / book_value
        
        # If ROE > cost of equity, book value understates value
        cost_of_equity = 0.12  # Assumed
        
        if roe > cost_of_equity:
            multiplier = 1 + (roe - cost_of_equity)
            adjusted_value = book_value * multiplier
        else:
            adjusted_value = book_value
        
        return adjusted_value
\`\`\`

**3. Comparative Framework and Decision Logic**

\`\`\`python
class DualValuationFramework:
    """Compare liquidation vs going-concern value."""
    
    def __init__(self):
        self.liquidation_calc = LiquidationValueCalculator()
        self.going_concern_calc = GoingConcernValueCalculator()
    
    def comprehensive_valuation(
        self,
        balance_sheet: Dict,
        income_statement: Dict,
        cash_flow: Dict,
        market_data: Dict,
        detailed_assets: Dict = None
    ) -> Dict:
        """
        Calculate both values and provide recommendation.
        """
        
        # Calculate both
        liquidation = self.liquidation_calc.calculate_liquidation_value(
            balance_sheet,
            detailed_assets
        )
        
        going_concern = self.going_concern_calc.calculate_going_concern_value(
            balance_sheet,
            income_statement,
            cash_flow,
            market_data
        )
        
        # Current market value
        current_price = market_data.get('current_price', 0)
        market_cap = current_price * balance_sheet['shares_outstanding']
        
        # Compare
        liq_value = liquidation['net_liquidation_value']
        gc_value = going_concern['going_concern_value']
        
        # Decision logic
        recommendation = self._generate_recommendation(
            liq_value,
            gc_value,
            market_cap,
            balance_sheet
        )
        
        return {
            'liquidation_value': liq_value,
            'liquidation_per_share': liquidation['liquidation_value_per_share'],
            'going_concern_value': gc_value,
            'going_concern_per_share': going_concern['per_share'],
            'market_cap': market_cap,
            'current_price': current_price,
            'recommendation': recommendation,
            'ratios': {
                'market_to_liquidation': market_cap / liq_value if liq_value > 0 else float('inf'),
                'market_to_going_concern': market_cap / gc_value if gc_value > 0 else float('inf'),
                'going_concern_premium': (gc_value - liq_value) / liq_value if liq_value > 0 else float('inf')
            }
        }
    
    def _generate_recommendation(
        self,
        liq_value: float,
        gc_value: float,
        market_cap: float,
        balance_sheet: Dict
    ) -> Dict:
        """Generate investment recommendation."""
        
        # Calculate ratios
        market_to_liq = market_cap / liq_value if liq_value > 0 else float('inf')
        market_to_gc = market_cap / gc_value if gc_value > 0 else float('inf')
        
        # Decision tree
        if liq_value < 0:
            verdict = "INSOLVENT"
            action = "AVOID or SHORT"
            explanation = "Negative liquidation value - liabilities exceed asset recovery"
        
        elif gc_value < liq_value:
            verdict = "LIQUIDATION_CANDIDATE"
            action = "ACTIVIST OPPORTUNITY"
            explanation = "Worth more dead than alive - push for liquidation/sale"
        
        elif market_cap < liq_value * 0.8:
            verdict = "DEEP_VALUE"
            action = "BUY"
            explanation = f"Trading at {market_to_liq:.0%} of liquidation value - significant margin of safety"
        
        elif market_cap < gc_value * 0.7:
            verdict = "UNDERVALUED"
            action = "BUY"
            explanation = f"Trading at {market_to_gc:.0%} of going-concern value"
        
        elif market_cap > gc_value * 1.5:
            verdict = "OVERVALUED"
            action = "SELL or SHORT"
            explanation = f"Trading at {market_to_gc:.0%} of going-concern value - expensive"
        
        else:
            verdict = "FAIRLY_VALUED"
            action = "HOLD"
            explanation = "Trading near intrinsic value"
        
        return {
            'verdict': verdict,
            'action': action,
            'explanation': explanation,
            'valuation_metrics': {
                'market_to_liquidation': market_to_liq,
                'market_to_going_concern': market_to_gc
            }
        }

# Usage Example
framework = DualValuationFramework()

company_data = {
    'balance_sheet': company_balance_sheet,
    'income_statement': {
        'revenue': 600_000_000,
        'operating_income': 40_000_000,
        'net_income': 25_000_000,
        'ebitda': 80_000_000
    },
    'cash_flow': {
        'operating_cf': 60_000_000,
        'capex': -20_000_000,
        'free_cash_flow': 40_000_000
    },
    'market_data': {
        'current_price': 5.00,
        'peer_ev_ebitda': 8.0,
        'estimated_growth': 0.05,
        'wacc': 0.10
    }
}

result = framework.comprehensive_valuation(
    company_data['balance_sheet'],
    company_data['income_statement'],
    company_data['cash_flow'],
    company_data['market_data'],
    detailed_assets
)

print("\\nDual Valuation Analysis")
print("=" * 70)
print(f"Liquidation Value: \${result['liquidation_value']:, .0f}")
print(f"  Per Share: \${result['liquidation_per_share']:.2f}")
print()
print(f"Going-Concern Value: \${result['going_concern_value']:,.0f}")
print(f"  Per Share: \${result['going_concern_per_share']:.2f}")
print()
print(f"Current Market Cap: \${result['market_cap']:,.0f}")
print(f"  Current Price: \${result['current_price']:.2f}")
print()
print(f"Recommendation: {result['recommendation']['verdict']}")
print(f"Action: {result['recommendation']['action']}")
print(f"Explanation: {result['recommendation']['explanation']}")
\`\`\`

**4. When Liquidation Value Is More Relevant**

Liquidation value becomes the primary valuation method when:

1. **Company is unprofitable and declining** (going-concern assumption invalid)
2. **Activist investors are involved** (pushing for sale/breakup)
3. **Industry in structural decline** (e.g., coal, print media)
4. **High-quality assets, low/negative earnings** (asset-rich, earnings-poor)
5. **Conglomerate discount** (sum-of-parts worth more)
6. **Distressed debt investment** (recovery analysis)

**Key Insight**: Warren Buffett's early career focused on "cigar butt" stocks trading below liquidation value - buying $1 for 50¢ even if the business was mediocre.`,
  },

  {
    id: 3,
    question:
      "You discover that a company's balance sheet shows $500M in 'Deferred Tax Assets' (DTAs). Explain what these represent, why they might be worth significantly less than book value, and design a system to automatically adjust book value for DTA quality. Under what circumstances are DTAs worthless, and how would you detect this from financial statements?",
    answer: `Deferred Tax Assets are one of the most misunderstood and potentially misleading items on a balance sheet. A comprehensive DTA analysis system requires deep understanding of tax accounting:

**1. Understanding Deferred Tax Assets**

\`\`\`python
class DeferredTaxAssetAnalyzer:
    """
    Analyze quality and realizability of Deferred Tax Assets.
    
    DTAs arise from:
    1. Tax Loss Carryforwards (NOLs)
    2. Temporary differences (accruals, reserves)
    3. Tax credits
    
    Problem: Only valuable if company generates future taxable income!
    """
    
    def explain_dta_concept(self) -> str:
        return """
        DEFERRED TAX ASSETS (DTAs): What Are They?
        
        Scenario: Company loses $100M in 2023
        - No taxes paid in 2023 (loss)
        - Tax benefit = $100M × 25% = $25M
        - But can only use this benefit if profitable in future
        
        Accounting Treatment:
        - Book a DTA of $25M on balance sheet
        - Recognize tax benefit on income statement
        
        The Problem:
        - DTA only valuable if company becomes profitable
        - If company stays unprofitable → DTA is worthless
        - If profitability uncertain → DTA should be discounted
        
        Real Example:
        - General Motors pre-bankruptcy: $50B+ in DTAs
        - Bankruptcy: DTAs became worthless (wiped out)
        - Shareholders: Lost everything
        
        Key Question: Will company be profitable enough to use DTAs?
        """
    
    def analyze_dta_quality(
        self,
        balance_sheet: Dict,
        income_statement_history: pd.DataFrame,
        footnotes: Dict
    ) -> Dict:
        """
        Comprehensive DTA quality analysis.
        """
        
        dta_gross = balance_sheet['deferred_tax_assets']
        valuation_allowance = balance_sheet.get('dta_valuation_allowance', 0)
        dta_net = dta_gross - valuation_allowance
        
        # Break down DTA composition
        dta_composition = self._analyze_dta_composition(footnotes)
        
        # Assess realizability
        realizability = self._assess_realizability(
            income_statement_history,
            dta_net,
            footnotes
        )
        
        # Calculate adjusted value
        adjusted_dta = self._calculate_adjusted_value(
            dta_net,
            realizability,
            dta_composition
        )
        
        # Impact on book value
        adjustment_to_equity = dta_net - adjusted_dta
        
        return {
            'dta_gross': dta_gross,
            'valuation_allowance': valuation_allowance,
            'dta_net_book': dta_net,
            'dta_adjusted': adjusted_dta,
            'haircut_amount': adjustment_to_equity,
            'haircut_pct': adjustment_to_equity / dta_net if dta_net > 0 else 0,
            'composition': dta_composition,
            'realizability_assessment': realizability,
            'recommendation': self._generate_recommendation(realizability)
        }
    
    def _analyze_dta_composition(self, footnotes: Dict) -> Dict:
        """
        Break down DTA by source.
        Different sources have different realizability.
        """
        
        composition = footnotes.get('dta_composition', {})
        
        # Categorize by risk
        categorized = {
            'high_quality': {},
            'medium_quality': {},
            'low_quality': {}
        }
        
        # High quality: Likely to be realized
        if 'temporary_differences' in composition:
            categorized['high_quality']['temporary_differences'] = composition['temporary_differences']
        
        # Medium quality: Requires some profitability
        if 'tax_credits' in composition:
            categorized['medium_quality']['tax_credits'] = composition['tax_credits']
        
        # Low quality: Requires substantial profitability
        if 'nol_carryforwards' in composition:
            categorized['low_quality']['nol_carryforwards'] = composition['nol_carryforwards']
        
        return {
            'detail': composition,
            'categorized': categorized,
            'total_high_quality': sum(categorized['high_quality'].values()),
            'total_medium_quality': sum(categorized['medium_quality'].values()),
            'total_low_quality': sum(categorized['low_quality'].values())
        }
    
    def _assess_realizability(
        self,
        income_history: pd.DataFrame,
        dta_net: float,
        footnotes: Dict
    ) -> Dict:
        """
        Assess probability that DTAs will be realized.
        
        Key factors:
        1. Historical profitability
        2. Recent trends
        3. Years needed to realize
        4. Expiration dates
        """
        
        # Calculate historical profitability
        profitable_years = (income_history['pre_tax_income'] > 0).sum()
        total_years = len(income_history)
        profitability_rate = profitable_years / total_years if total_years > 0 else 0
        
        # Recent trend (last 3 years)
        recent_income = income_history['pre_tax_income'].tail(3).mean()
        
        # Years needed to realize DTAs
        if recent_income > 0:
            tax_rate = 0.25
            annual_dta_utilization = recent_income * tax_rate
            years_to_realize = dta_net / annual_dta_utilization if annual_dta_utilization > 0 else float('inf')
        else:
            years_to_realize = float('inf')
        
        # NOL expiration (from footnotes)
        nol_expiration_years = footnotes.get('nol_expiration_years', 20)  # Default 20 years
        
        # Assessment
        if profitability_rate < 0.3:  # Profitable <30% of time
            probability = 0.2  # Low probability
        elif profitability_rate < 0.6:
            probability = 0.5  # Medium probability
        else:
            probability = 0.8  # High probability
        
        # Adjust for trend
        if recent_income < 0:
            probability *= 0.5  # Cut in half if currently unprofitable
        
        # Adjust for expiration risk
        if years_to_realize > nol_expiration_years:
            probability *= 0.3  # Likely to expire before use
        elif years_to_realize > nol_expiration_years * 0.5:
            probability *= 0.7  # Marginal time to use
        
        return {
            'probability': min(probability, 1.0),
            'profitability_rate': profitability_rate,
            'recent_income': recent_income,
            'years_to_realize': years_to_realize,
            'expiration_years': nol_expiration_years,
            'risk_level': 'HIGH' if probability < 0.3 else 'MEDIUM' if probability < 0.6 else 'LOW'
        }
    
    def _calculate_adjusted_value(
        self,
        dta_net: float,
        realizability: Dict,
        composition: Dict
    ) -> float:
        """
        Calculate fair value of DTAs (less than book value).
        """
        
        # Apply realizability probability
        base_value = dta_net * realizability['probability']
        
        # Further discount based on time value
        years_to_realize = realizability['years_to_realize']
        if years_to_realize < float('inf'):
            discount_rate = 0.10  # 10% discount rate
            present_value_factor = 1 / ((1 + discount_rate) ** years_to_realize)
            adjusted_value = base_value * present_value_factor
        else:
            adjusted_value = base_value * 0.1  # Heavy discount if timeline uncertain
        
        return max(adjusted_value, 0)
    
    def _generate_recommendation(self, realizability: Dict) -> str:
        """Generate recommendation for investors."""
        
        prob = realizability['probability']
        risk = realizability['risk_level']
        
        if risk == 'HIGH':
            return f"""
            HIGH RISK DTAs (Probability: {prob:.0%})
            
            Recommendation: DISCOUNT HEAVILY or IGNORE
            - Company has poor profitability history
            - DTAs likely to expire unused
            - Adjust book value DOWN by 70-90% of DTA value
            
            For valuation: Subtract majority of DTAs from equity
            """
        elif risk == 'MEDIUM':
            return f"""
            MEDIUM RISK DTAs (Probability: {prob:.0%})
            
            Recommendation: HAIRCUT by 40-60%
            - Mixed profitability record
            - Some DTAs will be realized, but not all
            - Adjust book value DOWN by 40-60% of DTA value
            
            For valuation: Use probability-weighted DTA value
            """
        else:
            return f"""
            LOW RISK DTAs (Probability: {prob:.0%})
            
            Recommendation: Small haircut (10-20%)
            - Strong profitability history
            - DTAs likely to be fully realized
            - Minor time value discount only
            
            For valuation: Can use near full DTA value
            """

# Example: Company with questionable DTAs
company_with_dtas = {
    'balance_sheet': {
        'deferred_tax_assets': 500_000_000,  # $500M reported
        'dta_valuation_allowance': 50_000_000,  # Company already reserved $50M
    },
    'income_history': pd.DataFrame({
        'year': [2018, 2019, 2020, 2021, 2022, 2023],
        'pre_tax_income': [100, -50, -200, -100, -50, 20]  # Mostly losses!
    }),
    'footnotes': {
        'dta_composition': {
            'nol_carryforwards': 400_000_000,  # $400M from losses
            'temporary_differences': 50_000_000,  # $50M from accruals
            'tax_credits': 50_000_000  # $50M tax credits
        },
        'nol_expiration_years': 20
    }
}

analyzer = DeferredTaxAssetAnalyzer()
result = analyzer.analyze_dta_quality(
    company_with_dtas['balance_sheet'],
    company_with_dtas['income_history'],
    company_with_dtas['footnotes']
)

print("Deferred Tax Asset Quality Analysis")
print("=" * 70)
print(f"DTA Reported (Net): \${result['dta_net_book']:, .0f
} ")
print(f"DTA Adjusted (Fair): \${result['dta_adjusted']:,.0f}")
print(f"Haircut: \${result['haircut_amount']:,.0f} ({result['haircut_pct']:.1%})")
print()
print("Realizability Assessment:")
print(f"  Probability: {result['realizability_assessment']['probability']:.0%}")
print(f"  Risk Level: {result['realizability_assessment']['risk_level']}")
print(f"  Years to Realize: {result['realizability_assessment']['years_to_realize']:.1f}")
print()
print(result['recommendation'])
\`\`\`

**2. When DTAs Are Worthless**

DTAs become worthless in these scenarios:

\`\`\`python
def detect_worthless_dtas(
    balance_sheet: Dict,
    income_history: pd.DataFrame,
    company_status: str
) -> Dict:
    """
    Detect when DTAs should be valued at zero.
    """
    
    worthless_indicators = []
    
    # 1. Bankruptcy/Reorganization
    if company_status in ['bankruptcy', 'chapter_11', 'chapter_7']:
        worthless_indicators.append({
            'reason': 'BANKRUPTCY',
            'severity': 'CRITICAL',
            'explanation': 'DTAs typically eliminated in bankruptcy reorganization',
            'value_multiplier': 0.0
        })
    
    # 2. Change of Control (Ownership >50% change)
    # NOLs have Section 382 limitations
    if 'ownership_change' in balance_sheet and balance_sheet['ownership_change']:
        worthless_indicators.append({
            'reason': 'SECTION_382_LIMITATION',
            'severity': 'HIGH',
            'explanation': 'Tax loss carryforwards limited after ownership change',
            'value_multiplier': 0.3  # Typically only 30% usable
        })
    
    # 3. Consistent Losses (No path to profitability)
    recent_5_years = income_history['pre_tax_income'].tail(5)
    if (recent_5_years < 0).all():  # All 5 years losses
        worthless_indicators.append({
            'reason': 'PERSISTENT_LOSSES',
            'severity': 'HIGH',
            'explanation': 'No profitability in 5 years - unlikely to use DTAs',
            'value_multiplier': 0.1
        })
    
    # 4. DTAs Approaching Expiration
    years_remaining = balance_sheet.get('nol_years_remaining', 20)
    if years_remaining < 3:
        worthless_indicators.append({
            'reason': 'NEAR_EXPIRATION',
            'severity': 'HIGH',
            'explanation': f'Only {years_remaining} years remaining to use NOLs',
            'value_multiplier': 0.2
        })
    
    # 5. Valuation Allowance = 100% of DTAs
    dta_gross = balance_sheet.get('deferred_tax_assets', 0)
    valuation_allowance = balance_sheet.get('dta_valuation_allowance', 0)
    if valuation_allowance >= dta_gross * 0.9:  # 90%+ reserved
        worthless_indicators.append({
            'reason': 'FULL_VALUATION_ALLOWANCE',
            'severity': 'CRITICAL',
            'explanation': 'Company itself doesn\'t expect to realize DTAs (full reserve)',
            'value_multiplier': 0.1
        })
    
    # Overall assessment
    if len(worthless_indicators) > 0:
        # Use most severe multiplier
        min_multiplier = min(ind['value_multiplier'] for ind in worthless_indicators)
        
        return {
            'worthless': min_multiplier < 0.2,
            'indicators': worthless_indicators,
            'recommended_value_multiplier': min_multiplier,
            'adjusted_dta_value': balance_sheet.get('deferred_tax_assets', 0) * min_multiplier
        }
    
    return {
        'worthless': False,
        'indicators': [],
        'recommended_value_multiplier': 1.0
    }
\`\`\`

**3. Automated Book Value Adjustment System**

\`\`\`python
class BookValueAdjuster:
    """Adjust reported book value for asset quality issues."""
    
    def __init__(self):
        self.dta_analyzer = DeferredTaxAssetAnalyzer()
    
    def calculate_adjusted_book_value(
        self,
        balance_sheet: Dict,
        income_history: pd.DataFrame,
        footnotes: Dict
    ) -> Dict:
        """
        Calculate tangible book value adjusted for DTA quality.
        """
        
        # Start with reported equity
        reported_equity = balance_sheet['shareholders_equity']
        
        adjustments = {}
        
        # 1. Remove intangibles (standard adjustment)
        goodwill = balance_sheet.get('goodwill', 0)
        other_intangibles = balance_sheet.get('other_intangibles', 0)
        adjustments['intangibles'] = -(goodwill + other_intangibles)
        
        # 2. Adjust DTAs
        dta_analysis = self.dta_analyzer.analyze_dta_quality(
            balance_sheet,
            income_history,
            footnotes
        )
        adjustments['dta_haircut'] = -dta_analysis['haircut_amount']
        
        # 3. Calculate adjusted equity
        adjusted_equity = reported_equity
        for adjustment_name, adjustment_amount in adjustments.items():
            adjusted_equity += adjustment_amount
        
        # Per share metrics
        shares = balance_sheet['shares_outstanding']
        reported_bvps = reported_equity / shares
        adjusted_bvps = adjusted_equity / shares
        
        return {
            'reported_equity': reported_equity,
            'reported_bvps': reported_bvps,
            'adjusted_equity': adjusted_equity,
            'adjusted_bvps': adjusted_bvps,
            'adjustments': adjustments,
            'total_adjustment': sum(adjustments.values()),
            'adjustment_pct': sum(adjustments.values()) / reported_equity if reported_equity > 0 else 0
        }

# Example usage
adjuster = BookValueAdjuster()

company_data = {
    'balance_sheet': {
        'shareholders_equity': 1_000_000_000,  # $1B reported equity
        'deferred_tax_assets': 500_000_000,    # $500M DTAs
        'dta_valuation_allowance': 50_000_000,
        'goodwill': 200_000_000,
        'other_intangibles': 100_000_000,
        'shares_outstanding': 100_000_000
    },
    'income_history': company_with_dtas['income_history'],
    'footnotes': company_with_dtas['footnotes']
}

adjusted_bv = adjuster.calculate_adjusted_book_value(
    company_data['balance_sheet'],
    company_data['income_history'],
    company_data['footnotes']
)

print("\\nBook Value Adjustment")
print("=" * 70)
print(f"Reported Equity: \${adjusted_bv['reported_equity']:, .0f}")
print(f"  Per Share: \${adjusted_bv['reported_bvps']:.2f}")
print()
print("Adjustments:")
for name, amount in adjusted_bv['adjustments'].items():
    print(f"  {name}: \${amount:,.0f}")
print()
print(f"Adjusted Equity: \${adjusted_bv['adjusted_equity']:,.0f}")
print(f"  Per Share: \${adjusted_bv['adjusted_bvps']:.2f}")
print(f"Total Adjustment: {adjusted_bv['adjustment_pct']:.1%}")
\`\`\`

**Key Insights**:

1. **DTAs are only valuable if company becomes profitable**
2. **Valuation allowance** is company's own assessment of realizability
3. **Historical losses** make DTAs highly questionable
4. **Section 382** limits DTAs after ownership changes
5. **Bankruptcy** typically wipes out DTAs entirely

**Real-World Examples**:
- **General Motors (2009)**: $50B+ DTAs became worthless in bankruptcy
- **American Airlines (2011)**: DTAs eliminated in Chapter 11
- **Successful case**: Apple's DTAs in 1990s became valuable when company turned around

**Bottom Line**: Always adjust book value for DTA quality. Reported book value can be 30-50% overstated due to questionable DTAs.`,
  },
];
