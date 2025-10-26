export const section8 = {
  slug: 'credit',
  title: 'Credit Analysis & Default Risk',
  content: `
# Credit Analysis & Default Risk

Learn to assess creditworthiness, predict defaults, and analyze bonds like a professional credit analyst.

## Section 1: Credit Analysis Framework

\`\`\`python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict

@dataclass
class CreditAnalyzer:
    """Comprehensive credit risk assessment."""
    
    # Income statement
    ebitda: float
    ebit: float
    interest_expense: float
    
    # Balance sheet
    total_debt: float
    cash: float
    total_assets: float
    total_liabilities: float
    shareholders_equity: float
    
    # Cash flow
    cfo: float
    fcf: float
    
    def interest_coverage (self) -> float:
        """EBIT / Interest Expense"""
        return self.ebit / self.interest_expense if self.interest_expense > 0 else float('inf')
    
    def debt_service_coverage (self) -> float:
        """EBITDA / Interest Expense"""
        return self.ebitda / self.interest_expense if self.interest_expense > 0 else float('inf')
    
    def leverage_ratio (self) -> float:
        """Total Debt / EBITDA"""
        return self.total_debt / self.ebitda if self.ebitda > 0 else float('inf')
    
    def net_leverage (self) -> float:
        """(Total Debt - Cash) / EBITDA"""
        net_debt = self.total_debt - self.cash
        return net_debt / self.ebitda if self.ebitda > 0 else float('inf')
    
    def debt_to_equity (self) -> float:
        """Total Debt / Shareholders' Equity"""
        return self.total_debt / self.shareholders_equity
    
    def fcf_to_debt (self) -> float:
        """FCF / Total Debt - measures debt paydown capability"""
        return self.fcf / self.total_debt if self.total_debt > 0 else float('inf')
    
    def credit_rating_estimate (self) -> str:
        """Estimate credit rating based on metrics."""
        
        ic = self.interest_coverage()
        leverage = self.leverage_ratio()
        fcf_ratio = self.fcf_to_debt()
        
        score = 0
        
        # Interest coverage scoring
        if ic > 12:
            score += 4
        elif ic > 8:
            score += 3
        elif ic > 5:
            score += 2
        elif ic > 2.5:
            score += 1
        
        # Leverage scoring
        if leverage < 1.5:
            score += 4
        elif leverage < 2.5:
            score += 3
        elif leverage < 4.0:
            score += 2
        elif leverage < 6.0:
            score += 1
        
        # FCF scoring
        if fcf_ratio > 0.20:
            score += 2
        elif fcf_ratio > 0.10:
            score += 1
        
        # Rating mapping
        if score >= 9:
            return "AAA/AA - Investment Grade (High)"
        elif score >= 7:
            return "A/BBB - Investment Grade (Medium)"
        elif score >= 5:
            return "BB/B - High Yield (Speculative)"
        else:
            return "CCC or below - Distressed"
    
    def comprehensive_analysis (self) -> Dict:
        """Full credit analysis report."""
        
        return {
            'interest_coverage': self.interest_coverage(),
            'debt_service_coverage': self.debt_service_coverage(),
            'leverage_ratio': self.leverage_ratio(),
            'net_leverage': self.net_leverage(),
            'debt_to_equity': self.debt_to_equity(),
            'fcf_to_debt': self.fcf_to_debt(),
            'estimated_rating': self.credit_rating_estimate()
        }

# Example: Investment grade company
ig_company = CreditAnalyzer(
    ebitda=500_000_000,
    ebit=400_000_000,
    interest_expense=30_000_000,
    total_debt=1_000_000_000,
    cash=300_000_000,
    total_assets=3_000_000_000,
    total_liabilities=1_500_000_000,
    shareholders_equity=1_500_000_000,
    cfo=450_000_000,
    fcf=350_000_000
)

print("Credit Analysis:")
analysis = ig_company.comprehensive_analysis()
for metric, value in analysis.items():
    if isinstance (value, float) and value != float('inf'):
        print(f"{metric}: {value:.2f}")
    else:
        print(f"{metric}: {value}")
\`\`\`

## Section 2: Default Probability Models

\`\`\`python
from scipy.stats import norm
import math

class MertonModel:
    """
    Merton Model: Structural credit risk model.
    Treats equity as call option on firm assets.
    """
    
    @staticmethod
    def calculate_default_probability(
        asset_value: float,
        asset_volatility: float,
        debt_face_value: float,
        risk_free_rate: float,
        time_horizon: float
    ) -> Dict:
        """Calculate probability of default using Merton model."""
        
        # Distance to default
        d1 = (math.log (asset_value / debt_face_value) + 
              (risk_free_rate + 0.5 * asset_volatility**2) * time_horizon) / \
             (asset_volatility * math.sqrt (time_horizon))
        
        d2 = d1 - asset_volatility * math.sqrt (time_horizon)
        
        # Default probability
        default_prob = norm.cdf(-d2)
        
        # Distance to default (in standard deviations)
        distance_to_default = -d2
        
        return {
            'default_probability': default_prob,
            'distance_to_default': distance_to_default,
            'd1': d1,
            'd2': d2
        }

# Example
merton = MertonModel.calculate_default_probability(
    asset_value=2_000_000_000,
    asset_volatility=0.25,  # 25% annual volatility
    debt_face_value=1_000_000_000,
    risk_free_rate=0.03,  # 3%
    time_horizon=1.0  # 1 year
)

print(f"\\nMerton Model Default Probability: {merton['default_probability']:.2%}")
print(f"Distance to Default: {merton['distance_to_default']:.2f} std devs")
\`\`\`

## Section 3: Credit Spread Analysis

\`\`\`python
class CreditSpreadAnalyzer:
    """Analyze credit spreads and bond pricing."""
    
    @staticmethod
    def calculate_credit_spread(
        corporate_yield: float,
        treasury_yield: float
    ) -> float:
        """Credit Spread = Corporate Yield - Treasury Yield"""
        return corporate_yield - treasury_yield
    
    @staticmethod
    def z_spread_analysis(
        bond_price: float,
        coupon_rate: float,
        face_value: float,
        years_to_maturity: int,
        treasury_curve: Dict[int, float]
    ) -> float:
        """Calculate Z-spread (spread over entire treasury curve)."""
        
        # Simplified Z-spread calculation
        # In practice, solve iteratively
        pass
    
    @staticmethod
    def interpret_spread (spread_bps: float) -> str:
        """Interpret credit spread level."""
        
        if spread_bps < 100:
            return "AAA/AA - Minimal credit risk"
        elif spread_bps < 200:
            return "A/BBB - Investment grade"
        elif spread_bps < 500:
            return "BB/B - High yield"
        elif spread_bps < 1000:
            return "CCC - Distressed"
        else:
            return "Near default"
    
    @staticmethod
    def spread_change_price_impact(
        duration: float,
        spread_change_bps: float
    ) -> float:
        """
        Estimate price change from spread widening/tightening.
        Price Change % ≈ -Duration × ΔSpread
        """
        return -duration * (spread_change_bps / 10000)

spread_analyzer = CreditSpreadAnalyzer()

# Example: BBB bond
corporate_yield = 0.055  # 5.5%
treasury_yield = 0.035  # 3.5%
spread_bps = (corporate_yield - treasury_yield) * 10000

print(f"\\nCredit Spread: {spread_bps:.0f} bps")
print(f"Interpretation: {spread_analyzer.interpret_spread (spread_bps)}")

# Price impact of 50bps spread widening on 7-year duration bond
duration = 7.0
spread_change = 50  # bps
price_impact = spread_analyzer.spread_change_price_impact (duration, spread_change)
print(f"Price impact of +50bps: {price_impact:.2%}")
\`\`\`

## Section 4: Covenant Analysis

\`\`\`python
class CovenantMonitor:
    """Monitor debt covenants and compliance."""
    
    def __init__(self, covenants: Dict):
        self.covenants = covenants
    
    def check_compliance (self, financials: Dict) -> Dict:
        """Check if company is complying with all covenants."""
        
        results = {}
        
        # Leverage covenant
        if 'max_leverage' in self.covenants:
            leverage = financials['total_debt'] / financials['ebitda']
            compliant = leverage <= self.covenants['max_leverage']
            cushion = self.covenants['max_leverage'] - leverage
            
            results['leverage_covenant'] = {
                'compliant': compliant,
                'actual': leverage,
                'threshold': self.covenants['max_leverage'],
                'cushion': cushion,
                'risk': 'HIGH' if cushion < 0.5 else 'MEDIUM' if cushion < 1.0 else 'LOW'
            }
        
        # Interest coverage covenant
        if 'min_interest_coverage' in self.covenants:
            coverage = financials['ebitda'] / financials['interest_expense']
            compliant = coverage >= self.covenants['min_interest_coverage']
            cushion = coverage - self.covenants['min_interest_coverage']
            
            results['interest_coverage_covenant'] = {
                'compliant': compliant,
                'actual': coverage,
                'threshold': self.covenants['min_interest_coverage'],
                'cushion': cushion
            }
        
        # Fixed charge coverage
        if 'min_fixed_charge_coverage' in self.covenants:
            fixed_charges = financials['interest_expense'] + financials.get('capex', 0)
            coverage = financials['ebitda'] / fixed_charges if fixed_charges > 0 else float('inf')
            compliant = coverage >= self.covenants['min_fixed_charge_coverage']
            
            results['fixed_charge_covenant'] = {
                'compliant': compliant,
                'actual': coverage,
                'threshold': self.covenants['min_fixed_charge_coverage']
            }
        
        # Overall status
        all_compliant = all (r['compliant'] for r in results.values())
        
        return {
            'overall_compliant': all_compliant,
            'covenant_details': results,
            'warning': 'Covenant breach risk' if not all_compliant else None
        }

# Example covenants
covenants = {
    'max_leverage': 4.0,  # Total Debt / EBITDA ≤ 4.0x
    'min_interest_coverage': 3.0,  # EBITDA / Interest ≥ 3.0x
    'min_fixed_charge_coverage': 1.25
}

monitor = CovenantMonitor (covenants)

financials = {
    'total_debt': 2_000_000_000,
    'ebitda': 600_000_000,
    'interest_expense': 150_000_000,
    'capex': 100_000_000
}

compliance = monitor.check_compliance (financials)
print("\\nCovenant Compliance:")
print(f"Overall: {'✓ COMPLIANT' if compliance['overall_compliant'] else '✗ BREACH'}")
\`\`\`

## Section 5: Recovery Analysis

\`\`\`python
class RecoveryAnalyzer:
    """Estimate recovery value in default scenarios."""
    
    @staticmethod
    def estimate_recovery_rate(
        seniority: str,
        secured_status: str,
        asset_coverage: float
    ) -> float:
        """
        Estimate recovery rate based on debt characteristics.
        
        Historical recovery rates:
        - Senior Secured: 60-70%
        - Senior Unsecured: 30-50%
        - Subordinated: 15-30%
        - Equity: 0-5%
        """
        
        base_recovery = {
            'senior_secured': 0.65,
            'senior_unsecured': 0.40,
            'subordinated': 0.20,
            'equity': 0.02
        }
        
        recovery = base_recovery.get (f"{seniority}_{secured_status}", 0.30)
        
        # Adjust for asset coverage
        if asset_coverage > 1.5:
            recovery *= 1.2
        elif asset_coverage < 0.8:
            recovery *= 0.7
        
        return min (recovery, 1.0)  # Cap at 100%
    
    @staticmethod
    def loss_given_default (recovery_rate: float) -> float:
        """LGD = 1 - Recovery Rate"""
        return 1 - recovery_rate
    
    @staticmethod
    def expected_loss(
        default_probability: float,
        exposure_at_default: float,
        loss_given_default: float
    ) -> float:
        """Expected Loss = PD × EAD × LGD"""
        return default_probability * exposure_at_default * loss_given_default

recovery_analyzer = RecoveryAnalyzer()

# Example: Senior secured bond
recovery = recovery_analyzer.estimate_recovery_rate(
    seniority='senior',
    secured_status='secured',
    asset_coverage=1.8
)

print(f"\\nEstimated Recovery Rate: {recovery:.1%}")
print(f"Loss Given Default: {recovery_analyzer.loss_given_default (recovery):.1%}")

# Expected loss calculation
expected_loss = recovery_analyzer.expected_loss(
    default_probability=0.05,  # 5% default probability
    exposure_at_default=1_000_000,  # $1M bond
    loss_given_default=1 - recovery
)

print(f"Expected Loss: \${expected_loss:,.0f}")
\`\`\`

## Key Takeaways

1. **Interest coverage is critical** - Below 3x is concerning, below 2x is dangerous
2. **Leverage matters** - Debt/EBITDA >4x for corporates is high risk
3. **Cash flow is king** - FCF/Debt ratio shows paydown capability
4. **Covenants protect lenders** - Monitor compliance closely
5. **Seniority determines recovery** - Senior secured recover 2-3x more than subordinated
6. **Spreads reflect risk** - Widening spreads = deteriorating credit
7. **Default models guide** - Merton model + financial ratios = comprehensive view

Master credit analysis and you can assess any bond or loan!
`,
  discussionQuestions: [],
  multipleChoiceQuestions: [],
};
