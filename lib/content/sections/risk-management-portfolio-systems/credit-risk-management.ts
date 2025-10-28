export const creditRiskManagement = `
# Credit Risk Management

## Introduction

Credit risk - the risk that a counterparty fails to meet obligations - is one of the oldest risks in finance. Unlike market risk (which you can see daily), credit risk is **latent and binary**: everything is fine until suddenly it isn't.

The 2008 financial crisis wasn't primarily a market risk event - it was a credit risk catastrophe. Lehman Brothers' bankruptcy triggered a cascade of counterparty failures that nearly collapsed the global financial system.

This section covers how financial institutions measure, manage, and mitigate credit risk across loans, bonds, derivatives, and trading relationships.

## Types of Credit Risk

### 1. Default Risk

The risk that borrower doesn't repay:

**Components**:
- **Probability of Default (PD)**: Likelihood of default
- **Loss Given Default (LGD)**: Recovery rate
- **Exposure at Default (EAD)**: Amount owed when default occurs

**Expected Loss Formula**:
\`\`\`
Expected Loss = PD × LGD × EAD
\`\`\`

\`\`\`python
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy import stats

@dataclass
class CreditExposure:
    """Credit exposure to single counterparty"""
    counterparty_id: str
    counterparty_name: str
    exposure_amount: float
    credit_rating: str
    pd_1year: float  # 1-year probability of default
    lgd: float  # Loss given default (1 - recovery rate)
    maturity_years: float
    collateral: float = 0.0

class CreditRiskManager:
    """
    Manage credit risk across portfolio of exposures
    """
    
    # Credit rating to PD mapping (approximate)
    RATING_PD_MAP = {
        'AAA': 0.0001,
        'AA+': 0.0002,
        'AA': 0.0003,
        'AA-': 0.0005,
        'A+': 0.0008,
        'A': 0.0012,
        'A-': 0.0020,
        'BBB+': 0.0035,
        'BBB': 0.0060,
        'BBB-': 0.0100,  # Bottom of investment grade
        'BB+': 0.0180,
        'BB': 0.0300,
        'BB-': 0.0500,
        'B+': 0.0850,
        'B': 0.1400,
        'B-': 0.2300,
        'CCC+': 0.3500,
        'CCC': 0.5000,
        'CC': 0.7000,
        'C': 0.9000,
        'D': 1.0000  # Already defaulted
    }
    
    def __init__(self, exposures: List[CreditExposure]):
        """
        Args:
            exposures: List of credit exposures
        """
        self.exposures = exposures
        
    def calculate_expected_loss(self, exposure: CreditExposure) -> float:
        """
        Expected Loss = PD × LGD × EAD
        
        Adjusted for collateral
        """
        effective_exposure = max(0, exposure.exposure_amount - exposure.collateral)
        expected_loss = exposure.pd_1year * exposure.lgd * effective_exposure
        return expected_loss
    
    def calculate_portfolio_expected_loss(self) -> Dict:
        """
        Total expected loss across portfolio
        """
        individual_els = []
        
        for exposure in self.exposures:
            el = self.calculate_expected_loss(exposure)
            individual_els.append({
                'counterparty': exposure.counterparty_name,
                'rating': exposure.credit_rating,
                'exposure': exposure.exposure_amount,
                'expected_loss': el,
                'el_rate': (el / exposure.exposure_amount) * 100 if exposure.exposure_amount > 0 else 0
            })
        
        total_el = sum(item['expected_loss'] for item in individual_els)
        total_exposure = sum(exp.exposure_amount for exp in self.exposures)
        
        return {
            'total_expected_loss': total_el,
            'total_exposure': total_exposure,
            'portfolio_el_rate': (total_el / total_exposure) * 100 if total_exposure > 0 else 0,
            'individual_els': pd.DataFrame(individual_els).sort_values('expected_loss', ascending=False)
        }
    
    def calculate_unexpected_loss(self, confidence_level: float = 0.99) -> float:
        """
        Unexpected Loss = Credit VaR - Expected Loss
        
        Using Basel II asymptotic single risk factor (ASRF) model
        """
        # For simplicity, using standard deviation approach
        # Real implementation would use copulas or Monte Carlo
        
        expected_losses = [self.calculate_expected_loss(exp) for exp in self.exposures]
        el_mean = np.mean(expected_losses)
        el_std = np.std(expected_losses)
        
        # VaR at confidence level
        z_score = stats.norm.ppf(confidence_level)
        credit_var = el_mean + z_score * el_std
        
        unexpected_loss = credit_var - el_mean
        
        return {
            'expected_loss': el_mean * len(self.exposures),
            'credit_var': credit_var * len(self.exposures),
            'unexpected_loss': unexpected_loss * len(self.exposures),
            'confidence_level': confidence_level
        }
    
    def calculate_concentration_risk(self) -> pd.DataFrame:
        """
        Analyze concentration by counterparty, rating, sector
        """
        # Counterparty concentration
        total_exposure = sum(exp.exposure_amount for exp in self.exposures)
        
        concentrations = []
        for exposure in self.exposures:
            concentration_pct = (exposure.exposure_amount / total_exposure) * 100
            concentrations.append({
                'counterparty': exposure.counterparty_name,
                'exposure': exposure.exposure_amount,
                'concentration_pct': concentration_pct,
                'rating': exposure.credit_rating
            })
        
        df = pd.DataFrame(concentrations).sort_values('exposure', ascending=False)
        
        # Top 10 concentration
        top_10_exposure = df.head(10)['exposure'].sum()
        top_10_pct = (top_10_exposure / total_exposure) * 100
        
        return {
            'counterparty_concentration': df,
            'top_10_concentration_pct': top_10_pct,
            'largest_exposure': df.iloc[0]['exposure'],
            'largest_counterparty': df.iloc[0]['counterparty']
        }
    
    def calculate_rating_distribution(self) -> pd.DataFrame:
        """
        Distribution of exposure by credit rating
        """
        total_exposure = sum(exp.exposure_amount for exp in self.exposures)
        
        rating_groups = {}
        for exposure in self.exposures:
            rating = exposure.credit_rating
            if rating not in rating_groups:
                rating_groups[rating] = 0
            rating_groups[rating] += exposure.exposure_amount
        
        distribution = []
        for rating, amount in rating_groups.items():
            distribution.append({
                'rating': rating,
                'exposure': amount,
                'percentage': (amount / total_exposure) * 100,
                'avg_pd': self.RATING_PD_MAP.get(rating, 0.05)
            })
        
        # Sort by rating quality
        rating_order = list(self.RATING_PD_MAP.keys())
        df = pd.DataFrame(distribution)
        df['rating_rank'] = df['rating'].map({r: i for i, r in enumerate(rating_order)})
        df = df.sort_values('rating_rank').drop('rating_rank', axis=1)
        
        # Investment grade vs high yield
        ig_ratings = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-']
        ig_exposure = df[df['rating'].isin(ig_ratings)]['exposure'].sum()
        hy_exposure = total_exposure - ig_exposure
        
        return {
            'rating_distribution': df,
            'investment_grade_pct': (ig_exposure / total_exposure) * 100,
            'high_yield_pct': (hy_exposure / total_exposure) * 100
        }
    
    def migration_analysis(self, 
                          transition_matrix: pd.DataFrame,
                          horizon_years: int = 1) -> Dict:
        """
        Analyze expected rating migrations
        
        Args:
            transition_matrix: Rating transition probabilities
            horizon_years: Time horizon
        """
        # Expected number of upgrades/downgrades/defaults
        migrations = {
            'upgrades': 0,
            'downgrades': 0,
            'defaults': 0,
            'stable': 0
        }
        
        for exposure in self.exposures:
            current_rating = exposure.credit_rating
            
            # Look up transition probabilities
            if current_rating in transition_matrix.index:
                probs = transition_matrix.loc[current_rating]
                
                # Probability of default
                migrations['defaults'] += exposure.exposure_amount * probs.get('D', 0)
                
                # Simplified upgrade/downgrade logic
                # (Real implementation would track full distribution)
        
        return migrations
    
    def stress_test_credit(self, 
                          rating_shock: int = 2,
                          pd_multiplier: float = 3.0) -> Dict:
        """
        Stress test: rating downgrade and PD increase
        
        Args:
            rating_shock: Number of notches to downgrade
            pd_multiplier: PD increase factor
        """
        # Calculate current expected loss
        baseline_el = sum(self.calculate_expected_loss(exp) for exp in self.exposures)
        
        # Stress scenario
        stressed_el = 0
        rating_order = list(self.RATING_PD_MAP.keys())
        
        for exposure in self.exposures:
            # Downgrade rating
            try:
                current_idx = rating_order.index(exposure.credit_rating)
                stressed_idx = min(current_idx + rating_shock, len(rating_order) - 1)
                stressed_rating = rating_order[stressed_idx]
                stressed_pd = self.RATING_PD_MAP[stressed_rating] * pd_multiplier
            except:
                stressed_pd = exposure.pd_1year * pd_multiplier
            
            # Calculate stressed EL
            effective_exposure = max(0, exposure.exposure_amount - exposure.collateral)
            stressed_el += stressed_pd * exposure.lgd * effective_exposure
        
        return {
            'baseline_el': baseline_el,
            'stressed_el': stressed_el,
            'el_increase': stressed_el - baseline_el,
            'el_increase_pct': ((stressed_el - baseline_el) / baseline_el) * 100 if baseline_el > 0 else 0,
            'scenario': f"{rating_shock} notch downgrade, {pd_multiplier}x PD"
        }

# Example Usage
if __name__ == "__main__":
    # Sample credit portfolio
    exposures = [
        CreditExposure(
            counterparty_id='CP001',
            counterparty_name='Tech Corp A',
            exposure_amount=10000000,
            credit_rating='A',
            pd_1year=0.0012,
            lgd=0.40,
            maturity_years=5.0,
            collateral=1000000
        ),
        CreditExposure(
            counterparty_id='CP002',
            counterparty_name='Bank B',
            exposure_amount=25000000,
            credit_rating='AA',
            pd_1year=0.0003,
            lgd=0.45,
            maturity_years=3.0,
            collateral=5000000
        ),
        CreditExposure(
            counterparty_id='CP003',
            counterparty_name='Energy Co C',
            exposure_amount=8000000,
            credit_rating='BBB',
            pd_1year=0.0060,
            lgd=0.50,
            maturity_years=7.0,
            collateral=0
        ),
        CreditExposure(
            counterparty_id='CP004',
            counterparty_name='Retailer D',
            exposure_amount=5000000,
            credit_rating='BBB-',
            pd_1year=0.0100,
            lgd=0.60,
            maturity_years=4.0,
            collateral=500000
        ),
        CreditExposure(
            counterparty_id='CP005',
            counterparty_name='Startup E',
            exposure_amount=2000000,
            credit_rating='B',
            pd_1year=0.1400,
            lgd=0.70,
            maturity_years=2.0,
            collateral=0
        )
    ]
    
    credit_risk = CreditRiskManager(exposures)
    
    print("Credit Risk Analysis")
    print("="*70)
    print()
    
    # Expected Loss
    el_analysis = credit_risk.calculate_portfolio_expected_loss()
    print(f"Portfolio Expected Loss: ${el_analysis['total_expected_loss']:,.0f}")
print(f"Total Exposure: ${el_analysis['total_exposure']:,.0f}")
print(f"Portfolio EL Rate: {el_analysis['portfolio_el_rate']:.2f}%")
print()

print("Expected Loss by Counterparty:")
print(el_analysis['individual_els'].to_string(index = False))
print()
    
    # Unexpected Loss
ul_analysis = credit_risk.calculate_unexpected_loss(confidence_level = 0.99)
print(f"Expected Loss: ${ul_analysis['expected_loss']:,.0f}")
print(f"Credit VaR (99%): ${ul_analysis['credit_var']:,.0f}")
print(f"Unexpected Loss: ${ul_analysis['unexpected_loss']:,.0f}")
print()
    
    # Concentration
concentration = credit_risk.calculate_concentration_risk()
print(f"Largest Exposure: {concentration['largest_counterparty']}")
print(f"  Amount: ${concentration['largest_exposure']:,.0f}")
print(f"Top 10 Concentration: {concentration['top_10_concentration_pct']:.1f}%")
print()
    
    # Rating distribution
rating_dist = credit_risk.calculate_rating_distribution()
print("Rating Distribution:")
print(rating_dist['rating_distribution'].to_string(index = False))
print()
print(f"Investment Grade: {rating_dist['investment_grade_pct']:.1f}%")
print(f"High Yield: {rating_dist['high_yield_pct']:.1f}%")
print()
    
    # Stress test
stress_result = credit_risk.stress_test_credit(rating_shock = 2, pd_multiplier = 3.0)
print("Credit Stress Test:")
print(f"  Scenario: {stress_result['scenario']}")
print(f"  Baseline EL: ${stress_result['baseline_el']:,.0f}")
print(f"  Stressed EL: ${stress_result['stressed_el']:,.0f}")
print(f"  Increase: ${stress_result['el_increase']:,.0f} ({stress_result['el_increase_pct']:.0f}%)")
\`\`\`

### 2. Counterparty Credit Risk (CCR)

Risk in derivatives and securities financing:

\`\`\`python
class CounterpartyCreditRisk:
    """
    Manage counterparty risk for derivatives
    """
    def __init__(self, derivative_positions: List[Dict]):
        """
        Args:
            derivative_positions: List of derivative trades
        """
        self.positions = derivative_positions
        
    def calculate_current_exposure(self) -> Dict:
        """
        Current Exposure = max(MTM, 0)
        
        If derivative is in-the-money, you have credit exposure
        """
        exposures_by_counterparty = {}
        
        for position in self.positions:
            counterparty = position['counterparty']
            mtm = position['mark_to_market']
            
            # Exposure only when positive MTM (owed to us)
            exposure = max(mtm, 0)
            
            if counterparty not in exposures_by_counterparty:
                exposures_by_counterparty[counterparty] = 0
            exposures_by_counterparty[counterparty] += exposure
        
        total_exposure = sum(exposures_by_counterparty.values())
        
        return {
            'total_current_exposure': total_exposure,
            'exposure_by_counterparty': exposures_by_counterparty,
            'num_counterparties': len(exposures_by_counterparty)
        }
    
    def calculate_potential_future_exposure(self,
                                           confidence_level: float = 0.95,
                                           time_horizon_days: int = 252) -> Dict:
        """
        PFE = Future exposure at confidence level
        
        Uses historical volatility to project
        """
        pfe_by_counterparty = {}
        
        for position in self.positions:
            counterparty = position['counterparty']
            current_mtm = position['mark_to_market']
            notional = position['notional']
            volatility = position.get('volatility', 0.20)
            
            # Project potential MTM at horizon
            # PFE = Current MTM + z × σ × √t × Notional
            from scipy import stats
            z_score = stats.norm.ppf(confidence_level)
            time_factor = np.sqrt(time_horizon_days / 252)
            
            pfe = current_mtm + z_score * volatility * time_factor * notional
            pfe = max(pfe, 0)  # Only positive exposures
            
            if counterparty not in pfe_by_counterparty:
                pfe_by_counterparty[counterparty] = 0
            pfe_by_counterparty[counterparty] += pfe
        
        return {
            'pfe_by_counterparty': pfe_by_counterparty,
            'total_pfe': sum(pfe_by_counterparty.values()),
            'confidence_level': confidence_level,
            'time_horizon_days': time_horizon_days
        }
    
    def calculate_cva(self, 
                     pd_curve: Dict[int, float],
                     lgd: float = 0.60,
                     discount_curve: Dict[int, float] = None) -> float:
        """
        Credit Valuation Adjustment
        
        CVA = LGD × Σ PD(t) × EE(t) × DF(t)
        
        Where:
        - PD(t) = marginal probability of default at time t
        - EE(t) = expected exposure at time t
        - DF(t) = discount factor at time t
        """
        if discount_curve is None:
            discount_curve = {i: np.exp(-0.03 * i) for i in range(1, 11)}  # 3% discount rate
        
        total_cva = 0
        
        # Group positions by counterparty
        counterparty_groups = {}
        for position in self.positions:
            cp = position['counterparty']
            if cp not in counterparty_groups:
                counterparty_groups[cp] = []
            counterparty_groups[cp].append(position)
        
        # Calculate CVA for each counterparty
        for counterparty, positions in counterparty_groups.items():
            # Simplified: use average expected exposure
            avg_exposure = np.mean([max(p['mark_to_market'], 0) for p in positions])
            
            # Sum over time buckets
            for year, pd in pd_curve.items():
                if year in discount_curve:
                    df = discount_curve[year]
                    # Marginal PD = PD(year) - PD(year-1)
                    prev_pd = pd_curve.get(year - 1, 0)
                    marginal_pd = pd - prev_pd
                    
                    cva_contribution = lgd * marginal_pd * avg_exposure * df
                    total_cva += cva_contribution
        
        return {
            'total_cva': total_cva,
            'cva_by_counterparty': {},  # Detailed breakdown
            'lgd': lgd
        }
    
    def calculate_dva(self,
                     own_pd_curve: Dict[int, float],
                     lgd: float = 0.60) -> float:
        """
        Debit Valuation Adjustment (DVA)
        
        DVA = Value of our own credit risk
        (When we owe money and might default)
        """
        # Similar to CVA but for negative MTM positions
        # (Positions where we owe counterparty)
        
        total_dva = 0
        
        for position in self.positions:
            mtm = position['mark_to_market']
            
            # DVA only for negative MTM (we owe them)
            if mtm < 0:
                exposure = abs(mtm)
                
                # Calculate DVA (our benefit from our own default risk)
                for year, pd in own_pd_curve.items():
                    prev_pd = own_pd_curve.get(year - 1, 0)
                    marginal_pd = pd - prev_pd
                    
                    dva_contribution = lgd * marginal_pd * exposure * np.exp(-0.03 * year)
                    total_dva += dva_contribution
        
        return {
            'total_dva': total_dva,
            'lgd': lgd
        }

# Example
if __name__ == "__main__":
    # Derivative positions
    derivatives = [
        {
            'trade_id': 'IRS001',
            'counterparty': 'Bank A',
            'product_type': 'Interest Rate Swap',
            'notional': 50000000,
            'mark_to_market': 1250000,  # Positive MTM = exposure
            'volatility': 0.15,
            'maturity_years': 5
        },
        {
            'trade_id': 'FX001',
            'counterparty': 'Bank A',
            'product_type': 'FX Forward',
            'notional': 20000000,
            'mark_to_market': -500000,  # Negative MTM = no current exposure
            'volatility': 0.12,
            'maturity_years': 1
        },
        {
            'trade_id': 'CDS001',
            'counterparty': 'Hedge Fund B',
            'product_type': 'Credit Default Swap',
            'notional': 10000000,
            'mark_to_market': 800000,
            'volatility': 0.25,
            'maturity_years': 3
        }
    ]
    
    ccr = CounterpartyCreditRisk(derivatives)
    
    print("Counterparty Credit Risk Analysis")
    print("="*70)
    
    # Current exposure
    current_exp = ccr.calculate_current_exposure()
    print(f"Total Current Exposure: ${current_exp['total_current_exposure']:, .0f}")
print("By Counterparty:")
for cp, exp in current_exp['exposure_by_counterparty'].items():
    print(f"  {cp}: ${exp:,.0f}")
print()
    
    # Potential Future Exposure
pfe = ccr.calculate_potential_future_exposure(confidence_level = 0.95, time_horizon_days = 252)
print(f"Total PFE (95%, 1Y): ${pfe['total_pfe']:,.0f}")
print("By Counterparty:")
for cp, exp in pfe['pfe_by_counterparty'].items():
    print(f"  {cp}: ${exp:,.0f}")
print()
    
    # CVA calculation
pd_curve = { 1: 0.01, 2: 0.025, 3: 0.045, 4: 0.07, 5: 0.10 }  # Cumulative PDs
cva_result = ccr.calculate_cva(pd_curve, lgd = 0.60)
print(f"Credit Valuation Adjustment (CVA): ${cva_result['total_cva']:,.0f}")
\`\`\`

## Credit Risk Mitigation

### Collateral Management

\`\`\`python
class CollateralManager:
    """
    Manage collateral for credit risk mitigation
    """
    def __init__(self, exposures: List[Dict], collateral_agreements: Dict):
        self.exposures = exposures
        self.agreements = collateral_agreements
        
    def calculate_collateral_required(self, 
                                     counterparty: str,
                                     current_exposure: float,
                                     threshold: float = 0,
                                     minimum_transfer: float = 100000) -> float:
        """
        Calculate collateral call amount
        
        Args:
            current_exposure: Current MTM exposure
            threshold: Unsecured credit threshold
            minimum_transfer: Minimum transfer amount
        """
        # Collateral requirement = max(Exposure - Threshold, 0)
        collateral_required = max(current_exposure - threshold, 0)
        
        # Round to minimum transfer amount
        if collateral_required > 0 and collateral_required < minimum_transfer:
            collateral_required = minimum_transfer
        
        return {
            'counterparty': counterparty,
            'current_exposure': current_exposure,
            'threshold': threshold,
            'collateral_required': collateral_required,
            'call_amount': collateral_required,
            'minimum_transfer': minimum_transfer
        }
    
    def haircut_collateral(self, 
                          collateral_value: float,
                          collateral_type: str) -> float:
        """
        Apply haircuts to collateral value
        """
        # Haircuts by collateral type
        haircuts = {
            'CASH': 0.00,
            'US_TREASURY': 0.02,
            'INVESTMENT_GRADE_BONDS': 0.08,
            'EQUITIES': 0.15,
            'HIGH_YIELD_BONDS': 0.25,
            'REAL_ESTATE': 0.35
        }
        
        haircut = haircuts.get(collateral_type, 0.50)  # Default 50% haircut
        adjusted_value = collateral_value * (1 - haircut)
        
        return {
            'gross_value': collateral_value,
            'haircut': haircut,
            'haircut_amount': collateral_value * haircut,
            'net_value': adjusted_value,
            'collateral_type': collateral_type
        }
\`\`\`

## Key Takeaways

1. **Expected vs Unexpected Loss**: EL is cost of doing business, UL requires capital
2. **Rating-Based Approach**: Map ratings to PDs
3. **Concentration Risk**: Don't put all eggs in one basket
4. **Counterparty Credit Risk**: Derivatives create bilateral exposure
5. **CVA/DVA**: Mark-to-market adjustments for credit risk
6. **Collateral**: Key risk mitigant but requires daily management
7. **Stress Testing**: Credit risk crystallizes in downturns

## Conclusion

Credit risk is insidious - it looks fine until it blows up. Unlike market risk (visible daily), credit deterioration happens slowly then suddenly.

The key is measuring expected loss (provision for), unexpected loss (hold capital for), and having robust collateral and limit frameworks.

Next: Operational Risk - the risk of process, system, and people failures.
`;

