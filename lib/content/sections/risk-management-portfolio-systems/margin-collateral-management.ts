export const marginCollateralManagement = `
# Margin and Collateral Management

## Introduction

"In good times, collateral is an afterthought. In bad times, it's the only thing that matters."

Margin and collateral management - the process of managing cash and securities posted to secure obligations - is critical infrastructure in modern finance. The 2008 crisis and subsequent regulations (Dodd-Frank, EMIR) transformed collateral management from back-office function to strategic priority.

**Why It Matters**:
- **$4+ trillion** in collateral circulating in financial system
- **MF Global (2011)**: Misuse of client collateral led to bankruptcy
- **Lehman Brothers (2008)**: $50B+ in collateral disputes
- **Dodd-Frank**: Mandatory margin for uncleared derivatives

## Types of Margin

### 1. Initial Margin (IM)

Upfront collateral to cover potential future exposure:

\`\`\`
IM protects against default during the "margin period of risk"
Typically covers 99% VaR over 5-10 day close-out period
\`\`\`

**Example**: Interest rate swap with $10M notional
- Potential future exposure: $500K (at 99% confidence)
- Initial margin required: $500K

### 2. Variation Margin (VM)

Daily settlement of mark-to-market changes:

\`\`\`
VM = Today's MTM - Yesterday's MTM
\`\`\`

**Example**: Derivative moves from +$100K to +$150K MTM
- Variation margin call: $50K (paid to you)

\`\`\`python
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class MarginAgreement:
    """ISDA Credit Support Annex (CSA) terms"""
    threshold: float = 0  # Unsecured credit amount
    minimum_transfer_amount: float = 100000  # Min transfer size
    independent_amount: float = 0  # Additional buffer
    rounding: float = 100000  # Round to nearest
    
class MarginCalculator:
    """
    Calculate margin requirements for derivatives portfolio
    """
    def __init__(self, agreement: MarginAgreement):
        self.agreement = agreement
        
    def calculate_variation_margin(self,
                                   current_mtm: float,
                                   previous_mtm: float,
                                   previous_collateral: float) -> Dict:
        """
        Calculate VM call/return
        
        VM = Current Exposure - Previous Collateral
        where Exposure = max(MTM - Threshold, 0)
        """
        # Current exposure
        current_exposure = max(current_mtm - self.agreement.threshold, 0)
        
        # Required collateral
        required_collateral = current_exposure + self.agreement.independent_amount
        
        # Margin call/return (before rounding and MTA)
        margin_movement = required_collateral - previous_collateral
        
        # Apply minimum transfer amount
        if abs(margin_movement) < self.agreement.minimum_transfer_amount:
            margin_movement = 0
        
        # Round
        if self.agreement.rounding > 0:
            margin_movement = np.round(margin_movement / self.agreement.rounding) * self.agreement.rounding
        
        return {
            'current_mtm': current_mtm,
            'current_exposure': current_exposure,
            'required_collateral': required_collateral,
            'previous_collateral': previous_collateral,
            'margin_movement': margin_movement,
            'direction': 'CALL' if margin_movement > 0 else ('RETURN' if margin_movement < 0 else 'NO_MOVEMENT'),
            'amount': abs(margin_movement)
        }
    
    def calculate_initial_margin(self,
                                trades: List[Dict],
                                confidence_level: float = 0.99,
                                mpor_days: int = 10) -> Dict:
        """
        Calculate IM using historical simulation
        
        MPOR = Margin Period of Risk (time to close out)
        """
        # Calculate potential future exposure for each trade
        im_by_trade = []
        
        for trade in trades:
            # Historical volatility
            volatility = trade.get('volatility', 0.02)
            notional = trade['notional']
            maturity_years = trade.get('maturity_years', 5)
            
            # Duration-adjusted notional
            duration = min(maturity_years, 10)
            risk_notional = notional * duration * 0.01  # 1% DV01 approximation
            
            # Potential move over MPOR
            from scipy import stats
            z_score = stats.norm.ppf(confidence_level)
            potential_move = z_score * volatility * np.sqrt(mpor_days / 252) * risk_notional
            
            im_by_trade.append({
                'trade_id': trade['trade_id'],
                'product': trade.get('product', 'Unknown'),
                'notional': notional,
                'im_requirement': abs(potential_move)
            })
        
        total_im = sum(t['im_requirement'] for t in im_by_trade)
        
        # Apply diversification benefit (simplified)
        diversification_benefit = 0.15  # 15% benefit
        net_im = total_im * (1 - diversification_benefit)
        
        return {
            'gross_im': total_im,
            'diversification_benefit': total_im * diversification_benefit,
            'net_im': net_im,
            'confidence_level': confidence_level,
            'mpor_days': mpor_days,
            'trade_breakdown': pd.DataFrame(im_by_trade)
        }

# Example Usage
if __name__ == "__main__":
    # CSA agreement terms
    csa = MarginAgreement(
        threshold=5000000,      # $5M unsecured
        minimum_transfer_amount=250000,  # $250K MTA
        independent_amount=1000000,  # $1M buffer
        rounding=100000
    )
    
    margin_calc = MarginCalculator(csa)
    
    print("Margin Calculation Examples")
    print("="*70)
    print()
    
    # Variation Margin
    print("1. VARIATION MARGIN")
    print("-"*70)
    
    vm_scenarios = [
        {'current': 8000000, 'previous': 6000000, 'prev_collateral': 1000000, 'desc': 'MTM increases'},
        {'current': 3000000, 'previous': 6000000, 'prev_collateral': 1000000, 'desc': 'MTM decreases'},
        {'current': 6100000, 'previous': 6000000, 'prev_collateral': 1000000, 'desc': 'Small change (below MTA)'}
    ]
    
    for scenario in vm_scenarios:
        result = margin_calc.calculate_variation_margin(
            scenario['current'],
            scenario['previous'],
            scenario['prev_collateral']
        )
        
        print(f"Scenario: {scenario['desc']}")
        print(f"  Current MTM: ${result['current_mtm']:,.0f}")
        print(f"  Required Collateral: ${result['required_collateral']:,.0f}")
        print(f"  Previous Collateral: ${result['previous_collateral']:,.0f}")
        print(f"  Margin {result['direction']}: ${result['amount']:,.0f}")
        print()
    
    # Initial Margin
    print("2. INITIAL MARGIN")
    print("-"*70)
    
    trades = [
        {
            'trade_id': 'IRS001',
            'product': 'Interest Rate Swap',
            'notional': 50000000,
            'maturity_years': 5,
            'volatility': 0.015
        },
        {
            'trade_id': 'IRS002',
            'product': 'Interest Rate Swap',
            'notional': 30000000,
            'maturity_years': 10,
            'volatility': 0.018
        },
        {
            'trade_id': 'CDS001',
            'product': 'Credit Default Swap',
            'notional': 10000000,
            'maturity_years': 5,
            'volatility': 0.025
        }
    ]
    
    im_result = margin_calc.calculate_initial_margin(trades)
    
    print(f"Gross Initial Margin: ${im_result['gross_im']:,.0f}")
    print(f"Diversification Benefit: -${im_result['diversification_benefit']:,.0f}")
    print(f"Net Initial Margin: ${im_result['net_im']:,.0f}")
    print()
    print("IM by Trade:")
    print(im_result['trade_breakdown'].to_string(index=False))
\`\`\`

## Collateral Optimization

Managing collateral efficiently is critical:

\`\`\`python
class CollateralOptimizer:
    """
    Optimize collateral allocation across counterparties
    """
    def __init__(self, 
                 available_collateral: Dict[str, float],
                 haircuts: Dict[str, float]):
        """
        Args:
            available_collateral: {asset_type: amount}
            haircuts: {asset_type: haircut_percentage}
        """
        self.available = available_collateral
        self.haircuts = haircuts
        
    def calculate_collateral_value(self,
                                   asset_type: str,
                                   amount: float) -> float:
        """
        Calculate post-haircut value
        
        Value = Amount × (1 - Haircut)
        """
        haircut = self.haircuts.get(asset_type, 0.50)
        return amount * (1 - haircut)
    
    def optimize_allocation(self,
                           margin_requirements: Dict[str, float]) -> Dict:
        """
        Allocate cheapest-to-deliver collateral first
        
        Optimization criteria:
        1. Use cash first (no haircut, no funding cost on return)
        2. Use government bonds (low haircut)
        3. Use corporate bonds
        4. Use equities (high haircut)
        """
        # Sort collateral by preference (haircut)
        collateral_preference = sorted(
            self.available.items(),
            key=lambda x: self.haircuts.get(x[0], 1.0)
        )
        
        allocations = {cp: {} for cp in margin_requirements.keys()}
        remaining_collateral = self.available.copy()
        
        for counterparty, requirement in margin_requirements.items():
            remaining_requirement = requirement
            
            for asset_type, available_amount in collateral_preference:
                if remaining_requirement <= 0:
                    break
                
                if remaining_collateral.get(asset_type, 0) <= 0:
                    continue
                
                # How much of this asset do we need (accounting for haircut)?
                haircut = self.haircuts.get(asset_type, 0.50)
                needed_gross = remaining_requirement / (1 - haircut)
                
                # Use what we have available
                to_allocate = min(needed_gross, remaining_collateral[asset_type])
                value_delivered = self.calculate_collateral_value(asset_type, to_allocate)
                
                allocations[counterparty][asset_type] = to_allocate
                remaining_collateral[asset_type] -= to_allocate
                remaining_requirement -= value_delivered
            
            if remaining_requirement > 0:
                allocations[counterparty]['SHORTFALL'] = remaining_requirement
        
        return {
            'allocations': allocations,
            'remaining_collateral': remaining_collateral
        }
    
    def calculate_funding_cost(self,
                              collateral_posted: Dict[str, float],
                              funding_rates: Dict[str, float]) -> float:
        """
        Calculate opportunity cost of posting collateral
        
        Cost = Σ (Amount × Funding Rate)
        """
        total_cost = 0
        
        for asset_type, amount in collateral_posted.items():
            funding_rate = funding_rates.get(asset_type, 0.03)  # 3% default
            cost = amount * funding_rate / 252  # Daily cost
            total_cost += cost
        
        return total_cost
    
    def collateral_transformation(self,
                                 from_asset: str,
                                 to_asset: str,
                                 amount: float) -> Dict:
        """
        Transform one collateral type to another
        
        E.g., borrow cash against bonds to post as margin
        """
        # Haircut on source asset
        source_haircut = self.haircuts.get(from_asset, 0.50)
        proceeds = amount * (1 - source_haircut)
        
        # Cost of transformation (repo rate, fees)
        transformation_cost_bps = 25  # 25 bps
        cost = proceeds * (transformation_cost_bps / 10000)
        
        net_proceeds = proceeds - cost
        
        return {
            'source_asset': from_asset,
            'source_amount': amount,
            'target_asset': to_asset,
            'proceeds': proceeds,
            'transformation_cost': cost,
            'net_amount': net_proceeds,
            'efficiency': (net_proceeds / amount) * 100
        }

# Example
if __name__ == "__main__":
    # Available collateral
    available_collateral = {
        'CASH': 10000000,
        'US_TREASURY': 50000000,
        'CORP_BONDS': 30000000,
        'EQUITIES': 20000000
    }
    
    # Haircuts
    haircuts = {
        'CASH': 0.00,
        'US_TREASURY': 0.02,
        'CORP_BONDS': 0.08,
        'EQUITIES': 0.20
    }
    
    optimizer = CollateralOptimizer(available_collateral, haircuts)
    
    print("Collateral Optimization")
    print("="*70)
    print()
    
    # Margin requirements for multiple counterparties
    requirements = {
        'Bank A': 15000000,
        'Bank B': 25000000,
        'Hedge Fund C': 10000000
    }
    
    allocation = optimizer.optimize_allocation(requirements)
    
    print("Optimal Collateral Allocation:")
    for cp, alloc in allocation['allocations'].items():
        print(f"  {cp}:")
        for asset, amount in alloc.items():
            if asset != 'SHORTFALL':
                value = optimizer.calculate_collateral_value(asset, amount)
                print(f"    {asset}: ${amount:,.0f} (value: ${value:,.0f})")
            else:
                print(f"    ⚠️ SHORTFALL: ${amount:,.0f}")
    
    print()
    print("Remaining Collateral:")
    for asset, amount in allocation['remaining_collateral'].items():
        print(f"  {asset}: ${amount:,.0f}")
\`\`\`

## Uncleared Margin Rules (UMR)

Post-2008 regulations mandate margin for uncleared derivatives:

\`\`\`python
class UMRCalculator:
    """
    Calculate margin under Uncleared Margin Rules (Dodd-Frank/EMIR)
    """
    
    # ISDA SIMM (Standard Initial Margin Model) risk weights
    SIMM_RISK_WEIGHTS = {
        'IR': {  # Interest Rates
            '2W': 115, '1M': 112, '3M': 96, '6M': 74,
            '1Y': 66, '2Y': 61, '3Y': 56, '5Y': 52,
            '10Y': 53, '15Y': 57, '20Y': 60, '30Y': 66
        },
        'FX': 0.21,  # 21% for FX
        'EQ': 0.32,  # 32% for Equities
        'CR': 0.38,  # 38% for Credit
        'CM': 0.36   # 36% for Commodities
    }
    
    def __init__(self):
        pass
    
    def calculate_simm_im(self, 
                         sensitivities: Dict[str, Dict],
                         correlations: Dict = None) -> Dict:
        """
        Calculate SIMM Initial Margin
        
        SIMM = Aggregate of risk class charges
        
        Args:
            sensitivities: {risk_class: {tenor/bucket: sensitivity}}
        """
        im_by_risk_class = {}
        
        for risk_class, sens in sensitivities.items():
            if risk_class == 'IR':
                # Interest rate risk
                risk_charge = self._calculate_ir_risk(sens)
            elif risk_class in ['FX', 'EQ', 'CR', 'CM']:
                # Delta risk for other classes
                total_sens = sum(sens.values())
                risk_weight = self.SIMM_RISK_WEIGHTS[risk_class]
                risk_charge = abs(total_sens) * risk_weight
            else:
                risk_charge = 0
            
            im_by_risk_class[risk_class] = risk_charge
        
        # Aggregate across risk classes (with diversification)
        total_im = sum(im_by_risk_class.values())
        
        # Add non-delta risks (vega, curvature)
        # Simplified - real SIMM has complex methodology
        
        return {
            'total_im': total_im,
            'by_risk_class': im_by_risk_class
        }
    
    def _calculate_ir_risk(self, sensitivities: Dict[str, float]) -> float:
        """
        Calculate interest rate risk charge
        """
        risk_charge = 0
        
        for tenor, sens in sensitivities.items():
            risk_weight = self.SIMM_RISK_WEIGHTS['IR'].get(tenor, 60)
            risk_charge += (sens * risk_weight) ** 2
        
        return np.sqrt(risk_charge)
    
    def check_threshold(self,
                       notional_outstanding: float,
                       threshold_1: float = 750000000000,  # $750B
                       threshold_2: float = 8000000000) -> Dict:
        """
        Check if UMR applies
        
        Phase 1 (2016): > $3T
        Phase 6 (2022): > $8B
        """
        if notional_outstanding >= threshold_1:
            phase = 1
            applies = True
        elif notional_outstanding >= threshold_2:
            phase = 6
            applies = True
        else:
            applies = False
            phase = None
        
        return {
            'notional': notional_outstanding,
            'umr_applies': applies,
            'phase': phase,
            'threshold': threshold_2 if applies and phase == 6 else (threshold_1 if applies else None)
        }

# Example
if __name__ == "__main__":
    umr = UMRCalculator()
    
    print("UMR / SIMM Calculation")
    print("="*70)
    print()
    
    # Interest rate swap sensitivities
    sensitivities = {
        'IR': {
            '1Y': 50000,   # $50K DV01 at 1Y
            '5Y': 120000,  # $120K DV01 at 5Y
            '10Y': 80000   # $80K DV01 at 10Y
        },
        'FX': {
            'EURUSD': 1000000  # $1M delta
        }
    }
    
    im_result = umr.calculate_simm_im(sensitivities)
    
    print(f"SIMM Initial Margin: ${im_result['total_im']:,.0f}")
    print()
    print("By Risk Class:")
    for risk_class, amount in im_result['by_risk_class'].items():
        print(f"  {risk_class}: ${amount:,.0f}")
    print()
    
    # Check threshold
    threshold_check = umr.check_threshold(notional_outstanding=50000000000)  # $50B
    print("UMR Threshold Check:")
    print(f"  Notional: ${threshold_check['notional']:,.0f}")
    print(f"  UMR Applies: {threshold_check['umr_applies']}")
    if threshold_check['umr_applies']:
        print(f"  Phase: {threshold_check['phase']}")
\`\`\`

## Collateral Disputes

Managing disputes when counterparties disagree on margin:

\`\`\`python
class CollateralDisputeManager:
    """
    Manage margin disputes
    """
    def __init__(self, tolerance_threshold: float = 500000):
        """
        Args:
            tolerance_threshold: Dispute if difference > threshold
        """
        self.tolerance = tolerance_threshold
        self.disputes = []
        
    def check_for_dispute(self,
                         our_calculation: float,
                         their_calculation: float,
                         counterparty: str) -> Dict:
        """
        Identify margin disputes
        """
        difference = our_calculation - their_calculation
        
        is_dispute = abs(difference) > self.tolerance
        
        if is_dispute:
            dispute = {
                'counterparty': counterparty,
                'our_calc': our_calculation,
                'their_calc': their_calculation,
                'difference': difference,
                'pct_difference': (difference / our_calculation) * 100 if our_calculation != 0 else 0,
                'date': datetime.now(),
                'status': 'OPEN'
            }
            self.disputes.append(dispute)
            return dispute
        
        return None
    
    def dispute_resolution_workflow(self, dispute: Dict) -> List[str]:
        """
        Standard dispute resolution process
        """
        workflow = [
            "1. Operations teams compare calculations",
            "2. Identify source of difference (MTM, rates, methodology)",
            "3. If unresolved in 1 day, escalate to middle office",
            "4. If unresolved in 3 days, senior management involvement",
            "5. If still unresolved, invoke dispute resolution clause in CSA",
            "6. Consider independent valuation agent"
        ]
        
        return workflow
\`\`\`

## Key Takeaways

1. **Two Types**: Initial Margin (upfront) and Variation Margin (daily MTM)
2. **CSA Terms**: Threshold, MTA, Independent Amount critical
3. **Collateral Optimization**: Use cheapest-to-deliver first
4. **Haircuts**: Asset quality determines haircut levels
5. **UMR**: Mandatory margin for large derivatives users
6. **SIMM**: Industry standard for IM calculation
7. **Disputes**: Common and require systematic resolution

## Common Pitfalls

❌ **Insufficient Collateral**: Not enough liquid assets  
❌ **Wrong Asset Mix**: Posting illiquid/high-haircut collateral  
❌ **Concentration**: All collateral in one asset type  
❌ **Dispute Delays**: Not resolving calculation differences quickly  
❌ **Operational Failures**: Missing margin calls = default

## Conclusion

Margin and collateral management has evolved from back-office plumbing to strategic function. Post-crisis regulations (Dodd-Frank UMR, EMIR) mandate sophisticated margin calculations and collateral optimization.

Key success factors:
- **Accurate calculation**: Get MTM and margin right
- **Efficient allocation**: Optimize collateral usage
- **Timely settlement**: Meet margin deadlines
- **Dispute resolution**: Quick escalation and resolution
- **Regulatory compliance**: Meet UMR requirements

Next: Position Limits and Risk Limits - controlling risk through hard constraints.
`;

