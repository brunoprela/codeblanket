export const operationalRisk = `
# Operational Risk

## Introduction

"Market risk you can hedge. Credit risk you can diversify. Operational risk can destroy you overnight."

Operational risk - the risk of loss from failed processes, people, systems, or external events - has caused some of the most spectacular failures in finance:

- **Société Générale (2008)**: Rogue trader Jérôme Kerviel lost €4.9B through unauthorized trades
- **JPMorgan London Whale (2012)**: Poor risk controls led to $6.2B loss
- **Knight Capital (2012)**: Software bug caused $440M loss in 45 minutes, bankrupt same day
- **Bangladesh Bank (2016)**: Cyber heist stole $81M via SWIFT system
- **Archegos Capital (2021)**: Risk management failures led to $10B+ loss

Unlike market or credit risk (which you accept for returns), operational risk is **pure downside** - it doesn't generate profit, only potential loss.

## Defining Operational Risk

### Basel Committee Definition

"The risk of loss resulting from inadequate or failed internal processes, people, and systems, or from external events."

### Four Sources of Operational Risk

1. **Process Risk**: Failed or inadequate procedures
2. **People Risk**: Human error, fraud, key person dependencies
3. **Systems Risk**: IT failures, cyber attacks, bugs
4. **External Risk**: Natural disasters, pandemics, terrorism

## Operational Risk Framework

\`\`\`python
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

class RiskEventCategory(Enum):
    """Basel operational risk event categories"""
    INTERNAL_FRAUD = "Internal Fraud"
    EXTERNAL_FRAUD = "External Fraud"
    EMPLOYMENT_PRACTICES = "Employment Practices and Workplace Safety"
    CLIENTS_PRODUCTS = "Clients, Products & Business Practices"
    DAMAGE_PHYSICAL_ASSETS = "Damage to Physical Assets"
    BUSINESS_DISRUPTION = "Business Disruption and System Failures"
    EXECUTION_DELIVERY = "Execution, Delivery & Process Management"

@dataclass
class OperationalLossEvent:
    """Single operational loss event"""
    event_id: str
    date: datetime
    category: RiskEventCategory
    business_line: str
    gross_loss: float
    recovery: float = 0.0
    description: str = ""
    root_cause: str = ""
    
    @property
    def net_loss(self) -> float:
        return self.gross_loss - self.recovery

class OperationalRiskManager:
    """
    Manage operational risk events and exposure
    """
    def __init__(self, loss_events: List[OperationalLossEvent] = None):
        """
        Args:
            loss_events: Historical operational loss events
        """
        self.loss_events = loss_events or []
        
    def add_event(self, event: OperationalLossEvent):
        """Record new operational loss event"""
        self.loss_events.append(event)
        
    def calculate_total_losses(self, 
                              start_date: datetime = None,
                              end_date: datetime = None) -> Dict:
        """
        Calculate total operational losses over period
        """
        # Filter by date range
        events = self.loss_events
        if start_date:
            events = [e for e in events if e.date >= start_date]
        if end_date:
            events = [e for e in events if e.date <= end_date]
        
        if not events:
            return {
                'num_events': 0,
                'total_gross_loss': 0,
                'total_recovery': 0,
                'total_net_loss': 0
            }
        
        total_gross = sum(e.gross_loss for e in events)
        total_recovery = sum(e.recovery for e in events)
        total_net = sum(e.net_loss for e in events)
        
        return {
            'num_events': len(events),
            'total_gross_loss': total_gross,
            'total_recovery': total_recovery,
            'total_net_loss': total_net,
            'recovery_rate': (total_recovery / total_gross) * 100 if total_gross > 0 else 0,
            'avg_loss_per_event': total_net / len(events)
        }
    
    def loss_by_category(self) -> pd.DataFrame:
        """
        Analyze losses by Basel category
        """
        category_losses = {}
        
        for event in self.loss_events:
            category = event.category.value
            if category not in category_losses:
                category_losses[category] = {
                    'num_events': 0,
                    'total_loss': 0,
                    'avg_loss': 0
                }
            
            category_losses[category]['num_events'] += 1
            category_losses[category]['total_loss'] += event.net_loss
        
        # Calculate averages
        for category in category_losses:
            n = category_losses[category]['num_events']
            category_losses[category]['avg_loss'] = category_losses[category]['total_loss'] / n
        
        df = pd.DataFrame.from_dict(category_losses, orient='index')
        df = df.sort_values('total_loss', ascending=False)
        
        return df
    
    def loss_by_business_line(self) -> pd.DataFrame:
        """
        Analyze losses by business line
        """
        bl_losses = {}
        
        for event in self.loss_events:
            bl = event.business_line
            if bl not in bl_losses:
                bl_losses[bl] = {
                    'num_events': 0,
                    'total_loss': 0
                }
            
            bl_losses[bl]['num_events'] += 1
            bl_losses[bl]['total_loss'] += event.net_loss
        
        df = pd.DataFrame.from_dict(bl_losses, orient='index')
        df = df.sort_values('total_loss', ascending=False)
        
        return df
    
    def calculate_frequency(self, period_days: int = 365) -> float:
        """
        Calculate event frequency (events per period)
        """
        if not self.loss_events:
            return 0
        
        # Get date range
        dates = [e.date for e in self.loss_events]
        min_date = min(dates)
        max_date = max(dates)
        
        total_days = (max_date - min_date).days
        if total_days == 0:
            return len(self.loss_events)
        
        # Annualized frequency
        frequency = (len(self.loss_events) / total_days) * period_days
        
        return frequency
    
    def calculate_severity_distribution(self) -> Dict:
        """
        Analyze loss severity distribution
        """
        losses = [e.net_loss for e in self.loss_events]
        
        if not losses:
            return {}
        
        return {
            'mean': np.mean(losses),
            'median': np.median(losses),
            'std': np.std(losses),
            'min': min(losses),
            'max': max(losses),
            'percentile_95': np.percentile(losses, 95),
            'percentile_99': np.percentile(losses, 99)
        }
    
    def calculate_operational_var(self, 
                                 confidence_level: float = 0.99,
                                 time_horizon_days: int = 365) -> Dict:
        """
        Operational VaR using Loss Distribution Approach (LDA)
        
        OpVaR = Frequency × Severity at confidence level
        """
        # Calculate frequency (lambda)
        frequency = self.calculate_frequency(period_days=time_horizon_days)
        
        # Calculate severity distribution
        severity = self.calculate_severity_distribution()
        
        if not severity:
            return {'operational_var': 0}
        
        # Simple approach: use percentile of historical severity
        severity_at_confidence = severity[f'percentile_{int(confidence_level*100)}']
        
        # Expected number of events in horizon
        expected_events = frequency
        
        # OpVaR = Expected Events × Severe Loss
        # (Simplified - real LDA uses Monte Carlo)
        operational_var = expected_events * severity_at_confidence
        
        return {
            'operational_var': operational_var,
            'expected_frequency': frequency,
            'expected_severity': severity['mean'],
            'severe_severity': severity_at_confidence,
            'confidence_level': confidence_level,
            'time_horizon_days': time_horizon_days
        }
    
    def identify_high_risk_areas(self, threshold_loss: float = 1000000) -> Dict:
        """
        Identify categories/business lines with high losses
        """
        # By category
        category_df = self.loss_by_category()
        high_risk_categories = category_df[category_df['total_loss'] > threshold_loss]
        
        # By business line
        bl_df = self.loss_by_business_line()
        high_risk_business_lines = bl_df[bl_df['total_loss'] > threshold_loss]
        
        return {
            'high_risk_categories': high_risk_categories.to_dict('index'),
            'high_risk_business_lines': high_risk_business_lines.to_dict('index'),
            'threshold': threshold_loss
        }

# Example Usage
if __name__ == "__main__":
    # Sample operational loss events
    events = [
        OperationalLossEvent(
            event_id='OE001',
            date=datetime(2023, 1, 15),
            category=RiskEventCategory.EXECUTION_DELIVERY,
            business_line='Trading',
            gross_loss=250000,
            recovery=50000,
            description='Trade settlement failure',
            root_cause='Manual process error'
        ),
        OperationalLossEvent(
            event_id='OE002',
            date=datetime(2023, 3, 22),
            category=RiskEventCategory.BUSINESS_DISRUPTION,
            business_line='Technology',
            gross_loss=1500000,
            recovery=0,
            description='System outage - 4 hours',
            root_cause='Infrastructure failure'
        ),
        OperationalLossEvent(
            event_id='OE003',
            date=datetime(2023, 5, 10),
            category=RiskEventCategory.INTERNAL_FRAUD,
            business_line='Operations',
            gross_loss=5000000,
            recovery=2000000,
            description='Employee fraud',
            root_cause='Lack of segregation of duties'
        ),
        OperationalLossEvent(
            event_id='OE004',
            date=datetime(2023, 7, 8),
            category=RiskEventCategory.EXTERNAL_FRAUD,
            business_line='Retail Banking',
            gross_loss=800000,
            recovery=600000,
            description='Phishing attack on customers',
            root_cause='Insufficient customer education'
        ),
        OperationalLossEvent(
            event_id='OE005',
            date=datetime(2023, 9, 30),
            category=RiskEventCategory.EXECUTION_DELIVERY,
            business_line='Trading',
            gross_loss=300000,
            recovery=0,
            description='Pricing error',
            root_cause='Model bug'
        )
    ]
    
    op_risk = OperationalRiskManager(events)
    
    print("Operational Risk Analysis")
    print("="*70)
    print()
    
    # Total losses
    total = op_risk.calculate_total_losses()
    print(f"Total Operational Losses:")
    print(f"  Number of Events: {total['num_events']}")
    print(f"  Gross Loss: ${total['total_gross_loss']:,.0f}")
print(f"  Recovery: ${total['total_recovery']:,.0f}")
print(f"  Net Loss: ${total['total_net_loss']:,.0f}")
print(f"  Recovery Rate: {total['recovery_rate']:.1f}%")
print(f"  Average Loss per Event: ${total['avg_loss_per_event']:,.0f}")
print()
    
    # By category
print("Losses by Category:")
print(op_risk.loss_by_category().to_string())
print()
    
    # Severity distribution
severity = op_risk.calculate_severity_distribution()
print("Loss Severity Distribution:")
print(f"  Mean: ${severity['mean']:,.0f}")
print(f"  Median: ${severity['median']:,.0f}")
print(f"  Std Dev: ${severity['std']:,.0f}")
print(f"  95th Percentile: ${severity['percentile_95']:,.0f}")
print(f"  99th Percentile: ${severity['percentile_99']:,.0f}")
print(f"  Max: ${severity['max']:,.0f}")
print()
    
    # Operational VaR
op_var = op_risk.calculate_operational_var(confidence_level = 0.99)
print("Operational VaR (99%, 1 year):")
print(f"  OpVaR: ${op_var['operational_var']:,.0f}")
print(f"  Expected Frequency: {op_var['expected_frequency']:.2f} events/year")
print(f"  Expected Severity: ${op_var['expected_severity']:,.0f}")
print()
    
    # High risk areas
high_risk = op_risk.identify_high_risk_areas(threshold_loss = 1000000)
print("High Risk Areas (>$1M loss):")
print(f"  Categories: {list(high_risk['high_risk_categories'].keys())}")
print(f"  Business Lines: {list(high_risk['high_risk_business_lines'].keys())}")
\`\`\`

## Key Risk Indicators (KRIs)

Leading indicators to predict operational risk:

\`\`\`python
class KeyRiskIndicators:
    """
    Monitor operational risk indicators
    """
    def __init__(self):
        self.kris = {}
        self.thresholds = {}
        
    def add_kri(self, 
                kri_name: str,
                value: float,
                threshold_yellow: float,
                threshold_red: float,
                is_higher_better: bool = False):
        """
        Add KRI with thresholds
        
        Args:
            is_higher_better: If False, exceeding threshold is bad
        """
        self.kris[kri_name] = value
        self.thresholds[kri_name] = {
            'yellow': threshold_yellow,
            'red': threshold_red,
            'higher_better': is_higher_better
        }
    
    def evaluate_kris(self) -> Dict:
        """
        Evaluate all KRIs against thresholds
        """
        results = []
        
        for kri_name, value in self.kris.items():
            threshold = self.thresholds[kri_name]
            higher_better = threshold['higher_better']
            
            if higher_better:
                # Higher is better (e.g., control effectiveness %)
                if value >= threshold['yellow']:
                    status = 'GREEN'
                elif value >= threshold['red']:
                    status = 'YELLOW'
                else:
                    status = 'RED'
            else:
                # Lower is better (e.g., error rate)
                if value <= threshold['yellow']:
                    status = 'GREEN'
                elif value <= threshold['red']:
                    status = 'YELLOW'
                else:
                    status = 'RED'
            
            results.append({
                'kri': kri_name,
                'value': value,
                'status': status,
                'yellow_threshold': threshold['yellow'],
                'red_threshold': threshold['red']
            })
        
        return pd.DataFrame(results)

# Example KRIs
if __name__ == "__main__":
    kri_monitor = KeyRiskIndicators()
    
    # Add various KRIs
    kri_monitor.add_kri('Failed Trades %', 0.05, threshold_yellow=0.10, threshold_red=0.20)
    kri_monitor.add_kri('System Downtime Hours/Month', 2.5, threshold_yellow=4.0, threshold_red=8.0)
    kri_monitor.add_kri('Employee Turnover %', 12, threshold_yellow=15, threshold_red=25)
    kri_monitor.add_kri('Control Test Pass Rate %', 95, threshold_yellow=90, threshold_red=85, is_higher_better=True)
    kri_monitor.add_kri('Audit Findings', 8, threshold_yellow=10, threshold_red=15)
    kri_monitor.add_kri('Reconciliation Breaks', 15, threshold_yellow=20, threshold_red=30)
    
    print("Key Risk Indicators Dashboard")
    print("="*70)
    kri_results = kri_monitor.evaluate_kris()
    print(kri_results.to_string(index=False))
\`\`\`

## Operational Risk Scenarios

\`\`\`python
class OperationalRiskScenarios:
    """
    Scenario analysis for operational risk
    """
    
    @staticmethod
    def cyber_attack_scenario(revenue_per_day: float) -> Dict:
        """
        Cyber attack scenario analysis
        """
        scenarios = {
            'Minor': {
                'probability': 0.20,  # 20% chance per year
                'downtime_days': 0.5,
                'data_breach': False,
                'regulatory_fine': 0
            },
            'Moderate': {
                'probability': 0.10,
                'downtime_days': 2,
                'data_breach': True,
                'regulatory_fine': 5000000
            },
            'Severe': {
                'probability': 0.02,
                'downtime_days': 5,
                'data_breach': True,
                'regulatory_fine': 50000000
            }
        }
        
        results = []
        for severity, params in scenarios.items():
            # Calculate impact
            revenue_loss = params['downtime_days'] * revenue_per_day
            recovery_cost = 1000000 * params['downtime_days']  # $1M per day recovery
            reputation_loss = revenue_per_day * 30 if params['data_breach'] else 0  # 30 days revenue
            
            total_loss = revenue_loss + recovery_cost + params['regulatory_fine'] + reputation_loss
            expected_loss = total_loss * params['probability']
            
            results.append({
                'severity': severity,
                'probability': params['probability'],
                'revenue_loss': revenue_loss,
                'recovery_cost': recovery_cost,
                'regulatory_fine': params['regulatory_fine'],
                'reputation_loss': reputation_loss,
                'total_loss': total_loss,
                'expected_loss': expected_loss
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def rogue_trader_scenario(trading_limit: float) -> Dict:
        """
        Rogue trader scenario (à la Société Générale)
        """
        # How much could rogue trader lose before detected?
        detection_scenarios = {
            'Quick Detection (1 day)': {
                'days_undetected': 1,
                'max_loss': trading_limit * 2,
                'probability': 0.70
            },
            'Delayed Detection (1 week)': {
                'days_undetected': 7,
                'max_loss': trading_limit * 10,
                'probability': 0.25
            },
            'Late Detection (1 month)': {
                'days_undetected': 30,
                'max_loss': trading_limit * 50,
                'probability': 0.05
            }
        }
        
        results = []
        for scenario, params in detection_scenarios.items():
            expected_loss = params['max_loss'] * params['probability']
            
            results.append({
                'scenario': scenario,
                'days_undetected': params['days_undetected'],
                'max_loss': params['max_loss'],
                'probability': params['probability'],
                'expected_loss': expected_loss
            })
        
        return pd.DataFrame(results)

# Example
if __name__ == "__main__":
    # Cyber attack scenario
    revenue_per_day = 10000000  # $10M revenue per day
    cyber_scenarios = OperationalRiskScenarios.cyber_attack_scenario(revenue_per_day)
    
    print("Cyber Attack Scenarios")
    print("="*70)
    print(cyber_scenarios.to_string(index=False))
    print()
    print(f"Total Expected Annual Loss: ${cyber_scenarios['expected_loss'].sum():, .0f}")
print()
    
    # Rogue trader scenario
trading_limit = 1000000  # $1M daily limit
rogue_scenarios = OperationalRiskScenarios.rogue_trader_scenario(trading_limit)

print("Rogue Trader Scenarios")
print("=" * 70)
print(rogue_scenarios.to_string(index = False))
print()
print(f"Total Expected Loss: ${rogue_scenarios['expected_loss'].sum():,.0f}")
\`\`\`

## Operational Risk Controls

### Three Lines of Defense

\`\`\`python
class OperationalRiskControls:
    """
    Manage operational risk controls
    """
    def __init__(self):
        self.controls = []
        
    def add_control(self,
                   control_id: str,
                   control_description: str,
                   control_type: str,  # 'Preventive' or 'Detective'
                   line_of_defense: int,  # 1, 2, or 3
                   frequency: str,  # 'Daily', 'Weekly', 'Monthly', etc.
                   effectiveness: float,  # 0-1
                   last_test_date: datetime = None,
                   test_result: str = 'Not Tested'):
        """
        Add operational control
        """
        control = {
            'control_id': control_id,
            'description': control_description,
            'type': control_type,
            'line_of_defense': line_of_defense,
            'frequency': frequency,
            'effectiveness': effectiveness,
            'last_test_date': last_test_date,
            'test_result': test_result
        }
        self.controls.append(control)
    
    def control_effectiveness_summary(self) -> pd.DataFrame:
        """
        Summary of control effectiveness
        """
        df = pd.DataFrame(self.controls)
        
        summary = df.groupby(['line_of_defense', 'type']).agg({
            'effectiveness': 'mean',
            'control_id': 'count'
        }).reset_index()
        
        summary.columns = ['Line of Defense', 'Type', 'Avg Effectiveness', 'Number of Controls']
        
        return summary
    
    def identify_weak_controls(self, threshold: float = 0.70) -> pd.DataFrame:
        """
        Identify controls with low effectiveness
        """
        df = pd.DataFrame(self.controls)
        weak_controls = df[df['effectiveness'] < threshold]
        
        return weak_controls[['control_id', 'description', 'effectiveness', 'test_result']]

# Example
if __name__ == "__main__":
    controls = OperationalRiskControls()
    
    # Add controls
    controls.add_control(
        'C001',
        'Daily trade reconciliation',
        'Detective',
        line_of_defense=1,
        frequency='Daily',
        effectiveness=0.95,
        last_test_date=datetime(2024, 1, 15),
        test_result='Passed'
    )
    
    controls.add_control(
        'C002',
        'Segregation of duties - trade entry vs approval',
        'Preventive',
        line_of_defense=1,
        frequency='Continuous',
        effectiveness=0.85,
        test_result='Passed'
    )
    
    controls.add_control(
        'C003',
        'Independent risk review',
        'Detective',
        line_of_defense=2,
        frequency='Daily',
        effectiveness=0.90,
        test_result='Passed'
    )
    
    controls.add_control(
        'C004',
        'Access rights review',
        'Preventive',
        line_of_defense=2,
        frequency='Quarterly',
        effectiveness=0.65,
        test_result='Deficiency Noted'
    )
    
    print("Control Effectiveness Summary")
    print("="*70)
    print(controls.control_effectiveness_summary().to_string(index=False))
    print()
    
    print("Weak Controls (< 70% effective)")
    print("="*70)
    weak = controls.identify_weak_controls(threshold=0.70)
    if not weak.empty:
        print(weak.to_string(index=False))
    else:
        print("No weak controls identified")
\`\`\`

## Basel III Operational Risk Capital

\`\`\`python
def calculate_operational_risk_capital_basel(revenue: List[float],
                                            approach: str = 'BIA') -> float:
    """
    Calculate operational risk capital under Basel III
    
    Args:
        revenue: List of annual gross income (last 3 years)
        approach: 'BIA' (Basic Indicator) or 'SA' (Standardized)
    
    Returns:
        Required capital for operational risk
    """
    if approach == 'BIA':
        # Basic Indicator Approach
        # Capital = 15% × Average Gross Income (last 3 years)
        avg_revenue = np.mean([r for r in revenue if r > 0])
        capital = 0.15 * avg_revenue
        
        return {
            'approach': 'Basic Indicator Approach',
            'avg_gross_income': avg_revenue,
            'capital_factor': 0.15,
            'required_capital': capital
        }
    
    elif approach == 'SA':
        # Standardized Approach
        # Different beta factors by business line
        business_line_betas = {
            'Corporate Finance': 0.18,
            'Trading & Sales': 0.18,
            'Retail Banking': 0.12,
            'Commercial Banking': 0.15,
            'Payment & Settlement': 0.18,
            'Agency Services': 0.15,
            'Asset Management': 0.12,
            'Retail Brokerage': 0.12
        }
        
        # Simplified - assume equal split across business lines
        total_capital = 0
        for beta in business_line_betas.values():
            bl_revenue = np.mean(revenue) / len(business_line_betas)
            total_capital += beta * bl_revenue
        
        return {
            'approach': 'Standardized Approach',
            'required_capital': total_capital,
            'business_lines': list(business_line_betas.keys())
        }

# Example
revenue_3years = [500000000, 520000000, 480000000]  # Last 3 years
capital_bia = calculate_operational_risk_capital_basel(revenue_3years, 'BIA')

print("Basel III Operational Risk Capital")
print("="*70)
print(f"Approach: {capital_bia['approach']}")
print(f"Average Gross Income: ${capital_bia['avg_gross_income']:, .0f}")
print(f"Required Capital: ${capital_bia['required_capital']:,.0f}")
\`\`\`

## Key Takeaways

1. **Pure Downside**: Operational risk doesn't generate returns
2. **Four Sources**: Process, people, systems, external events
3. **Loss Data**: Track and analyze all operational loss events
4. **KRIs**: Leading indicators to predict issues
5. **Scenario Analysis**: Model extreme operational risk events
6. **Three Lines of Defense**: Business, risk, audit
7. **Basel Capital**: Banks must hold capital for operational risk
8. **Continuous Monitoring**: Operational risk requires constant vigilance

## Common Pitfalls

❌ **Ignoring Near-Misses**: Learn from close calls  
❌ **Weak Controls**: Ineffective controls = no protection  
❌ **Technology Complacency**: "It hasn't failed yet" ≠ "It won't fail"  
❌ **Key Person Risk**: Over-reliance on individuals  
❌ **Poor Documentation**: Can't audit what isn't documented

## Conclusion

Operational risk is the "silent killer" - it accumulates slowly through weak processes, poor controls, and complacency, then strikes catastrophically.

Unlike market risk (which you're paid to take) or credit risk (which you can analyze and price), operational risk is pure loss. The goal is to **minimize** it through:
- Strong processes and controls
- Continuous monitoring (KRIs)
- Regular testing and audits
- Learning from losses and near-misses
- Culture of risk awareness

The best operational risk management is invisible - nothing goes wrong because risks are proactively identified and mitigated.

Next: Liquidity Risk - the risk of not being able to meet obligations when due.
`;

