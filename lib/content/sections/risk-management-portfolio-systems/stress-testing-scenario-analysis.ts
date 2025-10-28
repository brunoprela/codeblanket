export const stressTestingScenarioAnalysis = `
# Stress Testing and Scenario Analysis

## Introduction

"It wasn't that VaR failed - it's that people forgot VaR doesn't predict the future."

Historical data can only tell us what happened. Stress testing asks: **"What if something unprecedented happens?"**

While VaR and CVaR use historical data and statistical distributions, stress testing complements these with:
- **Historical scenarios**: What happened in past crises?
- **Hypothetical scenarios**: What could happen in the future?
- **Reverse stress tests**: What would break us?

Stress testing became mandatory after 2008 - regulators realized backward-looking risk models missed systemic risks.

## Types of Stress Tests

### 1. Historical Scenarios

Replay past crises on current portfolio:

**Major Historical Scenarios:**
- **Black Monday (October 1987)**: S&P 500 -20% in one day
- **Russian Default (August 1998)**: Emerging markets collapsed, flight to quality
- **Dot-com Crash (2000-2002)**: NASDAQ -78% from peak
- **9/11 Attacks (September 2001)**: Markets closed for days, sudden reopening
- **Global Financial Crisis (2007-2008)**: Credit markets froze, Lehman bankruptcy
- **Flash Crash (May 2010)**: S&P 500 -9% in minutes
- **European Sovereign Debt Crisis (2010-2012)**: Euro collapse fears
- **Taper Tantrum (May-June 2013)**: Bond yields spiked on Fed taper talks
- **Volmageddon (February 2018)**: VIX ETN collapse, volatility spike
- **COVID-19 Crash (March 2020)**: -34% in 33 days, fastest bear market ever
- **Russia-Ukraine War (February 2022)**: Energy crisis, commodity spike

\`\`\`python
import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class HistoricalScenario:
    """Historical crisis scenario"""
    name: str
    description: str
    date: str
    duration_days: int
    market_moves: Dict[str, float]  # asset_class -> return
    key_characteristics: List[str]

# Define major historical scenarios
HISTORICAL_SCENARIOS = {
    'black_monday_1987': HistoricalScenario(
        name='Black Monday 1987',
        description='Largest single-day stock market crash in history',
        date='1987-10-19',
        duration_days=1,
        market_moves={
            'US_EQUITIES': -0.2046,  # S&P 500 -20.46%
            'US_BONDS': 0.0050,      # Flight to safety
            'COMMODITIES': -0.015,
            'USD': 0.020,
            'VIX': 3.5  # Multiplier (vol spike)
        },
        key_characteristics=[
            'Single day event',
            'No warning',
            'Automated trading contributed',
            'Followed by recovery'
        ]
    ),
    
    'lehman_crisis_2008': HistoricalScenario(
        name='Lehman Bankruptcy & Credit Crisis',
        description='Global financial system near-collapse',
        date='2008-09-15',
        duration_days=60,
        market_moves={
            'US_EQUITIES': -0.30,
            'EUROPE_EQUITIES': -0.35,
            'EMERGING_EQUITIES': -0.40,
            'CREDIT_IG': -0.15,      # Investment grade credit
            'CREDIT_HY': -0.35,      # High yield spreads widened massively
            'US_BONDS': 0.10,        # Flight to quality
            'COMMODITIES': -0.25,
            'REAL_ESTATE': -0.20,
            'VIX': 5.0
        },
        key_characteristics=[
            'Systemic crisis',
            'Credit markets froze',
            'Counterparty risk dominated',
            'Correlations went to 1',
            'Liquidity disappeared'
        ]
    ),
    
    'covid_crash_2020': HistoricalScenario(
        name='COVID-19 Market Crash',
        description='Fastest bear market in history',
        date='2020-02-19',
        duration_days=33,
        market_moves={
            'US_EQUITIES': -0.34,    # S&P 500 peak to trough
            'EUROPE_EQUITIES': -0.36,
            'OIL': -0.60,            # Oil crashed to negative!
            'GOLD': 0.05,            # Safe haven
            'US_BONDS': 0.08,
            'CREDIT_HY': -0.25,
            'VIX': 8.0               # VIX peaked at 82
        },
        key_characteristics=[
            'Fastest bear market ever',
            'Unprecedented government response',
            'Circuit breakers triggered multiple times',
            'Fed intervened massively',
            'Followed by fastest recovery'
        ]
    ),
    
    'volmageddon_2018': HistoricalScenario(
        name='Volmageddon',
        description='VIX ETN collapse and volatility spike',
        date='2018-02-05',
        duration_days=7,
        market_moves={
            'US_EQUITIES': -0.10,
            'XIV_ETN': -0.96,        # XIV note went to near-zero
            'VIX': 2.5,
            'VOL_ETNS': -0.80,       # Volatility ETNs crushed
            'US_BONDS': 0.02
        },
        key_characteristics=[
            'Short volatility trade unwind',
            'Systematic strategies deleveraging',
            'Vol ETNs terminated',
            'Feedback loops',
            'Market structure issue'
        ]
    )
}

class HistoricalStressTester:
    """
    Apply historical scenarios to current portfolio
    """
    def __init__(self, portfolio: Dict[str, float], asset_class_mapping: Dict[str, str]):
        """
        Args:
            portfolio: Dict of {asset: dollar_position}
            asset_class_mapping: Dict of {asset: asset_class}
        """
        self.portfolio = portfolio
        self.asset_class_mapping = asset_class_mapping
        self.portfolio_value = sum(portfolio.values())
        
    def apply_scenario(self, scenario: HistoricalScenario) -> Dict:
        """
        Apply historical scenario to portfolio
        """
        # Calculate impact on each position
        position_impacts = {}
        total_pnl = 0
        
        for asset, position_value in self.portfolio.items():
            asset_class = self.asset_class_mapping.get(asset, 'US_EQUITIES')
            
            # Get scenario return for this asset class
            if asset_class in scenario.market_moves:
                scenario_return = scenario.market_moves[asset_class]
            else:
                # Default to equity-like behavior
                scenario_return = scenario.market_moves.get('US_EQUITIES', -0.10)
            
            # Calculate P&L
            pnl = position_value * scenario_return
            total_pnl += pnl
            
            position_impacts[asset] = {
                'position_value': position_value,
                'scenario_return': scenario_return,
                'pnl': pnl,
                'pnl_percentage': (pnl / position_value) if position_value != 0 else 0
            }
        
        # Calculate portfolio-level metrics
        portfolio_return = total_pnl / self.portfolio_value
        
        # Volatility adjustment (if specified)
        vol_multiplier = scenario.market_moves.get('VIX', 1.0)
        
        return {
            'scenario_name': scenario.name,
            'scenario_date': scenario.date,
            'scenario_description': scenario.description,
            'duration_days': scenario.duration_days,
            'portfolio_value': self.portfolio_value,
            'total_pnl': total_pnl,
            'portfolio_return': portfolio_return,
            'portfolio_return_pct': portfolio_return * 100,
            'position_impacts': position_impacts,
            'worst_position': min(position_impacts.items(), 
                                 key=lambda x: x[1]['pnl'])[0],
            'best_position': max(position_impacts.items(), 
                                key=lambda x: x[1]['pnl'])[0],
            'volatility_multiplier': vol_multiplier,
            'characteristics': scenario.key_characteristics
        }
    
    def run_all_scenarios(self) -> pd.DataFrame:
        """
        Run all historical scenarios
        """
        results = []
        
        for scenario_id, scenario in HISTORICAL_SCENARIOS.items():
            result = self.apply_scenario(scenario)
            results.append({
                'scenario_id': scenario_id,
                'scenario_name': result['scenario_name'],
                'date': result['scenario_date'],
                'duration_days': result['duration_days'],
                'total_pnl': result['total_pnl'],
                'portfolio_return_pct': result['portfolio_return_pct'],
                'worst_position': result['worst_position']
            })
        
        return pd.DataFrame(results).sort_values('total_pnl')

# Example Usage
if __name__ == "__main__":
    # Sample portfolio
    portfolio = {
        'AAPL': 500000,
        'GOOGL': 300000,
        'MSFT': 400000,
        'AMZN': 200000,
        'SPY': 600000,
        'TLT': 200000,   # Bonds
        'GLD': 100000    # Gold
    }
    
    # Asset class mapping
    asset_classes = {
        'AAPL': 'US_EQUITIES',
        'GOOGL': 'US_EQUITIES',
        'MSFT': 'US_EQUITIES',
        'AMZN': 'US_EQUITIES',
        'SPY': 'US_EQUITIES',
        'TLT': 'US_BONDS',
        'GLD': 'GOLD'
    }
    
    # Run stress tests
    stress_tester = HistoricalStressTester(portfolio, asset_classes)
    
    # Single scenario
    lehman_result = stress_tester.apply_scenario(HISTORICAL_SCENARIOS['lehman_crisis_2008'])
    
    print("Lehman Crisis 2008 Stress Test")
    print("="*60)
    print(f"Portfolio Value: ${lehman_result['portfolio_value']:,.0f}")
print(f"Scenario P&L: ${lehman_result['total_pnl']:,.0f}")
print(f"Portfolio Return: {lehman_result['portfolio_return_pct']:.2f}%")
print(f"Worst Position: {lehman_result['worst_position']}")
print()
    
    # All scenarios
all_scenarios = stress_tester.run_all_scenarios()
print("All Historical Scenarios:")
print("=" * 60)
print(all_scenarios.to_string(index = False))
\`\`\`

### 2. Hypothetical Scenarios

Forward-looking "what if" scenarios:

\`\`\`python
@dataclass
class HypotheticalScenario:
    """Forward-looking stress scenario"""
    name: str
    description: str
    probability: str  # 'Low', 'Medium', 'High'
    market_moves: Dict[str, float]
    macro_assumptions: Dict[str, str]
    potential_triggers: List[str]

# Define hypothetical scenarios
HYPOTHETICAL_SCENARIOS = {
    'fed_shock': HypotheticalScenario(
        name='Fed Policy Shock',
        description='Fed raises rates 200bp unexpectedly',
        probability='Low',
        market_moves={
            'US_EQUITIES': -0.15,
            'US_BONDS': -0.10,      # Bond prices fall (yields up)
            'CREDIT_IG': -0.12,
            'CREDIT_HY': -0.25,
            'USD': 0.10,            # Dollar strengthens
            'EMERGING_EQUITIES': -0.25,
            'GOLD': -0.05,
            'VIX': 2.5
        },
        macro_assumptions={
            'Fed_Funds_Rate': '+200 bps',
            '10Y_Treasury': '+150 bps',
            'Inflation': 'Persistent above 5%',
            'GDP_Growth': '-1% to -2%'
        },
        potential_triggers=[
            'Inflation expectations unanchor',
            'Fed credibility crisis',
            'Economic overheating',
            'Currency crisis'
        ]
    ),
    
    'china_crisis': HypotheticalScenario(
        name='China Economic Crisis',
        description='Chinese property bubble bursts, banking crisis',
        probability='Medium',
        market_moves={
            'CHINA_EQUITIES': -0.40,
            'EMERGING_EQUITIES': -0.30,
            'ASIA_EX_JAPAN': -0.25,
            'US_EQUITIES': -0.15,
            'EUROPE_EQUITIES': -0.20,
            'COMMODITIES': -0.30,    # Demand shock
            'COPPER': -0.40,
            'IRON_ORE': -0.50,
            'US_BONDS': 0.10,        # Safe haven
            'CREDIT_HY': -0.20,
            'CNY': -0.15,            # Yuan devaluation
            'VIX': 3.0
        },
        macro_assumptions={
            'China_GDP': 'Falls below 2%',
            'Property_Prices': '-30% to -40%',
            'Banking_System': 'NPLs spike to 15%+',
            'Capital_Controls': 'Implemented',
            'Trade': 'Global trade contracts'
        },
        potential_triggers=[
            'Evergrande contagion spreads',
            'Local government debt crisis',
            'Capital flight accelerates',
            'Political instability'
        ]
    ),
    
    'cyber_attack': HypotheticalScenario(
        name='Major Cyber Attack on Financial Infrastructure',
        description='Coordinated attack on payment systems',
        probability='Low',
        market_moves={
            'US_EQUITIES': -0.20,
            'FINANCIALS': -0.35,     # Banks hit hardest
            'TECH': -0.25,
            'CRYPTO': -0.40,
            'US_BONDS': 0.05,
            'GOLD': 0.15,
            'VIX': 4.0
        },
        macro_assumptions={
            'Markets': 'Potential temporary closure',
            'Settlement': 'Delayed or failed settlements',
            'Trust': 'Confidence in system shaken',
            'Liquidity': 'Severe disruption'
        },
        potential_triggers=[
            'State-sponsored attack',
            'Ransomware on major bank',
            'Swift system compromised',
            'Clearing house attacked'
        ]
    ),
    
    'oil_shock': HypotheticalScenario(
        name='Oil Price Shock',
        description='Oil spikes to $200/barrel',
        probability='Low',
        market_moves={
            'OIL': 1.5,             # 150% increase
            'ENERGY_STOCKS': 0.50,
            'US_EQUITIES': -0.12,
            'EUROPE_EQUITIES': -0.18,  # More dependent on energy imports
            'EMERGING_EQUITIES': -0.15,
            'AIRLINES': -0.40,
            'INDUSTRIALS': -0.25,
            'BONDS': -0.05,         # Inflation concerns
            'GOLD': 0.20,
            'VIX': 2.0
        },
        macro_assumptions={
            'Inflation': '+3% to +5%',
            'GDP_Growth': '-2% to -3%',
            'Consumer_Spending': 'Significant decline',
            'Central_Banks': 'Difficult policy choices'
        },
        potential_triggers=[
            'Middle East conflict',
            'OPEC supply cut',
            'Major refinery outages',
            'Strategic reserve depletion'
        ]
    )
}

class HypotheticalStressTester:
    """
    Run forward-looking stress scenarios
    """
    def __init__(self, portfolio: Dict[str, float], asset_class_mapping: Dict[str, str]):
        self.portfolio = portfolio
        self.asset_class_mapping = asset_class_mapping
        self.portfolio_value = sum(portfolio.values())
    
    def apply_scenario(self, scenario: HypotheticalScenario) -> Dict:
        """Apply hypothetical scenario"""
        position_impacts = {}
        total_pnl = 0
        
        for asset, position_value in self.portfolio.items():
            asset_class = self.asset_class_mapping.get(asset, 'US_EQUITIES')
            
            scenario_return = scenario.market_moves.get(asset_class, 
                                                       scenario.market_moves.get('US_EQUITIES', -0.10))
            
            pnl = position_value * scenario_return
            total_pnl += pnl
            
            position_impacts[asset] = {
                'position_value': position_value,
                'scenario_return': scenario_return,
                'pnl': pnl
            }
        
        portfolio_return = total_pnl / self.portfolio_value
        
        return {
            'scenario_name': scenario.name,
            'description': scenario.description,
            'probability': scenario.probability,
            'total_pnl': total_pnl,
            'portfolio_return': portfolio_return,
            'portfolio_return_pct': portfolio_return * 100,
            'macro_assumptions': scenario.macro_assumptions,
            'potential_triggers': scenario.potential_triggers,
            'position_impacts': position_impacts
        }
    
    def scenario_comparison(self) -> pd.DataFrame:
        """
        Compare all hypothetical scenarios
        """
        results = []
        
        for scenario_id, scenario in HYPOTHETICAL_SCENARIOS.items():
            result = self.apply_scenario(scenario)
            results.append({
                'scenario': result['scenario_name'],
                'probability': result['probability'],
                'total_pnl': result['total_pnl'],
                'return_pct': result['portfolio_return_pct']
            })
        
        return pd.DataFrame(results).sort_values('total_pnl')
\`\`\`

### 3. Reverse Stress Testing

Work backwards: "What would cause us to lose $X or fail?"

\`\`\`python
class ReverseStressTester:
    """
    Find scenarios that would cause catastrophic losses
    """
    def __init__(self, portfolio: Dict[str, float], 
                 asset_class_mapping: Dict[str, str],
                 failure_threshold: float = 0.30):
        """
        Args:
            portfolio: Current portfolio
            asset_class_mapping: Asset to asset class mapping
            failure_threshold: Loss percentage that defines "failure"
        """
        self.portfolio = portfolio
        self.asset_class_mapping = asset_class_mapping
        self.portfolio_value = sum(portfolio.values())
        self.failure_threshold = failure_threshold
        self.failure_loss = self.portfolio_value * failure_threshold
    
    def find_breaking_scenarios(self, n_simulations: int = 10000) -> List[Dict]:
        """
        Generate random scenarios and find ones that break portfolio
        """
        breaking_scenarios = []
        
        # Get unique asset classes in portfolio
        asset_classes = set(self.asset_class_mapping.values())
        
        for i in range(n_simulations):
            # Generate random scenario
            scenario_returns = {}
            for asset_class in asset_classes:
                # Random return between -60% and +40%
                # Skewed negative for stress testing
                scenario_returns[asset_class] = np.random.triangular(-0.60, -0.20, 0.40)
            
            # Calculate portfolio impact
            total_pnl = 0
            for asset, position_value in self.portfolio.items():
                asset_class = self.asset_class_mapping[asset]
                pnl = position_value * scenario_returns[asset_class]
                total_pnl += pnl
            
            # If this breaks portfolio, save it
            if total_pnl < -self.failure_loss:
                breaking_scenarios.append({
                    'scenario_id': i,
                    'total_loss': abs(total_pnl),
                    'loss_percentage': abs(total_pnl) / self.portfolio_value,
                    'asset_class_returns': scenario_returns.copy()
                })
        
        # Sort by severity
        breaking_scenarios.sort(key=lambda x: x['total_loss'], reverse=True)
        
        return breaking_scenarios
    
    def analyze_breaking_patterns(self, breaking_scenarios: List[Dict]) -> Dict:
        """
        What do breaking scenarios have in common?
        """
        if not breaking_scenarios:
            return {'message': 'No breaking scenarios found'}
        
        # Analyze common patterns
        asset_class_impacts = {}
        
        # Get all asset classes
        asset_classes = list(breaking_scenarios[0]['asset_class_returns'].keys())
        
        for asset_class in asset_classes:
            returns = [s['asset_class_returns'][asset_class] 
                      for s in breaking_scenarios]
            
            asset_class_impacts[asset_class] = {
                'mean_return': np.mean(returns),
                'median_return': np.median(returns),
                'worst_return': min(returns),
                'std_return': np.std(returns)
            }
        
        # Find most critical asset class
        critical_asset_class = min(asset_class_impacts.items(),
                                  key=lambda x: x[1]['mean_return'])[0]
        
        return {
            'total_breaking_scenarios': len(breaking_scenarios),
            'asset_class_impacts': asset_class_impacts,
            'critical_asset_class': critical_asset_class,
            'average_loss': np.mean([s['total_loss'] for s in breaking_scenarios]),
            'worst_case_loss': max([s['total_loss'] for s in breaking_scenarios])
        }
    
    def scenario_path_to_failure(self, target_asset_class: str) -> Dict:
        """
        How bad does a specific asset class need to move to break us?
        """
        # Calculate exposure to this asset class
        exposure = sum(
            position_value 
            for asset, position_value in self.portfolio.items()
            if self.asset_class_mapping[asset] == target_asset_class
        )
        
        # How much loss can this asset class cause?
        max_loss_from_asset_class = exposure  # If goes to zero
        
        # What return would cause failure threshold?
        # failure_loss = exposure * return
        # return = failure_loss / exposure
        breaking_return = -self.failure_loss / exposure if exposure > 0 else None
        
        return {
            'asset_class': target_asset_class,
            'exposure': exposure,
            'exposure_percentage': exposure / self.portfolio_value,
            'max_possible_loss': max_loss_from_asset_class,
            'breaking_return': breaking_return,
            'breaking_return_pct': breaking_return * 100 if breaking_return else None,
            'feasibility': 'Possible' if breaking_return and breaking_return > -1.0 else 'Extreme'
        }

# Example
if __name__ == "__main__":
    reverse_tester = ReverseStressTester(portfolio, asset_classes, failure_threshold=0.25)
    
    # Find breaking scenarios
    breaking = reverse_tester.find_breaking_scenarios(n_simulations=10000)
    
    print(f"Found {len(breaking)} scenarios that would cause 25%+ loss")
    print()
    
    # Analyze patterns
    patterns = reverse_tester.analyze_breaking_patterns(breaking)
    print("Breaking Scenario Patterns:")
    print("="*60)
    print(f"Critical asset class: {patterns['critical_asset_class']}")
    print(f"Average loss in breaking scenarios: ${patterns['average_loss']:, .0f}")
print(f"Worst case: ${patterns['worst_case_loss']:,.0f}")
print()
    
    # Path to failure for each asset class
    print("Path to Failure by Asset Class:")
    print("=" * 60)
for asset_class in set(asset_classes.values()):
    path = reverse_tester.scenario_path_to_failure(asset_class)
print(f"{path['asset_class']}:")
print(f"  Exposure: ${path['exposure']:,.0f} ({path['exposure_percentage']*100:.1f}%)")
if path['breaking_return_pct']:
    print(f"  Breaking return: {path['breaking_return_pct']:.1f}%")
print(f"  Feasibility: {path['feasibility']}")
print()
\`\`\`

### 4. Multi-Factor Stress Tests

Combine multiple risk factors:

\`\`\`python
class MultiFactorStressTest:
    """
    Stress test across multiple dimensions simultaneously
    """
    def __init__(self, portfolio, asset_class_mapping):
        self.portfolio = portfolio
        self.asset_class_mapping = asset_class_mapping
        self.portfolio_value = sum(portfolio.values())
    
    def generate_multifactor_scenario(self,
                                     equity_shock: float = -0.20,
                                     rates_shock: float = 0.02,  # 200bp increase
                                     credit_spread_shock: float = 0.03,  # 300bp widening
                                     fx_shock: float = 0.10,  # Dollar appreciation
                                     vol_shock: float = 2.0) -> Dict:
        """
        Generate comprehensive stress scenario
        """
        # Calculate impact on each asset class
        impacts = {}
        
        # Equities - direct impact
        impacts['US_EQUITIES'] = equity_shock
        impacts['EUROPE_EQUITIES'] = equity_shock * 1.1  # Slightly worse
        impacts['EMERGING_EQUITIES'] = equity_shock * 1.3 - fx_shock * 0.5  # EM hit by USD
        
        # Bonds - duration impact
        # For 10-year bond, duration ≈ 9
        # Price change = -Duration × ΔYield
        impacts['US_BONDS'] = -9 * rates_shock  # ≈ -18% for 10-year
        
        # Credit - equity shock + spread widening
        impacts['CREDIT_IG'] = equity_shock * 0.3 - 5 * credit_spread_shock
        impacts['CREDIT_HY'] = equity_shock * 0.6 - 4 * credit_spread_shock
        
        # Commodities
        impacts['COMMODITIES'] = equity_shock * 0.8  # Recession indicator
        impacts['GOLD'] = -fx_shock * 0.5  # Inverse to dollar
        
        # Calculate portfolio impact
        total_pnl = 0
        position_details = {}
        
        for asset, position_value in self.portfolio.items():
            asset_class = self.asset_class_mapping.get(asset, 'US_EQUITIES')
            impact = impacts.get(asset_class, equity_shock)
            
            pnl = position_value * impact
            total_pnl += pnl
            
            position_details[asset] = {
                'impact': impact,
                'pnl': pnl
            }
        
        return {
            'factors': {
                'equity_shock': equity_shock,
                'rates_shock': rates_shock,
                'credit_spread_shock': credit_spread_shock,
                'fx_shock': fx_shock,
                'vol_shock': vol_shock
            },
            'total_pnl': total_pnl,
            'portfolio_return': total_pnl / self.portfolio_value,
            'position_details': position_details,
            'asset_class_impacts': impacts
        }
    
    def scenario_grid(self) -> pd.DataFrame:
        """
        Test grid of scenarios with varying severity
        """
        results = []
        
        # Mild, Moderate, Severe scenarios
        scenarios = [
            ('Mild', -0.10, 0.01, 0.01, 0.05, 1.5),
            ('Moderate', -0.20, 0.02, 0.02, 0.10, 2.5),
            ('Severe', -0.35, 0.03, 0.04, 0.15, 4.0),
            ('Extreme', -0.50, 0.05, 0.06, 0.25, 6.0)
        ]
        
        for name, equity, rates, credit, fx, vol in scenarios:
            result = self.generate_multifactor_scenario(
                equity, rates, credit, fx, vol
            )
            results.append({
                'scenario': name,
                'equity_shock': f"{equity*100:.0f}%",
                'rates_shock': f"+{rates*100:.0f}bp",
                'total_pnl': result['total_pnl'],
                'return_pct': f"{result['portfolio_return']*100:.1f}%"
            })
        
        return pd.DataFrame(results)
\`\`\`

## Regulatory Stress Testing

### CCAR/DFAST (US Banks)

Comprehensive Capital Analysis and Review:

\`\`\`python
class CCARStressTest:
    """
    Federal Reserve stress testing framework
    """
    def __init__(self, balance_sheet: Dict, income_statement: Dict):
        self.balance_sheet = balance_sheet
        self.income_statement = income_statement
    
    def severely_adverse_scenario(self) -> Dict:
        """
        Fed's Severely Adverse Scenario (2024 example)
        """
        scenario = {
            'description': 'Fed Severely Adverse Scenario',
            'assumptions': {
                'Real_GDP': 'Decline of 6.5% from peak to trough',
                'Unemployment_Rate': 'Rises 5.75pp to 10%',
                'Equity_Prices': 'Decline 45%',
                'House_Prices': 'Decline 25%',
                'BBB_Corporate_Spreads': 'Widen to 5.5%',
                'VIX': 'Spikes to 70',
                'Treasury_10Y': 'Falls to 1.0%'
            },
            'horizon': '9 quarters',
            'focus': [
                'Credit losses',
                'Trading losses',
                'PPNR (Pre-Provision Net Revenue)',
                'Capital ratios'
            ]
        }
        
        # Calculate impact
        credit_losses = self.calculate_credit_losses(
            default_rate=0.08,  # 8% default rate
            loss_given_default=0.40
        )
        
        trading_losses = self.calculate_trading_losses(
            equity_shock=-0.45,
            rates_shock=-0.02,
            credit_shock=0.04
        )
        
        ppnr_impact = self.calculate_ppnr_impact(
            revenue_decline=-0.20,
            expense_increase=0.05
        )
        
        # Capital ratio impact
        tier1_capital = self.balance_sheet['tier1_capital']
        rwa = self.balance_sheet['risk_weighted_assets']
        
        total_losses = credit_losses + trading_losses
        new_tier1_capital = tier1_capital - total_losses - ppnr_impact
        
        current_tier1_ratio = tier1_capital / rwa
        stressed_tier1_ratio = new_tier1_capital / rwa
        
        return {
            'scenario': scenario,
            'credit_losses': credit_losses,
            'trading_losses': trading_losses,
            'ppnr_impact': ppnr_impact,
            'total_impact': total_losses + ppnr_impact,
            'current_tier1_ratio': current_tier1_ratio,
            'stressed_tier1_ratio': stressed_tier1_ratio,
            'passes': stressed_tier1_ratio >= 0.045,  # 4.5% minimum
            'buffer_remaining': stressed_tier1_ratio - 0.045
        }
    
    def calculate_credit_losses(self, default_rate: float, 
                               loss_given_default: float) -> float:
        """Calculate loan losses"""
        loan_portfolio = self.balance_sheet['total_loans']
        expected_loss = loan_portfolio * default_rate * loss_given_default
        return expected_loss
    
    def calculate_trading_losses(self, equity_shock: float,
                                rates_shock: float,
                                credit_shock: float) -> float:
        """Calculate trading book losses"""
        # Simplified - real CCAR is much more complex
        trading_assets = self.balance_sheet['trading_assets']
        
        # Weighted impact
        trading_loss = trading_assets * (
            0.4 * equity_shock +  # 40% equity exposure
            0.3 * rates_shock +   # 30% rates exposure
            0.3 * credit_shock    # 30% credit exposure
        )
        
        return abs(trading_loss)
    
    def calculate_ppnr_impact(self, revenue_decline: float,
                             expense_increase: float) -> float:
        """Pre-Provision Net Revenue impact"""
        baseline_revenue = self.income_statement['net_revenue']
        baseline_expenses = self.income_statement['operating_expenses']
        
        stressed_revenue = baseline_revenue * (1 + revenue_decline)
        stressed_expenses = baseline_expenses * (1 + expense_increase)
        
        baseline_ppnr = baseline_revenue - baseline_expenses
        stressed_ppnr = stressed_revenue - stressed_expenses
        
        ppnr_impact = baseline_ppnr - stressed_ppnr
        
        return ppnr_impact
\`\`\`

## Best Practices

### 1. Comprehensive Scenario Library

Maintain library of scenarios:
- Historical (updated regularly)
- Hypothetical (reviewed quarterly)
- Reverse stress tests (annual)
- Custom scenarios for specific risks

### 2. Regular Updates

- **Monthly**: Run key scenarios
- **Quarterly**: Full scenario library
- **Annual**: Develop new scenarios
- **Ad-hoc**: When risks emerge

### 3. Integration with Risk Limits

\`\`\`python
class StressTestIntegration:
    """
    Integrate stress testing with risk management
    """
    def __init__(self, portfolio, stress_test_results):
        self.portfolio = portfolio
        self.stress_test_results = stress_test_results
    
    def set_stress_based_limits(self, max_acceptable_loss: float) -> Dict:
        """
        Set position limits based on stress test results
        """
        # Find which positions contribute most to stress losses
        worst_scenario = min(self.stress_test_results, 
                           key=lambda x: x['total_pnl'])
        
        # Calculate contribution of each position
        contributions = worst_scenario['position_impacts']
        
        # Set limits
        limits = {}
        for asset, impact in contributions.items():
            current_position = self.portfolio[asset]
            loss_in_stress = abs(impact['pnl'])
            
            # If this position loses more than acceptable, reduce limit
            if loss_in_stress > max_acceptable_loss * 0.10:  # 10% of total acceptable
                # Reduce position limit
                reduction_factor = (max_acceptable_loss * 0.10) / loss_in_stress
                new_limit = current_position * reduction_factor
            else:
                new_limit = current_position * 1.2  # Allow 20% increase
            
            limits[asset] = {
                'current_position': current_position,
                'stress_loss': loss_in_stress,
                'new_limit': new_limit,
                'action': 'REDUCE' if new_limit < current_position else 'OK'
            }
        
        return limits
\`\`\`

## Key Takeaways

1. **Complement Historical Models**: VaR uses history; stress testing imagines future
2. **Multiple Scenarios**: Historical, hypothetical, reverse stress tests
3. **Regular Updates**: Scenarios must evolve with markets
4. **Regulatory Requirement**: CCAR, DFAST mandatory for banks
5. **Actionable**: Use stress tests to set limits and allocate capital
6. **Communication Tool**: Helps explain risks to management/board
7. **Crisis Planning**: Stress tests inform contingency plans

## Conclusion

Stress testing answers the question VaR cannot: "What happens in unprecedented scenarios?"

While VaR and CVaR tell us about normal market conditions, stress testing explores the tails and beyond. It's not about predicting the future - it's about being prepared for multiple possible futures.

As the saying goes: "The Fed's job is to take away the punch bowl just as the party gets going." Stress testing is the risk manager's job - to imagine the party ending badly and ensure you survive it.

Next: Market Risk Management - applying these concepts to specific risk types.
`;

