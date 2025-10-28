export const riskReportingDashboards = `
# Risk Reporting and Dashboards

## Introduction

"If you can't explain your risk clearly, you don't understand it."

Effective risk reporting is the bridge between technical risk systems and business decisions. Poor reporting contributed to the 2008 financial crisis - executives didn't understand the risks they were taking.

This section covers creating comprehensive risk reports and dashboards that communicate complex risks clearly to all stakeholders.

## Daily Risk Report Generation

\`\`\`python
import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

@dataclass
class PortfolioSnapshot:
    """Portfolio state snapshot"""
    date: datetime
    positions: Dict[str, float]
    market_values: Dict[str, float]
    total_value: float
    pnl: float
    var_95: float
    var_99: float
    cvar_99: float
    max_drawdown: float
    sharpe_ratio: float
    exposures: Dict[str, float]
    
class RiskReportGenerator:
    """
    Automated risk report generation
    """
    def __init__(self, portfolio_name: str):
        self.portfolio_name = portfolio_name
        self.snapshots = []
        
    def add_snapshot(self, snapshot: PortfolioSnapshot):
        """Add daily snapshot"""
        self.snapshots.append(snapshot)
        
    def generate_daily_report(self, date: datetime) -> str:
        """
        Generate comprehensive daily risk report
        
        Returns: HTML report string
        """
        # Find snapshot for date
        snapshot = None
        for s in self.snapshots:
            if s.date.date() == date.date():
                snapshot = s
                break
        
        if not snapshot:
            return f"No data for {date.date()}"
        
        # Generate report sections
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Daily Risk Report - {self.portfolio_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: right; }}
        th {{ background-color: #3498db; color: white; }}
        .metric {{ font-size: 24px; font-weight: bold; padding: 20px; margin: 10px 0; border-radius: 5px; }}
        .positive {{ background-color: #d4edda; color: #155724; }}
        .negative {{ background-color: #f8d7da; color: #721c24; }}
        .warning {{ background-color: #fff3cd; color: #856404; }}
        .status-green {{ background-color: #28a745; color: white; }}
        .status-yellow {{ background-color: #ffc107; color: black; }}
        .status-red {{ background-color: #dc3545; color: white; }}
    </style>
</head>
<body>
    <h1>Daily Risk Report</h1>
    <p><strong>Portfolio:</strong> {self.portfolio_name}</p>
    <p><strong>Date:</strong> {snapshot.date.strftime('%Y-%m-%d')}</p>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Executive Summary</h2>
    <div class="metric {'positive' if snapshot.pnl >= 0 else 'negative'}">
        Daily P&L: ${snapshot.pnl:,.2f}
</div>

    < table >
    <tr>
    <th>Metric </th>
    < th > Value </th>
    < th > Status </th>
    </tr>
    < tr >
    <td>Portfolio Value </td>
        < td > ${ snapshot.total_value:, .2f } </td>
            < td class="status-green" > OK </td>
                </tr>
                < tr >
                <td>99 % VaR </td>
                < td > ${ snapshot.var_99:, .2f } </td>
                    < td class="{'status-green' if snapshot.var_99 < 5000000 else 'status-yellow'}" >
                        { 'OK' if snapshot.var_99 < 5000000 else 'WARNING' }
                        </td>
                        </tr>
                        < tr >
                        <td>99 % CVaR </td>
                        < td > ${ snapshot.cvar_99:, .2f } </td>
                            < td class="status-green" > OK </td>
                                </tr>
                                < tr >
                                <td>Max Drawdown </td>
                                    < td > { snapshot.max_drawdown * 100: .2f } % </td>
                                    < td class="{'status-green' if snapshot.max_drawdown < 0.15 else 'status-red'}" >
                                        { 'OK' if snapshot.max_drawdown < 0.15 else 'BREACH' }
                                        </td>
                                        </tr>
                                        < tr >
                                        <td>Sharpe Ratio </td>
                                            < td > { snapshot.sharpe_ratio: .2f } </td>
                                            < td class="status-green" > OK </td>
                                                </tr>
                                                </table>

                                                < h2 > Top Positions </h2>
{ self._generate_positions_table(snapshot) }

<h2>Exposure Breakdown </h2>
{ self._generate_exposures_table(snapshot) }

<h2>Risk Metrics Trend </h2>
    < p > <em>Chart would be embedded here < /em></p >

        <h2>Limit Monitoring </h2>
{ self._generate_limits_table(snapshot) }

<p style="margin-top: 50px; color: #7f8c8d; font-size: 12px;" >
    This report is generated automatically.Please contact Risk Management with any questions.
    </p>
        </body>
        </html>
"""
return html
    
    def _generate_positions_table(self, snapshot: PortfolioSnapshot) -> str:
"""Generate top positions table"""
        # Sort by absolute value
sorted_positions = sorted(
    snapshot.positions.items(),
    key = lambda x: abs(snapshot.market_values.get(x[0], 0)),
    reverse = True
)[: 10]

html = "<table><tr><th>Symbol</th><th>Quantity</th><th>Market Value</th><th>% of Portfolio</th></tr>"

for symbol, quantity in sorted_positions:
    market_value = snapshot.market_values.get(symbol, 0)
pct = (abs(market_value) / snapshot.total_value * 100) if snapshot.total_value > 0 else 0

html += f"""
    < tr >
    <td style="text-align:left;" > { symbol } </td>
        < td > { quantity:, .0f } </td>
        < td > ${ market_value:, .2f } </td>
            < td > { pct: .2f } % </td>
            </tr>
"""

html += "</table>"
return html
    
    def _generate_exposures_table(self, snapshot: PortfolioSnapshot) -> str:
"""Generate exposure breakdown table"""
html = "<table><tr><th>Category</th><th>Exposure</th><th>% of Total</th></tr>"

total_exposure = sum(abs(v) for v in snapshot.exposures.values())

    for category, exposure in sorted(snapshot.exposures.items(), key = lambda x: abs(x[1]), reverse = True):
        pct = (abs(exposure) / total_exposure * 100) if total_exposure > 0 else 0
html += f"""
    < tr >
    <td style="text-align:left;" > { category } </td>
        < td > ${ exposure:, .2f } </td>
            < td > { pct: .2f } % </td>
            </tr>
"""

html += "</table>"
return html
    
    def _generate_limits_table(self, snapshot: PortfolioSnapshot) -> str:
"""Generate limits monitoring table"""
        # Example limits
limits = {
    '99% VaR': { 'current': snapshot.var_99, 'limit': 5000000 },
    'Max Drawdown': { 'current': snapshot.max_drawdown, 'limit': 0.15 },
    'Portfolio Value': { 'current': snapshot.total_value, 'limit': 100000000 }
}

html = "<table><tr><th>Limit</th><th>Current</th><th>Limit</th><th>Utilization</th><th>Status</th></tr>"

for limit_name, values in limits.items():
    current = values['current']
limit = values['limit']
utilization = (current / limit * 100) if limit > 0 else 0

if utilization < 80:
    status = '<span class="status-green">OK</span>'
            elif utilization < 100:
status = '<span class="status-yellow">WARNING</span>'
            else:
status = '<span class="status-red">BREACH</span>'

html += f"""
    < tr >
    <td style="text-align:left;" > { limit_name } </td>
        < td > { current:, .2f } </td>
        < td > { limit:, .2f } </td>
        < td > { utilization: .1f } % </td>
        < td > { status } </td>
        </tr>
"""

html += "</table>"
return html
    
    def generate_weekly_summary(self, end_date: datetime) -> str:
"""Generate weekly summary report"""
start_date = end_date - timedelta(days = 7)
        
        # Filter snapshots
week_snapshots = [
    s for s in self.snapshots 
            if start_date <= s.date <= end_date
        ]

if not week_snapshots:
    return "No data for week"
        
        # Calculate weekly metrics
total_pnl = sum(s.pnl for s in week_snapshots)
    avg_var = np.mean([s.var_99 for s in week_snapshots])
max_var = max(s.var_99 for s in week_snapshots)

    html = f"""
        < !DOCTYPE html >
            <html>
            <head><title>Weekly Risk Summary < /title></head >
                <body>
                <h1>Weekly Risk Summary </h1>
                    < p > <strong>Portfolio: </strong> {self.portfolio_name}</p >
                        <p><strong>Week Ending: </strong> {end_date.strftime('%Y-%m-%d')}</p >

                            <h2>Performance </h2>
                            < p > Total P & L: <strong>${ total_pnl:, .2f } </strong></p >

                                <h2>Risk Metrics </h2>
                                    < ul >
                                    <li>Average 99 % VaR: ${ avg_var:, .2f } </li>
                                        < li > Max 99 % VaR: ${ max_var:, .2f } </li>
                                            < li > Trading Days: { len(week_snapshots) } </li>
                                                </ul>
                                                </body>
                                                </html>
"""
return html

# Example Usage
if __name__ == "__main__":
    # Create report generator
generator = RiskReportGenerator("Hedge Fund Alpha")
    
    # Add sample snapshot
snapshot = PortfolioSnapshot(
    date = datetime.now(),
    positions = { 'AAPL': 10000, 'MSFT': -5000, 'GOOGL': 3000 },
    market_values = { 'AAPL': 1800000, 'MSFT': -1750000, 'GOOGL': 420000 },
    total_value = 10000000,
    pnl = 250000,
    var_95 = 3500000,
    var_99 = 4800000,
    cvar_99 = 5500000,
    max_drawdown = 0.12,
    sharpe_ratio = 1.8,
    exposures = { 'Technology': 5000000, 'Finance': 3000000, 'Healthcare': 2000000 }
)

generator.add_snapshot(snapshot)
    
    # Generate daily report
report = generator.generate_daily_report(datetime.now())
    
    # Save to file
with open('/tmp/risk_report.html', 'w') as f:
f.write(report)

print("Daily Risk Report Generated")
print(f"Saved to: /tmp/risk_report.html")
print()
print("Key Metrics:")
print(f"  Portfolio Value: ${snapshot.total_value:,.0f}")
print(f"  Daily P&L: ${snapshot.pnl:,.0f}")
print(f"  99% VaR: ${snapshot.var_99:,.0f}")
\`\`\`

## Interactive Risk Dashboard

\`\`\`python
class RiskDashboard:
    """
    Real-time risk dashboard
    
    In production, this would use:
    - Plotly/Dash for web interface
    - WebSocket for real-time updates
    - Redis for caching
    """
    def __init__(self):
        self.metrics = {}
        self.positions = {}
        
    def update_metric(self, metric_name: str, value: float, timestamp: datetime = None):
        """Update dashboard metric"""
        if timestamp is None:
            timestamp = datetime.now()
            
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append({
            'timestamp': timestamp,
            'value': value
        })
        
        # Keep last 1000 points
        if len(self.metrics[metric_name]) > 1000:
            self.metrics[metric_name] = self.metrics[metric_name][-1000:]
    
    def get_current_state(self) -> Dict:
        """Get current dashboard state"""
        state = {
            'timestamp': datetime.now(),
            'metrics': {}
        }
        
        for metric_name, history in self.metrics.items():
            if history:
                latest = history[-1]
                state['metrics'][metric_name] = {
                    'current': latest['value'],
                    'timestamp': latest['timestamp'],
                    'history_points': len(history)
                }
        
        return state
    
    def generate_charts(self) -> Dict[str, str]:
        """
        Generate charts for dashboard
        
        Returns: Dict of metric_name -> base64 encoded PNG
        """
        charts = {}
        
        for metric_name, history in self.metrics.items():
            if len(history) < 2:
                continue
            
            # Extract data
            timestamps = [h['timestamp'] for h in history]
            values = [h['value'] for h in history]
            
            # Create chart
            plt.figure(figsize=(10, 4))
            plt.plot(timestamps, values, linewidth=2)
            plt.title(f"{metric_name} Over Time")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            charts[metric_name] = image_base64
        
        return charts

# Example
dashboard = RiskDashboard()

# Simulate real-time updates
for i in range(50):
    timestamp = datetime.now() - timedelta(minutes=50-i)
    var = 4000000 + np.random.normal(0, 200000)
    dashboard.update_metric('99% VaR', var, timestamp)

print("Risk Dashboard")
print("="*70)
state = dashboard.get_current_state()
print(f"Metrics tracked: {len(state['metrics'])}")
for name, info in state['metrics'].items():
    print(f"  {name}: {info['current']:,.0f} ({info['history_points']} points)")
\`\`\`

## Regulatory Reporting

\`\`\`python
class RegulatoryReporter:
    """
    Generate regulatory risk reports
    
    Formats:
    - Basel III capital requirements
    - Dodd-Frank stress testing
    - EMIR reporting
    """
    def __init__(self, institution_name: str):
        self.institution_name = institution_name
        
    def generate_basel_capital_report(self, 
                                     market_risk_rwa: float,
                                     credit_risk_rwa: float,
                                     operational_risk_rwa: float,
                                     tier1_capital: float,
                                     total_capital: float) -> Dict:
        """
        Generate Basel III capital adequacy report
        
        Args:
            market_risk_rwa: Market risk-weighted assets
            credit_risk_rwa: Credit risk-weighted assets
            operational_risk_rwa: Operational risk-weighted assets
            tier1_capital: Tier 1 capital
            total_capital: Total regulatory capital
        """
        # Calculate total RWA
        total_rwa = market_risk_rwa + credit_risk_rwa + operational_risk_rwa
        
        # Calculate ratios
        cet1_ratio = tier1_capital / total_rwa if total_rwa > 0 else 0
        total_capital_ratio = total_capital / total_rwa if total_rwa > 0 else 0
        
        # Basel III minimums
        min_cet1 = 0.045  # 4.5%
        min_tier1 = 0.06  # 6%
        min_total = 0.08  # 8%
        
        report = {
            'institution': self.institution_name,
            'date': datetime.now(),
            'risk_weighted_assets': {
                'market_risk': market_risk_rwa,
                'credit_risk': credit_risk_rwa,
                'operational_risk': operational_risk_rwa,
                'total': total_rwa
            },
            'capital': {
                'tier1': tier1_capital,
                'total': total_capital
            },
            'ratios': {
                'cet1': cet1_ratio,
                'tier1': cet1_ratio,  # Simplified
                'total_capital': total_capital_ratio
            },
            'compliance': {
                'cet1': cet1_ratio >= min_cet1,
                'tier1': cet1_ratio >= min_tier1,
                'total_capital': total_capital_ratio >= min_total
            },
            'buffers': {
                'cet1_buffer': cet1_ratio - min_cet1,
                'tier1_buffer': cet1_ratio - min_tier1,
                'total_buffer': total_capital_ratio - min_total
            }
        }
        
        return report
    
    def generate_stress_test_report(self,
                                    baseline_var: float,
                                    stress_scenarios: Dict[str, float]) -> Dict:
        """
        Generate stress testing report (Dodd-Frank CCAR style)
        """
        report = {
            'institution': self.institution_name,
            'date': datetime.now(),
            'baseline_var': baseline_var,
            'stress_scenarios': {},
            'max_stress_loss': 0,
            'worst_scenario': None
        }
        
        for scenario_name, stress_var in stress_scenarios.items():
            loss_increase = stress_var - baseline_var
            report['stress_scenarios'][scenario_name] = {
                'stressed_var': stress_var,
                'loss_increase': loss_increase,
                'increase_pct': (loss_increase / baseline_var * 100) if baseline_var > 0 else 0
            }
            
            if loss_increase > report['max_stress_loss']:
                report['max_stress_loss'] = loss_increase
                report['worst_scenario'] = scenario_name
        
        return report

# Example
reporter = RegulatoryReporter("Bank of Examples")

# Basel III report
basel_report = reporter.generate_basel_capital_report(
    market_risk_rwa=500000000,
    credit_risk_rwa=2000000000,
    operational_risk_rwa=300000000,
    tier1_capital=180000000,
    total_capital=250000000
)

print("Basel III Capital Adequacy Report")
print("="*70)
print(f"Institution: {basel_report['institution']}")
print(f"Total RWA: ${basel_report['risk_weighted_assets']['total']:, .0f}")
print()
print("Capital Ratios:")
print(f"  CET1: {basel_report['ratios']['cet1']*100:.2f}% (min 4.5%)")
print(f"  Total Capital: {basel_report['ratios']['total_capital']*100:.2f}% (min 8.0%)")
print()
print("Compliance:")
for ratio_type, compliant in basel_report['compliance'].items():
    status = "✓ PASS" if compliant else "✗ FAIL"
print(f"  {ratio_type}: {status}")
\`\`\`

## Key Takeaways

1. **Clear Communication**: Reports must be understandable by non-technical stakeholders
2. **Automation**: Daily reports generated automatically
3. **Multiple Formats**: HTML, PDF, Excel, dashboards
4. **Real-Time Dashboards**: Live risk metrics
5. **Regulatory Compliance**: Basel III, CCAR, EMIR formats
6. **Visual**: Charts and graphs for quick understanding
7. **Actionable**: Highlight limit breaches and required actions

## Production Checklist

- [ ] Automated daily risk reports
- [ ] Weekly/monthly summary reports
- [ ] Real-time web dashboard
- [ ] Regulatory report templates
- [ ] Export to multiple formats (HTML, PDF, Excel)
- [ ] Email distribution lists
- [ ] Mobile-friendly dashboards
- [ ] Historical report archive
- [ ] Custom report builder
- [ ] Role-based access control

## Conclusion

Effective risk reporting bridges the gap between technical risk measurement and business decisions. Clear, timely, accurate reporting enables informed decision-making and regulatory compliance.

The 2008 crisis showed that complex risks poorly communicated lead to catastrophe. Modern risk reporting must be both comprehensive and comprehensible.

Next: BlackRock Aladdin Architecture - studying the world's largest risk platform.
`;

