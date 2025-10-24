export const automatedReportGeneration = {
  title: 'Automated Report Generation',
  id: 'automated-report-generation',
  content: `
# Automated Report Generation

## Introduction

Financial professionals spend significant time creating reports: portfolio performance reports, risk reports, market commentary, investment theses, and regulatory filings. These reports follow predictable structures but require domain expertise to interpret data and provide insights.

LLMs can automate report generation at scale while maintaining professional quality, freeing analysts to focus on strategy and decision-making. This section covers generating various financial reports using LLMs: portfolio reports, risk analysis, market commentary, performance attribution, and regulatory reporting.

### Why Automate Report Generation

**Time Savings**: Generate in minutes what takes hours manually
**Consistency**: Standardized format and quality across reports
**Scale**: Generate personalized reports for thousands of clients
**Timeliness**: Real-time reports as data updates
**Insights**: LLMs can identify patterns humans might miss

---

## Portfolio Performance Reports

### Generating Client Portfolio Reports

\`\`\`python
"""
Generate comprehensive portfolio performance reports
"""

import anthropic
from typing import Dict, List
import pandas as pd
from datetime import datetime, timedelta

class PortfolioReportGenerator:
    """
    Generate portfolio performance reports using LLMs
    """
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def generate_performance_report(self, portfolio_data: Dict,
                                   period: str = "Q4 2023") -> str:
        """
        Generate comprehensive portfolio performance report
        
        Args:
            portfolio_data: Portfolio holdings and performance data
            period: Reporting period
            
        Returns:
            Full report text in Markdown
        """
        # Format portfolio data for LLM
        holdings_summary = self._format_holdings(portfolio_data['holdings'])
        performance_metrics = self._format_metrics(portfolio_data['performance'])
        
        prompt = f"""Generate a comprehensive portfolio performance report for {period}.

Portfolio Data:
{holdings_summary}

Performance Metrics:
{performance_metrics}

The report should include:

1. Executive Summary (2-3 paragraphs)
   - Overall performance vs benchmark
   - Key wins and challenges
   - Portfolio changes made

2. Performance Analysis
   - Total return and breakdown (price appreciation vs dividends)
   - Comparison to benchmark
   - Risk-adjusted returns (Sharpe ratio analysis)
   - Monthly/quarterly performance table

3. Holdings Review
   - Top performers and underperformers
   - Sector allocation vs benchmark
   - Geographic diversification
   - Notable position changes

4. Market Context
   - How major market events affected the portfolio
   - Sector trends impacting performance

5. Risk Metrics
   - Portfolio volatility vs benchmark
   - Maximum drawdown
   - Beta and correlation analysis

6. Looking Ahead
   - Current positioning
   - Key risks and opportunities
   - Areas of focus for next period

Use professional investment management language. Include specific numbers and percentages.
Format as a polished Markdown document suitable for client presentation."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def _format_holdings(self, holdings: List[Dict]) -> str:
        """Format holdings data for prompt"""
        df = pd.DataFrame(holdings)
        
        summary = f"""
Total Portfolio Value: \\\${df['market_value'].sum():,.2f}
Number of Holdings: {len(holdings)}

Top 10 Holdings:
{df.nlargest(10, 'market_value')[['ticker', 'name', 'weight', 'return', 'market_value']].to_string(index=False)}

Sector Allocation:
{df.groupby('sector')['weight'].sum().to_string()}
"""
        return summary
    
    def _format_metrics(self, metrics: Dict) -> str:
        """Format performance metrics"""
        return f"""
Period Return: {metrics.get('return', 0):.2f}%
Benchmark Return: {metrics.get('benchmark_return', 0):.2f}%
Outperformance: {metrics.get('alpha', 0):.2f}%

Risk Metrics:
- Volatility: {metrics.get('volatility', 0):.2f}%
- Sharpe Ratio: {metrics.get('sharpe', 0):.2f}
- Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%
- Beta: {metrics.get('beta', 1.0):.2f}

Income Generated:
- Dividend Income: \\\${metrics.get('dividend_income', 0):,.2f}
- Dividend Yield: {metrics.get('dividend_yield', 0):.2f}%
"""
    
    def generate_personalized_report(self, client_data: Dict,
                                    portfolio_data: Dict) -> str:
        """
        Generate personalized report for individual client
        
        Args:
            client_data: Client profile and preferences
            portfolio_data: Portfolio performance data
            
        Returns:
            Personalized report
        """
        prompt = f"""Generate a personalized portfolio report for this client.

Client Profile:
- Name: {client_data.get('name')}
- Risk Tolerance: {client_data.get('risk_tolerance')}
- Investment Objective: {client_data.get('objective')}
- Time Horizon: {client_data.get('time_horizon')}
- Age: {client_data.get('age')}
- Special Considerations: {client_data.get('special_notes', 'None')}

Portfolio Performance:
{self._format_metrics(portfolio_data['performance'])}

Key Holdings:
{self._format_holdings(portfolio_data['holdings'])}

Create a personalized report that:
1. Speaks directly to the client's goals and concerns
2. Explains performance in context of their risk tolerance
3. Addresses how the portfolio aligns with their objectives
4. Provides actionable insights relevant to their situation
5. Uses appropriate tone for the client's sophistication level

Format as a professional client letter."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text

# Example usage
generator = PortfolioReportGenerator(api_key="your-key")

# Sample portfolio data
portfolio_data = {
    'holdings': [
        {
            'ticker': 'AAPL',
            'name': 'Apple Inc.',
            'sector': 'Technology',
            'weight': 15.5,
            'return': 48.2,
            'market_value': 155000
        },
        {
            'ticker': 'MSFT',
            'name': 'Microsoft Corp.',
            'sector': 'Technology',
            'weight': 12.3,
            'return': 57.1,
            'market_value': 123000
        },
        # ... more holdings
    ],
    'performance': {
        'return': 22.5,
        'benchmark_return': 18.3,
        'alpha': 4.2,
        'volatility': 15.2,
        'sharpe': 1.35,
        'max_drawdown': -12.4,
        'beta': 1.08,
        'dividend_income': 12500,
        'dividend_yield': 1.8
    }
}

# Generate report
report = generator.generate_performance_report(portfolio_data, period="Q4 2023")
print(report)

# Generate personalized report
client_data = {
    'name': 'John Smith',
    'risk_tolerance': 'Moderate-Aggressive',
    'objective': 'Long-term growth with some income',
    'time_horizon': '15 years until retirement',
    'age': 50
}

personalized = generator.generate_personalized_report(client_data, portfolio_data)
print("\\n\\n" + "="*60)
print("PERSONALIZED REPORT")
print("="*60)
print(personalized)
\`\`\`

---

## Risk Analysis Reports

### Automated Risk Assessment Documentation

\`\`\`python
"""
Generate risk analysis and risk management reports
"""

class RiskReportGenerator:
    """
    Generate comprehensive risk reports
    """
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def generate_risk_report(self, portfolio_data: Dict,
                            market_data: Dict,
                            risk_metrics: Dict) -> str:
        """
        Generate comprehensive risk analysis report
        
        Args:
            portfolio_data: Portfolio holdings and exposures
            market_data: Current market conditions
            risk_metrics: Calculated risk metrics (VaR, CVaR, etc.)
            
        Returns:
            Risk report in Markdown
        """
        prompt = f"""Generate a comprehensive risk analysis report for this portfolio.

Portfolio Composition:
- Total Value: \\${portfolio_data.get('total_value'):,.2f}
- Number of Positions: {portfolio_data.get('num_positions')}
- Largest Position: {portfolio_data.get('largest_position')} ({portfolio_data.get('largest_weight')}%)

Sector Exposures:
{self._format_dict(portfolio_data.get('sector_exposure', {}))}

Geographic Exposures:
{self._format_dict(portfolio_data.get('geo_exposure', {}))}

Risk Metrics:
- Value at Risk (95% confidence, 1-day): \\${risk_metrics.get('var_95'):,.2f}
- Conditional VaR (Expected Shortfall): \\${risk_metrics.get('cvar_95'):,.2f}
- Portfolio Beta: {risk_metrics.get('beta', 1.0):.2f}
- Portfolio Volatility: {risk_metrics.get('volatility', 0):.1f}%
- Maximum Drawdown (1Y): {risk_metrics.get('max_drawdown', 0):.1f}%
- Correlation with Market: {risk_metrics.get('market_correlation', 0):.2f}

Current Market Environment:
- VIX Level: {market_data.get('vix', 0):.1f}
- Market Trend: {market_data.get('trend', 'Unknown')}
- Recent Volatility: {market_data.get('recent_volatility', 'Normal')}

Generate a risk report including:

1. Executive Summary
   - Overall risk assessment (Low/Medium/High)
   - Key risk factors
   - Immediate concerns

2. Risk Analysis
   - Market risk exposure
   - Concentration risk
   - Sector and geographic risks
   - Liquidity risk
   - Currency risk (if applicable)

3. Stress Testing
   - Historical scenario analysis
   - Hypothetical shock scenarios
   - Expected portfolio impact

4. Risk-Adjusted Performance
   - Sharpe and Sortino ratios
   - Risk-return tradeoff analysis

5. Risk Mitigation Recommendations
   - Hedging suggestions
   - Rebalancing recommendations
   - Position size adjustments

Use professional risk management terminology.
Format as Markdown suitable for risk committee presentation."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def generate_var_explanation(self, var_metrics: Dict,
                                 client_level: str = "sophisticated") -> str:
        """
        Generate explanation of VaR for clients
        
        Args:
            var_metrics: VaR calculations
            client_level: "sophisticated", "intermediate", or "novice"
            
        Returns:
            VaR explanation appropriate for audience
        """
        technical_detail = {
            'sophisticated': 'Include technical methodology and assumptions',
            'intermediate': 'Balance technical accuracy with accessibility',
            'novice': 'Use simple analogies and avoid jargon'
        }
        
        prompt = f"""Explain Value at Risk (VaR) to a {client_level} investor.

VaR Metrics:
- 1-Day VaR (95%): \\${var_metrics.get('var_1d_95'):,.2f}
- 1-Day VaR (99%): \\${var_metrics.get('var_1d_99'):,.2f}
- 10-Day VaR (95%): \\${var_metrics.get('var_10d_95'):,.2f}
- Portfolio Value: \\${var_metrics.get('portfolio_value'):,.2f}

Instructions: {technical_detail[client_level]}

Explain:
1. What VaR means in simple terms
2. How to interpret these specific numbers
3. What assumptions are made
4. Limitations of VaR
5. How this should inform decisions

Write 2-3 paragraphs."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def _format_dict(self, data: Dict) -> str:
        """Format dictionary for prompt"""
        return "\\n".join([f"- {k}: {v}" for k, v in data.items()])

# Example usage
risk_generator = RiskReportGenerator(api_key="your-key")

portfolio_data = {
    'total_value': 1000000,
    'num_positions': 25,
    'largest_position': 'AAPL',
    'largest_weight': 15.5,
    'sector_exposure': {
        'Technology': 40.2,
        'Healthcare': 22.1,
        'Financials': 18.5,
        'Consumer': 12.2,
        'Other': 7.0
    },
    'geo_exposure': {
        'United States': 72.3,
        'Europe': 15.8,
        'Asia': 8.2,
        'Other': 3.7
    }
}

risk_metrics = {
    'var_95': 18500,
    'cvar_95': 25300,
    'beta': 1.15,
    'volatility': 18.2,
    'max_drawdown': -15.3,
    'market_correlation': 0.87
}

market_data = {
    'vix': 18.5,
    'trend': 'Upward with increasing volatility',
    'recent_volatility': 'Above average'
}

risk_report = risk_generator.generate_risk_report(
    portfolio_data, market_data, risk_metrics
)
print(risk_report)
\`\`\`

---

## Market Commentary Generation

### Automated Market Analysis and Commentary

\`\`\`python
"""
Generate market commentary and analysis
"""

class MarketCommentaryGenerator:
    """
    Generate market commentary and analysis reports
    """
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def generate_daily_commentary(self, market_data: Dict,
                                  news_highlights: List[str],
                                  economic_calendar: List[str]) -> str:
        """
        Generate daily market commentary
        
        Args:
            market_data: Market performance data
            news_highlights: Key news items
            economic_calendar: Upcoming events
            
        Returns:
            Market commentary
        """
        market_summary = f"""
Market Performance:
- S&P 500: {market_data.get('sp500_change', 0):+.2f}%
- Dow Jones: {market_data.get('dow_change', 0):+.2f}%
- NASDAQ: {market_data.get('nasdaq_change', 0):+.2f}%
- Russell 2000: {market_data.get('russell_change', 0):+.2f}%

Sector Performance (Best to Worst):
{self._format_sectors(market_data.get('sector_performance', {}))}

Volume: {market_data.get('volume_vs_avg', 0):+.1f}% vs average
VIX: {market_data.get('vix', 0):.2f} ({market_data.get('vix_change', 0):+.2f})
"""

        news_summary = "\\n".join([f"- {item}" for item in news_highlights])
        calendar_summary = "\\n".join([f"- {item}" for item in economic_calendar])
        
        prompt = f"""Write a daily market commentary for {datetime.now().strftime('%B %d, %Y')}.

{market_summary}

Key News Today:
{news_summary}

Upcoming Events:
{calendar_summary}

Write a professional market commentary (300-400 words) that:
1. Summarizes the day's market action with key drivers
2. Analyzes sector rotation and what it signals
3. Discusses key news impacts
4. Previews upcoming events and their potential impact
5. Provides market outlook and key levels to watch

Use active voice and professional investment language.
Be objective but insightful."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def generate_weekly_outlook(self, weekly_data: Dict,
                               technical_levels: Dict,
                               sentiment_data: Dict) -> str:
        """
        Generate weekly market outlook
        
        Args:
            weekly_data: Week's performance summary
            technical_levels: Key technical levels
            sentiment_data: Market sentiment indicators
            
        Returns:
            Weekly outlook report
        """
        prompt = f"""Generate a weekly market outlook report.

Week's Performance:
{self._format_dict(weekly_data)}

Technical Levels to Watch:
{self._format_dict(technical_levels)}

Sentiment Indicators:
{self._format_dict(sentiment_data)}

Write a comprehensive weekly outlook (600-800 words) covering:

1. Week in Review
   - Major market moves and themes
   - What worked and what didn't

2. Technical Analysis
   - Key support/resistance levels
   - Trend analysis
   - Chart patterns

3. Sentiment Analysis
   - Bull/bear indicators
   - Positioning and flows
   - Contrarian signals

4. Week Ahead Preview
   - Earnings calendar highlights
   - Economic data releases
   - Potential catalysts

5. Trading Strategies
   - Sectors to watch
   - Risk management considerations
   - Key levels for entry/exit

Professional investment publication style."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def generate_thematic_research(self, theme: str,
                                  supporting_data: Dict) -> str:
        """
        Generate thematic investment research report
        
        Args:
            theme: Investment theme (e.g., "AI Revolution", "Energy Transition")
            supporting_data: Data and analysis supporting the theme
            
        Returns:
            Research report
        """
        prompt = f"""Write an investment research report on the theme: "{theme}"

Supporting Data and Analysis:
{json.dumps(supporting_data, indent=2)}

Create a comprehensive research report (1000-1500 words) with:

1. Executive Summary
   - Investment thesis in 3-4 sentences
   - Key recommendations

2. Theme Overview
   - What is driving this trend
   - Why it matters now
   - Long-term potential

3. Market Analysis
   - Current state of the market
   - Growth projections
   - Key metrics and indicators

4. Investment Opportunities
   - Specific sectors/companies positioned to benefit
   - Risk/reward assessment
   - Valuation considerations

5. Risks and Challenges
   - Potential headwinds
   - Competitive threats
   - Execution risks

6. Investment Strategy
   - How to gain exposure
   - Portfolio allocation suggestions
   - Time horizon considerations

Write in professional sell-side research style.
Include specific actionable insights."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=3500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def _format_sectors(self, sectors: Dict) -> str:
        """Format sector performance"""
        sorted_sectors = sorted(sectors.items(), key=lambda x: x[1], reverse=True)
        return "\\n".join([f"  {sector}: {perf:+.2f}%" 
                          for sector, perf in sorted_sectors])
    
    def _format_dict(self, data: Dict) -> str:
        """Format dictionary"""
        return "\\n".join([f"- {k}: {v}" for k, v in data.items()])

# Example usage
commentary_gen = MarketCommentaryGenerator(api_key="your-key")

market_data = {
    'sp500_change': 1.25,
    'dow_change': 0.85,
    'nasdaq_change': 1.85,
    'russell_change': 0.65,
    'sector_performance': {
        'Technology': 2.3,
        'Communication Services': 1.8,
        'Consumer Discretionary': 1.2,
        'Financials': 0.5,
        'Healthcare': 0.2,
        'Energy': -0.5,
        'Utilities': -0.8
    },
    'volume_vs_avg': 5.2,
    'vix': 16.5,
    'vix_change': -1.2
}

news_highlights = [
    "Fed Chair Powell signals data-dependent approach",
    "Strong earnings from mega-cap tech drive rally",
    "Oil prices decline on demand concerns"
]

economic_calendar = [
    "Wednesday: FOMC Minutes Release",
    "Thursday: Weekly Jobless Claims",
    "Friday: PCE Inflation Data"
]

daily_commentary = commentary_gen.generate_daily_commentary(
    market_data, news_highlights, economic_calendar
)

print("Daily Market Commentary:")
print("="*60)
print(daily_commentary)
\`\`\`

---

## Performance Attribution Reports

### Explaining Portfolio Performance

\`\`\`python
"""
Generate performance attribution analysis
"""

class AttributionReportGenerator:
    """
    Generate performance attribution reports
    """
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def generate_attribution_report(self, attribution_data: Dict) -> str:
        """
        Generate performance attribution report
        
        Args:
            attribution_data: Attribution analysis data
            
        Returns:
            Attribution report
        """
        prompt = f"""Generate a performance attribution report explaining sources of portfolio returns.

Portfolio Return: {attribution_data.get('total_return', 0):.2f}%
Benchmark Return: {attribution_data.get('benchmark_return', 0):.2f}%
Active Return (Alpha): {attribution_data.get('active_return', 0):.2f}%

Attribution Breakdown:

Asset Allocation Effect: {attribution_data.get('allocation_effect', 0):.2f}%
(Difference from being overweight/underweight sectors vs benchmark)

Security Selection Effect: {attribution_data.get('selection_effect', 0):.2f}%
(Difference from picking outperforming/underperforming stocks)

Interaction Effect: {attribution_data.get('interaction_effect', 0):.2f}%

Sector Attribution:
{self._format_sector_attribution(attribution_data.get('sector_attribution', {}))}

Top Contributors to Performance:
{self._format_contributors(attribution_data.get('top_contributors', []))}

Top Detractors from Performance:
{self._format_contributors(attribution_data.get('top_detractors', []))}

Generate a report (800-1000 words) explaining:

1. Performance Overview
   - How the portfolio performed vs benchmark
   - Key sources of outperformance/underperformance

2. Attribution Analysis
   - Allocation decisions impact
   - Security selection impact
   - Which was more important this period

3. Sector Analysis
   - Which sector bets paid off
   - Which didn't work
   - Lessons learned

4. Individual Position Analysis
   - Biggest winners and why
   - Biggest losers and what went wrong
   - Position sizing impact

5. Looking Forward
   - Adjustments being made
   - Areas of opportunity
   - Risk management considerations

Use clear explanations suitable for client communication."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def _format_sector_attribution(self, sectors: Dict) -> str:
        """Format sector attribution"""
        output = []
        for sector, data in sectors.items():
            output.append(f"{sector}:")
            output.append(f"  Total Effect: {data.get('total', 0):+.2f}%")
            output.append(f"  Allocation: {data.get('allocation', 0):+.2f}%")
            output.append(f"  Selection: {data.get('selection', 0):+.2f}%")
        return "\\n".join(output)
    
    def _format_contributors(self, contributors: List[Dict]) -> str:
        """Format top contributors/detractors"""
        return "\\n".join([
            f"- {c['ticker']} ({c['name']}): {c['contribution']:+.2f}%"
            for c in contributors
        ])

# Example usage
attribution_gen = AttributionReportGenerator(api_key="your-key")

attribution_data = {
    'total_return': 24.5,
    'benchmark_return': 18.3,
    'active_return': 6.2,
    'allocation_effect': 1.8,
    'selection_effect': 4.1,
    'interaction_effect': 0.3,
    'sector_attribution': {
        'Technology': {
            'total': 3.2,
            'allocation': 1.5,
            'selection': 1.7
        },
        'Healthcare': {
            'total': 1.1,
            'allocation': 0.3,
            'selection': 0.8
        },
        'Financials': {
            'total': -0.5,
            'allocation': -0.2,
            'selection': -0.3
        }
    },
    'top_contributors': [
        {'ticker': 'NVDA', 'name': 'NVIDIA Corp', 'contribution': 2.8},
        {'ticker': 'MSFT', 'name': 'Microsoft Corp', 'contribution': 1.9},
        {'ticker': 'AAPL', 'name': 'Apple Inc', 'contribution': 1.5}
    ],
    'top_detractors': [
        {'ticker': 'XYZ', 'name': 'XYZ Corp', 'contribution': -0.8},
        {'ticker': 'ABC', 'name': 'ABC Inc', 'contribution': -0.5}
    ]
}

attribution_report = attribution_gen.generate_attribution_report(attribution_data)
print(attribution_report)
\`\`\`

---

## Production Report Generation System

### Automated Multi-Report Pipeline

\`\`\`python
"""
Production system for automated report generation
"""

from typing import List, Dict
import schedule
from datetime import datetime
import os

class ReportGenerationPipeline:
    """
    Automated report generation pipeline
    """
    
    def __init__(self, api_key: str, output_dir: str = "./reports"):
        self.portfolio_gen = PortfolioReportGenerator(api_key)
        self.risk_gen = RiskReportGenerator(api_key)
        self.commentary_gen = MarketCommentaryGenerator(api_key)
        self.attribution_gen = AttributionReportGenerator(api_key)
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_daily_reports(self, clients: List[Dict]):
        """
        Generate all daily reports
        
        Args:
            clients: List of client data
        """
        print(f"\\n[{datetime.now()}] Generating daily reports...")
        
        # Generate market commentary
        market_commentary = self._generate_market_commentary()
        self._save_report(market_commentary, "market_commentary_daily.md")
        
        print(f"Generated reports for {len(clients)} clients")
    
    def generate_monthly_reports(self, clients: List[Dict]):
        """
        Generate all monthly reports
        
        Args:
            clients: List of client data with portfolio info
        """
        print(f"\\n[{datetime.now()}] Generating monthly reports...")
        
        for client in clients:
            # Portfolio performance report
            portfolio_report = self.portfolio_gen.generate_personalized_report(
                client_data=client,
                portfolio_data=client['portfolio']
            )
            
            # Risk report
            risk_report = self.risk_gen.generate_risk_report(
                portfolio_data=client['portfolio'],
                market_data=self._get_market_data(),
                risk_metrics=client['risk_metrics']
            )
            
            # Save reports
            client_dir = os.path.join(self.output_dir, client['id'])
            os.makedirs(client_dir, exist_ok=True)
            
            self._save_report(
                portfolio_report,
                f"{client_dir}/portfolio_report_{datetime.now().strftime('%Y%m')}.md"
            )
            self._save_report(
                risk_report,
                f"{client_dir}/risk_report_{datetime.now().strftime('%Y%m')}.md"
            )
            
            print(f"Generated reports for client {client['name']}")
    
    def generate_quarterly_reports(self, clients: List[Dict]):
        """
        Generate comprehensive quarterly reports
        
        Args:
            clients: List of client data
        """
        print(f"\\n[{datetime.now()}] Generating quarterly reports...")
        
        for client in clients:
            # Comprehensive quarterly report
            quarterly_report = self._generate_quarterly_report(client)
            
            client_dir = os.path.join(self.output_dir, client['id'])
            self._save_report(
                quarterly_report,
                f"{client_dir}/quarterly_report_Q{self._get_quarter()}.md"
            )
        
        print(f"Generated quarterly reports for {len(clients)} clients")
    
    def _generate_market_commentary(self) -> str:
        """Generate daily market commentary"""
        market_data = self._get_market_data()
        news = self._get_news_highlights()
        calendar = self._get_economic_calendar()
        
        return self.commentary_gen.generate_daily_commentary(
            market_data, news, calendar
        )
    
    def _generate_quarterly_report(self, client: Dict) -> str:
        """Generate comprehensive quarterly report"""
        # Combine multiple report types
        sections = []
        
        # Performance section
        sections.append("# Quarterly Portfolio Review\\n")
        sections.append(self.portfolio_gen.generate_performance_report(
            client['portfolio'], period=f"Q{self._get_quarter()} 2023"
        ))
        
        # Attribution section
        sections.append("\\n\\n# Performance Attribution\\n")
        sections.append(self.attribution_gen.generate_attribution_report(
            client.get('attribution_data', {})
        ))
        
        # Risk section
        sections.append("\\n\\n# Risk Analysis\\n")
        sections.append(self.risk_gen.generate_risk_report(
            client['portfolio'],
            self._get_market_data(),
            client['risk_metrics']
        ))
        
        return "\\n".join(sections)
    
    def _save_report(self, content: str, filepath: str):
        """Save report to file"""
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Saved: {filepath}")
    
    def _get_market_data(self) -> Dict:
        """Get current market data"""
        # In production, fetch from data provider
        return {}
    
    def _get_news_highlights(self) -> List[str]:
        """Get news highlights"""
        return []
    
    def _get_economic_calendar(self) -> List[str]:
        """Get upcoming economic events"""
        return []
    
    def _get_quarter(self) -> int:
        """Get current quarter"""
        return (datetime.now().month - 1) // 3 + 1
    
    def run_scheduled(self):
        """
        Run report generation on schedule
        """
        # Schedule daily reports
        schedule.every().day.at("17:00").do(
            self.generate_daily_reports, clients=[]
        )
        
        # Schedule monthly reports (1st of month)
        schedule.every().day.at("08:00").do(
            self._check_monthly_reports
        )
        
        # Schedule quarterly reports
        schedule.every().day.at("08:00").do(
            self._check_quarterly_reports
        )
        
        print("Report generation pipeline started")
        
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    def _check_monthly_reports(self):
        """Check if it's time for monthly reports"""
        if datetime.now().day == 1:
            self.generate_monthly_reports(clients=[])
    
    def _check_quarterly_reports(self):
        """Check if it's time for quarterly reports"""
        # First day of quarter
        if datetime.now().day == 1 and datetime.now().month in [1, 4, 7, 10]:
            self.generate_quarterly_reports(clients=[])

# Initialize pipeline
# pipeline = ReportGenerationPipeline(api_key="your-key")
# pipeline.run_scheduled()
\`\`\`

---

## Best Practices

1. **Templates**: Use consistent templates for each report type
2. **Data Quality**: Verify all data before generating reports
3. **Personalization**: Tailor language and detail to audience
4. **Validation**: Review generated reports before distribution
5. **Version Control**: Keep historical versions of all reports
6. **Formatting**: Ensure professional formatting and layout
7. **Compliance**: Include required disclosures and disclaimers
8. **Timeliness**: Generate and distribute on consistent schedule
9. **Feedback Loop**: Incorporate client feedback to improve
10. **Human Oversight**: Critical reports need human review

---

## Summary

We covered:
- Generating portfolio performance reports
- Risk analysis and VaR reporting
- Market commentary generation
- Performance attribution reports
- Building production report generation pipelines
- Best practices for automated reporting

Next: Trading signal generation using LLMs.
`,
};

