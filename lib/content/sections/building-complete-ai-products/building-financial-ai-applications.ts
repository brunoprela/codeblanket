export const buildingFinancialAiApplications = {
  title: 'Building Financial AI Applications',
  id: 'building-financial-ai-applications',
  content: `
# Building Financial AI Applications

## Introduction

Financial AI applications have unique requirements: extreme accuracy demands, regulatory compliance, real-time market data, complex calculations, and high stakes (errors = money lost). This section covers building production-grade financial AI tools.

### Use Cases

**Trading Assistants**: Analyze markets, generate insights
**Risk Analysis**: Portfolio risk, stress testing
**Document Processing**: Extract data from 10-Ks, earnings calls
**Financial Modeling**: Build models from natural language
**Compliance**: Flag suspicious transactions, regulatory reporting

---

## Financial Data Integration

### Market Data APIs

\`\`\`python
"""
Integrate real-time and historical financial data
"""

import yfinance as yf
import alpha_vantage
import pandas as pd
from datetime import datetime, timedelta

class MarketDataProvider:
    """
    Unified interface for financial data
    """
    
    def __init__(self):
        self.yf = yf
        self.alpha = alpha_vantage.AlphaVantage (api_key=os.getenv("ALPHA_VANTAGE_KEY"))
    
    def get_stock_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get stock price data
        
        period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        """
        
        ticker = self.yf.Ticker (symbol)
        df = ticker.history (period=period, interval=interval)
        
        return df
    
    def get_company_info (self, symbol: str) -> dict:
        """Get company fundamentals"""
        
        ticker = self.yf.Ticker (symbol)
        
        return {
            "name": ticker.info.get("longName"),
            "sector": ticker.info.get("sector"),
            "industry": ticker.info.get("industry"),
            "market_cap": ticker.info.get("marketCap"),
            "pe_ratio": ticker.info.get("trailingPE"),
            "dividend_yield": ticker.info.get("dividendYield"),
            "52_week_high": ticker.info.get("fiftyTwoWeekHigh"),
            "52_week_low": ticker.info.get("fiftyTwoWeekLow"),
        }
    
    def get_financial_statements (self, symbol: str) -> dict:
        """Get income statement, balance sheet, cash flow"""
        
        ticker = self.yf.Ticker (symbol)
        
        return {
            "income_statement": ticker.financials,
            "balance_sheet": ticker.balance_sheet,
            "cash_flow": ticker.cashflow
        }
    
    def get_news (self, symbol: str, limit: int = 10) -> list:
        """Get recent news for symbol"""
        
        ticker = self.yf.Ticker (symbol)
        news = ticker.news[:limit]
        
        return [
            {
                "title": item.get("title"),
                "publisher": item.get("publisher"),
                "link": item.get("link"),
                "published": datetime.fromtimestamp (item.get("providerPublishTime"))
            }
            for item in news
        ]

# Usage
data_provider = MarketDataProvider()

# Get Apple stock data
aapl_data = data_provider.get_stock_data("AAPL", period="1y")
aapl_info = data_provider.get_company_info("AAPL")
aapl_financials = data_provider.get_financial_statements("AAPL")
aapl_news = data_provider.get_news("AAPL")
\`\`\`

---

## Financial Analysis with LLMs

### Market Analysis Assistant

\`\`\`python
"""
Generate market insights using LLM + financial data
"""

import anthropic

class FinancialAnalyst:
    """
    AI-powered financial analysis
    """
    
    def __init__(self, data_provider: MarketDataProvider):
        self.data_provider = data_provider
        self.client = anthropic.Anthropic()
    
    async def analyze_stock (self, symbol: str) -> dict:
        """
        Comprehensive stock analysis
        """
        
        # Gather data
        price_data = self.data_provider.get_stock_data (symbol, period="1y")
        company_info = self.data_provider.get_company_info (symbol)
        financials = self.data_provider.get_financial_statements (symbol)
        news = self.data_provider.get_news (symbol, limit=5)
        
        # Calculate technical indicators
        technical = self.calculate_technical_indicators (price_data)
        
        # Build analysis prompt
        prompt = f"""
Analyze this stock as a financial analyst:

Company: {company_info['name']} ({symbol})
Sector: {company_info['sector']}
Market Cap: \${company_info['market_cap']:,.0f}

Price Performance(1 year):
- Current: \${ price_data['Close'].iloc[-1]:.2f }
- 52 - week high: \${ company_info['52_week_high']:.2f }
- 52 - week low: \${ company_info['52_week_low']:.2f }
- YTD return: { ((price_data['Close'].iloc[-1] / price_data['Close'].iloc[0]) - 1) * 100: .1f }%

  Valuation:
- P / E Ratio: { company_info['pe_ratio']: .2f }
- Dividend Yield: { company_info['dividend_yield'] * 100 if company_info['dividend_yield'] else 0: .2f }%

  Technical Indicators:
- RSI: { technical['rsi']: .1f }
- MACD: { technical['macd']: .2f }
- 50 - day MA: \${ technical['ma_50']:.2f }
- 200 - day MA: \${ technical['ma_200']:.2f }

Recent News:
{ self._format_news (news) }

Provide a comprehensive analysis:
1. Executive Summary(2 - 3 sentences)
2. Valuation Assessment (undervalued / fairly valued / overvalued)
3. Technical Analysis (bullish / neutral / bearish)
4. Key Risks(3 - 5 points)
5. Key Opportunities(3 - 5 points)
6. Rating(Strong Buy / Buy / Hold / Sell / Strong Sell)

Format as JSON.
"""

response = self.client.messages.create(
  model = "claude-3-5-sonnet-20241022",
  max_tokens = 2000,
  messages = [{ "role": "user", "content": prompt }]
)

analysis = json.loads (response.content[0].text)

return {
            ** analysis,
  "symbol": symbol,
    "data": {
  "price_data": price_data.to_dict(),
    "company_info": company_info,
      "technical": technical
}
        }
    
    def calculate_technical_indicators (self, df: pd.DataFrame) -> dict:
"""Calculate common technical indicators"""
        
        # RSI
delta = df['Close'].diff()
gain = (delta.where (delta > 0, 0)).rolling (window = 14).mean()
loss = (-delta.where (delta < 0, 0)).rolling (window = 14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))
        
        # Moving averages
ma_50 = df['Close'].rolling (window = 50).mean()
ma_200 = df['Close'].rolling (window = 200).mean()
        
        # MACD
ema_12 = df['Close'].ewm (span = 12).mean()
ema_26 = df['Close'].ewm (span = 26).mean()
macd = ema_12 - ema_26

return {
  "rsi": rsi.iloc[-1],
  "ma_50": ma_50.iloc[-1],
  "ma_200": ma_200.iloc[-1],
  "macd": macd.iloc[-1]
}
    
    def _format_news (self, news: list) -> str:
return "\\n".join([
  f"- {item['title']} ({item['publisher']})"
            for item in news
        ])

# API endpoint
@app.post("/api/analyze")
async def analyze_stock (symbol: str):
"""
    Analyze stock
"""
analyst = FinancialAnalyst (data_provider)
analysis = await analyst.analyze_stock (symbol)

return analysis
\`\`\`

---

## Portfolio Management

### AI Portfolio Assistant

\`\`\`python
"""
Portfolio analysis and optimization
"""

class PortfolioManager:
    """
    Manage and optimize investment portfolios
    """
    
    def __init__(self, data_provider: MarketDataProvider):
        self.data_provider = data_provider
        self.client = anthropic.Anthropic()
    
    def analyze_portfolio (self, holdings: list[dict]) -> dict:
        """
        Analyze portfolio composition and risk
        
        holdings: [{"symbol": "AAPL", "shares": 100, "cost_basis": 150.00}, ...]
        """
        
        # Get current prices
        portfolio_data = []
        for holding in holdings:
            current_price = self.data_provider.get_stock_data(
                holding['symbol'],
                period="1d"
            )['Close'].iloc[-1]
            
            portfolio_data.append({
                **holding,
                "current_price": current_price,
                "market_value": holding['shares'] * current_price,
                "gain_loss": (current_price - holding['cost_basis']) * holding['shares'],
                "gain_loss_pct": ((current_price / holding['cost_basis']) - 1) * 100
            })
        
        # Calculate portfolio metrics
        total_value = sum (h['market_value'] for h in portfolio_data)
        total_gain_loss = sum (h['gain_loss'] for h in portfolio_data)
        
        # Calculate diversification
        sector_allocation = self._calculate_sector_allocation (portfolio_data)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics (portfolio_data)
        
        return {
            "total_value": total_value,
            "total_gain_loss": total_gain_loss,
            "total_gain_loss_pct": (total_gain_loss / (total_value - total_gain_loss)) * 100,
            "holdings": portfolio_data,
            "sector_allocation": sector_allocation,
            "risk_metrics": risk_metrics
        }
    
    def _calculate_sector_allocation (self, holdings: list) -> dict:
        """Calculate portfolio allocation by sector"""
        
        sector_values = {}
        total_value = sum (h['market_value'] for h in holdings)
        
        for holding in holdings:
            info = self.data_provider.get_company_info (holding['symbol'])
            sector = info.get('sector', 'Unknown')
            
            if sector not in sector_values:
                sector_values[sector] = 0
            
            sector_values[sector] += holding['market_value']
        
        # Convert to percentages
        return {
            sector: (value / total_value) * 100
            for sector, value in sector_values.items()
        }
    
    def _calculate_risk_metrics (self, holdings: list) -> dict:
        """Calculate portfolio risk metrics"""
        
        # Get historical returns for all holdings
        returns_data = {}
        for holding in holdings:
            data = self.data_provider.get_stock_data(
                holding['symbol'],
                period="1y"
            )
            returns = data['Close'].pct_change().dropna()
            returns_data[holding['symbol']] = returns
        
        # Portfolio volatility (simplified)
        portfolio_returns = []
        for symbol, returns in returns_data.items():
            holding = next (h for h in holdings if h['symbol'] == symbol)
            weight = holding['market_value'] / sum (h['market_value'] for h in holdings)
            portfolio_returns.append (returns * weight)
        
        combined_returns = sum (portfolio_returns)
        
        return {
            "volatility_annual": combined_returns.std() * (252 **0.5),  # Annualized
            "sharpe_ratio": (combined_returns.mean() * 252) / (combined_returns.std() * (252 **0.5)),
            "max_drawdown": (combined_returns.cumsum().max() - combined_returns.cumsum().min())
        }
    
    async def get_recommendations (self, portfolio: dict) -> list:
        """
        Get AI recommendations for portfolio improvements
        """
        
        prompt = f"""
Analyze this investment portfolio and provide recommendations:

Portfolio Summary:
- Total Value: \${portfolio['total_value']:,.2f}
- Total Gain / Loss: \${ portfolio['total_gain_loss']:,.2f } ({ portfolio['total_gain_loss_pct']: .1f } %)

Holdings:
{ self._format_holdings (portfolio['holdings']) }

Sector Allocation:
{ self._format_sector_allocation (portfolio['sector_allocation']) }

Risk Metrics:
- Annual Volatility: { portfolio['risk_metrics']['volatility_annual']: .1 %}
- Sharpe Ratio: { portfolio['risk_metrics']['sharpe_ratio']: .2f }
- Max Drawdown: { portfolio['risk_metrics']['max_drawdown']: .1 %}

Provide:
1. Diversification assessment (well - diversified / concentrated / over - diversified)
2. Risk assessment (conservative / moderate / aggressive)
3. Top 3 recommendations for improvement
4. Suggested actions (buy, sell, rebalance)

Format as JSON.
"""

response = self.client.messages.create(
  model = "claude-3-5-sonnet-20241022",
  max_tokens = 1500,
  messages = [{ "role": "user", "content": prompt }]
)

recommendations = json.loads (response.content[0].text)

return recommendations
\`\`\`

---

## Regulatory Compliance

### Compliance Monitoring

\`\`\`python
"""
Financial regulatory compliance
"""

class ComplianceMonitor:
    """
    Monitor for suspicious activity and regulatory compliance
    """
    
    def __init__(self):
        self.client = anthropic.Anthropic()
    
    async def check_transaction (self, transaction: dict) -> dict:
        """
        Check transaction for compliance issues
        
        Checks for:
        - Unusual patterns
        - Large transactions (SAR threshold)
        - Wash sales
        - Insider trading indicators
        """
        
        flags = []
        
        # Check amount threshold (SAR = $10,000)
        if transaction['amount'] > 10000:
            flags.append({
                "type": "large_transaction",
                "severity": "high",
                "description": f"Transaction exceeds $10,000 SAR threshold",
                "requires_filing": True
            })
        
        # Check for wash sale (same security within 30 days)
        if self._is_wash_sale (transaction):
            flags.append({
                "type": "wash_sale",
                "severity": "medium",
                "description": "Potential wash sale (sold at loss, repurchased within 30 days)",
                "tax_implication": True
            })
        
        # Use AI to detect unusual patterns
        if transaction['amount'] > 1000:
            pattern_analysis = await self._analyze_pattern (transaction)
            if pattern_analysis['suspicious']:
                flags.append (pattern_analysis)
        
        return {
            "compliant": len (flags) == 0,
            "flags": flags,
            "risk_score": self._calculate_risk_score (flags)
        }
    
    def _is_wash_sale (self, transaction: dict) -> bool:
        # Check transaction history for same security within 30 days
        # Implementation depends on database structure
        return False
    
    async def _analyze_pattern (self, transaction: dict) -> dict:
        """Use AI to detect unusual trading patterns"""
        
        prompt = f"""
Analyze this financial transaction for suspicious patterns:

Transaction:
- Amount: \${transaction['amount']:,.2f}
- Security: { transaction.get('symbol') }
- Type: { transaction['type'] }
- Time: { transaction['timestamp'] }
- User: { transaction['user_id'] }

Recent activity:
{ transaction.get('recent_history', []) }

Is this suspicious ? Consider :
  1. Unusual size relative to account history
2. Timing (end of quarter, before earnings)
3. Pattern (rapid buy / sell, structuring)
4. Context (news, market events)

Respond with JSON:
{
  "suspicious": true / false,
    "confidence": 0 - 100,
      "reasoning": "...",
        "severity": "low/medium/high"
}
"""

response = self.client.messages.create(
  model = "claude-3-haiku-20240307",
  max_tokens = 500,
  messages = [{ "role": "user", "content": prompt }]
)

analysis = json.loads (response.content[0].text)

return {
  "type": "unusual_pattern",
  "severity": analysis['severity'],
  "description": analysis['reasoning'],
  "confidence": analysis['confidence']
}
    
    def _calculate_risk_score (self, flags: list) -> int:
"""Calculate overall risk score (0-100)"""

severity_weights = {
  "low": 10,
  "medium": 30,
  "high": 60
}

score = sum (severity_weights.get (flag['severity'], 0) for flag in flags)

  return min (score, 100)
\`\`\`

---

## Conclusion

Financial AI applications require:

1. **Data Integration**: Market data APIs (yfinance, Alpha Vantage)
2. **Analysis**: Technical + fundamental + AI insights
3. **Portfolio Management**: Risk metrics, optimization
4. **Compliance**: Regulatory monitoring, SAR filing
5. **Accuracy**: Financial errors = money lost

**Key Challenges**:
- **Accuracy**: Must be extremely accurate (financial decisions)
- **Real-time**: Market data changes every second
- **Regulations**: FINRA, SEC compliance
- **Liability**: Disclaimers, not financial advice

**Best Practices**:
- Always include disclaimer: "Not financial advice"
- Show sources/calculations (transparency)
- Log all recommendations (audit trail)
- Rate limit to prevent market manipulation
- Never auto-execute trades without confirmation

Financial AI is high-stakes but extremely valuable when done correctly.
`,
};
