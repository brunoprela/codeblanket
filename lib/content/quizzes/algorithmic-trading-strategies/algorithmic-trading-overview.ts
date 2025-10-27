export const algorithmicTradingOverviewQuiz = [
  {
    id: 'ats-1-1-q-1',
    question:
      "You're building an algorithmic trading system for a small hedge fund ($5M AUM). They want to trade US equities with a holding period of 1-5 days. Design the complete technology stack including: (1) Data infrastructure, (2) Order execution system, (3) Risk management, (4) Monitoring. Include specific technologies, estimated costs, and justify why each component is necessary.",
    sampleAnswer: `**System Design for Medium-Frequency Equity Trading:**

**1. Data Infrastructure**

**Real-Time Market Data:**
- **Provider**: Polygon.io or IEX Cloud ($200-500/month for real-time)
- **Justification**: Affordable real-time data with WebSocket support
- **Alternative**: yfinance (free but delayed 15min - OK for backtesting, not live)

**Historical Data Storage:**
- **Database**: PostgreSQL with TimescaleDB extension
- **Schema**: 
  \`\`\`sql
  CREATE TABLE market_data (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT,
    PRIMARY KEY (timestamp, symbol)
  );
  SELECT create_hypertable('market_data', 'timestamp');
  \`\`\`
- **Cost**: Self-hosted on AWS EC2 ($50-100/month)
- **Justification**: TimescaleDB optimized for time-series queries, better than vanilla PostgreSQL

**Data Pipeline:**
\`\`\`python
import asyncio
import websockets
from polygon import WebSocketClient

class MarketDataPipeline:
    """Ingest real-time data and store in TimescaleDB"""
    
    async def stream_quotes(self, symbols: list):
        async with websockets.connect('wss://socket.polygon.io/stocks') as ws:
            # Subscribe to quotes
            await ws.send(json.dumps({
                'action': 'subscribe',
                'params': ','.join([f'Q.{s}' for s in symbols])
            }))
            
            async for message in ws:
                quote = json.loads(message)
                await self.store_quote(quote)
    
    async def store_quote(self, quote):
        await self.db.execute(
            "INSERT INTO market_data VALUES ($1, $2, $3, $4, $5, $6, $7)",
            quote['timestamp'], quote['symbol'], 
            quote['open'], quote['high'], quote['low'], 
            quote['close'], quote['volume']
        )
\`\`\`

**2. Order Execution System**

**Broker**: Interactive Brokers (IBKR)
- **Why**: Lowest commissions ($0.0035/share, $0.35 min), excellent API
- **API**: ib_insync Python library (cleaner than native IB API)

**Order Management System (OMS):**
\`\`\`python
from ib_insync import IB, Order, MarketOrder, LimitOrder

class OrderManagementSystem:
    """Handle order lifecycle"""
    
    def __init__(self):
        self.ib = IB()
        self.ib.connect('127.0.0.1', 7497, clientId=1)
        self.open_orders = {}
    
    async def place_order(self, symbol: str, quantity: int, 
                         order_type: str, limit_price: float = None):
        """Place order with IBKR"""
        contract = Stock(symbol, 'SMART', 'USD')
        
        if order_type == 'MARKET':
            order = MarketOrder('BUY' if quantity > 0 else 'SELL', 
                               abs(quantity))
        else:
            order = LimitOrder('BUY' if quantity > 0 else 'SELL',
                              abs(quantity), limit_price)
        
        # Place with pre-trade risk checks
        if self.pre_trade_risk_check(contract, quantity, limit_price):
            trade = self.ib.placeOrder(contract, order)
            self.open_orders[trade.order.orderId] = trade
            return trade.order.orderId
        else:
            raise ValueError("Failed pre-trade risk check")
\`\`\`

**Smart Order Routing**: Not needed for $5M fund - IBKR's built-in routing sufficient

**3. Risk Management**

**Pre-Trade Controls:**
\`\`\`python
class RiskManagement:
    def __init__(self, capital: float):
        self.capital = capital
        self.max_position_size = 0.10  # 10% per position
        self.max_leverage = 1.0  # No leverage for now
        self.max_daily_loss = 0.02  # 2% max daily loss
        self.current_daily_pnl = 0.0
    
    def check_position_limit(self, symbol: str, quantity: int, 
                            price: float) -> bool:
        position_value = quantity * price
        return position_value <= self.capital * self.max_position_size
    
    def check_daily_loss_limit(self) -> bool:
        return self.current_daily_pnl >= -(self.capital * self.max_daily_loss)
    
    def check_concentration(self, positions: dict) -> bool:
        # No more than 20% in any single sector
        sector_exposure = self.calculate_sector_exposure(positions)
        return all(exp <= 0.20 for exp in sector_exposure.values())
\`\`\`

**Real-Time P&L Monitoring:**
- Update every 1 second from IBKR account data
- Alert if approaching daily loss limit (-1.5%)
- Kill switch at -2% daily loss (flatten all positions)

**4. Monitoring & Alerting**

**Dashboard**: Grafana + Prometheus
- Real-time P&L chart
- Open positions and exposure
- Strategy performance (Sharpe, drawdown)
- System health (latency, error rates)

**Alerting**: PagerDuty ($0 - $21/month for 1 user)
- Daily loss approaching limit
- Strategy stopped generating signals
- Connection to broker lost
- Unusual order rejection rate

**Logging**: Structured JSON logs to S3
\`\`\`python
import logging
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# All trading decisions logged
logger.info(json.dumps({
    'timestamp': datetime.now().isoformat(),
    'event': 'ORDER_PLACED',
    'symbol': 'AAPL',
    'quantity': 100,
    'price': 180.50,
    'strategy': 'mean_reversion_v1',
    'reason': 'Z-score crossed -2.0'
}))
\`\`\`

**Technology Stack Summary:**

| Component | Technology | Monthly Cost | Justification |
|-----------|-----------|--------------|---------------|
| Market Data | Polygon.io | $500 | Real-time WebSocket, affordable |
| Database | TimescaleDB on EC2 | $100 | Time-series optimized |
| Broker | Interactive Brokers | $10 min | Lowest fees, best API |
| Compute | AWS EC2 t3.medium | $30 | Sufficient for 1-5 day strategies |
| Monitoring | Grafana Cloud | $0 | Free tier adequate |
| Alerting | PagerDuty | $0 | Free tier adequate |
| **Total** | | **~$640/month** | 0.15% of AUM annually |

**Infrastructure Cost**: $640/month = $7,680/year = 0.15% of $5M AUM (very reasonable)

**Why This Stack:**
- **Affordable**: <$10K/year for entire infrastructure
- **Scalable**: Can handle up to $50M AUM without changes
- **Reliable**: Enterprise-grade components
- **Maintainable**: Python throughout, well-documented
- **Sufficient**: 1-5 day holding period doesn't need HFT infrastructure

**What We're NOT Building (and why):**
- Custom FIX connectivity: IBKR API sufficient
- Co-location: Not needed for medium-frequency
- Tick data storage: Daily/hourly bars sufficient
- Microservices: Single process fine for now (premature optimization)
- Kubernetes: EC2 instance adequate for small fund

**Critical Success Factors:**
1. Comprehensive pre-trade risk checks (prevent fat fingers)
2. Real-time P&L monitoring (know your exposure)
3. Kill switch (protect capital in disasters)
4. Complete audit trail (debugging and compliance)
5. Start simple, add complexity only when needed`,
    keyPoints: [
      'Medium-frequency trading (~$5M) needs affordable real-time data (Polygon/IEX), TimescaleDB for storage',
      'Interactive Brokers best for small funds: low fees ($0.0035/share), excellent API',
      'Risk management critical: position limits (10%), daily loss limits (2%), sector concentration (20%)',
      'Monitoring via Grafana + PagerDuty, comprehensive logging to S3 for audit trail',
      'Total infrastructure cost ~$640/month (0.15% AUM) - very reasonable for complete stack',
    ],
  },
  {
    id: 'ats-1-1-q-2',
    question:
      'Compare High-Frequency Trading (HFT) vs Medium-Frequency Trading (MFT) across: (1) Capital requirements, (2) Technology stack, (3) Regulatory burden, (4) Profitability per trade, (5) Capacity constraints. If you had $500K to start, which would you choose and why? What are the realistic expectations for returns?',
    sampleAnswer: `**Comparison: HFT vs MFT**

**1. Capital Requirements**

**HFT:**
- **Minimum capital**: $10M-$50M
- **Infrastructure**: $500K-$2M initial + $50K-$200K/month
- **Co-location fees**: $10K-$50K/month per exchange
- **Why so high**: Need scale to amortize infrastructure costs

**MFT:**
- **Minimum capital**: $100K-$1M
- **Infrastructure**: $10K-$50K initial + $500-$5K/month
- **Co-location**: Not needed
- **Why accessible**: Leverage cloud infrastructure, no co-location

\`\`\`python
class CapitalRequirements:
    HFT = {
        'min_capital': 10_000_000,
        'initial_infrastructure': 1_000_000,
        'monthly_infrastructure': 100_000,
        'colocation_monthly': 30_000,
        'break_even_trades_per_day': 10_000,  # Need volume to profit
    }
    
    MFT = {
        'min_capital': 500_000,
        'initial_infrastructure': 20_000,
        'monthly_infrastructure': 2_000,
        'colocation_monthly': 0,
        'break_even_trades_per_day': 10,  # Much lower volume needed
    }
\`\`\`

**2. Technology Stack**

**HFT:**
- **Language**: C++ or FPGA (nanosecond optimization)
- **Network**: Direct exchange connections, fiber optic co-location
- **Latency requirement**: <100 microseconds
- **Hardware**: Custom servers ($50K+ each), 10G/40G network cards
- **Data**: Level 3 market data (full depth of book), every tick
- **Storage**: In-memory (Redis, KDB+), minimal disk
- **Team**: 10+ engineers (kernel developers, network engineers)

**MFT:**
- **Language**: Python (numpy, pandas) or Java
- **Network**: Internet + VPN to broker
- **Latency requirement**: <1 second (acceptable)
- **Hardware**: AWS EC2 instances ($30-$200/month)
- **Data**: Level 1 or 2 data, minute/hourly bars
- **Storage**: PostgreSQL/TimescaleDB
- **Team**: 1-2 developers

**3. Regulatory Burden**

**HFT:**
- **Registration**: Market maker registration required
- **Testing**: Extensive pre-deployment testing (MiFID II)
- **Controls**: Mandatory circuit breakers, kill switches
- **Reporting**: Microsecond-precision audit trail
- **Capital**: Significant regulatory capital requirements
- **Compliance cost**: $100K+/year (lawyers, auditors)

**MFT:**
- **Registration**: Minimal (broker handles most)
- **Testing**: Self-imposed testing
- **Controls**: Recommended but not legally mandated
- **Reporting**: Standard broker reporting sufficient
- **Capital**: Broker margin requirements only
- **Compliance cost**: $5K-$20K/year

**4. Profitability Per Trade**

**HFT:**
- **Profit per trade**: $0.0001 - $0.001 per share (0.1-1 cent)
- **Win rate**: 50-52% (barely positive)
- **Volume required**: Millions of shares daily
- **Example**: Make $0.0005/share on 10M shares = $5K/day
- **After costs**: $5K - $3K (infrastructure) = $2K/day profit

**MFT:**
- **Profit per trade**: $0.10 - $1.00 per share (10-100 cents)
- **Win rate**: 55-65% (higher edge)
- **Volume required**: Thousands of shares daily
- **Example**: Make $0.50/share on 10K shares = $5K/day
- **After costs**: $5K - $100 (infrastructure) = $4.9K/day profit

\`\`\`python
import numpy as np

def calculate_expected_profit(strategy_type: str, capital: float, 
                             daily_volume_shares: int) -> dict:
    if strategy_type == 'HFT':
        profit_per_share = 0.0005
        win_rate = 0.51
        infrastructure_daily = 3000
        
    else:  # MFT
        profit_per_share = 0.50
        win_rate = 0.60
        infrastructure_daily = 100
    
    # Expected value per trade
    expected_profit = (win_rate * profit_per_share - 
                      (1 - win_rate) * profit_per_share)
    
    gross_profit_daily = daily_volume_shares * expected_profit
    net_profit_daily = gross_profit_daily - infrastructure_daily
    
    annual_return_pct = (net_profit_daily * 252 / capital) * 100
    
    return {
        'daily_gross': gross_profit_daily,
        'daily_net': net_profit_daily,
        'annual_return_pct': annual_return_pct,
        'sharpe_estimate': annual_return_pct / 15  # Rough estimate
    }

# HFT scenario
hft = calculate_expected_profit('HFT', 10_000_000, 10_000_000)
print(f"HFT: {hft['annual_return_pct']:.1f}% annual, Sharpe: {hft['sharpe_estimate']:.1f}")

# MFT scenario  
mft = calculate_expected_profit('MFT', 500_000, 10_000)
print(f"MFT: {mft['annual_return_pct']:.1f}% annual, Sharpe: {mft['sharpe_estimate']:.1f}")
\`\`\`

**5. Capacity Constraints**

**HFT:**
- **Typical capacity**: $10M-$100M per strategy
- **Why limited**: Strategies exploit small inefficiencies that disappear with size
- **Scaling**: Must develop new strategies, can't just add capital

**MFT:**
- **Typical capacity**: $50M-$500M per strategy
- **Why larger**: Longer holding periods allow gradual accumulation
- **Scaling**: Can scale within limits by spreading trades

**Decision with $500K:**

**Recommendation: Medium-Frequency Trading (MFT)**

**Why MFT:**
1. **Capital adequate**: $500K sufficient to start
2. **Infrastructure affordable**: $2K/month vs $100K/month
3. **Manageable alone**: 1-2 people can operate
4. **Higher profit per trade**: $0.50/share vs $0.0005/share
5. **Regulatory simpler**: Avoid market maker registration

**Why NOT HFT:**
1. **Capital insufficient**: Need $10M+ minimum
2. **Infrastructure too expensive**: $100K/month infrastructure + $30K colocation
3. **Requires team**: Need C++/kernel developers
4. **Regulatory complex**: Market maker registration, extensive testing
5. **Competitive**: Competing with Citadel, Jump, Virtu (billions invested)

**Realistic Return Expectations with $500K MFT:**

**Conservative Scenario (Realistic):**
- **Annual return**: 20-30% (after costs)
- **Sharpe ratio**: 1.5-2.0
- **Max drawdown**: 15-20%
- **Monthly return**: $8K-$12K ($100K-$150K/year)

**Optimistic Scenario (Good year):**
- **Annual return**: 40-60%
- **Sharpe ratio**: 2.5-3.0
- **Max drawdown**: 10-15%
- **Monthly return**: $17K-$25K ($200K-$300K/year)

**Realistic First Year:**
- **Q1-Q2**: Break-even to small profit (learning, debugging)
- **Q3-Q4**: 10-20% return if strategy works
- **Year 2+**: 20-40% if strategy remains robust

**Path Forward with $500K:**

1. **Start**: Develop 2-3 uncorrelated MFT strategies
2. **Paper trade**: 3-6 months to validate
3. **Go live**: Start with $50K (10% of capital)
4. **Scale**: Increase to $500K over 6-12 months if profitable
5. **Diversify**: Add more strategies as capital grows

**Key Insight**: With $500K, MFT is realistic. HFT requires $10M+ and a team. Most successful retail/small funds use MFT (1-5 day holding periods) not HFT.`,
    keyPoints: [
      'HFT requires $10M+ capital + $100K+/month infrastructure; MFT viable with $500K + $2K/month',
      'HFT profits tiny per trade ($0.0005/share) but massive volume; MFT larger per trade ($0.50/share)',
      'HFT needs C++, co-location, microsecond latency; MFT works with Python, cloud, second latency',
      'With $500K choose MFT: affordable, manageable alone, higher profit margins, less regulatory burden',
      'Realistic MFT returns: 20-30% annually (conservative), 40-60% (optimistic), Sharpe 1.5-2.5',
    ],
  },
  {
    id: 'ats-1-1-q-3',
    question:
      'A strategy has a Sharpe ratio of 2.5 in backtesting (2010-2020) but only 0.8 in live trading (2021-2023). Analyze potential causes: (1) Overfitting, (2) Regime change, (3) Transaction costs, (4) Market impact, (5) Data quality. How would you diagnose which is the primary cause? Design a systematic testing framework to identify the issue.',
    sampleAnswer: `**Diagnosing Strategy Degradation: 2.5 → 0.8 Sharpe**

This is a common and painful problem in algorithmic trading. A 70% drop in Sharpe (2.5 → 0.8) suggests fundamental issues, not just noise.

**Potential Causes & Diagnosis:**

**1. Overfitting (Most Common)**

**Symptoms:**
- Performance degrades immediately out-of-sample
- Many parameters were optimized
- Strategy works perfectly in-sample
- No economic rationale for why strategy should work

**Diagnostic Tests:**
\`\`\`python
class OverfittingDiagnostic:
    """Test if strategy is overfit"""
    
    def walk_forward_analysis(self, returns_by_period: dict) -> dict:
        """
        Split data into 6-month in-sample, 6-month out-of-sample periods
        
        If strategy is overfit:
        - In-sample Sharpe: 2.5
        - Out-of-sample Sharpe: 0.8
        - Consistent pattern across all periods
        """
        results = []
        for i in range(0, len(returns_by_period) - 1, 2):
            in_sample = returns_by_period[i]
            out_sample = returns_by_period[i + 1]
            
            results.append({
                'in_sample_sharpe': self.sharpe(in_sample),
                'out_sample_sharpe': self.sharpe(out_sample),
                'degradation': self.sharpe(in_sample) - self.sharpe(out_sample)
            })
        
        # If consistent degradation → overfitting
        avg_degradation = np.mean([r['degradation'] for r in results])
        return {
            'is_overfit': avg_degradation > 1.0,  # >1.0 Sharpe drop
            'avg_degradation': avg_degradation,
            'consistency': np.std([r['degradation'] for r in results])
        }
    
    def parameter_sensitivity(self, strategy, param_name: str, 
                             param_range: list) -> dict:
        """
        Test if small parameter changes destroy performance
        
        Overfit strategy: Sharpe drops from 2.5 to 0.5 with 5% param change
        Robust strategy: Sharpe stays 1.5-2.0 across wide param range
        """
        results = []
        for param_value in param_range:
            sharpe = strategy.backtest(param_override={param_name: param_value})
            results.append({'param': param_value, 'sharpe': sharpe})
        
        # Calculate sensitivity
        sharpe_std = np.std([r['sharpe'] for r in results])
        max_sharpe = max([r['sharpe'] for r in results])
        min_sharpe = min([r['sharpe'] for r in results])
        
        return {
            'is_fragile': (max_sharpe - min_sharpe) / max_sharpe > 0.5,
            'sharpe_range': (min_sharpe, max_sharpe),
            'stability_score': 1 - (sharpe_std / max_sharpe)
        }
    
    def data_snooping_test(self, n_strategies_tested: int, 
                          best_sharpe: float) -> dict:
        """
        Deflated Sharpe Ratio (Bailey & Lopez de Prado)
        
        Accounts for multiple testing bias
        """
        # If tested 100 strategies, best one likely overfit
        # Deflate Sharpe by number of trials
        deflated_sharpe = best_sharpe / np.sqrt(np.log(n_strategies_tested))
        
        return {
            'original_sharpe': best_sharpe,
            'deflated_sharpe': deflated_sharpe,
            'likely_overfit': deflated_sharpe < 1.0
        }
\`\`\`

**2. Regime Change (Market Environment Shifted)**

**Symptoms:**
- Strategy worked well 2010-2020 (low vol, QE era)
- Broke down 2021+ (inflation, rate hikes, volatility)
- Specific strategy types affected (e.g., momentum works in trends, fails in chop)

**Diagnostic Tests:**
\`\`\`python
class RegimeAnalysis:
    """Detect if market regime changed"""
    
    def volatility_regime(self, returns: pd.Series) -> dict:
        """Compare volatility regimes"""
        vol_2010_2020 = returns['2010':'2020'].std() * np.sqrt(252)
        vol_2021_2023 = returns['2021':'2023'].std() * np.sqrt(252)
        
        return {
            'historical_vol': vol_2010_2020,
            'recent_vol': vol_2021_2023,
            'vol_increase': vol_2021_2023 / vol_2010_2020,
            'regime_change': vol_2021_2023 / vol_2010_2020 > 1.5
        }
    
    def correlation_regime(self, returns: pd.Series, 
                          market_returns: pd.Series) -> dict:
        """Check if market correlations changed"""
        corr_2010_2020 = returns['2010':'2020'].corr(market_returns['2010':'2020'])
        corr_2021_2023 = returns['2021':'2023'].corr(market_returns['2021':'2023'])
        
        # If correlation flipped (e.g., -0.3 → +0.8), regime changed
        return {
            'historical_corr': corr_2010_2020,
            'recent_corr': corr_2021_2023,
            'correlation_shift': abs(corr_2021_2023 - corr_2010_2020),
            'significant_change': abs(corr_2021_2023 - corr_2010_2020) > 0.5
        }
    
    def trend_vs_chop(self, prices: pd.Series) -> dict:
        """
        Measure trendiness vs choppiness
        
        Trend Following strategies fail in choppy markets
        """
        # ADX (Average Directional Index) measures trend strength
        adx_2010_2020 = self.calculate_adx(prices['2010':'2020'])
        adx_2021_2023 = self.calculate_adx(prices['2021':'2023'])
        
        return {
            'historical_adx': adx_2010_2020.mean(),
            'recent_adx': adx_2021_2023.mean(),
            'market_became_choppy': adx_2021_2023.mean() < 20  # <20 = choppy
        }
\`\`\`

**3. Transaction Costs Underestimated**

**Symptoms:**
- Backtest didn't include realistic costs
- High turnover strategy (100+ trades/day)
- Performance degrades proportionally to trade frequency

**Diagnostic Tests:**
\`\`\`python
class TransactionCostAudit:
    """Compare modeled vs actual transaction costs"""
    
    def backtest_cost_assumptions(self) -> dict:
        return {
            'commission_modeled': 0.001,  # $0.001/share
            'slippage_modeled': 0.0,  # NONE (mistake!)
            'total_cost_modeled_bps': 1.0
        }
    
    def actual_costs_from_live_trading(self, fills: pd.DataFrame) -> dict:
        """Extract actual costs from execution data"""
        # Calculate slippage: execution price vs mid-price at signal time
        fills['slippage'] = (fills['exec_price'] - fills['signal_mid']) / fills['signal_mid']
        
        actual_commission = fills['commission'].sum() / fills['notional'].sum()
        actual_slippage_bps = fills['slippage'].mean() * 10000
        
        return {
            'commission_actual_bps': actual_commission * 10000,
            'slippage_actual_bps': actual_slippage_bps,
            'total_cost_actual_bps': actual_commission * 10000 + actual_slippage_bps
        }
    
    def rerun_backtest_with_actual_costs(self, strategy, actual_costs_bps: float):
        """Re-run backtest with realistic costs"""
        # Original backtest: 1 bps costs → 2.5 Sharpe
        # With actual costs: 10 bps costs → ? Sharpe
        
        sharpe_with_realistic_costs = strategy.backtest(
            transaction_cost_bps=actual_costs_bps
        )
        
        return {
            'original_sharpe': 2.5,
            'realistic_sharpe': sharpe_with_realistic_costs,
            'costs_explain_degradation': sharpe_with_realistic_costs < 1.0
        }
\`\`\`

**4. Market Impact (Size Increased)**

**Symptoms:**
- Backtest used $100K account
- Live trading with $10M account
- Larger orders move the market

**Diagnostic Tests:**
\`\`\`python
class MarketImpactAnalysis:
    def compare_order_sizes(self, backtest_capital: float, 
                           live_capital: float) -> dict:
        """
        Market impact ∝ sqrt(order size / ADV)
        
        ADV = Average Daily Volume
        """
        # Example: AAPL, ADV = 50M shares
        adv = 50_000_000
        
        backtest_order_size = backtest_capital * 0.1 / 180  # 10% position, $180/share
        live_order_size = live_capital * 0.1 / 180
        
        backtest_impact_bps = self.estimate_impact(backtest_order_size, adv)
        live_impact_bps = self.estimate_impact(live_order_size, adv)
        
        return {
            'backtest_impact_bps': backtest_impact_bps,
            'live_impact_bps': live_impact_bps,
            'impact_increase': live_impact_bps / backtest_impact_bps
        }
    
    def estimate_impact(self, order_size: float, adv: float) -> float:
        """
        Market impact model (square root law)
        
        Impact (bps) ≈ 10 * sqrt(order_size / ADV)
        """
        participation_rate = order_size / adv
        impact_bps = 10 * np.sqrt(participation_rate)
        return impact_bps
\`\`\`

**5. Data Quality Issues**

**Symptoms:**
- Backtest used survivorship-biased data
- Live trading includes delisted stocks
- Corporate actions not handled correctly

**Diagnostic Tests:**
\`\`\`python
class DataQualityCheck:
    def survivorship_bias_test(self, backtest_universe: list, 
                               actual_universe_2010: list) -> dict:
        """
        Check if backtest only included survivors
        
        Example: Backtest used 2023 S&P 500 constituents for 2010-2020 data
        """
        # S&P 500 turnover: ~5%/year × 10 years = 50% different
        survivors_only = set(backtest_universe)
        actual_2010 = set(actual_universe_2010)
        
        missing_from_backtest = actual_2010 - survivors_only
        
        # Survivorship bias: missing stocks likely underperformers
        return {
            'pct_missing': len(missing_from_backtest) / len(actual_2010),
            'survivorship_bias': len(missing_from_backtest) > 0.3 * len(actual_2010)
        }
\`\`\`

**Systematic Diagnostic Framework:**

\`\`\`python
class StrategyDegradationDiagnostic:
    """Complete diagnostic system"""
    
    def run_all_tests(self, strategy, live_results: dict) -> dict:
        """Run all diagnostic tests"""
        
        results = {
            'overfitting': self.test_overfitting(strategy),
            'regime_change': self.test_regime_change(strategy),
            'transaction_costs': self.test_transaction_costs(strategy, live_results),
            'market_impact': self.test_market_impact(strategy, live_results),
            'data_quality': self.test_data_quality(strategy)
        }
        
        # Identify primary cause
        primary_cause = self.identify_primary_cause(results)
        
        return {
            'diagnostic_results': results,
            'primary_cause': primary_cause,
            'recommended_action': self.recommend_action(primary_cause)
        }
    
    def identify_primary_cause(self, results: dict) -> str:
        """Identify most likely cause"""
        
        # Priority order (most common first)
        if results['overfitting']['is_overfit']:
            return 'OVERFITTING'
        
        if results['transaction_costs']['costs_explain_degradation']:
            return 'TRANSACTION_COSTS'
        
        if results['regime_change']['regime_changed']:
            return 'REGIME_CHANGE'
        
        if results['market_impact']['impact_significant']:
            return 'MARKET_IMPACT'
        
        if results['data_quality']['survivorship_bias']:
            return 'DATA_QUALITY'
        
        return 'UNKNOWN'
    
    def recommend_action(self, cause: str) -> str:
        actions = {
            'OVERFITTING': 'Simplify strategy, reduce parameters, use walk-forward analysis',
            'TRANSACTION_COSTS': 'Reduce turnover, use limit orders, implement VWAP execution',
            'REGIME_CHANGE': 'Add regime detection, adapt parameters, diversify across regimes',
            'MARKET_IMPACT': 'Reduce position sizes, scale into positions, use execution algos',
            'DATA_QUALITY': 'Use point-in-time data, include delistings, fix corporate actions'
        }
        return actions.get(cause, 'Further investigation needed')
\`\`\`

**Most Likely Cause: OVERFITTING (70% probability)**

Given: 2.5 Sharpe in backtest, 0.8 in live → classic overfitting signature

**Recommended Actions:**
1. Run walk-forward analysis (should see consistent degradation)
2. Simplify strategy (fewer parameters)
3. Test on different markets/periods
4. Require economic rationale (why should this work?)
5. Consider strategy dead, develop new one

**Key Lesson**: A Sharpe of 2.5 in backtest should be suspicious (too good). Realistic strategies: 1.0-1.5 Sharpe. Anything >2.0 likely overfit.`,
    keyPoints: [
      'Sharpe drop from 2.5 → 0.8 suggests overfitting (most common), transaction costs, or regime change',
      'Overfitting diagnosed via walk-forward analysis, parameter sensitivity, data snooping tests',
      'Regime change detected by volatility analysis, correlation shifts, trend vs chop metrics',
      'Transaction costs often underestimated: model 1 bps, actual 10+ bps destroys high-frequency strategies',
      'Systematic diagnostic framework: test all causes, identify primary issue, take corrective action',
    ],
  },
];
