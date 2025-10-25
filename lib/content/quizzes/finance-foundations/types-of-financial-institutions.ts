export const typesOfFinancialInstitutionsQuiz = [
  {
    id: 'tfi-q-1',
    question:
      "You're joining a quantitative hedge fund as a researcher. Design a complete systematic trading strategy development pipeline: (1) research environment setup (data, tools, libraries), (2) alpha signal discovery process (hypothesis → testing → validation), (3) backtesting framework with realistic assumptions (transaction costs, slippage, market impact), (4) risk management integration (position sizing, stop losses, portfolio constraints), (5) production deployment considerations (latency, monitoring, fail-safes). Include code architecture, statistical rigor requirements, and common pitfalls to avoid.",
    sampleAnswer: `Systematic Trading Strategy Development Pipeline:

**1. Research Environment Setup**

Infrastructure:
\`\`\`python
# Environment: Linux workstation with GPU
# Tools: Jupyter Lab, VS Code, Git, Docker

# Data stack
import pandas as pd
import numpy as np
import polars as pl  # Faster than pandas for large datasets
import dask  # Distributed computing for huge datasets

# Market data sources
import yfinance as yf  # Free EOD data
from alpaca_trade_api import REST  # Real-time & historical
import ccxt  # Crypto data (200+ exchanges)

# Backtesting frameworks
import backtrader  # Event-driven backtesting
import vectorbt as vbt  # Vectorized backtesting (faster)
import zipline  # Quantopian\'s framework

# ML/Statistical tools
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

# Risk & Portfolio
import PyPortfolioOpt as ppo
import pyfolio  # Performance analysis
import empyrical  # Performance metrics

# Databases
import psycopg2  # PostgreSQL for fundamentals
from influxdb_client import InfluxDBClient  # Time-series data
import redis  # Caching

# Version control
# Git for code, DVC for data versioning
\`\`\`

Research workflow:
- Jupyter notebooks for exploration (prototype)
- .py files for production code (refactor)
- Git branches for each strategy hypothesis
- DVC to track datasets and model versions

**2. Alpha Signal Discovery Process**

Step 1: Hypothesis Generation
Sources of ideas:
- Academic papers (SSRN, arXiv, Journal of Finance)
- Market anomalies (momentum, mean reversion, volatility)
- Alternative data (sentiment, options flow, positioning)
- Pattern recognition (technical indicators, regime shifts)

Example hypothesis: "Stocks with high short interest that gap down >5% on earnings tend to mean-revert within 5 days"

Step 2: Data Collection
\`\`\`python
def collect_earnings_data (start_date, end_date):
    """Collect earnings dates, surprises, and price gaps"""
    # Get earnings calendar
    earnings = fetch_earnings_calendar (start_date, end_date)
    
    # Get price data
    prices = fetch_price_data (earnings['tickers'], start_date, end_date)
    
    # Get short interest
    short_interest = fetch_short_interest (earnings['tickers'])
    
    # Calculate gaps
    gaps = calculate_earnings_gaps (prices, earnings['dates'])
    
    # Merge datasets
    data = pd.merge (earnings, gaps, on=['ticker', 'date'])
    data = pd.merge (data, short_interest, on='ticker')
    
    return data

# Statistical requirements
# - Minimum sample size: 1000+ observations
# - Multiple market regimes: bull, bear, sideways
# - Time period: 5+ years (avoid overfitting to recent regime)
\`\`\`

Step 3: Exploratory Analysis
\`\`\`python
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_signal (data):
    """Analyze signal strength and characteristics"""
    # 1. Return distribution conditional on signal
    high_short = data[data['short_interest'] > 0.20]
    low_short = data[data['short_interest'] < 0.10]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Forward returns by gap size
    axes[0, 0].scatter (data['gap_pct'], data['fwd_5d_return'])
    axes[0, 0].set_xlabel('Earnings Gap (%)')
    axes[0, 0].set_ylabel('5-Day Forward Return (%)')
    axes[0, 0].set_title('Gap vs Forward Return')
    
    # Distribution comparison
    axes[0, 1].hist([high_short['fwd_5d_return'], low_short['fwd_5d_return']], 
                    bins=50, alpha=0.7, label=['High Short', 'Low Short'])
    axes[0, 1].set_xlabel('5-Day Return (%)')
    axes[0, 1].legend()
    
    # Statistical tests
    t_stat, p_value = stats.ttest_ind(
        high_short['fwd_5d_return'],
        low_short['fwd_5d_return']
    )
    
    print(f"T-statistic: {t_stat:.2f}")
    print(f"P-value: {p_value:.4f}")
    
    # Effect size (Cohen\'s d)
    mean_diff = high_short['fwd_5d_return'].mean() - low_short['fwd_5d_return'].mean()
    pooled_std = np.sqrt((high_short['fwd_5d_return'].std()**2 + 
                          low_short['fwd_5d_return'].std()**2) / 2)
    cohens_d = mean_diff / pooled_std
    
    print(f"Effect size (Cohen\'s d): {cohens_d:.3f}")
    
    # Decay analysis (how long does signal last?)
    for days in [1, 2, 5, 10, 20]:
        corr = data['gap_pct'].corr (data[f'fwd_{days}d_return'])
        print(f"Signal decay at {days} days: {corr:.3f}")
    
    return t_stat, p_value, cohens_d
\`\`\`

Step 4: Statistical Validation
Requirements for signal acceptance:
- P-value < 0.05 (statistically significant)
- Cohen\'s d > 0.2 (meaningful effect size)
- Consistent across market regimes (bull/bear/sideways)
- Information ratio > 0.5 (Sharpe ratio of signal alone)
- Not explained by known factors (CAPM, Fama-French 5-factor)

**3. Backtesting Framework with Realistic Assumptions**

\`\`\`python
class RealisticBacktester:
    """
    Backtesting with realistic market conditions
    Critical: Avoid lookahead bias, survivorship bias, overfitting
    """
    
    def __init__(self, initial_capital=1_000_000):
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        # Transaction costs
        self.commission_rate = 0.0005  # 5 bps per trade
        self.min_commission = 1.0  # $1 minimum
        self.slippage_bps = 2.0  # 2 bps slippage
        
        # Market impact model (square root model)
        self.market_impact_coef = 0.1
    
    def calculate_transaction_costs (self, price, shares, adv):
        """
        Calculate total transaction costs
        
        Parameters:
        -----------
        price : float
            Execution price
        shares : int
            Number of shares
        adv : float
            Average daily volume
        """
        notional = price * shares
        
        # Commission
        commission = max (notional * self.commission_rate, self.min_commission)
        
        # Slippage (fixed)
        slippage = notional * (self.slippage_bps / 10000)
        
        # Market impact (depends on order size relative to ADV)
        participation_rate = shares / adv if adv > 0 else 0
        market_impact = notional * self.market_impact_coef * np.sqrt (participation_rate)
        
        total_cost = commission + slippage + market_impact
        
        return {
            'commission': commission,
            'slippage': slippage,
            'market_impact': market_impact,
            'total_cost': total_cost,
            'total_cost_bps': (total_cost / notional) * 10000
        }
    
    def backtest_strategy (self, signals, prices, volumes):
        """
        Run backtest with realistic execution
        
        Key considerations:
        1. No lookahead bias: Use only data available at trade time
        2. Execute at next day's open (after signal generation)
        3. Account for halts, delistings, suspensions
        4. Capacity constraints (can't trade more than 10% ADV)
        """
        for date in signals.index:
            # Get signal at market close
            signal = signals.loc[date]
            
            # Execute next day at open (delay)
            next_date = get_next_trading_day (date)
            if next_date not in prices.index:
                continue
            
            exec_price = prices.loc[next_date, 'Open']
            adv = volumes.loc[next_date]
            
            # Position sizing (Kelly criterion or fixed risk)
            position_size = self.calculate_position_size(
                signal['score'],
                signal['confidence'],
                self.capital
            )
            
            # Capacity check: Don't exceed 10% of ADV
            max_shares = adv * 0.10
            desired_shares = position_size / exec_price
            actual_shares = min (desired_shares, max_shares)
            
            if actual_shares < 100:  # Skip if less than 1 lot
                continue
            
            # Calculate costs
            costs = self.calculate_transaction_costs (exec_price, actual_shares, adv)
            
            # Execute trade
            self.execute_trade(
                ticker=signal['ticker'],
                shares=actual_shares,
                price=exec_price,
                costs=costs['total_cost'],
                date=next_date
            )
            
            # Update equity curve
            self.update_equity (date, prices, volumes)
        
        return self.analyze_performance()
    
    def analyze_performance (self):
        """Calculate performance metrics"""
        returns = pd.Series (self.equity_curve).pct_change()
        
        metrics = {
            'total_return': (self.equity_curve[-1] / self.equity_curve[0]) - 1,
            'sharpe_ratio': empyrical.sharpe_ratio (returns),
            'sortino_ratio': empyrical.sortino_ratio (returns),
            'max_drawdown': empyrical.max_drawdown (returns),
            'calmar_ratio': empyrical.calmar_ratio (returns),
            'win_rate': len([t for t in self.trades if t['pnl'] > 0]) / len (self.trades),
            'avg_trade_pnl': np.mean([t['pnl'] for t in self.trades]),
            'total_costs': sum([t['costs'] for t in self.trades]),
            'costs_as_pct_return': sum([t['costs'] for t in self.trades]) / self.equity_curve[-1]
        }
        
        return metrics
\`\`\`

Critical backtesting checks:
- Walk-forward analysis: Train on 3 years, test on 1 year, roll forward
- Out-of-sample validation: Hold out 20% of data, never look until final validation
- Monte Carlo: Shuffle trade order 10,000 times, ensure consistent performance
- Regime analysis: Performance in bull, bear, high vol, low vol periods
- Capacity analysis: How does performance degrade as AUM grows?

**4. Risk Management Integration**

\`\`\`python
class RiskManager:
    """Portfolio-level risk management"""
    
    def __init__(self, max_portfolio_var=0.02, max_position_size=0.05):
        self.max_portfolio_var = max_portfolio_var  # 2% daily VaR
        self.max_position_size = max_position_size  # 5% per position
        self.max_sector_exposure = 0.30  # 30% per sector
        self.max_leverage = 2.0
        
    def position_sizing_kelly (self, win_rate, avg_win, avg_loss):
        """
        Kelly Criterion for position sizing
        
        Kelly% = (p * b - q) / b
        where p = win rate, q = 1-p, b = avg_win / avg_loss
        
        Use fractional Kelly (0.25 - 0.5) to reduce volatility
        """
        if avg_loss == 0:
            return 0
        
        b = avg_win / abs (avg_loss)
        kelly_full = (win_rate * b - (1 - win_rate)) / b
        kelly_fraction = 0.25  # Use 25% Kelly for safety
        
        return max(0, min (kelly_full * kelly_fraction, self.max_position_size))
    
    def position_sizing_var (self, strategy_volatility, correlation_matrix, portfolio):
        """
        Position sizing based on portfolio VaR target
        
        Ensures adding new position doesn't exceed VaR limit
        """
        # Calculate current portfolio VaR
        weights = np.array([p['weight'] for p in portfolio.values()])
        cov_matrix = correlation_matrix * np.outer(
            [p['volatility'] for p in portfolio.values()],
            [p['volatility'] for p in portfolio.values()]
        )
        
        portfolio_var = np.sqrt (weights @ cov_matrix @ weights.T)
        
        # Marginal VaR: How much does this position add to risk?
        if portfolio_var < self.max_portfolio_var:
            # Room for more risk
            remaining_var = self.max_portfolio_var - portfolio_var
            position_size = remaining_var / strategy_volatility
            return min (position_size, self.max_position_size)
        else:
            # Already at limit
            return 0
    
    def stop_loss_placement (self, entry_price, atr, confidence):
        """
        Dynamic stop loss based on volatility
        
        Higher confidence → wider stops
        Lower confidence → tighter stops
        """
        base_stop_multiple = 2.0  # 2x ATR
        confidence_adj = 0.5 + (confidence * 1.5)  # 0.5x to 2x
        
        stop_distance = atr * base_stop_multiple * confidence_adj
        stop_loss = entry_price - stop_distance
        
        return stop_loss
    
    def portfolio_constraints (self, proposed_portfolio):
        """Check if portfolio violates constraints"""
        violations = []
        
        # Position size check
        for ticker, position in proposed_portfolio.items():
            if position['weight'] > self.max_position_size:
                violations.append (f"{ticker}: Position too large ({position['weight']:.1%})")
        
        # Sector exposure check
        sector_exposure = {}
        for ticker, position in proposed_portfolio.items():
            sector = position['sector']
            sector_exposure[sector] = sector_exposure.get (sector, 0) + position['weight']
        
        for sector, exposure in sector_exposure.items():
            if exposure > self.max_sector_exposure:
                violations.append (f"{sector}: Sector exposure too high ({exposure:.1%})")
        
        # Leverage check
        gross_leverage = sum([abs (p['weight']) for p in proposed_portfolio.values()])
        if gross_leverage > self.max_leverage:
            violations.append (f"Leverage too high: {gross_leverage:.2f}x")
        
        return violations
\`\`\`

**5. Production Deployment Considerations**

Architecture:
\`\`\`
Research (Jupyter) → Refactor (Python modules) → Backtest (validation) → Paper Trading (dry run) → Live Trading (production)
\`\`\`

Production system components:
\`\`\`python
class ProductionTradingSystem:
    """
    Production-grade trading system
    
    Requirements:
    - Latency: <100ms for signal → order
    - Uptime: 99.9% during market hours
    - Monitoring: Real-time alerts for anomalies
    - Fail-safes: Kill switch, max loss limits
    """
    
    def __init__(self):
        self.redis_client = redis.Redis()  # State management
        self.postgres_conn = psycopg2.connect()  # Trade log
        self.broker_api = AlpacaAPI()  # Execution
        
        # Monitoring
        self.datadog = DatadogClient()
        self.pagerduty = PagerDutyClient()
        
        # Safety limits
        self.max_daily_loss = 50000  # $50K daily loss limit
        self.max_order_size = 100000  # $100K per order
        self.trading_enabled = True
        
    def run_strategy_loop (self):
        """Main trading loop"""
        while self.trading_enabled:
            try:
                # 1. Fetch latest market data (< 10ms)
                data = self.fetch_market_data()
                
                # 2. Generate signals (< 50ms)
                signals = self.generate_signals (data)
                
                # 3. Risk checks (< 20ms)
                approved_signals = self.risk_check (signals)
                
                # 4. Execute orders (< 20ms)
                orders = self.execute_orders (approved_signals)
                
                # 5. Update monitoring (< 10ms)
                self.update_monitoring (orders)
                
                # 6. Safety checks
                self.check_safety_limits()
                
            except Exception as e:
                self.handle_error (e)
                self.emergency_shutdown()
            
            time.sleep(1)  # Run every second
    
    def check_safety_limits (self):
        """Emergency shutdown conditions"""
        # Check daily loss
        daily_pnl = self.calculate_daily_pnl()
        if daily_pnl < -self.max_daily_loss:
            self.emergency_shutdown("Max daily loss exceeded")
        
        # Check position concentration
        largest_position = max([p['value'] for p in self.get_positions().values()])
        if largest_position > self.max_order_size * 2:
            self.alert_risk_team("Position concentration high")
        
        # Check market connectivity
        if not self.broker_api.is_connected():
            self.emergency_shutdown("Lost broker connection")
        
        # Check data freshness
        if time.time() - self.last_data_update > 60:
            self.emergency_shutdown("Stale market data")
    
    def emergency_shutdown (self, reason):
        """Kill switch: Flatten all positions"""
        self.trading_enabled = False
        self.pagerduty.alert (f"EMERGENCY SHUTDOWN: {reason}")
        
        # Close all positions
        for ticker, position in self.get_positions().items():
            self.broker_api.market_order (ticker, -position['shares'])
        
        self.datadog.event("Trading halted", reason)
\`\`\`

Monitoring dashboard (real-time):
- P&L (intraday, MTD, YTD)
- Sharpe ratio (rolling 30/60/90 days)
- Win rate, avg trade P&L
- Execution quality (slippage vs expected)
- System health (latency, uptime, data freshness)
- Risk metrics (VaR, portfolio delta, sector exposure)

**Common Pitfalls to Avoid**

1. **Lookahead Bias**: Using future information
   - Bad: Use today's close to generate signal, execute at today's close
   - Good: Use yesterday's close to generate signal, execute tomorrow's open

2. **Survivorship Bias**: Only testing on stocks that still exist
   - Bad: Backtest on current S&P 500 constituents
   - Good: Use point-in-time S&P 500 membership

3. **Overfitting**: Too many parameters, too little data
   - Bad: 20 parameters optimized on 2 years of data
   - Good: 3-5 parameters validated on 10+ years across multiple regimes

4. **Ignoring Transaction Costs**: Assume zero cost
   - Bad: Trade 100 times/day with 5 bps costs → 12.6% annual drag!
   - Good: Model realistic costs, account for market impact

5. **Data Mining**: Testing 1000 signals, picking the one that worked
   - Bad: Try every combination, pick best backtest
   - Good: Hypothesis-driven research, validate out-of-sample

6. **Regime Dependency**: Only works in bull markets
   - Bad: Great Sharpe 2017-2021 (bull market), terrible 2022 (bear)
   - Good: Consistent performance across regimes

7. **Capacity Ignorance**: Strategy works at $10M, fails at $100M
   - Bad: Trade illiquid stocks, large position sizes
   - Good: Model capacity explicitly, stress test at scale

**Final Validation Checklist**
- [ ] Sharpe ratio > 1.0 (after costs)
- [ ] Positive returns in ≥60% of quarters
- [ ] Max drawdown < 20%
- [ ] Works across market regimes
- [ ] Passes walk-forward validation
- [ ] Out-of-sample Sharpe > 70% of in-sample
- [ ] Monte Carlo robust (95% of shuffles have Sharpe > 0.8)
- [ ] Capacity > $50M AUM
- [ ] Can explain WHY strategy works (not just curve-fit)`,
    keyPoints: [
      'Research environment: Jupyter for exploration, refactor to .py modules for production, Git + DVC for version control',
      "Statistical validation: p-value < 0.05, Cohen\'s d > 0.2, consistent across regimes, not explained by known factors",
      'Realistic backtesting: Model transaction costs (commission + slippage + market impact), avoid lookahead/survivorship bias, walk-forward validation',
      'Risk management: Kelly criterion for position sizing (use 25% fraction for safety), VaR-based portfolio limits, dynamic stops based on ATR',
      'Production deployment: <100ms latency, 99.9% uptime, real-time monitoring, emergency kill switch at max daily loss limit',
    ],
  },
  {
    id: 'tfi-q-2',
    question:
      'Compare career paths at (1) Goldman Sachs (investment bank strat), (2) Two Sigma (quant hedge fund), (3) BlackRock (asset manager), (4) Stripe (fintech), (5) Citadel Securities (market maker). For each, analyze: compensation structure (base/bonus/equity), work-life balance, engineering culture, learning opportunities, career progression, exit opportunities, and risk of role obsolescence. Which path would you choose for a 10-year career and why? Include realistic comp numbers and pros/cons analysis.',
    sampleAnswer: `Answer to be completed (comprehensive comparison covering: Goldman Sachs strat $150-400K mostly bonus-weighted with hierarchical progression to VP/MD but long hours 60-80/week, Two Sigma quant researcher $200-500K+ with profit sharing and cutting-edge ML but high pressure and publish-or-perish, BlackRock risk engineer $120-280K with better work-life balance 45-55 hours but slower career progression, Stripe senior engineer $200-400K base + equity potentially worth millions but startup risk and hyper-growth burnout, Citadel Securities low-latency engineer $200-500K+ with P&L share and ultimate technical challenge but market-hours pressure and specialized skills. Choice depends on: compensation priority vs work-life balance, desire for equity upside vs stable cash comp, preference for trading excitement vs product building, risk tolerance for startup equity vs established firm security, and long-term career goals - recommend Two Sigma for pure quant technical challenge, Stripe for equity upside and product building, BlackRock for work-life balance and stability).`,
    keyPoints: [
      'Investment bank (GS): $150-400K heavily bonus-weighted, 60-80 hr weeks, hierarchical (Analyst→VP→MD), exit to buy-side/corp dev',
      'Quant fund (Two Sigma): $200-500K+ with profit sharing, cutting-edge ML/research, publish-or-perish culture, exit to big tech AI/research',
      'Asset manager (BlackRock): $120-280K, best work-life balance (45-55 hrs), slower progression but stable, exit to fintech/corp finance',
      'Fintech (Stripe): $200-400K base + equity (potentially millions), startup culture, equity risk, exit to other startups/FAANG',
      'Market maker (Citadel): $200-500K+ with P&L share, ultimate low-latency challenge, market-hours pressure, exit to HFT/exchanges',
    ],
  },
  {
    id: 'tfi-q-3',
    question:
      'Design a market making system for a new crypto exchange. Cover: (1) order book data structure for O(1) best bid/ask lookup, (2) pricing algorithm considering inventory risk and adverse selection, (3) latency optimization techniques (C++, zero-copy, kernel bypass), (4) risk management (inventory limits, P&L circuit breakers), (5) profitability analysis showing minimum spread needed given trading volume. Include code architecture, expected latency targets, and competitive analysis vs Binance/Coinbase.',
    sampleAnswer: `Answer to be completed (comprehensive market maker system design covering: Order book using two heaps (max-heap for bids, min-heap for asks) with hash map for O(1) lookup and O(log n) insert/delete, pricing algorithm using Avellaneda-Stoikov model accounting for inventory skew (quote wider on long inventory side to shed risk), latency optimization with C++17 lock-free structures + DPDK kernel bypass + SIMD vectorization targeting <10μs exchange→quote update, risk management with max inventory $1M per asset + max daily loss $50K kill switch + max position drift 5% triggers rebalancing, profitability model showing need 2-5 bps spread on $1B daily volume = $200K-500K daily revenue minus infrastructure costs $50K/month, competitive analysis shows Binance <1ms latency via co-located matching engine vs Coinbase 5-10ms cloud-based gives latency advantage for professional market makers).`,
    keyPoints: [
      'Order book: Two heaps (max-heap bids, min-heap asks) + hash map for O(1) best bid/ask, O(log n) insert/delete',
      'Pricing: Avellaneda-Stoikov model - quote wider on inventory-heavy side to mean-revert, adjust for volatility and adverse selection',
      'Latency: C++ with lock-free structures, DPDK kernel bypass, SIMD, target <10μs quote update, <50μs order-to-execution',
      'Risk: Max inventory $1M per asset, daily loss limit $50K (kill switch), inventory drift >5% triggers forced rebalancing',
      'Economics: Need 2-5 bps spread on $1B daily volume = $200-500K daily revenue, infrastructure costs $50K/mo, 80%+ profit margin',
    ],
  },
];
