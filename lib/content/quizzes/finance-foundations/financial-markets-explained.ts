export const financialMarketsExplainedQuiz = [
  {
    id: 'fme-q-1',
    question:
      'Build a multi-asset portfolio optimization system that handles equities, bonds, commodities, and crypto. Design: (1) data fetching infrastructure (handle different APIs, data formats, update frequencies), (2) return calculation methodology (arithmetic vs geometric, handling different time zones for crypto), (3) correlation matrix computation (handling missing data, non-synchronous trading), (4) portfolio optimization (mean-variance, risk parity, Black-Litterman), (5) rebalancing logic (tax-aware, transaction cost-aware). Include code architecture, handle edge cases (delisted stocks, market holidays, crypto downtime), and explain trade-offs between different optimization methods.',
    sampleAnswer: `Multi-Asset Portfolio Optimization System Design:

**1. Data Fetching Infrastructure**

\`\`\`python
from dataclasses import dataclass
from typing import Dict, List
from abc import ABC, abstractmethod
import pandas as pd
import yfinance as yf
import ccxt  # Crypto exchange library

@dataclass
class AssetData:
    """Standardized asset data format"""
    ticker: str
    asset_class: str  # 'equity', 'bond', 'commodity', 'crypto'
    prices: pd.Series
    returns: pd.Series
    last_updated: pd.Timestamp

class DataProvider(ABC):
    """Abstract base class for data providers"""
    
    @abstractmethod
    def fetch_data (self, ticker: str, start_date, end_date) -> AssetData:
        pass

class EquityDataProvider(DataProvider):
    """Fetch equity data from Yahoo Finance"""
    
    def fetch_data (self, ticker, start_date, end_date) -> AssetData:
        df = yf.download (ticker, start=start_date, end=end_date, progress=False)
        prices = df['Adj Close']
        returns = prices.pct_change().dropna()
        
        return AssetData(
            ticker=ticker,
            asset_class='equity',
            prices=prices,
            returns=returns,
            last_updated=pd.Timestamp.now()
        )

class CryptoDataProvider(DataProvider):
    """Fetch crypto data (24/7 data)"""
    
    def __init__(self):
        self.exchange = ccxt.binance()
    
    def fetch_data (self, ticker, start_date, end_date) -> AssetData:
        # Crypto tickers like 'BTC/USDT'
        # Fetch OHLCV data
        since = int (start_date.timestamp() * 1000)
        ohlcv = self.exchange.fetch_ohlcv (ticker, '1d', since=since)
        
        df = pd.DataFrame (ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime (df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        prices = df['close']
        returns = prices.pct_change().dropna()
        
        return AssetData(
            ticker=ticker,
            asset_class='crypto',
            prices=prices,
            returns=returns,
            last_updated=pd.Timestamp.now()
        )

class MultiAssetDataManager:
    """Centralized data management for multiple asset classes"""
    
    def __init__(self):
        self.providers = {
            'equity': EquityDataProvider(),
            'bond': EquityDataProvider(),  # Bonds also on Yahoo
            'commodity': EquityDataProvider(),
            'crypto': CryptoDataProvider(),
        }
        self.cache = {}  # Cache fetched data
    
    def fetch_portfolio_data (self, portfolio: Dict[str, str], 
                             start_date, end_date) -> Dict[str, AssetData]:
        """
        Fetch data for entire portfolio
        
        portfolio = {
            'AAPL': 'equity',
            'BTC/USDT': 'crypto',
            'GC=F': 'commodity',  # Gold
            'TLT': 'bond'  # 20+ Year Treasury ETF
        }
        """
        data = {}
        for ticker, asset_class in portfolio.items():
            try:
                provider = self.providers[asset_class]
                asset_data = provider.fetch_data (ticker, start_date, end_date)
                data[ticker] = asset_data
                self.cache[ticker] = asset_data
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
                # Use cached data if available
                if ticker in self.cache:
                    data[ticker] = self.cache[ticker]
        
        return data
\`\`\`

**2. Return Calculation Methodology**

Challenge: Different assets trade at different times (crypto 24/7, stocks M-F 9:30-4pm ET)

\`\`\`python
import numpy as np

class ReturnCalculator:
    """Handle different return calculation methods"""
    
    @staticmethod
    def arithmetic_return (prices: pd.Series) -> float:
        """Simple average return"""
        returns = prices.pct_change().dropna()
        return returns.mean()
    
    @staticmethod
    def geometric_return (prices: pd.Series) -> float:
        """Compound average return (CAGR)"""
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        years = (prices.index[-1] - prices.index[0]).days / 365.25
        cagr = (1 + total_return) ** (1 / years) - 1
        return cagr
    
    @staticmethod
    def log_return (prices: pd.Series) -> pd.Series:
        """Log returns (better for statistical properties)"""
        return np.log (prices / prices.shift(1)).dropna()
    
    @staticmethod
    def align_returns (data: Dict[str, AssetData], 
                      frequency: str = 'D') -> pd.DataFrame:
        """
        Align returns across assets with different trading schedules
        
        Key challenge: Crypto trades 24/7, stocks don't
        Solution: Resample all to common frequency, forward-fill weekends
        """
        returns_dict = {}
        
        for ticker, asset_data in data.items():
            returns = asset_data.returns
            
            # Resample to common frequency
            if asset_data.asset_class == 'crypto':
                # Crypto: Resample to daily, use last price of day
                returns = returns.resample (frequency).last()
            
            returns_dict[ticker] = returns
        
        # Create aligned DataFrame
        df = pd.DataFrame (returns_dict)
        
        # Forward-fill missing values (weekends, holidays)
        # This assumes asset maintains value when not trading
        df = df.fillna (method='ffill')
        
        # Drop rows where any asset has no data
        df = df.dropna()
        
        return df

# Usage
calculator = ReturnCalculator()
portfolio_data = manager.fetch_portfolio_data (portfolio, start_date, end_date)

# Get aligned returns
aligned_returns = calculator.align_returns (portfolio_data)
print(aligned_returns.head())
\`\`\`

**3. Correlation Matrix with Missing Data**

\`\`\`python
class CorrelationCalculator:
    """Robust correlation calculation"""
    
    @staticmethod
    def pairwise_correlation (returns: pd.DataFrame, 
                             method: str = 'pearson',
                             min_periods: int = 30) -> pd.DataFrame:
        """
        Calculate correlation with minimum period requirement
        
        Handles missing data gracefully
        """
        return returns.corr (method=method, min_periods=min_periods)
    
    @staticmethod
    def shrinkage_correlation (returns: pd.DataFrame, 
                              shrinkage_factor: float = 0.1) -> pd.DataFrame:
        """
        Ledoit-Wolf shrinkage for more stable correlation estimates
        
        Shrinks sample correlation toward identity matrix
        Useful when you have more assets than observations
        """
        sample_corr = returns.corr()
        identity = np.eye (len (sample_corr))
        
        # Shrink toward identity matrix
        shrunk_corr = (1 - shrinkage_factor) * sample_corr + shrinkage_factor * identity
        
        return pd.DataFrame (shrunk_corr, 
                            index=sample_corr.index, 
                            columns=sample_corr.columns)
    
    @staticmethod
    def rolling_correlation (returns: pd.DataFrame, 
                            window: int = 60) -> Dict[str, pd.DataFrame]:
        """
        Calculate rolling correlation matrices
        
        Useful for understanding how correlations change over time
        """
        rolling_corr = {}
        for i in range (window, len (returns)):
            window_data = returns.iloc[i-window:i]
            corr_matrix = window_data.corr()
            rolling_corr[returns.index[i]] = corr_matrix
        
        return rolling_corr

# Usage
corr_calc = CorrelationCalculator()

# Sample correlation
correlation_matrix = corr_calc.pairwise_correlation (aligned_returns)
print("\\nCorrelation Matrix:")
print(correlation_matrix)

# Shrunk correlation (more stable for optimization)
shrunk_corr = corr_calc.shrinkage_correlation (aligned_returns, shrinkage_factor=0.2)
\`\`\`

**4. Portfolio Optimization Methods**

\`\`\`python
from scipy.optimize import minimize
import cvxpy as cp

class PortfolioOptimizer:
    """Multiple optimization strategies"""
    
    def mean_variance_optimization (self, returns: pd.DataFrame, 
                                    target_return: float = None) -> np.ndarray:
        """
        Markowitz mean-variance optimization
        
        Minimize: w' Σ w (portfolio variance)
        Subject to: w' μ = target_return
                    Σ w_i = 1
                    w_i >= 0 (long-only)
        """
        mu = returns.mean() * 252  # Annualize
        Sigma = returns.cov() * 252  # Annualize
        n = len (mu)
        
        # Define optimization variables
        w = cp.Variable (n)
        
        # Objective: minimize variance
        portfolio_variance = cp.quad_form (w, Sigma)
        
        # Constraints
        constraints = [
            cp.sum (w) == 1,  # Weights sum to 1
            w >= 0  # Long-only (no short positions)
        ]
        
        if target_return:
            constraints.append (mu @ w >= target_return)
        
        # Solve
        problem = cp.Problem (cp.Minimize (portfolio_variance), constraints)
        problem.solve()
        
        return w.value
    
    def risk_parity (self, returns: pd.DataFrame) -> np.ndarray:
        """
        Risk parity: Equal risk contribution from all assets
        
        Each asset contributes equally to portfolio risk
        """
        Sigma = returns.cov() * 252  # Annualized covariance
        n = len(Sigma)
        
        def risk_parity_objective (w):
            # Portfolio variance
            portfolio_var = w @ Sigma @ w
            
            # Marginal risk contribution
            marginal_contrib = Sigma @ w
            
            # Risk contribution
            risk_contrib = w * marginal_contrib / np.sqrt (portfolio_var)
            
            # Target: equal risk from all assets
            target_risk = np.sqrt (portfolio_var) / n
            
            # Minimize squared differences
            return np.sum((risk_contrib - target_risk) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum (w) - 1},  # Sum to 1
        ]
        bounds = tuple((0, 1) for _ in range (n))  # Long-only
        
        # Initial guess: equal weights
        w0 = np.ones (n) / n
        
        result = minimize (risk_parity_objective, w0, 
                          method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x
    
    def maximum_sharpe (self, returns: pd.DataFrame, 
                       risk_free_rate: float = 0.04) -> np.ndarray:
        """
        Maximize Sharpe ratio
        
        Sharpe = (return - risk_free) / volatility
        """
        mu = returns.mean() * 252
        Sigma = returns.cov() * 252
        n = len (mu)
        
        w = cp.Variable (n)
        
        # Expected return
        portfolio_return = mu @ w
        
        # Volatility
        portfolio_std = cp.sqrt (cp.quad_form (w, Sigma))
        
        # Sharpe ratio (we minimize negative Sharpe)
        sharpe = (portfolio_return - risk_free_rate) / portfolio_std
        
        constraints = [
            cp.sum (w) == 1,
            w >= 0
        ]
        
        problem = cp.Problem (cp.Maximize (sharpe), constraints)
        problem.solve()
        
        return w.value

# Usage
optimizer = PortfolioOptimizer()

# Mean-variance
mv_weights = optimizer.mean_variance_optimization (aligned_returns, target_return=0.10)
print("\\nMean-Variance Weights:")
for ticker, weight in zip (aligned_returns.columns, mv_weights):
    print(f"  {ticker}: {weight:.2%}")

# Risk parity
rp_weights = optimizer.risk_parity (aligned_returns)
print("\\nRisk Parity Weights:")
for ticker, weight in zip (aligned_returns.columns, rp_weights):
    print(f"  {ticker}: {weight:.2%}")

# Maximum Sharpe
ms_weights = optimizer.maximum_sharpe (aligned_returns)
print("\\nMaximum Sharpe Weights:")
for ticker, weight in zip (aligned_returns.columns, ms_weights):
    print(f"  {ticker}: {weight:.2%}")
\`\`\`

**5. Rebalancing Logic (Tax & Cost Aware)**

\`\`\`python
class Rebalancer:
    """Smart rebalancing with cost awareness"""
    
    def __init__(self, transaction_cost: float = 0.001, 
                 tax_rate_short: float = 0.37,
                 tax_rate_long: float = 0.20):
        self.transaction_cost = transaction_cost
        self.tax_rate_short = tax_rate_short
        self.tax_rate_long = tax_rate_long
    
    def calculate_rebalance_trades (self, current_weights: Dict[str, float],
                                    target_weights: Dict[str, float],
                                    current_prices: Dict[str, float],
                                    cost_basis: Dict[str, float],
                                    holding_period_days: Dict[str, int],
                                    portfolio_value: float,
                                    threshold: float = 0.05) -> Dict[str, float]:
        """
        Calculate optimal rebalancing trades
        
        Parameters:
        -----------
        threshold : float
            Only rebalance if drift > threshold (e.g., 5%)
        """
        trades = {}
        total_cost = 0
        total_tax = 0
        
        for ticker in current_weights:
            current_weight = current_weights[ticker]
            target_weight = target_weights.get (ticker, 0)
            
            # Check if rebalancing needed
            drift = abs (current_weight - target_weight)
            if drift < threshold:
                continue  # Skip if within threshold
            
            # Calculate trade size
            weight_change = target_weight - current_weight
            dollar_change = weight_change * portfolio_value
            shares_change = dollar_change / current_prices[ticker]
            
            # Calculate costs
            transaction_cost_dollar = abs (dollar_change) * self.transaction_cost
            
            # Calculate taxes (only on gains when selling)
            if dollar_change < 0:  # Selling
                current_value = current_weight * portfolio_value
                basis_value = (current_value / current_prices[ticker]) * cost_basis[ticker]
                gain = current_value - basis_value
                
                if gain > 0:
                    # Long-term vs short-term capital gains
                    if holding_period_days[ticker] > 365:
                        tax = gain * (abs (dollar_change) / current_value) * self.tax_rate_long
                    else:
                        tax = gain * (abs (dollar_change) / current_value) * self.tax_rate_short
                    
                    total_tax += tax
            
            total_cost += transaction_cost_dollar
            
            trades[ticker] = {
                'shares': shares_change,
                'dollars': dollar_change,
                'transaction_cost': transaction_cost_dollar,
                'tax': total_tax
            }
        
        return trades, total_cost, total_tax
    
    def tax_loss_harvesting (self, positions: Dict[str, dict]) -> List[str]:
        """
        Identify positions for tax-loss harvesting
        
        Sell positions with losses to offset gains
        """
        harvestable = []
        
        for ticker, position in positions.items():
            if position['current_value'] < position['cost_basis']:
                loss = position['cost_basis'] - position['current_value']
                harvestable.append((ticker, loss))
        
        # Sort by loss amount (harvest biggest losses first)
        harvestable.sort (key=lambda x: x[1], reverse=True)
        
        return [ticker for ticker, loss in harvestable]

# Usage
rebalancer = Rebalancer()

current_weights = {'AAPL': 0.30, 'BTC': 0.20, 'GLD': 0.30, 'TLT': 0.20}
target_weights = {'AAPL': 0.25, 'BTC': 0.15, 'GLD': 0.35, 'TLT': 0.25}

trades, transaction_costs, taxes = rebalancer.calculate_rebalance_trades(
    current_weights, target_weights, current_prices, cost_basis, 
    holding_period_days, portfolio_value=1_000_000, threshold=0.03
)

print("\\nRebalancing Trades:")
for ticker, trade in trades.items():
    print(f"  {ticker}: {trade['shares']:.0f} shares (\${trade['dollars']:,.0f}) ")
print(f"\\nTotal Transaction Costs: \${transaction_costs:,.2f}")
print(f"Total Taxes: \${taxes:,.2f}")
\`\`\`

**Trade-offs Summary:**

1. Mean-Variance: Best for maximizing return per risk, but sensitive to input estimates
2. Risk Parity: More stable, better diversification, but may have lower returns
3. Maximum Sharpe: Optimizes risk-adjusted returns, but concentrated in high-Sharpe assets
4. Black-Litterman: Incorporates views, more realistic, but complex to implement

**Recommendation**: Start with Mean-Variance for simplicity, add shrinkage to correlation matrix for stability, implement tax-aware rebalancing for real-world portfolios.`,
    keyPoints: [
      'Data infrastructure: Separate providers for each asset class (EquityDataProvider, CryptoDataProvider), handle 24/7 crypto vs M-F stocks',
      'Return alignment: Resample to common frequency, forward-fill weekends/holidays, use geometric returns for long-term, log returns for modeling',
      'Correlation: Use Ledoit-Wolf shrinkage for stability, pairwise calculation with min 30 periods, rolling correlation for regime analysis',
      'Optimization: Mean-variance (maximize return/risk), risk parity (equal risk contribution), max Sharpe (best risk-adjusted), each has trade-offs',
      'Rebalancing: Only rebalance if drift >5%, tax-aware (long-term gains 20% vs short-term 37%), transaction costs reduce returns by 0.1-0.5%/year',
    ],
  },
  {
    id: 'fme-q-2',
    question:
      'Explain why bond prices and interest rates move inversely. Design a bond portfolio immunization strategy that: (1) calculates duration and convexity, (2) constructs a portfolio matching a liability duration, (3) handles interest rate shocks (+/-2%), (4) rebalances as duration drifts, (5) compares to naive strategies. Include mathematical derivation, code implementation, and analysis of when immunization fails.',
    sampleAnswer: `Answer to be completed (comprehensive bond immunization covering: inverse price-yield relationship because present value denominator increases when rates rise reducing PV of future cash flows, duration measures price sensitivity (modified duration = -dP/P per 1% yield change), convexity captures second-order curvature, immunization matches portfolio duration to liability duration so parallel shifts cancel out, implementation requires: calculate Macaulay duration of all bonds, use optimization to find portfolio weights matching target duration, handle non-parallel shifts with key rate duration, rebalance quarterly as duration changes with time passage, analysis shows immunization works for parallel shifts but fails for yield curve twists or large non-parallel moves, naive strategy of just buying zero-coupon bond matching liability date is simpler but has reinvestment risk).`,
    keyPoints: [
      'Inverse relationship: Bond price = PV of cash flows; when discount rate (yield) rises, PV falls, so price falls',
      'Duration: First-order sensitivity (-10% duration means 1% yield rise → 10% price fall), Macaulay = weighted average time to cash flows',
      'Immunization: Match portfolio duration to liability duration, so interest rate changes affect assets and liabilities equally',
      'Convexity: Second-order effect (positive convexity = benefit from large rate moves), adds ~0.5 × Convexity × (Δy)²',
      'Limitations: Only works for parallel shifts (all rates move together), fails for yield curve twists, need to rebalance as duration drifts',
    ],
  },
  {
    id: 'fme-q-3',
    question:
      'Build a cryptocurrency trading system that handles 24/7 markets. Cover: (1) WebSocket connections for real-time data (handle disconnects, reconnects), (2) order execution across multiple exchanges (Binance, Coinbase, Kraken), (3) arbitrage detection (triangular arbitrage, cross-exchange), (4) risk management (volatility filters, position limits), (5) monitoring and alerting. Include handling extreme volatility (50%+ moves), exchange downtime, and flash crashes.',
    sampleAnswer: `Answer to be completed (24/7 crypto trading system covering: WebSocket implementation using ccxt.pro for real-time order book + trades streaming, handle disconnections with exponential backoff reconnection, execute orders via REST API with retry logic for failed orders, triangular arbitrage checks BTC/USD → ETH/BTC → ETH/USD for profit opportunities requiring <100ms execution, cross-exchange arbitrage compares Binance BTC price vs Coinbase accounting for fees + transfer time, risk management includes: circuit breaker stops trading if volatility >5% in 1 minute, max position $100K per exchange, stop-loss at -2%, monitoring uses Prometheus + Grafana for latency metrics + P&L tracking, alerting via PagerDuty for: exchange disconnection >30 seconds, position limit breach, daily loss >$10K, edge cases include flash crash handling via price reasonableness checks rejecting trades >10% from VWAP, exchange downtime gracefully fails to other exchanges, extreme volatility pauses trading until volatility <3% for 5 minutes).`,
    keyPoints: [
      'WebSocket: Use ccxt.pro for real-time streaming, handle disconnects with exponential backoff, heartbeat every 30s to detect stale connections',
      'Multi-exchange: Binance (lowest fees), Coinbase (most liquid USD), Kraken (backup), route orders to best price after fees',
      'Arbitrage: Triangular (BTC→ETH→USD→BTC) needs <100ms execution, cross-exchange needs <30s for profitable after transfer costs',
      'Risk: Circuit breaker at 5% volatility, max position $100K, stop-loss -2%, position sizing based on volatility (higher vol = smaller size)',
      'Monitoring: Track: latency (<50ms target), order fill rate (>95%), P&L (real-time), exchange health, alert on anomalies',
    ],
  },
];
