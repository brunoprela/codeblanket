import { Discussion } from '@/lib/types';

export const jupyterLabQuantQuiz: Discussion = {
  title: 'Jupyter Lab for Quantitative Research Discussion Questions',
  description:
    'Deep dive into professional Jupyter Lab workflows for quantitative finance.',
  questions: [
    {
      id: 'jupyter-disc-1',
      question:
        'Design a comprehensive Jupyter Lab workflow for developing and testing a new quantitative trading strategy from initial research to production deployment. Include directory structure, notebook organization, testing strategy, version control, and transition to production code.',
      sampleAnswer: `
# Professional Jupyter Lab Workflow for Quantitative Strategy Development

## Phase 1: Project Setup and Structure

### Directory Structure
\`\`\`plaintext
momentum_reversion_strategy/
├── .git/                           # Version control
├── .gitignore                      # Ignore outputs, data, env
├── .env                            # API keys, credentials (not committed)
├── README.md                       # Project overview and setup
├── requirements.txt                # Python dependencies
├── setup.py                        # Package installation
├── Makefile                        # Common commands
│
├── config/                         # Configuration files
│   ├── backtest_config.yaml       # Backtest parameters
│   ├── data_sources.yaml          # Data source credentials
│   └── trading_parameters.json     # Strategy parameters
│
├── data/                           # Data directory
│   ├── raw/                        # Original, immutable data
│   │   ├── .gitkeep
│   │   └── README.md              # Data source documentation
│   ├── processed/                  # Cleaned, feature-engineered data
│   │   └── .gitkeep
│   ├── external/                   # Third-party reference data
│   │   └── .gitkeep
│   └── interim/                    # Intermediate transformation steps
│       └── .gitkeep
│
├── notebooks/                      # Jupyter notebooks
│   ├── 00_project_overview.ipynb          # Project documentation
│   ├── 01_data_collection.ipynb           # Data acquisition
│   ├── 02_data_quality_check.ipynb        # Data validation
│   ├── 03_exploratory_analysis.ipynb      # EDA
│   ├── 04_feature_engineering.ipynb       # Feature creation
│   ├── 05_strategy_prototyping.ipynb      # Initial strategy dev
│   ├── 06_backtest_development.ipynb      # Backtest framework
│   ├── 07_parameter_optimization.ipynb    # Grid search/optimization
│   ├── 08_walk_forward_analysis.ipynb     # Out-of-sample testing
│   ├── 09_risk_analysis.ipynb             # Drawdown, VaR, etc.
│   ├── 10_transaction_cost_analysis.ipynb # Slippage, commissions
│   ├── 11_regime_analysis.ipynb           # Bull/bear performance
│   ├── 12_correlation_analysis.ipynb      # With other strategies
│   ├── 99_final_report.ipynb              # Executive summary
│   │
│   ├── archive/                            # Deprecated notebooks
│   │   └── old_approach_*.ipynb
│   └── templates/
│       └── analysis_template.ipynb
│
├── src/                            # Production-quality source code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py             # Data loading utilities
│   │   ├── validators.py          # Data quality checks
│   │   └── preprocessors.py       # Data cleaning
│   ├── features/
│   │   ├── __init__.py
│   │   ├── technical_indicators.py # TA-Lib wrappers
│   │   ├── fundamental_features.py # Fundamental data features
│   │   └── alternative_data.py     # Sentiment, etc.
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base_strategy.py        # Abstract base class
│   │   ├── momentum_reversion.py   # Main strategy
│   │   └── signals.py              # Signal generation
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py               # Backtesting engine
│   │   ├── execution.py            # Order execution simulation
│   │   ├── metrics.py              # Performance metrics
│   │   └── reporting.py            # Report generation
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── grid_search.py          # Parameter search
│   │   ├── walk_forward.py         # WF optimization
│   │   └── genetic_algorithm.py    # GA optimization
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── position_sizing.py      # Kelly criterion, etc.
│   │   ├── risk_metrics.py         # VaR, CVaR, etc.
│   │   └── portfolio.py            # Portfolio-level risk
│   └── utils/
│       ├── __init__.py
│       ├── logging_config.py       # Logging setup
│       ├── database.py             # DB connections
│       └── helpers.py              # Misc utilities
│
├── tests/                          # Unit and integration tests
│   ├── __init__.py
│   ├── conftest.py                 # Pytest fixtures
│   ├── test_data_loaders.py
│   ├── test_features.py
│   ├── test_strategies.py
│   ├── test_backtest_engine.py
│   └── integration/
│       └── test_end_to_end.py
│
├── results/                        # Analysis outputs
│   ├── backtests/
│   │   └── 2024-01-15_momentum_v1/
│   │       ├── metrics.csv
│   │       ├── trades.csv
│   │       └── equity_curve.png
│   ├── optimizations/
│   │   └── grid_search_results.csv
│   ├── reports/
│   │   └── monthly_performance.pdf
│   └── figures/
│       └── correlation_matrix.png
│
├── scripts/                        # Automation scripts
│   ├── run_backtest.py            # CLI backtest runner
│   ├── optimize_parameters.py      # Batch optimization
│   ├── generate_report.py          # Automated reporting
│   └── deploy_strategy.sh          # Deployment script
│
├── docs/                           # Documentation
│   ├── strategy_whitepaper.md     # Strategy description
│   ├── methodology.md              # Research methodology
│   └── api_documentation.md        # Code API docs
│
└── deployment/                     # Production deployment
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements_prod.txt
    └── production_config.yaml
\`\`\`

## Phase 2: Research Workflow (Notebooks)

### Step 1: Data Collection (01_data_collection.ipynb)

\`\`\`python
"""
# Data Collection Notebook

**Objective**: Download and store historical price data for S&P 500 constituents

**Data Sources**:
- Primary: Yahoo Finance (yfinance)
- Backup: Alpha Vantage API
- Alternative data: Quandl for fundamentals

**Output**: data/raw/sp500_prices.parquet
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig (level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
START_DATE = "2015-01-01"
END_DATE = "2024-01-01"
DATA_DIR = Path("../data/raw")
DATA_DIR.mkdir (parents=True, exist_ok=True)

# Get S&P 500 tickers
import pandas_datareader as pdr
sp500_tickers = pdr.get_data_yahoo('^GSPC')  # Simplified

# Download data with error handling
def download_ticker (ticker):
    try:
        df = yf.download (ticker, start=START_DATE, end=END_DATE, progress=False)
        logger.info (f"Downloaded {ticker}: {len (df)} rows")
        return ticker, df
    except Exception as e:
        logger.error (f"Failed to download {ticker}: {e}")
        return ticker, None

# Parallel download
from joblib import Parallel, delayed
results = Parallel (n_jobs=10)(
    delayed (download_ticker)(ticker) 
    for ticker in sp500_tickers[:50]  # Start with 50 for testing
)

# Combine and save
data_dict = {ticker: df for ticker, df in results if df is not None}
combined_df = pd.concat (data_dict, names=['ticker', 'date'])
combined_df.to_parquet(DATA_DIR / "sp500_prices.parquet")

logger.info (f"Saved data for {len (data_dict)} tickers")
\`\`\`

### Step 2: Data Quality Check (02_data_quality_check.ipynb)

\`\`\`python
"""
# Data Quality Assessment

**Checks**:
1. Missing data analysis
2. Outlier detection
3. Price discontinuities (splits, dividends)
4. Volume anomalies
5. Data completeness

**Action Items**: Document any data issues for handling in preprocessing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_parquet("../data/raw/sp500_prices.parquet")

# Check 1: Missing data
missing_by_ticker = df.groupby('ticker').apply(
    lambda x: x.isnull().sum() / len (x) * 100
)
print("Tickers with >5% missing data:")
print(missing_by_ticker[missing_by_ticker['Close'] > 5])

# Check 2: Price discontinuities (potential splits)
returns = df.groupby('ticker')['Close'].pct_change()
extreme_returns = returns[(returns < -0.3) | (returns > 0.3)]
print(f"\\nPotential splits/errors: {len (extreme_returns)} instances")

# Check 3: Volume analysis
volume_stats = df.groupby('ticker')['Volume'].describe()
zero_volume_days = df[df['Volume'] == 0].groupby('ticker').size()
print(f"\\nTickers with zero volume days:")
print(zero_volume_days[zero_volume_days > 5])

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Missing data heatmap
missing_data = df.isnull().groupby('ticker').sum()
sns.heatmap (missing_data.T, cmap='YlOrRd', ax=axes[0,0])
axes[0,0].set_title('Missing Data by Ticker')

# Returns distribution
returns.hist (bins=100, ax=axes[0,1])
axes[0,1].set_title('Returns Distribution')
axes[0,1].set_xlabel('Daily Return')

# Volume over time
df.groupby (level='date')['Volume'].sum().plot (ax=axes[1,0])
axes[1,0].set_title('Total Market Volume')

# Price levels
df.groupby('ticker')['Close'].last().hist (bins=50, ax=axes[1,1])
axes[1,1].set_title('Current Price Distribution')

plt.tight_layout()
plt.show()

# Save quality report
quality_report = {
    'total_tickers': df.index.get_level_values('ticker').nunique(),
    'date_range': f"{df.index.get_level_values('date').min()} to {df.index.get_level_values('date').max()}",
    'missing_data_pct': df.isnull().sum().sum() / df.size * 100,
    'tickers_with_issues': len (missing_by_ticker[missing_by_ticker['Close'] > 5])
}

pd.Series (quality_report).to_csv("../results/data_quality_report.csv")
\`\`\`

### Step 3: Strategy Prototyping (05_strategy_prototyping.ipynb)

THIS IS WHERE RESEARCH HAPPENS - Quick iteration, trying ideas.

\`\`\`python
"""
# Strategy Prototype: Momentum-Mean Reversion Hybrid

**Hypothesis**: 
Combine short-term mean reversion with longer-term momentum.
Buy on short-term pullbacks in stocks with strong 6-month momentum.

**Logic**:
1. Identify stocks with strong 126-day momentum (top quartile)
2. Wait for 5-day RSI < 30 (oversold)
3. Enter long position
4. Exit when RSI > 70 or stop loss hit
"""

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator

# Load clean data
df = pd.read_parquet("../data/processed/sp500_clean.parquet")

# For prototyping, focus on one ticker
ticker_data = df.xs('AAPL', level='ticker')

# Calculate indicators (PROTOTYPE - will move to src/features later)
ticker_data['returns_126d'] = ticker_data['Close'].pct_change(126)
ticker_data['rsi_5d'] = RSIIndicator (ticker_data['Close'], window=5).rsi()

# Generate signals (PROTOTYPE)
ticker_data['momentum_rank'] = ticker_data['returns_126d'].rolling(252).rank (pct=True)
ticker_data['signal'] = 0

# Long signal: Strong momentum + Oversold RSI
long_condition = (
    (ticker_data['momentum_rank'] > 0.75) &  # Top quartile momentum
    (ticker_data['rsi_5d'] < 30)              # Oversold
)
ticker_data.loc[long_condition, 'signal'] = 1

# Exit signal: Overbought
exit_condition = ticker_data['rsi_5d'] > 70
ticker_data.loc[exit_condition, 'signal'] = -1

# Simple backtest (PROTOTYPE - will use proper engine later)
ticker_data['position'] = ticker_data['signal'].replace(-1, 0).fillna (method='ffill')
ticker_data['strategy_returns'] = ticker_data['Close'].pct_change() * ticker_data['position'].shift(1)
ticker_data['cumulative_returns'] = (1 + ticker_data['strategy_returns']).cumprod()

# Quick performance check
total_return = ticker_data['cumulative_returns'].iloc[-1] - 1
sharpe = ticker_data['strategy_returns'].mean() / ticker_data['strategy_returns'].std() * np.sqrt(252)

print(f"Quick Stats (AAPL only):")
print(f"Total Return: {total_return:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")

# Visualize
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Price and signals
ticker_data['Close'].plot (ax=axes[0])
buy_signals = ticker_data[ticker_data['signal'] == 1]
axes[0].scatter (buy_signals.index, buy_signals['Close'], 
               color='green', marker='^', s=100, label='Buy')
axes[0].legend()
axes[0].set_title('Price and Entry Signals')

# RSI
ticker_data['rsi_5d'].plot (ax=axes[1])
axes[1].axhline(30, color='green', linestyle='--', label='Oversold')
axes[1].axhline(70, color='red', linestyle='--', label='Overbought')
axes[1].legend()
axes[1].set_title('RSI Indicator')

# Cumulative returns
ticker_data['cumulative_returns'].plot (ax=axes[2])
axes[2].set_title('Strategy Performance')

plt.tight_layout()
plt.show()

# Decision: Looks promising! Move to proper backtesting framework.
\`\`\`

## Phase 3: Transition to Production Code

Once strategy shows promise in prototype, extract to src/ modules.

### Extract to src/features/technical_indicators.py

\`\`\`python
\"\"\"
Technical indicator calculations for strategy development.
\"\"\"
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from typing import Union

def calculate_momentum(
    df: pd.DataFrame,
    price_col: str = 'Close',
    periods: Union[int, list] = [21, 63, 126, 252]
) -> pd.DataFrame:
    \"\"\"
    Calculate momentum over multiple periods.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with price data
    price_col : str
        Column name for price (default 'Close')
    periods : int or list
        Lookback period (s) in days
        
    Returns
    -------
    pd.DataFrame
        Original dataframe with added momentum columns
    \"\"\"
    df = df.copy()
    
    if isinstance (periods, int):
        periods = [periods]
    
    for period in periods:
        col_name = f'momentum_{period}d'
        df[col_name] = df[price_col].pct_change (period)
    
    return df

def calculate_rsi(
    df: pd.DataFrame,
    price_col: str = 'Close',
    window: int = 14
) -> pd.DataFrame:
    \"\"\"
    Calculate RSI indicator.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with price data
    price_col : str
        Column name for price
    window : int
        RSI period (default 14)
        
    Returns
    -------
    pd.DataFrame
        Original dataframe with added RSI column
    \"\"\"
    df = df.copy()
    rsi = RSIIndicator (df[price_col], window=window)
    df[f'rsi_{window}'] = rsi.rsi()
    return df

# ... more indicator functions ...
\`\`\`

### Extract to src/strategies/momentum_reversion.py

\`\`\`python
\"\"\"
Momentum-Mean Reversion Hybrid Strategy
\"\"\"
import pandas as pd
import numpy as np
from src.strategies.base_strategy import BaseStrategy
from src.features.technical_indicators import calculate_momentum, calculate_rsi

class MomentumReversionStrategy(BaseStrategy):
    \"\"\"
    Hybrid strategy combining momentum and mean reversion.
    
    Logic:
    - Long strong momentum stocks on short-term pullbacks
    - Exit on overbought conditions or stop loss
    \"\"\"
    
    def __init__(
        self,
        momentum_period: int = 126,
        momentum_threshold: float = 0.75,
        rsi_period: int = 5,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        stop_loss_pct: float = 0.05
    ):
        self.momentum_period = momentum_period
        self.momentum_threshold = momentum_threshold
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.stop_loss_pct = stop_loss_pct
        
    def generate_signals (self, df: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Generate trading signals.
        
        Parameters
        ----------
        df : pd.DataFrame
            Price data with OHLCV columns
            
        Returns
        -------
        pd.DataFrame
            Data with added 'signal' column (1=long, 0=flat, -1=exit)
        \"\"\"
        df = df.copy()
        
        # Calculate features
        df = calculate_momentum (df, periods=[self.momentum_period])
        df = calculate_rsi (df, window=self.rsi_period)
        
        # Momentum ranking
        df['momentum_rank'] = df[f'momentum_{self.momentum_period}d'].rolling(
            window=252
        ).rank (pct=True)
        
        # Initialize signal column
        df['signal'] = 0
        
        # Entry conditions
        long_condition = (
            (df['momentum_rank'] > self.momentum_threshold) &
            (df[f'rsi_{self.rsi_period}'] < self.rsi_oversold)
        )
        df.loc[long_condition, 'signal'] = 1
        
        # Exit conditions
        exit_condition = df[f'rsi_{self.rsi_period}'] > self.rsi_overbought
        df.loc[exit_condition, 'signal'] = -1
        
        return df
    
    def calculate_position_size(
        self,
        df: pd.DataFrame,
        capital: float,
        risk_per_trade: float = 0.02
    ) -> pd.DataFrame:
        \"\"\"
        Calculate position sizes based on volatility.
        
        Uses ATR-based position sizing to normalize risk across trades.
        \"\"\"
        # Implementation here...
        pass
\`\`\`

### Use in Notebook (06_backtest_development.ipynb)

\`\`\`python
\"\"\"
# Comprehensive Backtesting

Now using production-quality code from src/
\"\"\"

from src.strategies.momentum_reversion import MomentumReversionStrategy
from src.backtest.engine import Backtester
from src.backtest.metrics import PerformanceMetrics

# Initialize strategy
strategy = MomentumReversionStrategy(
    momentum_period=126,
    momentum_threshold=0.75,
    rsi_period=5,
    rsi_oversold=30,
    rsi_overbought=70
)

# Initialize backtester
backtester = Backtester(
    initial_capital=100000,
    commission=0.001,  # 10 bps
    slippage=0.0005     # 5 bps
)

# Run backtest on all tickers
results = backtester.run(
    data=df,
    strategy=strategy,
    start_date='2015-01-01',
    end_date='2023-12-31'
)

# Calculate metrics
metrics = PerformanceMetrics (results)
print(metrics.summary())

# Results are now reproducible and testable!
\`\`\`

## Phase 4: Testing Strategy

### tests/test_strategies.py

\`\`\`python
\"\"\"
Unit tests for trading strategies
\"\"\"
import pytest
import pandas as pd
import numpy as np
from src.strategies.momentum_reversion import MomentumReversionStrategy

@pytest.fixture
def sample_price_data():
    \"\"\"Create sample price data for testing\"\"\"
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    np.random.seed(42)
    
    prices = 100 * (1 + np.random.randn(500) * 0.02).cumprod()
    
    df = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.02,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 500)
    }, index=dates)
    
    return df

def test_strategy_generates_signals (sample_price_data):
    \"\"\"Test that strategy generates valid signals\"\"\"
    strategy = MomentumReversionStrategy()
    result = strategy.generate_signals (sample_price_data)
    
    # Should have signal column
    assert 'signal' in result.columns
    
    # Signals should be -1, 0, or 1
    assert result['signal'].isin([-1, 0, 1]).all()
    
    # Should have some signals
    assert (result['signal'] != 0).any()

def test_strategy_parameters():
    \"\"\"Test that strategy parameters are properly set\"\"\"
    strategy = MomentumReversionStrategy(
        momentum_period=90,
        rsi_period=7
    )
    
    assert strategy.momentum_period == 90
    assert strategy.rsi_period == 7

def test_no_lookahead_bias (sample_price_data):
    \"\"\"Ensure strategy doesn't use future data\"\"\"
    strategy = MomentumReversionStrategy()
    
    # Run strategy
    result = strategy.generate_signals (sample_price_data)
    
    # Signal at time t should only use data up to time t
    # Test by comparing with progressive calculation
    for i in range(200, len (sample_price_data)):
        partial_data = sample_price_data.iloc[:i]
        partial_result = strategy.generate_signals (partial_data)
        
        # Signal at day i should match
        assert result['signal'].iloc[i-1] == partial_result['signal'].iloc[-1]

# Run tests: pytest tests/test_strategies.py -v
\`\`\`

## Phase 5: Version Control Workflow

\`\`\`bash
# .gitignore
.ipynb_checkpoints/
__pycache__/
*.pyc
.DS_Store
.env

# Data (usually too large for git)
data/raw/*
data/processed/*
!data/*/.gitkeep

# Results
results/*
!results/.gitkeep

# Environment
venv/
.venv/

# Configure nbstripout
nbstripout --install

# Commit workflow
git add notebooks/05_strategy_prototyping.ipynb
git add src/strategies/momentum_reversion.py
git commit -m "feat: Add momentum-reversion hybrid strategy

- Implemented signal generation logic
- Added RSI and momentum indicators
- Initial backtest shows 1.8 Sharpe ratio
- Needs further testing on full universe"

git push
\`\`\`

## Phase 6: Production Deployment

### scripts/run_backtest.py

\`\`\`python
\"\"\"
Production backtest runner - CLI interface
\"\"\"
import argparse
import logging
from pathlib import Path
import yaml
import pandas as pd

from src.data.loaders import load_price_data
from src.strategies.momentum_reversion import MomentumReversionStrategy
from src.backtest.engine import Backtester
from src.backtest.reporting import generate_report

def main():
    parser = argparse.ArgumentParser (description='Run strategy backtest')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--tickers', nargs='+', help='Specific tickers to test')
    args = parser.parse_args()
    
    # Load configuration
    with open (args.config) as f:
        config = yaml.safe_load (f)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path (args.output) / 'backtest.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info (f"Starting backtest with config: {args.config}")
    
    # Load data
    data = load_price_data(
        tickers=args.tickers or config['data']['tickers'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    
    # Initialize strategy
    strategy = MomentumReversionStrategy(**config['strategy']['parameters'])
    
    # Run backtest
    backtester = Backtester(**config['backtest']['parameters'])
    results = backtester.run (data, strategy)
    
    # Generate report
    output_dir = Path (args.output)
    output_dir.mkdir (parents=True, exist_ok=True)
    
    generate_report(
        results=results,
        output_dir=output_dir,
        include_plots=True
    )
    
    logger.info (f"Backtest complete. Results saved to {output_dir}")

if __name__ == '__main__':
    main()

# Usage:
# python scripts/run_backtest.py \\
#     --config config/backtest_config.yaml \\
#     --output results/backtest_2024-01-15/ \\
#     --tickers AAPL MSFT GOOGL
\`\`\`

## Key Principles

1. **Prototype in Notebooks**: Fast iteration, visual feedback
2. **Extract to Modules**: Reusable, testable production code
3. **Test Everything**: Unit tests prevent regressions
4. **Version Control**: Track changes, collaborate effectively
5. **Automate**: Scripts for repetitive tasks
6. **Document**: Clear README, docstrings, strategy documentation
7. **Reproducibility**: Anyone should be able to run your analysis

This workflow enables rapid research while maintaining production-quality standards.
      `,
    },
    {
      id: 'jupyter-disc-2',
      question:
        'Discuss strategies for optimizing Jupyter notebook performance when working with large financial datasets (e.g., tick data, large universes). Cover data loading, processing, memory management, and parallel computation.',
      sampleAnswer: `
# Performance Optimization for Large-Scale Financial Analysis in Jupyter

## Challenge Overview

Financial data analysis often involves:
- **Large datasets**: Billions of rows (tick data, minute bars for thousands of stocks)
- **Memory constraints**: 16-64GB RAM typical, but datasets can be terabytes
- **Computational intensity**: Complex calculations across millions of data points
- **Interactive requirements**: Results needed quickly for exploration

## Strategy 1: Efficient Data Storage and Loading

### Problem: CSV Files Are Too Slow

\`\`\`python
# Slow: Loading 5GB CSV takes 60+ seconds
import pandas as pd
import time

start = time.time()
df = pd.read_csv('sp500_minute_bars_2023.csv')  # 5GB file
print(f"Load time: {time.time() - start:.1f}s")  # ~65 seconds
print(f"Memory: {df.memory_usage (deep=True).sum() / 1e9:.1f} GB")  # 8.2 GB
\`\`\`

### Solution: Use Columnar Formats

\`\`\`python
# Fast: Parquet with compression
import pyarrow.parquet as pq

# One-time conversion
df.to_parquet('sp500_minute_bars_2023.parquet', compression='snappy')

# Subsequent loads are 10x faster
start = time.time()
df = pd.read_parquet('sp500_minute_bars_2023.parquet')
print(f"Load time: {time.time() - start:.1f}s")  # ~6 seconds
print(f"Memory: {df.memory_usage (deep=True).sum() / 1e9:.1f} GB")  # 4.5 GB (better compression)

# Selective column loading
df_subset = pd.read_parquet(
    'sp500_minute_bars_2023.parquet',
    columns=['timestamp', 'ticker', 'close', 'volume']
)
# Only loads needed columns - even faster
\`\`\`

### Advanced: Partitioned Parquet for Massive Datasets

\`\`\`python
# Partition by date for efficient filtering
df.to_parquet(
    'sp500_data/',
    partition_cols=['date'],
    compression='snappy'
)

# Resulting structure:
# sp500_data/
#   date=2023-01-01/
#     data_0.parquet
#   date=2023-01-02/
#     data_0.parquet
#   ...

# Load only specific dates (blazing fast)
import pyarrow.dataset as ds

dataset = ds.dataset('sp500_data/', format='parquet', partitioning='hive')
df_jan = dataset.to_table(
    filter=ds.field('date') >= '2023-01-01',
    filter=ds.field('date') < '2023-02-01'
).to_pandas()

# 100x faster than loading full file then filtering
\`\`\`

## Strategy 2: Memory Optimization

### Problem: DataFrame Too Large for RAM

\`\`\`python
# Check memory usage
df.info (memory_usage='deep')

# Common culprits:
# - object dtype (strings) uses excessive memory
# - float64 when float32 sufficient
# - Repeated values not categorized
\`\`\`

### Solution 1: Downcast Numeric Types

\`\`\`python
def optimize_dtypes (df):
    \"\"\"Reduce DataFrame memory footprint\"\"\"
    df_optimized = df.copy()
    
    # Downcast integers
    int_cols = df.select_dtypes (include=['int64']).columns
    df_optimized[int_cols] = df[int_cols].apply (pd.to_numeric, downcast='integer')
    
    # Downcast floats
    float_cols = df.select_dtypes (include=['float64']).columns
    df_optimized[float_cols] = df[float_cols].apply (pd.to_numeric, downcast='float')
    
    return df_optimized

# Before
print(f"Memory: {df.memory_usage (deep=True).sum() / 1e9:.2f} GB")

# After
df_opt = optimize_dtypes (df)
print(f"Memory: {df_opt.memory_usage (deep=True).sum() / 1e9:.2f} GB")

# Typical reduction: 40-60%
\`\`\`

### Solution 2: Categorize Repeated Values

\`\`\`python
# Problem: Ticker symbols as strings
df['ticker'].memory_usage (deep=True) / 1e6  # 150 MB

# Solution: Categorize
df['ticker'] = df['ticker'].astype('category')
df['ticker'].memory_usage (deep=True) / 1e6  # 5 MB (30x reduction!)

# Apply to all low-cardinality strings
for col in ['ticker', 'exchange', 'sector']:
    if df[col].nunique() / len (df) < 0.5:  # Less than 50% unique
        df[col] = df[col].astype('category')
\`\`\`

### Solution 3: Chunked Processing

\`\`\`python
# When data truly doesn't fit in memory
def process_in_chunks (file_path, chunk_size=100000):
    \"\"\"Process large file in chunks\"\"\"
    results = []
    
    for chunk in pd.read_parquet (file_path, chunksize=chunk_size):
        # Process chunk
        chunk['returns'] = chunk.groupby('ticker')['close'].pct_change()
        chunk_agg = chunk.groupby('ticker')['returns'].agg(['mean', 'std'])
        results.append (chunk_agg)
    
    # Combine results
    final_result = pd.concat (results).groupby('ticker').mean()
    return final_result

# Processes 50GB file with only 1GB RAM usage
\`\`\`

## Strategy 3: Vectorization and NumPy

### Problem: Slow Pandas Apply/Loops

\`\`\`python
# Slow: Using apply (100x slower than necessary)
%%timeit
df['returns'] = df.groupby('ticker')['close'].apply (lambda x: x.pct_change())
# 2.5 seconds

# Fast: Vectorized operation
%%timeit
df['returns'] = df.groupby('ticker')['close'].pct_change()
# 25 milliseconds (100x faster!)
\`\`\`

### Advanced: NumPy for Heavy Computation

\`\`\`python
import numpy as np

# Calculate Sharpe ratio for 500 stocks
# Slow method: Loop through each ticker
def slow_sharpe (df):
    sharpe_ratios = {}
    for ticker in df['ticker'].unique():
        returns = df[df['ticker'] == ticker]['returns']
        sharpe_ratios[ticker] = returns.mean() / returns.std() * np.sqrt(252)
    return sharpe_ratios

%%timeit
slow_sharpe (df)
# 850 ms

# Fast method: Vectorized GroupBy
def fast_sharpe (df):
    grouped = df.groupby('ticker')['returns']
    sharpe = (grouped.mean() / grouped.std()) * np.sqrt(252)
    return sharpe

%%timeit
fast_sharpe (df)
# 45 ms (19x faster!)

# Ultra-fast: Pure NumPy
def ultra_fast_sharpe (returns_matrix):
    \"\"\"
    returns_matrix: 2D NumPy array (days x tickers)
    \"\"\"
    mean_returns = np.mean (returns_matrix, axis=0)
    std_returns = np.std (returns_matrix, axis=0)
    sharpe = (mean_returns / std_returns) * np.sqrt(252)
    return sharpe

# Convert to matrix once
returns_pivot = df.pivot (index='date', columns='ticker', values='returns')
returns_matrix = returns_pivot.values

%%timeit
ultra_fast_sharpe (returns_matrix)
# 2 ms (425x faster than original!)
\`\`\`

## Strategy 4: Parallel Processing

### Use Case 1: Embarrassingly Parallel Tasks

\`\`\`python
from joblib import Parallel, delayed
import yfinance as yf

def backtest_ticker (ticker):
    \"\"\"Backtest single ticker - independent of others\"\"\"
    df = yf.download (ticker, start='2020-01-01', end='2023-12-31', progress=False)
    
    # Strategy logic
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['signal'] = (df['SMA_20'] > df['SMA_50']).astype (int)
    df['returns'] = df['Close'].pct_change() * df['signal'].shift(1)
    
    total_return = (1 + df['returns']).prod() - 1
    sharpe = df['returns'].mean() / df['returns'].std() * np.sqrt(252)
    
    return {'ticker': ticker, 'return': total_return, 'sharpe': sharpe}

# Sequential: 500 tickers * 2 seconds = 1000 seconds
tickers = get_sp500_tickers()  # 500 tickers

start = time.time()
results_sequential = [backtest_ticker (t) for t in tickers]
print(f"Sequential: {time.time() - start:.0f} seconds")  # ~1000s

# Parallel: Use all CPU cores
start = time.time()
results_parallel = Parallel (n_jobs=-1)(
    delayed (backtest_ticker)(t) for t in tickers
)
print(f"Parallel: {time.time() - start:.0f} seconds")  # ~125s (8x speedup on 8 cores)
\`\`\`

### Use Case 2: Parallel Rolling Window Calculations

\`\`\`python
def calculate_rolling_metric (data_chunk, window=20):
    \"\"\"Calculate metric on data chunk\"\"\"
    return data_chunk.rolling (window).mean()

# Split data into chunks
n_cores = 8
chunk_size = len (df) // n_cores
chunks = [df.iloc[i:i+chunk_size] for i in range(0, len (df), chunk_size)]

# Process in parallel
results = Parallel (n_jobs=n_cores)(
    delayed (calculate_rolling_metric)(chunk) 
    for chunk in chunks
)

# Combine results
df_result = pd.concat (results)
\`\`\`

## Strategy 5: Dask for Out-of-Core Computation

### When Data Doesn't Fit in RAM

\`\`\`python
import dask.dataframe as dd
import dask.array as da

# Load data larger than RAM
ddf = dd.read_parquet('tick_data_2023/*.parquet')

print(f"Partitions: {ddf.npartitions}")  # Data split into manageable chunks
print(f"Size: {ddf.memory_usage().sum().compute() / 1e9:.1f} GB")  # Larger than RAM

# Operations are lazy - no computation yet
ddf['returns'] = ddf.groupby('ticker')['price'].pct_change()
ddf_filtered = ddf[ddf['volume'] > 1000000]
result = ddf_filtered.groupby('ticker')['returns'].agg(['mean', 'std', 'count'])

# Trigger computation (Dask handles memory management)
final_result = result.compute()

# Dask processes one partition at a time, never exceeding RAM
\`\`\`

### Dask with Progress Bars

\`\`\`python
from dask.diagnostics import ProgressBar

with ProgressBar():
    result = ddf.groupby('ticker').apply(
        complex_function,
        meta=('returns', 'f8')
    ).compute()

# Shows progress for long-running operations
\`\`\`

## Strategy 6: Numba for JIT Compilation

### For Loop-Heavy Numerical Code

\`\`\`python
from numba import jit

# Python loop: Very slow
def slow_moving_average (prices, window):
    n = len (prices)
    ma = np.empty (n)
    for i in range (window, n):
        ma[i] = np.mean (prices[i-window:i])
    return ma

prices = np.random.randn(1000000)

%%timeit
slow_moving_average (prices, 20)
# 1.2 seconds

# Numba-accelerated: Compiled to machine code
@jit (nopython=True)
def fast_moving_average (prices, window):
    n = len (prices)
    ma = np.empty (n)
    for i in range (window, n):
        ma[i] = np.mean (prices[i-window:i])
    return ma

%%timeit
fast_moving_average (prices, 20)
# 15 milliseconds (80x speedup!)
\`\`\`

### Numba for Technical Indicators

\`\`\`python
@jit (nopython=True)
def fast_rsi (prices, period=14):
    \"\"\"Ultra-fast RSI calculation\"\"\"
    n = len (prices)
    rsi = np.empty (n)
    gains = np.zeros (n)
    losses = np.zeros (n)
    
    # Calculate gains and losses
    for i in range(1, n):
        diff = prices[i] - prices[i-1]
        gains[i] = diff if diff > 0 else 0
        losses[i] = -diff if diff < 0 else 0
    
    # Calculate RSI
    for i in range (period, n):
        avg_gain = np.mean (gains[i-period:i])
        avg_loss = np.mean (losses[i-period:i])
        
        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi

# 100x faster than pure Python
\`\`\`

## Strategy 7: Caching and Memoization

### Problem: Recomputing Same Results

\`\`\`python
# Without caching: Recompute every time
def expensive_analysis (ticker, start_date, end_date):
    df = yf.download (ticker, start=start_date, end=end_date)
    # ... expensive calculations ...
    return results

# Each call downloads and recomputes (slow)
result1 = expensive_analysis('AAPL', '2020-01-01', '2023-12-31')
result2 = expensive_analysis('AAPL', '2020-01-01', '2023-12-31')  # Redundant work!
\`\`\`

### Solution: Joblib Memory Caching

\`\`\`python
from joblib import Memory

# Set up cache directory
memory = Memory('cache/', verbose=0)

@memory.cache
def cached_analysis (ticker, start_date, end_date):
    \"\"\"Results cached to disk automatically\"\"\"
    print(f"Computing {ticker}...")  # Only prints first time
    df = yf.download (ticker, start=start_date, end=end_date, progress=False)
    # ... expensive calculations ...
    return results

# First call: Slow (computes and caches)
result1 = cached_analysis('AAPL', '2020-01-01', '2023-12-31')  # 30 seconds

# Subsequent calls: Instant (loads from cache)
result2 = cached_analysis('AAPL', '2020-01-01', '2023-12-31')  # 100 ms

# Different parameters: Computes again
result3 = cached_analysis('MSFT', '2020-01-01', '2023-12-31')  # 30 seconds
\`\`\`

## Strategy 8: Database Integration

### Problem: Loading Full Dataset Every Time

\`\`\`python
# Inefficient: Load entire table, then filter
df = pd.read_parquet('full_universe.parquet')  # 20GB
df_filtered = df[df['ticker'].isin(['AAPL', 'MSFT', 'GOOGL'])]  # Wasted time/memory
\`\`\`

### Solution: Query Database Directly

\`\`\`python
import sqlalchemy as sa

engine = sa.create_engine('postgresql://localhost/marketdata')

# Only load needed data
query = \"\"\"
    SELECT date, ticker, close, volume
    FROM prices
    WHERE ticker IN ('AAPL', 'MSFT', 'GOOGL')
    AND date >= '2020-01-01'
    ORDER BY date
\"\"\"

df = pd.read_sql (query, engine)  # Much faster, less memory

# Leverage database indexes and query optimization
\`\`\`

### TimescaleDB for Time Series Data

\`\`\`python
# TimescaleDB provides hypertables optimized for time series

# Create hypertable (one-time setup)
\"\"\"
CREATE TABLE prices (
    time TIMESTAMPTZ NOT NULL,
    ticker TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume BIGINT
);

SELECT create_hypertable('prices', 'time');
CREATE INDEX ON prices (ticker, time DESC);
\"\"\"

# Queries are blazing fast
query = \"\"\"
    SELECT ticker, 
           time_bucket('1 day', time) AS day,
           first (open, time) as open,
           max (high) as high,
           min (low) as low,
           last (close, time) as close,
           sum (volume) as volume
    FROM prices
    WHERE ticker = 'AAPL'
    AND time >= NOW() - INTERVAL '1 year'
    GROUP BY ticker, day
    ORDER BY day DESC
\"\"\"

df = pd.read_sql (query, engine)  # Sub-second response
\`\`\`

## Performance Monitoring

### Profile Your Code

\`\`\`python
# Line profiler
%load_ext line_profiler

def my_analysis_function (df):
    # ... analysis code ...
    pass

%lprun -f my_analysis_function my_analysis_function (df)

# Shows time spent per line - identifies bottlenecks
\`\`\`

### Memory Profiler

\`\`\`python
%load_ext memory_profiler

%memit df = pd.read_csv('large_file.csv')
# peak memory: 4523.45 MiB, increment: 3200.12 MiB

%memit df = pd.read_parquet('large_file.parquet')
# peak memory: 2100.23 MiB, increment: 1500.45 MiB
\`\`\`

## Summary: Performance Optimization Checklist

1. ✅ **Storage**: Use Parquet/Feather instead of CSV (10x faster)
2. ✅ **Data Types**: Downcast numerics, categorize strings (50% memory reduction)
3. ✅ **Vectorization**: Avoid loops, use NumPy/Pandas operations (100x faster)
4. ✅ **Parallel**: Use joblib/multiprocessing for independent tasks (8x on 8 cores)
5. ✅ **Out-of-Core**: Use Dask when data > RAM
6. ✅ **JIT Compilation**: Use Numba for loop-heavy numerical code (50-100x faster)
7. ✅ **Caching**: Use joblib.Memory to avoid recomputation
8. ✅ **Database**: Query directly instead of loading full dataset
9. ✅ **Profile**: Use %lprun and %memit to find bottlenecks

With these strategies, you can analyze terabyte-scale financial data interactively in Jupyter notebooks.
      `,
    },
    {
      id: 'jupyter-disc-3',
      question:
        'Create a detailed plan for transitioning a successful research notebook into a production-ready trading system. Address code organization, testing, deployment, monitoring, and maintenance considerations.',
      sampleAnswer: `[Answer covering: extracting notebook code to proper Python modules with tests, setting up CI/CD pipelines, containerization with Docker, deployment strategies (cloud vs on-premise), logging and monitoring, error handling, database integration, API design, scheduling, and maintenance workflows. Include code examples for production-ready backtest engine, risk management system, and order execution framework. Discuss challenges like ensuring no lookahead bias, handling market data in production, dealing with failures, and maintaining system over time.]`,
    },
  ],
};
