import { Content } from '@/lib/types';

export const jupyterLabQuantContent: Content = {
    title: "Jupyter Lab for Quantitative Research",
    subtitle: "Professional computational research environment for finance",
    description: "Master Jupyter Lab and Jupyter Notebook for quantitative financial research, from basic data analysis to sophisticated backtesting and strategy development. Learn the tools used by quantitative researchers at leading hedge funds and banks.",
    sections: [
        {
            title: "Jupyter Ecosystem Overview",
            content: `
# Understanding the Jupyter Ecosystem

## Jupyter Notebook vs Jupyter Lab

### Jupyter Notebook
The original web-based interactive development environment.

\`\`\`plaintext
Jupyter Notebook Features:
├── Single Document Interface
│   └── One notebook at a time
├── Linear Workflow
│   └── Top-to-bottom execution
├── Simpler Interface
│   └── Easier for beginners
├── Extensions
│   └── Limited customization
└── File Management
    └── Basic file browser
\`\`\`

### Jupyter Lab
The next-generation interface with IDE-like features.

\`\`\`plaintext
Jupyter Lab Features:
├── Multi-Document Interface
│   ├── Multiple notebooks side-by-side
│   ├── Split views
│   └── Tab management
├── Integrated Tools
│   ├── Text editor
│   ├── Terminal
│   ├── File browser
│   └── Output viewer
├── Extensions System
│   ├── Git integration
│   ├── Table of contents
│   ├── Variable inspector
│   └── Debugger
├── Flexible Layout
│   ├── Drag and drop panels
│   ├── Customizable workspace
│   └── Multiple views of same notebook
└── Advanced Features
    ├── Real-time collaboration
    ├── Code debugging
    └── Interactive widgets
\`\`\`

## Installation and Setup

### Basic Installation
\`\`\`bash
# Install Jupyter Lab
pip install jupyterlab

# Install with conda (recommended for financial work)
conda install -c conda-forge jupyterlab

# Launch Jupyter Lab
jupyter lab

# Launch on specific port
jupyter lab --port=8889

# Launch without opening browser
jupyter lab --no-browser
\`\`\`

### Professional Setup for Quantitative Finance
\`\`\`bash
# Create dedicated environment
conda create -n quant_research python=3.10
conda activate quant_research

# Install core scientific stack
conda install -c conda-forge jupyterlab numpy pandas scipy matplotlib seaborn

# Install financial libraries
pip install yfinance pandas-datareader ta-lib quantlib zipline-reloaded

# Install machine learning
pip install scikit-learn tensorflow torch

# Install database connectors
pip install sqlalchemy psycopg2-binary pymongo

# Install additional tools
pip install jupyter-lsp-python  # Language server protocol
pip install jupyterlab-git  # Git integration
pip install jupyterlab_execute_time  # Execution time display
pip install aquirdturtle_collapsible_headings  # Collapsible sections

# Start Jupyter Lab
jupyter lab
\`\`\`

## Jupyter Lab Interface Components

### 1. Launcher
Create new notebooks, consoles, terminals, and files.

### 2. File Browser (Left Sidebar)
Navigate your project structure:
\`\`\`plaintext
quant_research/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_backtesting.ipynb
│   └── 04_live_trading.ipynb
├── src/
│   ├── data_loader.py
│   ├── indicators.py
│   ├── strategies.py
│   └── backtest.py
├── tests/
│   └── test_strategies.py
├── results/
│   ├── backtest_results.csv
│   └── performance_charts/
├── requirements.txt
└── README.md
\`\`\`

### 3. Running Kernels and Terminals
Monitor and manage active notebooks and terminals.

### 4. Commands Palette (Cmd+Shift+C / Ctrl+Shift+C)
Quick access to all Jupyter Lab commands.

### 5. Property Inspector
View notebook metadata and cell properties.
      `
        },
        {
            title: "Essential Jupyter Lab Features",
            content: `
# Power User Features for Quantitative Research

## Cell Types and Execution

### Code Cells
Execute Python code and display results.

\`\`\`python
# Standard execution
import pandas as pd
import numpy as np
import yfinance as yf

# Download data
ticker = "AAPL"
df = yf.download(ticker, start="2020-01-01", end="2024-01-01")
print(f"Downloaded {len(df)} rows for {ticker}")
\`\`\`

**Keyboard Shortcuts:**
- `Shift + Enter`: Execute cell and move to next
- `Ctrl + Enter`: Execute cell and stay
- `Alt + Enter`: Execute cell and insert new cell below
- `DD`: Delete selected cell (in command mode)
- `A`: Insert cell above
- `B`: Insert cell below

### Markdown Cells
Document your research with formatted text.

\`\`\`markdown
# Strategy Analysis

## Hypothesis
The momentum strategy outperforms buy-and-hold during trending markets.

## Methodology
1. Calculate 20-day and 50-day moving averages
2. Generate signals when fast MA crosses slow MA
3. Backtest from 2015-2023

### Key Findings
- Sharpe Ratio: **1.85**
- Max Drawdown: **-12.3%**
- Win Rate: **58.7%**

The strategy shows promise but requires further optimization for risk management.
\`\`\`

### Raw Cells
Pass-through content (useful for LaTeX or nbconvert).

## Magic Commands

### Line Magics (%)
Single-line commands.

\`\`\`python
# Time single line execution
%timeit [x**2 for x in range(1000)]

# Run external Python file
%run ./src/data_loader.py

# Load external code
%load ./src/strategies.py

# Display matplotlib plots inline
%matplotlib inline

# Interactive matplotlib plots
%matplotlib widget

# Enable autoreload for external modules
%load_ext autoreload
%autoreload 2

# Display all variables
%who

# Detailed variable information
%whos

# Show execution history
%history

# Change working directory
%cd ../data/

# List files
%ls

# Get system information
%env
\`\`\`

### Cell Magics (%%)
Multi-line commands.

\`\`\`python
%%time
# Time entire cell execution
result = []
for i in range(1000000):
    result.append(i**2)

%%timeit
# Benchmark cell with multiple runs
result = [x**2 for x in range(1000)]

%%writefile strategy.py
# Write cell contents to file
def moving_average_crossover(df, fast=20, slow=50):
    df['fast_ma'] = df['Close'].rolling(fast).mean()
    df['slow_ma'] = df['Close'].rolling(slow).mean()
    df['signal'] = 0
    df.loc[df['fast_ma'] > df['slow_ma'], 'signal'] = 1
    df.loc[df['fast_ma'] < df['slow_ma'], 'signal'] = -1
    return df

%%bash
# Execute bash commands
ls -la
pwd
echo "Current directory contents"

%%html
# Render HTML
<div style="background: #f0f0f0; padding: 20px;">
    <h2>Custom HTML Content</h2>
    <p>Useful for reports and dashboards</p>
</div>

%%javascript
# Execute JavaScript
element.text("JavaScript output in notebook");

%%latex
# Render LaTeX
\begin{equation}
R_p = \sum_{i=1}^{n} w_i R_i
\end{equation}
\`\`\`

## Interactive Output and Display

### IPython Display System

\`\`\`python
from IPython.display import display, HTML, Image, Video, Markdown, Math, Code

# Display HTML
display(HTML('<h2 style="color: blue;">Portfolio Performance</h2>'))

# Display markdown
display(Markdown("""
## Results
- **Return**: 15.7%
- **Volatility**: 12.3%
- **Sharpe**: 1.28
"""))

# Display mathematical notation
display(Math(r'SR = \\frac{E[R_p - R_f]}{\\sigma_p}'))

# Display images
display(Image(filename='./charts/returns_distribution.png'))

# Display multiple objects
display(df.head(), df.describe(), df.corr())
\`\`\`

### Rich Output

\`\`\`python
import matplotlib.pyplot as plt
import seaborn as sns

# Multiple plots in one cell
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Price chart
df['Close'].plot(ax=axes[0, 0], title='Price History')

# Returns distribution
df['Returns'].hist(ax=axes[0, 1], bins=50, title='Returns Distribution')

# Cumulative returns
(1 + df['Returns']).cumprod().plot(ax=axes[1, 0], title='Cumulative Returns')

# Volatility over time
df['Returns'].rolling(30).std().plot(ax=axes[1, 1], title='Rolling Volatility')

plt.tight_layout()
plt.show()
\`\`\`

## Extensions for Quantitative Research

### 1. Variable Inspector
See all variables in memory.

\`\`\`bash
# Install
pip install lckr-jupyterlab-variableinspector

# Restart Jupyter Lab
# Enable: View -> Activate Command Palette -> Show Variable Inspector
\`\`\`

### 2. Table of Contents
Navigate large notebooks easily.

\`\`\`bash
# Install
pip install jupyterlab-toc

# Enable: View -> Show Table of Contents
\`\`\`

### 3. Execution Time
Display cell execution time automatically.

\`\`\`bash
# Install
pip install jupyterlab_execute_time

# Restart Jupyter Lab
# Execution time appears automatically for each cell
\`\`\`

### 4. Git Integration
Version control within Jupyter Lab.

\`\`\`bash
# Install
pip install jupyterlab-git

# Restart Jupyter Lab
# Git panel appears in left sidebar
\`\`\`

### 5. Code Formatter (Black/Autopep8)
Format code cells automatically.

\`\`\`bash
# Install
pip install jupyterlab_code_formatter black isort

# Restart Jupyter Lab
# Right-click on cell -> Format Cell
\`\`\`

### 6. Debugger
Interactive debugging like VS Code.

\`\`\`bash
# Built into Jupyter Lab 3.0+
# Enable: View -> Activate Debugger
# Set breakpoints by clicking line numbers
\`\`\`
      `
    },
    {
        title: "Organizing Quantitative Research",
        content: `
# Professional Notebook Organization

## Project Structure for Quant Research

### Directory Layout
\`\`\`plaintext
momentum_strategy/
├── data/
│   ├── raw/                    # Original, immutable data
│   │   ├── sp500_prices.csv
│   │   └── market_data.h5
│   ├── processed/              # Cleaned, transformed data
│   │   ├── features.parquet
│   │   └── train_test_split.pkl
│   └── external/               # Third-party data
│       └── fred_economic_data.csv
├── notebooks/
│   ├── 00_data_collection.ipynb      # Data acquisition
│   ├── 01_exploratory_analysis.ipynb  # EDA
│   ├── 02_feature_engineering.ipynb   # Feature creation
│   ├── 03_strategy_development.ipynb  # Strategy logic
│   ├── 04_backtesting.ipynb          # Historical testing
│   ├── 05_optimization.ipynb         # Parameter tuning
│   ├── 06_walk_forward.ipynb         # Out-of-sample validation
│   └── 99_final_report.ipynb         # Presentation notebook
├── src/                        # Production-quality code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # Data loading utilities
│   │   └── preprocessor.py    # Data cleaning
│   ├── features/
│   │   ├── __init__.py
│   │   ├── technical.py       # Technical indicators
│   │   └── fundamental.py     # Fundamental features
│   ├── models/
│   │   ├── __init__.py
│   │   ├── strategies.py      # Trading strategies
│   │   └── ml_models.py       # ML models
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py          # Backtesting engine
│   │   └── metrics.py         # Performance metrics
│   └── visualization/
│       ├── __init__.py
│       └── charts.py          # Plotting functions
├── tests/
│   ├── test_data_loader.py
│   ├── test_strategies.py
│   └── test_backtest_engine.py
├── results/
│   ├── backtest_results/
│   │   ├── 2024-01-15_momentum_v1.csv
│   │   └── 2024-01-16_momentum_v2.csv
│   ├── charts/
│   │   └── equity_curve.png
│   └── reports/
│       └── monthly_performance.pdf
├── config/
│   ├── config.yaml
│   └── trading_parameters.json
├── .gitignore
├── requirements.txt
├── README.md
└── setup.py
\`\`\`

## Notebook Naming Conventions

### Sequential Numbering
\`\`\`plaintext
00-09: Data acquisition and exploration
10-19: Data cleaning and preprocessing
20-29: Feature engineering
30-39: Model development
40-49: Model evaluation
50-59: Optimization
60-69: Production preparation
70-79: Documentation
80-89: Archive (deprecated notebooks)
90-99: Reports and presentations
\`\`\`

### Descriptive Names
\`\`\`plaintext
Good Examples:
├── 01_load_sp500_constituents.ipynb
├── 02_eda_price_distributions.ipynb
├── 03_feature_momentum_indicators.ipynb
├── 04_strategy_moving_average_crossover.ipynb
└── 05_backtest_initial_results.ipynb

Bad Examples:
├── notebook1.ipynb
├── test.ipynb
├── untitled.ipynb
└── Copy_of_notebook.ipynb
\`\`\`

## Notebook Structure Template

### Research Notebook Template

\`\`\`python
# Cell 1: Title and Description
"""
# Moving Average Crossover Strategy - Initial Development

**Author**: Your Name
**Date**: 2024-01-15
**Version**: 1.0

## Objective
Develop and test a moving average crossover strategy on S&P 500 stocks.

## Hypothesis
Fast MA crossing above slow MA predicts short-term price increases.

## Data
- Source: Yahoo Finance
- Universe: S&P 500
- Period: 2015-01-01 to 2023-12-31
- Frequency: Daily

## Methodology
1. Calculate 20-day and 50-day moving averages
2. Generate buy signal when fast > slow
3. Generate sell signal when fast < slow
4. Backtest with transaction costs
5. Evaluate performance metrics

## Results Summary
(To be filled after analysis)
"""

# Cell 2: Imports and Configuration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from datetime import datetime, timedelta
from pathlib import Path

# Local modules
import sys
sys.path.append('../src')
from data.loader import load_price_data
from features.technical import calculate_moving_averages
from backtest.engine import Backtester
from backtest.metrics import calculate_performance_metrics

# Configuration
%matplotlib inline
%load_ext autoreload
%autoreload 2

sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (14, 7)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Random seed for reproducibility
np.random.seed(42)

# Cell 3: Configuration Parameters
# Parameters
TICKER = "SPY"
START_DATE = "2015-01-01"
END_DATE = "2023-12-31"
FAST_PERIOD = 20
SLOW_PERIOD = 50
INITIAL_CAPITAL = 100000
COMMISSION = 0.001  # 10 basis points

# Paths
DATA_DIR = Path("../data")
RESULTS_DIR = Path("../results")

# Cell 4: Data Loading
# Load data
df = yf.download(TICKER, start=START_DATE, end=END_DATE)

# Basic data validation
print(f"Data shape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"Missing values: {df.isnull().sum().sum()}")

df.head()

# Cell 5: Exploratory Data Analysis
# Price chart
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

df['Close'].plot(ax=axes[0], title=f'{TICKER} Price History')
axes[0].set_ylabel('Price ($)')

df['Volume'].plot(ax=axes[1], title='Volume')
axes[1].set_ylabel('Volume')

plt.tight_layout()
plt.show()

# Cell 6: Feature Engineering
# Calculate moving averages
df['MA_fast'] = df['Close'].rolling(window=FAST_PERIOD).mean()
df['MA_slow'] = df['Close'].rolling(window=SLOW_PERIOD).mean()

# Generate signals
df['signal'] = 0
df.loc[df['MA_fast'] > df['MA_slow'], 'signal'] = 1
df.loc[df['MA_fast'] < df['MA_slow'], 'signal'] = -1

# Position changes (when signal changes)
df['position_change'] = df['signal'].diff()

# Visualization
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Close'], label='Price', alpha=0.5)
plt.plot(df.index, df['MA_fast'], label=f'{FAST_PERIOD}-day MA')
plt.plot(df.index, df['MA_slow'], label=f'{SLOW_PERIOD}-day MA')

# Mark buy signals
buy_signals = df[df['position_change'] == 2]
plt.scatter(buy_signals.index, buy_signals['Close'], 
           color='green', marker='^', s=100, label='Buy', zorder=5)

# Mark sell signals
sell_signals = df[df['position_change'] == -2]
plt.scatter(sell_signals.index, sell_signals['Close'],
           color='red', marker='v', s=100, label='Sell', zorder=5)

plt.legend()
plt.title(f'{TICKER} - Moving Average Crossover Signals')
plt.show()

# Cell 7: Backtesting
# Calculate returns
df['returns'] = df['Close'].pct_change()
df['strategy_returns'] = df['returns'] * df['signal'].shift(1)

# Apply transaction costs
df['trades'] = df['position_change'].abs()
df['transaction_costs'] = df['trades'] * COMMISSION
df['strategy_returns_net'] = df['strategy_returns'] - df['transaction_costs']

# Cumulative returns
df['cumulative_market'] = (1 + df['returns']).cumprod()
df['cumulative_strategy'] = (1 + df['strategy_returns_net']).cumprod()

# Plot performance
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['cumulative_market'], label='Buy & Hold', linewidth=2)
plt.plot(df.index, df['cumulative_strategy'], label='Strategy', linewidth=2)
plt.legend()
plt.title('Cumulative Returns: Strategy vs Buy & Hold')
plt.ylabel('Cumulative Return')
plt.grid(True, alpha=0.3)
plt.show()

# Cell 8: Performance Metrics
def calculate_metrics(returns, periods_per_year=252):
    """Calculate strategy performance metrics"""
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
    annual_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'Total Return': f'{total_return:.2%}',
        'Annual Return': f'{annual_return:.2%}',
        'Annual Volatility': f'{annual_vol:.2%}',
        'Sharpe Ratio': f'{sharpe:.2f}',
        'Max Drawdown': f'{max_drawdown:.2%}',
        'Number of Trades': int((returns != 0).sum())
    }

# Calculate metrics
strategy_metrics = calculate_metrics(df['strategy_returns_net'].dropna())
market_metrics = calculate_metrics(df['returns'].dropna())

# Display results
results_df = pd.DataFrame([strategy_metrics, market_metrics],
                          index=['Strategy', 'Buy & Hold'])
display(results_df)

# Cell 9: Conclusions and Next Steps
"""
## Key Findings

1. **Performance**: Strategy returned X% vs Y% for buy-and-hold
2. **Risk**: Sharpe ratio of X.XX indicates [strong/weak] risk-adjusted returns
3. **Drawdown**: Maximum drawdown of XX% [better/worse] than market's YY%
4. **Trade Frequency**: N trades over the period (avg X per year)

## Observations

- Strategy performs well during trending markets
- Whipsaws occur during consolidation periods
- Transaction costs significantly impact returns

## Next Steps

1. Test on expanded universe (all S&P 500 stocks)
2. Optimize MA parameters via grid search
3. Add volatility filter to reduce whipsaws
4. Implement position sizing based on volatility
5. Conduct walk-forward analysis

## Questions for Further Research

1. Does performance persist out-of-sample?
2. How sensitive is performance to parameter changes?
3. Can we identify market regimes where strategy works best?
"""

# Cell 10: Save Results
# Save processed data
df.to_parquet(RESULTS_DIR / f'backtest_{TICKER}_{datetime.now().strftime("%Y%m%d")}.parquet')

# Save metrics
results_df.to_csv(RESULTS_DIR / f'metrics_{TICKER}_{datetime.now().strftime("%Y%m%d")}.csv')

print("Results saved successfully!")
\`\`\`

## Cell Organization Best Practices

### 1. One Logical Operation Per Cell
\`\`\`python
# Good: Single focused operation
df = load_data('SPY')

# Good: Next logical step
df_clean = clean_data(df)

# Bad: Multiple unrelated operations in one cell
df = load_data('SPY')
model = train_model(df)
results = backtest(model)
charts = create_visualizations(results)
\`\`\`

### 2. Restart and Run All
Your notebook should run from top to bottom without errors.

\`\`\`python
# Use Kernel -> Restart Kernel and Run All Cells regularly
# Ensures reproducibility
\`\`\`

### 3. Clear Outputs Before Committing
\`\`\`bash
# Install nbstripout to auto-clear outputs
pip install nbstripout

# Set up git filter
nbstripout --install

# Now outputs won't be committed to git
\`\`\`
      `
    },
{
    title: "Advanced Workflow Techniques",
        content: `
# Professional Workflows for Quantitative Research

## Converting Notebooks to Production Code

### Extract Functions to .py Modules

**From Notebook:**
\`\`\`python
# Cell in notebook
df['MA_20'] = df['Close'].rolling(20).mean()
df['MA_50'] = df['Close'].rolling(50).mean()
df['signal'] = np.where(df['MA_20'] > df['MA_50'], 1, -1)
\`\`\`

**To Module (src/features/technical.py):**
\`\`\`python
def moving_average_crossover(df, fast_period=20, slow_period=50):
    """
    Generate trading signals based on moving average crossover.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Close' column
    fast_period : int
        Fast moving average period
    slow_period : int
        Slow moving average period
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with added MA columns and signal column
    """
    df = df.copy()
    df[f'MA_{fast_period}'] = df['Close'].rolling(fast_period).mean()
    df[f'MA_{slow_period}'] = df['Close'].rolling(slow_period).mean()
    df['signal'] = np.where(df[f'MA_{fast_period}'] > df[f'MA_{slow_period}'], 1, -1)
    return df
\`\`\`

**Back in Notebook:**
\`\`\`python
from src.features.technical import moving_average_crossover

df = moving_average_crossover(df, fast_period=20, slow_period=50)
\`\`\`

### Workflow: Notebook → Module → Test

\`\`\`plaintext
1. Prototype in notebook
   └── Explore, experiment, iterate quickly

2. Extract working code to module
   └── Add docstrings, type hints, error handling

3. Write tests
   └── Ensure code works correctly

4. Use module in notebook
   └── Clean, reusable, tested code

5. Repeat
   └── Continue prototyping with confidence
\`\`\`

## Parameterized Notebooks with Papermill

Execute notebooks programmatically with different parameters.

### Setup
\`\`\`bash
pip install papermill
\`\`\`

### Create Parameterized Notebook

\`\`\`python
# Cell tagged as "parameters" (View -> Property Inspector -> Add Tag: "parameters")
TICKER = "AAPL"
START_DATE = "2020-01-01"
END_DATE = "2023-12-31"
FAST_MA = 20
SLOW_MA = 50
\`\`\`

### Execute with Different Parameters

\`\`\`bash
# Run for different ticker
papermill template_strategy.ipynb output_spy.ipynb -p TICKER SPY

# Run for multiple tickers
papermill template_strategy.ipynb output_aapl.ipynb -p TICKER AAPL
papermill template_strategy.ipynb output_msft.ipynb -p TICKER MSFT
papermill template_strategy.ipynb output_googl.ipynb -p TICKER GOOGL
\`\`\`

### Automate with Python

\`\`\`python
import papermill as pm
from datetime import datetime

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

for ticker in tickers:
    output_path = f'results/backtest_{ticker}_{datetime.now().strftime("%Y%m%d")}.ipynb'
    
    pm.execute_notebook(
        'template_strategy.ipynb',
        output_path,
        parameters=dict(
            TICKER=ticker,
            START_DATE='2020-01-01',
            END_DATE='2023-12-31'
        )
    )
    print(f"Completed: {ticker}")
\`\`\`

## Interactive Dashboards with ipywidgets

Create interactive controls for parameter exploration.

\`\`\`python
import ipywidgets as widgets
from IPython.display import display

# Download data once
df = yf.download('AAPL', start='2020-01-01', end='2024-01-01')

def plot_ma_crossover(fast_period, slow_period):
    """Interactive MA crossover visualization"""
    df_copy = df.copy()
    df_copy['MA_fast'] = df_copy['Close'].rolling(fast_period).mean()
    df_copy['MA_slow'] = df_copy['Close'].rolling(slow_period).mean()
    
    plt.figure(figsize=(14, 7))
    plt.plot(df_copy.index, df_copy['Close'], label='Price', alpha=0.5)
    plt.plot(df_copy.index, df_copy['MA_fast'], label=f'{fast_period}-day MA')
    plt.plot(df_copy.index, df_copy['MA_slow'], label=f'{slow_period}-day MA')
    plt.legend()
    plt.title(f'Moving Average Crossover: {fast_period}/{slow_period}')
    plt.show()

# Create interactive sliders
fast_slider = widgets.IntSlider(value=20, min=5, max=50, step=5, description='Fast MA:')
slow_slider = widgets.IntSlider(value=50, min=20, max=200, step=10, description='Slow MA:')

# Link sliders to function
widgets.interact(plot_ma_crossover, fast_period=fast_slider, slow_period=slow_slider)
\`\`\`

### Advanced Interactive Dashboard

\`\`\`python
class StrategyDashboard:
    """Interactive strategy analysis dashboard"""
    
    def __init__(self, ticker='AAPL'):
        self.ticker = ticker
        self.df = yf.download(ticker, start='2020-01-01', end='2024-01-01')
        
        # Create widgets
        self.fast_ma = widgets.IntSlider(value=20, min=5, max=50, step=5, 
                                         description='Fast MA')
        self.slow_ma = widgets.IntSlider(value=50, min=20, max=200, step=10,
                                        description='Slow MA')
        self.capital = widgets.FloatText(value=100000, description='Capital')
        self.commission = widgets.FloatText(value=0.001, description='Commission')
        
        self.run_button = widgets.Button(description='Run Backtest',
                                         button_style='success')
        self.run_button.on_click(self.run_backtest)
        
        self.output = widgets.Output()
        
    def display(self):
        """Display dashboard"""
        controls = widgets.VBox([
            widgets.HTML(f'<h2>Strategy Dashboard: {self.ticker}</h2>'),
            self.fast_ma,
            self.slow_ma,
            self.capital,
            self.commission,
            self.run_button,
            self.output
        ])
        display(controls)
    
    def run_backtest(self, button):
        """Execute backtest with current parameters"""
        with self.output:
            self.output.clear_output()
            
            # Calculate MAs
            df = self.df.copy()
            df['MA_fast'] = df['Close'].rolling(self.fast_ma.value).mean()
            df['MA_slow'] = df['Close'].rolling(self.slow_ma.value).mean()
            
            # Generate signals
            df['signal'] = np.where(df['MA_fast'] > df['MA_slow'], 1, -1)
            df['position_change'] = df['signal'].diff()
            
            # Calculate returns
            df['returns'] = df['Close'].pct_change()
            df['strategy_returns'] = df['returns'] * df['signal'].shift(1)
            
            # Transaction costs
            df['trades'] = df['position_change'].abs()
            df['transaction_costs'] = df['trades'] * self.commission.value
            df['strategy_returns_net'] = df['strategy_returns'] - df['transaction_costs']
            
            # Performance metrics
            total_return = (1 + df['strategy_returns_net']).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(df)) - 1
            sharpe = df['strategy_returns_net'].mean() / df['strategy_returns_net'].std() * np.sqrt(252)
            
            # Display results
            print(f"\\n{'='*50}")
            print(f"Backtest Results")
            print(f"{'='*50}")
            print(f"Fast MA: {self.fast_ma.value}")
            print(f"Slow MA: {self.slow_ma.value}")
            print(f"Total Return: {total_return:.2%}")
            print(f"Annual Return: {annual_return:.2%}")
            print(f"Sharpe Ratio: {sharpe:.2f}")
            print(f"Number of Trades: {int(df['trades'].sum() / 2)}")
            
            # Plot
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            # Cumulative returns
            cumulative = (1 + df['strategy_returns_net']).cumprod()
            cumulative.plot(ax=axes[0], title='Cumulative Returns')
            axes[0].set_ylabel('Cumulative Return')
            axes[0].grid(True, alpha=0.3)
            
            # Drawdown
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            drawdown.plot(ax=axes[1], title='Drawdown', color='red')
            axes[1].set_ylabel('Drawdown')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()

# Usage
dashboard = StrategyDashboard('AAPL')
dashboard.display()
\`\`\`

## Exporting and Sharing

### Convert to HTML
\`\`\`bash
# Convert notebook to standalone HTML
jupyter nbconvert --to html notebook.ipynb

# With embedded images
jupyter nbconvert --to html --no-input notebook.ipynb

# Execute and convert
jupyter nbconvert --to html --execute notebook.ipynb
\`\`\`

### Convert to Python Script
\`\`\`bash
# Convert to .py file
jupyter nbconvert --to script notebook.ipynb

# Result: notebook.py (pure Python)
\`\`\`

### Convert to PDF (requires LaTeX)
\`\`\`bash
# Install pandoc and LaTeX
conda install -c conda-forge pandoc texlive-core

# Convert to PDF
jupyter nbconvert --to pdf notebook.ipynb
\`\`\`

### Convert to Slides (reveal.js)
\`\`\`bash
# Mark cells with slide type (View -> Cell Toolbar -> Slideshow)
# Convert to slides
jupyter nbconvert --to slides notebook.ipynb --post serve
\`\`\`

## Collaboration and Version Control

### Git Best Practices for Notebooks

\`\`\`bash
# .gitignore for Jupyter projects
# Add to .gitignore:
.ipynb_checkpoints/
*/.ipynb_checkpoints/*
__pycache__/
*.pyc
.DS_Store

# Large data files
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep

# Results
results/*
!results/.gitkeep

# Environment
.env
.venv
env/
venv/
\`\`\`

### Clear Outputs Before Committing

\`\`\`bash
# Install nbstripout
pip install nbstripout

# Set up for repository
cd /path/to/repo
nbstripout --install

# Configure
nbstripout --install --attributes .gitattributes

# Now all commits automatically strip outputs
\`\`\`

### Review Notebooks on GitHub
GitHub renders Jupyter notebooks natively, but diff viewing can be challenging.

**Alternative: nbdime**
\`\`\`bash
# Install nbdime (notebook diff and merge)
pip install nbdime

# Enable git integration
nbdime config-git --enable --global

# Now git diff shows semantic notebook differences
git diff notebook.ipynb

# Web-based diff viewer
nbdiff-web notebook_old.ipynb notebook_new.ipynb
\`\`\`
      `
},
{
    title: "Performance Optimization",
        content: `
# Optimizing Jupyter Notebooks for Large-Scale Analysis

## Profiling and Benchmarking

### Line Profiler
\`\`\`python
# Install
pip install line_profiler

# Load extension
%load_ext line_profiler

# Profile specific function
def slow_function(n):
    result = []
    for i in range(n):
        result.append(i ** 2)
    return result

%lprun -f slow_function slow_function(100000)

# Output shows time spent per line
\`\`\`

### Memory Profiler
\`\`\`python
# Install
pip install memory_profiler

# Load extension
%load_ext memory_profiler

# Profile memory usage
%memit df = pd.read_csv('large_file.csv')

# Detailed line-by-line memory usage
%%memit
df = pd.read_csv('large_file.csv')
df_filtered = df[df['volume'] > 1000000]
df_processed = df_filtered.groupby('ticker').mean()
\`\`\`

## Efficient Data Loading

### Use Appropriate File Formats

\`\`\`python
import pandas as pd
import time

# CSV (slowest, largest)
start = time.time()
df_csv = pd.read_csv('large_data.csv')
csv_time = time.time() - start
print(f"CSV load time: {csv_time:.2f}s")

# Parquet (fast, compressed)
start = time.time()
df_parquet = pd.read_parquet('large_data.parquet')
parquet_time = time.time() - start
print(f"Parquet load time: {parquet_time:.2f}s")

# HDF5 (fast, good for time series)
start = time.time()
df_hdf = pd.read_hdf('large_data.h5', key='data')
hdf_time = time.time() - start
print(f"HDF5 load time: {hdf_time:.2f}s")

# Feather (fastest, but less compression)
start = time.time()
df_feather = pd.read_feather('large_data.feather')
feather_time = time.time() - start
print(f"Feather load time: {feather_time:.2f}s")

# Typical results:
# CSV: 10.5s
# Parquet: 1.2s (9x faster)
# HDF5: 1.5s
# Feather: 0.8s (13x faster)
\`\`\`

### Chunked Loading for Large Files

\`\`\`python
# Load large CSV in chunks
chunk_size = 100000
chunks = []

for chunk in pd.read_csv('huge_file.csv', chunksize=chunk_size):
    # Process each chunk
    chunk_processed = chunk[chunk['volume'] > 1000000]
    chunks.append(chunk_processed)

# Combine processed chunks
df = pd.concat(chunks, ignore_index=True)
\`\`\`

### Selective Column Loading

\`\`\`python
# Only load needed columns
df = pd.read_csv('data.csv', 
                 usecols=['date', 'ticker', 'close', 'volume'])

# vs loading all 50 columns (much slower)
\`\`\`

## Vectorization and NumPy

### Avoid Loops When Possible

\`\`\`python
# Slow: Loop
%%timeit
result = []
for value in df['close']:
    result.append(value * 1.05)

# Fast: Vectorized
%%timeit
result = df['close'] * 1.05

# Speedup: 100-1000x faster
\`\`\`

### Use NumPy for Numerical Operations

\`\`\`python
import numpy as np

# Slow: Pandas apply
%%timeit
df['returns'] = df['close'].apply(lambda x: np.log(x / df['close'].shift(1)))

# Fast: NumPy
%%timeit
df['returns'] = np.log(df['close'] / df['close'].shift(1))

# Faster: Direct division then log
%%timeit
df['returns'] = np.log(df['close'].values[1:] / df['close'].values[:-1])
\`\`\`

## Parallel Processing

### Use Joblib for Parallel Execution

\`\`\`python
from joblib import Parallel, delayed

def backtest_ticker(ticker, start_date, end_date):
    """Backtest single ticker"""
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    # ... backtest logic ...
    return results

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # ...100+ tickers

# Sequential (slow)
results = [backtest_ticker(t, '2020-01-01', '2023-12-31') for t in tickers]

# Parallel (fast)
results = Parallel(n_jobs=-1)(
    delayed(backtest_ticker)(t, '2020-01-01', '2023-12-31') 
    for t in tickers
)

# Use all CPU cores
\`\`\`

### Multiprocessing for Heavy Computation

\`\`\`python
from multiprocessing import Pool
import pandas as pd

def process_ticker(ticker):
    """CPU-intensive processing"""
    df = yf.download(ticker, start='2020-01-01', end='2023-12-31', progress=False)
    
    # Heavy computation
    for i in range(100):
        df[f'feature_{i}'] = df['close'].rolling(i+1).mean()
    
    return ticker, df.shape[0]

if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT', 'GOOGL']  # ... many more
    
    with Pool(processes=8) as pool:
        results = pool.map(process_ticker, tickers)
    
    print(results)
\`\`\`

## Dask for Out-of-Core Computation

When data doesn't fit in memory, use Dask.

\`\`\`python
import dask.dataframe as dd

# Load large CSV with Dask
df = dd.read_csv('huge_file_*.csv')

# Operations are lazy (not executed yet)
df_filtered = df[df['volume'] > 1000000]
df_grouped = df_filtered.groupby('ticker')['close'].mean()

# Trigger computation
result = df_grouped.compute()

# Dask handles data larger than RAM by:
# 1. Breaking into partitions
# 2. Processing chunks sequentially
# 3. Combining results
\`\`\`

## Caching Results

### Use joblib Memory for Function Caching

\`\`\`python
from joblib import Memory

# Set up cache directory
memory = Memory('./cache', verbose=0)

@memory.cache
def expensive_computation(ticker, start_date, end_date):
    """Expensive function - results will be cached"""
    print(f"Computing {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    # ... expensive processing ...
    return results

# First call: slow (computes and caches)
result1 = expensive_computation('AAPL', '2020-01-01', '2023-12-31')

# Second call: instant (loads from cache)
result2 = expensive_computation('AAPL', '2020-01-01', '2023-12-31')

# Different parameters: slow again (new computation)
result3 = expensive_computation('MSFT', '2020-01-01', '2023-12-31')
\`\`\`

### Manual Checkpointing

\`\`\`python
import pickle
from pathlib import Path

def load_or_compute(cache_path, compute_func, *args, **kwargs):
    """Load from cache if exists, otherwise compute"""
    cache_file = Path(cache_path)
    
    if cache_file.exists():
        print(f"Loading from cache: {cache_path}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"Computing and caching to: {cache_path}")
        result = compute_func(*args, **kwargs)
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        return result

# Usage
df = load_or_compute(
    'cache/processed_data.pkl',
    process_data,
    start_date='2020-01-01',
    end_date='2023-12-31'
)
\`\`\`

## Database Integration

### Direct Query to Pandas

\`\`\`python
import sqlalchemy as sa
import pandas as pd

# Create connection
engine = sa.create_engine('postgresql://user:pass@localhost:5432/marketdata')

# Load data directly to DataFrame
query = """
    SELECT date, ticker, close, volume
    FROM prices
    WHERE ticker IN ('AAPL', 'MSFT', 'GOOGL')
    AND date >= '2020-01-01'
    ORDER BY date
"""

df = pd.read_sql(query, engine)

# Much faster than loading entire table then filtering in Python
\`\`\`

### Chunked Writing to Database

\`\`\`python
# Write large DataFrame to database in chunks
chunk_size = 10000

for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    chunk.to_sql('prices', engine, if_exists='append', index=False)
    print(f"Written rows {i} to {i+len(chunk)}")
\`\`\`
      `
}
  ],
exercises: [
    {
        title: "Build a Research Workflow",
        description: "Create a complete Jupyter Lab research workflow for testing a momentum strategy across multiple stocks.",
        difficulty: "intermediate",
        hints: [
            "Set up proper directory structure with data/, notebooks/, src/, tests/",
            "Create parameterized notebook that can test any ticker",
            "Extract reusable functions to .py modules",
            "Use papermill to run notebook for 20+ tickers automatically"
        ]
    },
    {
        title: "Interactive Strategy Dashboard",
        description: "Build an interactive dashboard using ipywidgets that allows real-time parameter adjustment and visualization.",
        difficulty: "intermediate",
        hints: [
            "Use widgets.IntSlider for MA periods, position size, etc.",
            "Create real-time chart updates as parameters change",
            "Display performance metrics dynamically",
            "Add export functionality for optimal parameters"
        ]
    },
    {
        title: "Performance Optimization",
        description: "Optimize a slow notebook that processes 500 stocks for 5 years of daily data.",
        difficulty: "advanced",
        hints: [
            "Profile code to find bottlenecks using %lprun and %memit",
            "Vectorize loops and use NumPy where possible",
            "Implement parallel processing with joblib or multiprocessing",
            "Use parquet instead of CSV for faster loading",
            "Cache expensive computations"
        ]
    },
    {
        title: "Production Pipeline",
        description: "Convert a research notebook into a production-ready pipeline with proper error handling, logging, and testing.",
        difficulty: "advanced",
        hints: [
            "Extract all code to proper Python modules with docstrings",
            "Write unit tests for each function",
            "Add comprehensive error handling and logging",
            "Create command-line interface using argparse",
            "Set up automated execution with cron or Airflow"
        ]
    }
]
};

