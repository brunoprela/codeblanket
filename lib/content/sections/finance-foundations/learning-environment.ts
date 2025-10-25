export const learningEnvironment = {
    title: 'Your Finance Learning Environment',
    id: 'learning-environment',
    content: `
# Your Finance Learning Environment

## Introduction

This section helps you set up a **professional-grade development environment** for finance engineering:

1. Hardware requirements
2. Software stack (Python, Jupyter, VS Code, Git)
3. Essential libraries
4. Market data sources
5. Paper trading accounts
6. Development workflow

By the end, you'll have everything needed to build and test trading systems.

---

## Hardware Requirements

### Minimum Specs

**For learning/backtesting**:
- CPU: 4+ cores (Intel i5/Ryzen 5 or better)
- RAM: 16GB (32GB recommended for large datasets)
- Storage: 256GB SSD (500GB+ recommended)
- OS: Windows 10/11, macOS, or Linux

**For production trading**:
- CPU: 8+ cores (Intel i7/Ryzen 7 or better)
- RAM: 32GB+ (64GB for high-frequency)
- Storage: 1TB+ NVMe SSD
- Network: Low-latency internet (fiber, <10ms to exchanges)
- Backup: UPS (uninterruptible power supply)

### Cloud Alternatives

**AWS/GCP/Azure** for production:
- **EC2 c6i.2xlarge**: 8 vCPUs, 16GB RAM, $0.34/hour
- **EC2 c6i.4xlarge**: 16 vCPUs, 32GB RAM, $0.68/hour
- **Spot instances**: 70% discount (risk of termination)

**Why cloud**:
- Co-location possible (close to exchanges)
- Scalable (add compute for backtesting)
- Always-on (no power outages)
- Cheaper for experimentation

---

## Software Stack

### Python Setup

**Install Python 3.10+**:
\`\`\`bash
# macOS (using Homebrew)
brew install python@3.11

# Linux (Ubuntu/Debian)
sudo apt update
sudo apt install python3.11 python3-pip

# Windows (download from python.org)
# Or use Anaconda: https://www.anaconda.com/download
\`\`\`

**Virtual environments** (isolate project dependencies):
\`\`\`bash
# Create virtual environment
python3 -m venv financeenv

# Activate (macOS/Linux)
source financeenv/bin/activate

# Activate (Windows)
financeenv\\Scripts\\activate

# Verify
which python  # Should show financeenv path
python --version  # Should show 3.10+
\`\`\`

### Essential Libraries

**Install data science stack**:
\`\`\`bash
pip install --upgrade pip

# Core data science
pip install numpy pandas scipy matplotlib seaborn

# Finance-specific
pip install yfinance alpha_vantage fredapi pandas-datareader

# Technical analysis
pip install ta-lib TA-Lib  # Note: ta-lib requires C library first

# Machine learning
pip install scikit-learn xgboost lightgbm

# Deep learning (optional)
pip install torch tensorflow

# Backtesting
pip install backtrader zipline-reloaded pyfolio

# Live trading
pip install alpaca-trade-api ccxt  # Alpaca for stocks, CCXT for crypto

# Utilities
pip install requests beautifulsoup4 selenium jupyter notebook

# Database
pip install psycopg2-binary sqlalchemy redis

# Save requirements
pip freeze > requirements.txt
\`\`\`

**TA-Lib installation** (technical indicators):
\`\`\`bash
# macOS
brew install ta-lib
pip install TA-Lib

# Linux (Ubuntu)
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib

# Windows (use conda)
conda install -c conda-forge ta-lib
\`\`\`

### Development Tools

**VS Code** (recommended editor):
\`\`\`bash
# Download from: https://code.visualstudio.com/

# Install extensions:
# - Python (Microsoft)
# - Jupyter
# - GitLens
# - Excel Viewer (for CSV files)
# - Rainbow CSV
\`\`\`

**Jupyter Notebook** (interactive development):
\`\`\`bash
# Install
pip install jupyter jupyterlab

# Launch
jupyter notebook  # Classic interface
jupyter lab       # Modern interface

# Access at: http://localhost:8888
\`\`\`

**Git** (version control):
\`\`\`bash
# macOS
brew install git

# Linux
sudo apt install git

# Windows
# Download from: https://git-scm.com/download/win

# Configure
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Create repo
mkdir my-trading-system
cd my-trading-system
git init
git add .
git commit -m "Initial commit"

# Push to GitHub
git remote add origin https://github.com/yourusername/my-trading-system.git
git push -u origin main
\`\`\`

---

## Market Data Sources

### Free Data Sources

**Yahoo Finance** (historical prices):
\`\`\`python
import yfinance as yf

# Single ticker
aapl = yf.download('AAPL', start='2020-01-01', end='2024-01-01')

# Multiple tickers
tickers = ['AAPL', 'MSFT', 'GOOGL']
data = yf.download(tickers, start='2023-01-01')

# Save to CSV
aapl.to_csv('aapl_prices.csv')
\`\`\`

**Alpha Vantage** (free API, 500 calls/day):
\`\`\`python
from alpha_vantage.timeseries import TimeSeries

ts = TimeSeries(key='YOUR_API_KEY', output_format='pandas')

# Daily prices
data, meta = ts.get_daily('AAPL', outputsize='full')

# Intraday (1min, 5min, 15min, 30min, 60min)
data, meta = ts.get_intraday('AAPL', interval='5min', outputsize='full')
\`\`\`

**FRED** (economic data):
\`\`\`python
from fredapi import Fred

fred = Fred(api_key='YOUR_FRED_API_KEY')

# Get data
gdp = fred.get_series('GDP')
unemployment = fred.get_series('UNRATE')
treasury_10y = fred.get_series('DGS10')
\`\`\`

### Paid Data Sources (Production)

**Polygon.io**:
- Cost: $199/month (stocks), $299/month (options)
- Features: Real-time, historical, options, aggregates
- Quality: Professional-grade, used by hedge funds

**IEX Cloud**:
- Cost: Pay-per-call (~$0.001 per quote)
- Features: Real-time, fundamentals, news
- Quality: High, built for developers

**Alpaca Markets** (also broker):
- Cost: Free for basic, $99/month for unlimited
- Features: Real-time quotes, trades, bars
- Quality: Good for retail traders

---

## Paper Trading Accounts

### Alpaca

**Best for stocks** (commission-free):
\`\`\`python
import alpaca_trade_api as tradeapi

# Setup (get keys from alpaca.markets)
API_KEY = 'your_api_key'
SECRET_KEY = 'your_secret_key'
BASE_URL = 'https://paper-api.alpaca.markets'  # Paper trading

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# Get account info
account = api.get_account()
print(f"Buying power: ${float(account.buying_power):, .2f
}")

# Place order
api.submit_order(
    symbol = 'AAPL',
    qty = 10,
    side = 'buy',
    type = 'market',
    time_in_force = 'gtc'
)

# Get positions
positions = api.list_positions()
for position in positions:
    print(f"{position.symbol}: {position.qty} shares @ ${float(position.avg_entry_price):.2f}")
\`\`\`

### Interactive Brokers (IBKR)

**Best for global markets** (stocks, options, futures, forex):
- Paper account: Free at interactivebrokers.com
- API: TWS API (Java/Python)
- Python wrapper: `ib_insync` library

\`\`\`python
from ib_insync import *

# Connect to TWS or IB Gateway
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # 7497 = paper trading port

# Get account info
account_values = ib.accountValues()
for value in account_values:
    print(f"{value.tag}: {value.value}")

# Place order
contract = Stock('AAPL', 'SMART', 'USD')
order = MarketOrder('BUY', 100)
trade = ib.placeOrder(contract, order)

# Wait for fill
while not trade.isDone():
    ib.waitOnUpdate()

print(f"Order filled at {trade.orderStatus.avgFillPrice}")
\`\`\`

### Binance (Crypto)

**Best for cryptocurrency**:
\`\`\`python
import ccxt

# Setup (testnet)
exchange = ccxt.binance({
    'apiKey': 'your_testnet_key',
    'secret': 'your_testnet_secret',
    'options': {'defaultType': 'future'},
})

exchange.set_sandbox_mode(True)  # Testnet

# Get balance
balance = exchange.fetch_balance()
print(f"USDT: {balance['USDT']['free']}")

# Place order
order = exchange.create_market_buy_order('BTC/USDT', 0.001)  # 0.001 BTC
print(f"Bought at ${order['average']}")
\`\`\`

---

## Development Workflow

### Project Structure

\`\`\`
my-trading-system/
├── data/                 # Downloaded market data
│   ├── prices/
│   └── fundamentals/
├── notebooks/            # Jupyter notebooks for research
│   ├── exploration.ipynb
│   └── backtests.ipynb
├── src/                  # Source code
│   ├── data/            # Data fetching
│   ├── indicators/      # Technical indicators
│   ├── strategies/      # Trading strategies
│   ├── backtesting/     # Backtest engine
│   └── execution/       # Live trading
├── tests/               # Unit tests
├── config/              # Configuration files
├── logs/                # Log files
├── requirements.txt     # Python dependencies
├── README.md            # Documentation
└── .gitignore          # Git ignore file
\`\`\`

### Daily Workflow

1. **Morning** (before market open 9:30am ET):
   - Check overnight news
   - Review positions
   - Update watchlist

2. **Market hours** (9:30am-4pm ET):
   - Monitor running strategies
   - Check for alerts
   - Log any manual interventions

3. **After market close**:
   - Download end-of-day data
   - Run backtests on new ideas
   - Update research notebooks
   - Commit code changes

4. **Weekly**:
   - Review performance metrics
   - Analyze what worked/didn't
   - Research new strategies
   - Update documentation

---

## Key Takeaways

1. **Hardware**: 16GB+ RAM, SSD storage, consider cloud for production
2. **Software**: Python 3.10+, Jupyter, VS Code, Git, virtual environments
3. **Libraries**: NumPy, pandas, yfinance, TA-Lib, scikit-learn, backtrader
4. **Data**: Yahoo Finance (free), Alpha Vantage (free API), FRED (economic)
5. **Paper trading**: Alpaca (stocks), IBKR (global), Binance (crypto)
6. **Workflow**: Research in Jupyter, develop in VS Code, version control with Git

**Next section**: Module Project - build a complete personal finance dashboard!
`,
};

