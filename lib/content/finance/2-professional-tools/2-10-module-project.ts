import { Content } from '@/lib/types';

export const moduleProjectContent: Content = {
    title: 'Module Project: Financial Data Aggregator',
    subtitle: 'Build a complete data pipeline from scratch',
    description:
        'Capstone project for Module 2: Build a production-quality financial data aggregator that downloads, stores, and serves market data. Integrates all professional tools learned in this module.',
    sections: [
        {
            title: 'Project Overview',
            content: `
# Financial Data Aggregator Project

## What You'll Build

A complete end-to-end system for collecting, storing, and analyzing financial market data:

\`\`\`plaintext
Data Aggregator Architecture:

┌─────────────────────────────────────────────────────────┐
│                    Data Sources                          │
├──────────┬──────────┬───────────┬──────────────────────┤
│ Yahoo    │ Alpha    │  Quandl   │     FRED             │
│ Finance  │ Vantage  │           │  (Economic Data)     │
└────┬─────┴────┬─────┴─────┬─────┴──────┬──────────────┘
     │          │           │            │
     └──────────┴───────────┴────────────┘
                       │
            ┌──────────▼──────────┐
            │   Data Collectors   │
            │   (Python Scripts)   │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │   Validation &      │
            │   Preprocessing     │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │  TimescaleDB        │
            │  (Storage)          │
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │   REST API          │
            │   (FastAPI)         │
            └──────────┬──────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼─────┐ ┌─────▼──────┐ ┌────▼─────┐
│  Jupyter    │ │  Dashboards │ │  Backtest│
│  Notebooks  │ │  (Plotly)   │ │  Engine  │
└─────────────┘ └────────────┘ └──────────┘
\`\`\`

## Features You'll Implement

1. **Data Collection**
   - Multi-source data fetching (Yahoo Finance, Alpha Vantage, FRED)
   - Error handling and retry logic
   - Rate limiting and API quota management
   - Parallel downloads for speed

2. **Data Storage**
   - TimescaleDB for time-series data
   - Efficient schema design
   - Automatic deduplication
   - Data validation

3. **Data API**
   - RESTful API using FastAPI
   - Query endpoints for historical data
   - Real-time data streaming
   - Authentication

4. **Analysis Interface**
   - Jupyter notebooks for exploration
   - Pre-built technical indicators
   - Visualization dashboards
   - Export capabilities

5. **DevOps**
   - Docker containerization
   - Automated testing
   - CI/CD pipeline
   - Monitoring and logging

## Learning Objectives

By completing this project, you'll demonstrate mastery of:
- Python data pipelines
- Database design and optimization
- API development
- Version control with Git
- Docker and deployment
- Testing and quality assurance

## Time Estimate

- Basic version: 8-12 hours
- Full featured: 20-30 hours
- Production grade: 40+ hours

## Prerequisites

Before starting, ensure you have:
- Python 3.10+ installed
- PostgreSQL/TimescaleDB running
- Git configured
- API keys for data sources
- Basic familiarity with all Module 2 topics
      `,
        },
        {
            title: 'Phase 1: Project Setup',
            content: `
# Setting Up the Project

## Directory Structure

Create a professional project structure:

\`\`\`bash
mkdir financial-data-aggregator
cd financial-data-aggregator

# Create directory structure
mkdir -p {src,tests,data,config,scripts,notebooks,docs}
mkdir -p src/{collectors,database,api,utils}
mkdir -p data/{raw,processed}
mkdir -p tests/{unit,integration}

# Initialize Git
git init
git branch -m main

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# Environment
.env
.venv
venv/
ENV/

# Data
data/raw/*
data/processed/*
*.csv
*.parquet
*.h5

# Jupyter
.ipynb_checkpoints
*/.ipynb_checkpoints/*

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
*.log
logs/

# Database
*.db
*.sqlite

# OS
.DS_Store
Thumbs.db

# Config
config/secrets.yaml
config/production.yaml
EOF

# Create README
cat > README.md << 'EOF'
# Financial Data Aggregator

A production-quality system for collecting, storing, and serving financial market data.

## Features
- Multi-source data collection (Yahoo Finance, Alpha Vantage, FRED)
- TimescaleDB storage optimized for time-series
- RESTful API for data access
- Docker containerized
- Comprehensive testing

## Setup
\`\`\`bash
# Create environment
conda create -n data-aggregator python=3.10
conda activate data-aggregator

# Install dependencies
pip install -r requirements.txt

# Configure
cp config/config.yaml.template config/config.yaml
# Edit config/config.yaml with your settings

# Start database
docker-compose up -d postgres

# Initialize database
python scripts/init_db.py

# Run collectors
python scripts/collect_daily_data.py
\`\`\`

## Usage
See [docs/usage.md](docs/usage.md) for detailed usage instructions.

## License
MIT
EOF

# First commit
git add .
git commit -m "Initial commit: Project structure"
\`\`\`

## Dependencies

Create `requirements.txt`:

\`\`\`text
# Core
pandas==2.0.3
numpy==1.24.3
python-dateutil==2.8.2

# Data sources
yfinance==0.2.28
alpha-vantage==2.3.1
pandas-datareader==0.10.0
requests==2.31.0

# Database
sqlalchemy==2.0.19
psycopg2-binary==2.9.7
alembic==1.11.1

# API
fastapi==0.100.0
uvicorn[standard]==0.23.1
pydantic==2.0.3
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6

# Analysis
ta-lib==0.4.28
scipy==1.11.1
scikit-learn==1.3.0

# Visualization
plotly==5.15.0
dash==2.11.1

# Development
pytest==7.4.0
pytest-cov==4.1.0
pytest-asyncio==0.21.1
black==23.7.0
flake8==6.0.0
mypy==1.4.1

# Utilities
python-dotenv==1.0.0
pyyaml==6.0.1
loguru==0.7.0
click==8.1.6
tenacity==8.2.2
schedule==1.2.0
\`\`\`

## Configuration

Create `config/ config.yaml.template`:

\`\`\`yaml
# Configuration Template
# Copy to config.yaml and fill in your values

# Database
database:
  host: localhost
  port: 5432
  database: marketdata
  user: postgres
  password: your_password_here

# Data Sources
alpha_vantage:
  api_key: YOUR_KEY_HERE
  calls_per_minute: 5

fred:
  api_key: YOUR_KEY_HERE

# Collection
collection:
  tickers:
    - AAPL
    - MSFT
    - GOOGL
    - AMZN
    - TSLA
  start_date: "2020-01-01"
  
# API
api:
  host: 0.0.0.0
  port: 8000
  secret_key: generate_secure_key_here
  
# Logging
logging:
  level: INFO
  file: logs/aggregator.log
\`\`\`

## Docker Setup

Create `docker - compose.yml`:

\`\`\`yaml
version: '3.8'

services:
  postgres:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_PASSWORD: password
      POSTGRES_DB: marketdata
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:latest
    ports:
      - "6379:6379"
    volumes:
      - redisdata:/data

  api:
    build: .
    command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
    volumes:
      - .:/app
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      DATABASE_URL: postgresql://postgres:password@postgres:5432/marketdata

volumes:
  pgdata:
  redisdata:
\`\`\`

Create `Dockerfile`:

\`\`\`dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    postgresql-client \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Default command
CMD ["python", "scripts/collect_daily_data.py"]
\`\`\`
      `,
        },
    {
        title: 'Phase 2: Data Collectors',
        content: `
# Building Data Collectors

## Base Collector Class

Create `src / collectors / base.py`:

\`\`\`python
"""Base collector class with common functionality"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import time

class BaseCollector(ABC):
    """Base class for data collectors"""
    
    def __init__(self, rate_limit: int = 5):
        """
        Initialize collector
        
        Args:
            rate_limit: Maximum API calls per minute
        """
        self.rate_limit = rate_limit
        self.call_times: List[float] = []
        
    def _check_rate_limit(self):
        """Ensure we don't exceed rate limit"""
        now = time.time()
        
        # Remove calls older than 1 minute
        self.call_times = [t for t in self.call_times if now - t < 60]
        
        # If at limit, wait
        if len(self.call_times) >= self.rate_limit:
            sleep_time = 60 - (now - self.call_times[0]) + 1
            logger.info(f"Rate limit reached, sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
            self.call_times = []
        
        self.call_times.append(now)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def _fetch_with_retry(self, fetch_func, *args, **kwargs):
        """Fetch data with automatic retry on failure"""
        try:
            self._check_rate_limit()
            return fetch_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Fetch failed: {e}")
            raise
    
    @abstractmethod
    def fetch_historical(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """Fetch historical data for ticker"""
        pass
    
    @abstractmethod
    def fetch_latest(self, ticker: str) -> Dict[str, Any]:
        """Fetch latest data point for ticker"""
        pass
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate fetched data"""
        if df is empty or len(df) == 0:
            logger.warning("Empty dataframe")
            return False
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = set(required_cols) - set(df.columns)
        if missing:
            logger.error(f"Missing columns: {missing}")
            return False
        
        # Check for nulls
        if df[required_cols].isnull().any().any():
            logger.warning("Null values detected")
            return False
        
        # Check OHLC relationships
        if (df['high'] < df['low']).any():
            logger.error("High < Low detected")
            return False
        
        if ((df['high'] < df['open']) | (df['high'] < df['close'])).any():
            logger.error("High not highest")
            return False
        
        if ((df['low'] > df['open']) | (df['low'] > df['close'])).any():
            logger.error("Low not lowest")
            return False
        
        return True
\`\`\`

## Yahoo Finance Collector

Create `src / collectors / yahoo.py`:

\`\`\`python
"""Yahoo Finance data collector"""
import pandas as pd
import yfinance as yf
from typing import Dict, Any
from loguru import logger
from .base import BaseCollector

class YahooCollector(BaseCollector):
    """Collect data from Yahoo Finance"""
    
    def __init__(self):
        super().__init__(rate_limit=2000)  # Very generous limit
        
    def fetch_historical(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data
        
        Args:
            ticker: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching {ticker} from {start_date} to {end_date}")
        
        try:
            df = self._fetch_with_retry(
                yf.download,
                ticker,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()
            
            # Standardize column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            df = df.reset_index()
            df['ticker'] = ticker
            df['source'] = 'yahoo'
            
            # Rename columns
            df = df.rename(columns={'date': 'time'})
            
            if self.validate_data(df):
                logger.success(f"Fetched {len(df)} rows for {ticker}")
                return df
            else:
                logger.error(f"Validation failed for {ticker}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            return pd.DataFrame()
    
    def fetch_latest(self, ticker: str) -> Dict[str, Any]:
        """Fetch latest price data"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'ticker': ticker,
                'price': info.get('regularMarketPrice'),
                'volume': info.get('regularMarketVolume'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'source': 'yahoo'
            }
        except Exception as e:
            logger.error(f"Failed to fetch latest for {ticker}: {e}")
            return {}
\`\`\`

## Alpha Vantage Collector

Create `src / collectors / alpha_vantage.py`:

\`\`\`python
"""Alpha Vantage data collector"""
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from loguru import logger
from .base import BaseCollector

class AlphaVantageCollector(BaseCollector):
    """Collect data from Alpha Vantage"""
    
    def __init__(self, api_key: str):
        super().__init__(rate_limit=5)  # 5 calls per minute (free tier)
        self.ts = TimeSeries(key=api_key, output_format='pandas')
        
    def fetch_historical(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch historical daily data"""
        logger.info(f"Fetching {ticker} from Alpha Vantage")
        
        try:
            data, meta = self._fetch_with_retry(
                self.ts.get_daily,
                symbol=ticker,
                outputsize='full'
            )
            
            # Process data
            df = data.reset_index()
            df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
            df['ticker'] = ticker
            df['source'] = 'alpha_vantage'
            
            # Filter date range
            df['time'] = pd.to_datetime(df['time'])
            df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]
            
            if self.validate_data(df):
                logger.success(f"Fetched {len(df)} rows for {ticker}")
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            return pd.DataFrame()
    
    def fetch_latest(self, ticker: str) -> Dict[str, Any]:
        """Fetch latest quote"""
        try:
            data, meta = self._fetch_with_retry(
                self.ts.get_quote_endpoint,
                symbol=ticker
            )
            
            return {
                'ticker': ticker,
                'price': float(data['05. price'][0]),
                'volume': int(data['06. volume'][0]),
                'source': 'alpha_vantage'
            }
        except Exception as e:
            logger.error(f"Failed to fetch latest for {ticker}: {e}")
            return {}
\`\`\`

## Collector Manager

Create `src / collectors / manager.py`:

\`\`\`python
"""Manage multiple collectors"""
from typing import List, Dict
import pandas as pd
from loguru import logger
from .yahoo import YahooCollector
from .alpha_vantage import AlphaVantageCollector

class CollectorManager:
    """Manage data collection from multiple sources"""
    
    def __init__(self, config: Dict):
        self.collectors = {
            'yahoo': YahooCollector(),
        }
        
        # Add Alpha Vantage if API key provided
        if config.get('alpha_vantage', {}).get('api_key'):
            self.collectors['alpha_vantage'] = AlphaVantageCollector(
                api_key=config['alpha_vantage']['api_key']
            )
        
        self.primary_source = config.get('primary_source', 'yahoo')
        
    def fetch_historical(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch historical data, trying multiple sources if needed
        """
        # Try primary source first
        df = self.collectors[self.primary_source].fetch_historical(
            ticker, start_date, end_date
        )
        
        if not df.empty:
            return df
        
        # Try fallback sources
        for name, collector in self.collectors.items():
            if name == self.primary_source:
                continue
                
            logger.info(f"Trying fallback source: {name}")
            df = collector.fetch_historical(ticker, start_date, end_date)
            
            if not df.empty:
                return df
        
        logger.error(f"All sources failed for {ticker}")
        return pd.DataFrame()
    
    def fetch_multiple(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch data for multiple tickers"""
        from joblib import Parallel, delayed
        
        results = Parallel(n_jobs=4)(
            delayed(self.fetch_historical)(ticker, start_date, end_date)
            for ticker in tickers
        )
        
        valid_results = [df for df in results if not df.empty]
        
        if valid_results:
            return pd.concat(valid_results, ignore_index=True)
        else:
            return pd.DataFrame()
\`\`\`
      `,
        },
{
    title: 'Phase 3: Database Layer',
        content: `
# Database Integration

## Database Schema

Create `src / database / schema.py`:

\`\`\`python
"""Database schema definitions"""
from sqlalchemy import create_engine, Column, Integer, String, Float, BigInteger, DateTime, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Ticker(Base):
    """Ticker information"""
    __tablename__ = 'tickers'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True, nullable=False)
    name = Column(String(255))
    exchange = Column(String(50))
    sector = Column(String(100))
    industry = Column(String(100))
    
class Price(Base):
    """Price data"""
    __tablename__ = 'prices'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    time = Column(DateTime(timezone=True), nullable=False)
    ticker_id = Column(Integer, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(BigInteger)
    adj_close = Column(Float)
    source = Column(String(50))
    
    __table_args__ = (
        UniqueConstraint('time', 'ticker_id', name='uix_time_ticker'),
    )
\`\`\`

## Database Manager

Create `src / database / manager.py`:

\`\`\`python
"""Database operations manager"""
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from loguru import logger
from typing import List, Dict
from .schema import Base, Ticker, Price

class DatabaseManager:
    """Manage database operations"""
    
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string, pool_pre_ping=True)
        self.Session = sessionmaker(bind=self.engine)
        
    def init_database(self):
        """Initialize database schema"""
        logger.info("Initializing database")
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Enable TimescaleDB extension
        with self.engine.connect() as conn:
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
                conn.commit()
                logger.success("TimescaleDB extension enabled")
            except Exception as e:
                logger.warning(f"Could not enable TimescaleDB: {e}")
        
        # Convert prices table to hypertable
        with self.engine.connect() as conn:
            try:
                conn.execute(text(
                    "SELECT create_hypertable('prices', 'time', "
                    "if_not_exists => TRUE)"
                ))
                conn.commit()
                logger.success("Hypertable created")
            except Exception as e:
                logger.warning(f"Could not create hypertable: {e}")
        
        # Create indexes
        with self.engine.connect() as conn:
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_prices_ticker_time "
                "ON prices(ticker_id, time DESC)"
            ))
            conn.commit()
        
        logger.success("Database initialized")
    
    def insert_ticker(self, symbol: str, **kwargs) -> int:
        """Insert or get ticker ID"""
        session = self.Session()
        
        try:
            ticker = session.query(Ticker).filter_by(symbol=symbol).first()
            
            if not ticker:
                ticker = Ticker(symbol=symbol, **kwargs)
                session.add(ticker)
                session.commit()
                logger.info(f"Inserted ticker: {symbol}")
            
            return ticker.id
            
        finally:
            session.close()
    
    def insert_prices(self, df: pd.DataFrame) -> int:
        """Insert price data"""
        if df.empty:
            return 0
        
        session = self.Session()
        
        try:
            # Ensure ticker exists
            ticker_id = self.insert_ticker(df['ticker'].iloc[0])
            
            # Prepare data
            df = df.copy()
            df['ticker_id'] = ticker_id
            
            # Use raw SQL for bulk insert (faster)
            df[['time', 'ticker_id', 'open', 'high', 'low', 'close', 'volume', 'source']].to_sql(
                'prices',
                self.engine,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=1000
            )
            
            logger.success(f"Inserted {len(df)} price records")
            return len(df)
            
        except Exception as e:
            logger.error(f"Failed to insert prices: {e}")
            return 0
            
        finally:
            session.close()
    
    def get_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Retrieve price data"""
        query = text("""
            SELECT p.time, p.open, p.high, p.low, p.close, p.volume
            FROM prices p
            JOIN tickers t ON p.ticker_id = t.id
            WHERE t.symbol = :ticker
              AND p.time >= :start_date
              AND p.time <= :end_date
            ORDER BY p.time
        """)
        
        return pd.read_sql(
            query,
            self.engine,
            params={
                'ticker': ticker,
                'start_date': start_date,
                'end_date': end_date
            }
        )
    
    def get_latest_date(self, ticker: str) -> str:
        """Get latest date for ticker"""
        query = text("""
            SELECT MAX(p.time) as latest
            FROM prices p
            JOIN tickers t ON p.ticker_id = t.id
            WHERE t.symbol = :ticker
        """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {'ticker': ticker})
            row = result.fetchone()
            return str(row[0]) if row and row[0] else None
\`\`\`

## Database initialization script

Create `scripts / init_db.py`:

\`\`\`python
#!/usr/bin/env python3
"""Initialize database"""
import yaml
from pathlib import Path
from src.database.manager import DatabaseManager
from loguru import logger

def main():
    # Load config
    config_path = Path("config/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Build connection string
    db_config = config['database']
    conn_string = (
        f"postgresql://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    
    # Initialize database
    db = DatabaseManager(conn_string)
    db.init_database()
    
    logger.success("Database initialized successfully")

if __name__ == '__main__':
    main()
\`\`\`
      `,
        },
{
    title: 'Phase 4: Complete Implementation',
        content: `
# Final Integration and Testing

## Data Collection Script

Create `scripts / collect_daily_data.py`:

\`\`\`python
#!/usr/bin/env python3
"""Collect daily market data"""
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
from src.collectors.manager import CollectorManager
from src.database.manager import DatabaseManager

def main():
    # Load config
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logger.add("logs/collection.log", rotation="1 day")
    
    # Initialize components
    collector = CollectorManager(config)
    
    db_config = config['database']
    conn_string = (
        f"postgresql://{db_config['user']}:{db_config['password']}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
    db = DatabaseManager(conn_string)
    
    # Get tickers
    tickers = config['collection']['tickers']
    logger.info(f"Processing {len(tickers)} tickers")
    
    # Collect data
    for ticker in tickers:
        try:
            # Get latest date in database
            latest = db.get_latest_date(ticker)
            
            if latest:
                start_date = (datetime.fromisoformat(latest) + timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                start_date = config['collection']['start_date']
            
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"Collecting {ticker} from {start_date} to {end_date}")
            
            # Fetch data
            df = collector.fetch_historical(ticker, start_date, end_date)
            
            if not df.empty:
                # Insert to database
                rows = db.insert_prices(df)
                logger.success(f"Inserted {rows} rows for {ticker}")
            else:
                logger.warning(f"No new data for {ticker}")
                
        except Exception as e:
            logger.error(f"Failed to process {ticker}: {e}")
            continue
    
    logger.success("Collection complete")

if __name__ == '__main__':
    main()
\`\`\`

## Testing

Create `tests / test_collectors.py`:

\`\`\`python
"""Test data collectors"""
import pytest
from src.collectors.yahoo import YahooCollector

def test_yahoo_collector():
    """Test Yahoo Finance collector"""
    collector = YahooCollector()
    
    df = collector.fetch_historical('AAPL', '2023-01-01', '2023-01-31')
    
    assert not df.empty
    assert 'open' in df.columns
    assert 'high' in df.columns
    assert 'low' in df.columns
    assert 'close' in df.columns
    assert 'volume' in df.columns
    assert len(df) > 0

def test_data_validation():
    """Test data validation"""
    collector = YahooCollector()
    
    df = collector.fetch_historical('AAPL', '2023-01-01', '2023-01-31')
    
    # Should pass validation
    assert collector.validate_data(df)
    
    # High should be >= low
    assert (df['high'] >= df['low']).all()
    
    # High should be >= open and close
    assert (df['high'] >= df['open']).all()
    assert (df['high'] >= df['close']).all()
\`\`\`

## Running the Project

### Initial Setup
\`\`\`bash
# Clone or initialize project
git clone https://github.com/yourusername/financial-data-aggregator.git
cd financial-data-aggregator

# Create environment
conda create -n data-aggregator python=3.10
conda activate data-aggregator

# Install dependencies
pip install -r requirements.txt

# Configure
cp config/config.yaml.template config/config.yaml
# Edit config/config.yaml with your settings

# Start database
docker-compose up -d postgres

# Wait for database to be ready
sleep 10

# Initialize database
python scripts/init_db.py
\`\`\`

### Collect Data
\`\`\`bash
# Collect historical data
python scripts/collect_daily_data.py

# Schedule daily collection (crontab)
crontab -e
# Add: 0 18 * * * cd /path/to/project && /path/to/conda/envs/data-aggregator/bin/python scripts/collect_daily_data.py
\`\`\`

### Test
\`\`\`bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
\`\`\`

## Success Criteria

Your project is complete when:

✅ **Data Collection**
- [x] Fetches data from at least 2 sources
- [x] Handles errors gracefully with retries
- [x] Respects rate limits
- [x] Validates data quality
- [x] Processes multiple tickers in parallel

✅ **Database**
- [x] Uses TimescaleDB hypertables
- [x] Proper schema with constraints
- [x] Efficient indexes
- [x] Handles duplicates correctly
- [x] Fast queries (<100ms for year of data)

✅ **Code Quality**
- [x] Follows PEP 8 style
- [x] Comprehensive docstrings
- [x] Unit tests with >80% coverage
- [x] Proper error handling and logging
- [x] Configuration separate from code

✅ **DevOps**
- [x] Docker containerized
- [x] Git version controlled
- [x] README with setup instructions
- [x] Automated daily collection
- [x] Monitoring and logs

## Extensions (Optional)

Take your project further:

1. **REST API** - Build FastAPI endpoints to serve data
2. **Real-time Updates** - Add WebSocket for live prices
3. **Dashboard** - Create Plotly Dash visualization
4. **ML Features** - Calculate technical indicators
5. **Alerts** - Email/SMS notifications for price movements
6. **Multi-asset** - Add forex, crypto, commodities
7. **Cloud Deploy** - Deploy to AWS/GCP/Azure
8. **Monitoring** - Add Grafana dashboards

## Submission

When complete, create a repository with:
- All source code
- README with setup instructions
- Configuration templates
- Database schema
- Tests
- Example notebooks

Share your GitHub repository link!
      `,
        },
    ],
exercises: [
    {
        title: 'Complete Basic Implementation',
        description:
            'Build the core data aggregator with Yahoo Finance collector, TimescaleDB storage, and daily collection script.',
        difficulty: 'intermediate',
        hints: [
            'Start with project structure and Git init',
            'Implement Yahoo collector first (simpler)',
            'Set up database with TimescaleDB',
            'Write collection script with error handling',
            'Add logging throughout',
            'Test on 5 stocks before scaling up',
        ],
    },
    {
        title: 'Add Multiple Data Sources',
        description: 'Extend the aggregator to fetch from Alpha Vantage and FRED, with automatic fallback between sources.',
        difficulty: 'advanced',
        hints: [
            'Implement Alpha Vantage collector',
            'Add collector manager with fallback logic',
            'Respect rate limits for each source',
            'Handle different data formats',
            'Merge data from multiple sources',
            'Track data source in database',
        ],
    },
    {
        title: 'Build REST API',
        description:
            'Create a FastAPI application that serves historical data, latest prices, and allows data queries with various filters.',
        difficulty: 'advanced',
        hints: [
            'Create FastAPI app with endpoints',
            'Add authentication with JWT tokens',
            'Implement query parameters (ticker, date range, interval)',
            'Add caching with Redis',
            'Write API documentation',
            'Add rate limiting',
        ],
    },
    {
        title: 'Production Deployment',
        description:
            'Deploy the complete system to cloud (AWS/GCP/Azure) with monitoring, automated backups, and CI/CD pipeline.',
        difficulty: 'advanced',
        hints: [
            'Choose cloud provider',
            'Set up managed PostgreSQL',
            'Deploy API with Docker',
            'Configure domain and SSL',
            'Add Grafana monitoring',
            'Set up GitHub Actions for CI/CD',
            'Implement automated backups',
        ],
    },
],
};

