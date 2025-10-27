export const historicalDataManagement = {
    title: 'Historical Data Management',
    slug: 'historical-data-management',
    description:
        'Master the complexities of managing historical market data for backtesting, including handling corporate actions, avoiding survivorship bias, and ensuring data quality',
    content: `
# Historical Data Management

## Introduction: Why Data Quality Makes or Breaks Backtests

In backtesting, **garbage in = garbage out**. The quality of your historical data directly determines the reliability of your backtest results. A single data error can make a losing strategy appear profitable, or vice versa.

**What you'll learn:**
- How to clean and validate market data
- Handling corporate actions (splits, dividends, mergers)
- Avoiding survivorship bias with point-in-time data
- Building production-grade data pipelines
- Cost-effective data storage strategies

**Why this matters for engineers:**
- Data errors are the #1 cause of backtest failures
- Renaissance Technologies spends 40% of engineering time on data quality
- A mishandled stock split can corrupt

 months of analysis
- Production systems need automated data validation

**Real-World Impact:**
A hedge fund lost $50M+ because their data vendor didn't properly adjust for a stock split. Their "profitable" strategy was actually just buying stocks before splits and selling after - a paper-only profit that disappeared in live trading.

---

## Data Quality and Validation

### The Data Quality Pyramid

\`\`\`python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from enum import Enum

class DataQualityLevel(Enum):
    """Data quality tiers"""
    CRITICAL = "critical"  # Will break backtest
    HIGH = "high"          # Will distort results
    MEDIUM = "medium"      # May cause issues
    LOW = "low"            # Minor concerns

@dataclass
class DataQualityIssue:
    """Represents a data quality problem"""
    level: DataQualityLevel
    issue_type: str
    description: str
    affected_rows: int
    recommendation: str

class DataValidator:
    """
    Comprehensive data quality validation for backtesting
    """
    
    def __init__(self, data: pd.DataFrame, ticker: str):
        """
        Initialize validator
        
        Args:
            data: DataFrame with OHLCV data (DateTimeIndex)
            ticker: Stock ticker symbol
        """
        self.data = data.copy()
        self.ticker = ticker
        self.issues: List[DataQualityIssue] = []
    
    def validate_all(self) -> List[DataQualityIssue]:
        """
        Run all validation checks
        
        Returns:
            List of identified issues
        """
        print(f"Validating data for {self.ticker}...")
        print(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
        print(f"Total rows: {len(self.data)}\\n")
        
        # Run all checks
        self.check_missing_data()
        self.check_price_continuity()
        self.check_volume_anomalies()
        self.check_price_anomalies()
        self.check_data_gaps()
        self.check_timestamp_ordering()
        self.check_ohlc_consistency()
        
        # Summarize results
        self.print_summary()
        
        return self.issues
    
    def check_missing_data(self):
        """Check for missing values"""
        missing = self.data.isnull().sum()
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if missing[col] > 0:
                self.issues.append(DataQualityIssue(
                    level=DataQualityLevel.CRITICAL,
                    issue_type="missing_data",
                    description=f"Missing {missing[col]} values in {col}",
                    affected_rows=int(missing[col]),
                    recommendation="Forward-fill or interpolate missing values"
                ))
    
    def check_price_continuity(self):
        """
        Check for unrealistic price movements
        
        Large price jumps might indicate:
        - Stock splits not adjusted
        - Data errors
        - Actual price shocks (need to verify)
        """
        pct_change = self.data['Close'].pct_change().abs()
        
        # Flag moves > 50% in one day (likely data error or split)
        large_moves = pct_change[pct_change > 0.5]
        
        if len(large_moves) > 0:
            for date, move in large_moves.items():
                self.issues.append(DataQualityIssue(
                    level=DataQualityLevel.HIGH,
                    issue_type="large_price_jump",
                    description=f"Price moved {move*100:.1f}% on {date}",
                    affected_rows=1,
                    recommendation="Verify if this is a stock split or data error"
                ))
    
    def check_volume_anomalies(self):
        """Check for volume anomalies"""
        volume = self.data['Volume']
        
        # Zero volume days
        zero_volume = (volume == 0).sum()
        if zero_volume > 0:
            self.issues.append(DataQualityIssue(
                level=DataQualityLevel.MEDIUM,
                issue_type="zero_volume",
                description=f"{zero_volume} days with zero volume",
                affected_rows=int(zero_volume),
                recommendation="Verify if stock was halted or data is missing"
            ))
        
        # Extremely high volume (> 10x median)
        median_volume = volume.median()
        high_volume = volume[volume > median_volume * 10]
        
        if len(high_volume) > 0:
            self.issues.append(DataQualityIssue(
                level=DataQualityLevel.LOW,
                issue_type="high_volume",
                description=f"{len(high_volume)} days with volume > 10x median",
                affected_rows=len(high_volume),
                recommendation="Verify if these are legitimate (earnings, news)"
            ))
    
    def check_price_anomalies(self):
        """Check for unrealistic prices"""
        # Negative or zero prices
        for col in ['Open', 'High', 'Low', 'Close']:
            invalid = (self.data[col] <= 0).sum()
            if invalid > 0:
                self.issues.append(DataQualityIssue(
                    level=DataQualityLevel.CRITICAL,
                    issue_type="invalid_price",
                    description=f"{invalid} rows with {col} <= 0",
                    affected_rows=int(invalid),
                    recommendation="Remove or correct invalid prices"
                ))
        
        # Extremely low prices (penny stocks or data error)
        low_price = (self.data['Close'] < 0.01).sum()
        if low_price > 0:
            self.issues.append(DataQualityIssue(
                level=DataQualityLevel.MEDIUM,
                issue_type="penny_stock",
                description=f"{low_price} days with price < $0.01",
                affected_rows=int(low_price),
                recommendation="Be cautious with penny stocks in backtests"
            ))
    
    def check_data_gaps(self):
        """Check for gaps in time series"""
        dates = pd.date_range(
            start=self.data.index[0],
            end=self.data.index[-1],
            freq='B'  # Business days
        )
        
        missing_dates = dates.difference(self.data.index)
        
        # Exclude known holidays
        # (In production, maintain holiday calendar)
        actual_gaps = len(missing_dates)
        
        if actual_gaps > 10:  # More than 10 missing business days
            self.issues.append(DataQualityIssue(
                level=DataQualityLevel.MEDIUM,
                issue_type="data_gaps",
                description=f"{actual_gaps} missing business days",
                affected_rows=actual_gaps,
                recommendation="Fill gaps or adjust backtest logic"
            ))
    
    def check_timestamp_ordering(self):
        """Verify timestamps are properly ordered"""
        if not self.data.index.is_monotonic_increasing:
            self.issues.append(DataQualityIssue(
                level=DataQualityLevel.CRITICAL,
                issue_type="unordered_timestamps",
                description="Timestamps are not in chronological order",
                affected_rows=len(self.data),
                recommendation="Sort data by timestamp before using"
            ))
    
    def check_ohlc_consistency(self):
        """
        Verify OHLC relationships
        High should be >= Open, Close, Low
        Low should be <= Open, Close, High
        """
        # High >= all others
        high_violations = (
            (self.data['High'] < self.data['Open']) |
            (self.data['High'] < self.data['Close']) |
            (self.data['High'] < self.data['Low'])
        ).sum()
        
        # Low <= all others
        low_violations = (
            (self.data['Low'] > self.data['Open']) |
            (self.data['Low'] > self.data['Close']) |
            (self.data['Low'] > self.data['High'])
        ).sum()
        
        total_violations = high_violations + low_violations
        
        if total_violations > 0:
            self.issues.append(DataQualityIssue(
                level=DataQualityLevel.HIGH,
                issue_type="ohlc_inconsistency",
                description=f"{total_violations} rows with invalid OHLC relationships",
                affected_rows=int(total_violations),
                recommendation="Correct or remove rows with invalid OHLC"
            ))
    
    def print_summary(self):
        """Print validation summary"""
        if not self.issues:
            print("✓ All validation checks passed!")
            return
        
        print(f"\\n{'='*60}")
        print(f"Data Quality Report for {self.ticker}")
        print(f"{'='*60}\\n")
        
        by_level = {
            DataQualityLevel.CRITICAL: [],
            DataQualityLevel.HIGH: [],
            DataQualityLevel.MEDIUM: [],
            DataQualityLevel.LOW: []
        }
        
        for issue in self.issues:
            by_level[issue.level].append(issue)
        
        for level in [DataQualityLevel.CRITICAL, DataQualityLevel.HIGH, 
                     DataQualityLevel.MEDIUM, DataQualityLevel.LOW]:
            issues = by_level[level]
            if issues:
                print(f"\\n{level.value.upper()} Issues ({len(issues)}):")
                for i, issue in enumerate(issues, 1):
                    print(f"  {i}. {issue.description}")
                    print(f"     → {issue.recommendation}")
        
        # Overall assessment
        critical = len(by_level[DataQualityLevel.CRITICAL])
        high = len(by_level[DataQualityLevel.HIGH])
        
        print(f"\\n{'='*60}")
        if critical > 0:
            print("⚠️  CRITICAL: Data quality issues must be fixed before backtesting")
        elif high > 0:
            print("⚠️  WARNING: Significant data quality issues present")
        else:
            print("✓ Data quality is acceptable (minor issues only)")
        print(f"{'='*60}\\n")

# Example usage
def validate_stock_data(ticker: str, data: pd.DataFrame):
    """
    Validate historical stock data
    """
    validator = DataValidator(data, ticker)
    issues = validator.validate_all()
    
    return {
        'ticker': ticker,
        'total_issues': len(issues),
        'critical_issues': len([i for i in issues if i.level == DataQualityLevel.CRITICAL]),
        'high_issues': len([i for i in issues if i.level == DataQualityLevel.HIGH]),
        'is_usable': len([i for i in issues if i.level == DataQualityLevel.CRITICAL]) == 0
    }

# In production, run validation on all tickers before backtesting
\`\`\`

---

## Handling Corporate Actions

### Stock Splits

**The Problem:** A 2-for-1 stock split doubles the number of shares and halves the price. If not adjusted, it looks like the stock crashed 50%.

\`\`\`python
from typing import Dict, Tuple
from datetime import datetime, timedelta

class CorporateActionsHandler:
    """
    Handle stock splits, dividends, and other corporate actions
    """
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.splits: Dict[datetime, float] = {}
        self.dividends: Dict[datetime, float] = {}
    
    def add_split(self, date: datetime, ratio: float):
        """
        Add stock split
        
        Args:
            date: Split date
            ratio: Split ratio (2.0 for 2-for-1 split)
        """
        self.splits[date] = ratio
    
    def add_dividend(self, date: datetime, amount: float):
        """
        Add dividend payment
        
        Args:
            date: Ex-dividend date
            amount: Dividend per share
        """
        self.dividends[date] = amount
    
    def adjust_prices_for_splits(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust historical prices for stock splits
        
        Important: Adjust BACKWARDS from most recent split
        This ensures current prices are accurate
        
        Example:
        - Current price: $100
        - 2-for-1 split on Jan 1, 2023
        - Historical prices before Jan 1 should be divided by 2
        - So a $200 price on Dec 31, 2022 becomes $100 (split-adjusted)
        """
        df = data.copy()
        
        if not self.splits:
            return df
        
        # Sort splits by date (most recent first)
        sorted_splits = sorted(self.splits.items(), reverse=True)
        
        for split_date, ratio in sorted_splits:
            # Adjust all prices BEFORE split date
            mask = df.index < split_date
            
            # Divide prices by split ratio
            df.loc[mask, ['Open', 'High', 'Low', 'Close']] /= ratio
            
            # Multiply volume by split ratio
            df.loc[mask, 'Volume'] *= ratio
            
            print(f"Adjusted for {ratio}-for-1 split on {split_date}")
        
        return df
    
    def adjust_prices_for_dividends(self, 
                                    data: pd.DataFrame,
                                    include_dividends: bool = True) -> pd.DataFrame:
        """
        Adjust prices for dividends
        
        Two approaches:
        1. Price-adjusted: Adjust historical prices down by dividend amount
        2. Total return: Include dividend payments in returns
        
        Most backtests use approach #1 for consistency
        """
        if not include_dividends or not self.dividends:
            return data
        
        df = data.copy()
        
        # Sort dividends by date (most recent first)
        sorted_divs = sorted(self.dividends.items(), reverse=True)
        
        for div_date, amount in sorted_divs:
            # Find price on ex-dividend date
            try:
                ex_div_price = df.loc[div_date, 'Close']
            except KeyError:
                # Ex-dividend date not in dataset (weekend/holiday)
                # Find next available date
                future_dates = df.index[df.index > div_date]
                if len(future_dates) == 0:
                    continue
                ex_div_price = df.loc[future_dates[0], 'Close']
            
            # Calculate adjustment factor
            adjustment_factor = 1 - (amount / ex_div_price)
            
            # Adjust all prices BEFORE ex-dividend date
            mask = df.index < div_date
            df.loc[mask, ['Open', 'High', 'Low', 'Close']] *= adjustment_factor
            
            print(f"Adjusted for \${amount:.2f} dividend on {div_date}")
        
        return df
    
    def get_total_return_adjusted_prices(self,
    data: pd.DataFrame) -> pd.DataFrame:
"""
        Calculate total return adjusted prices
    (includes reinvested dividends)
        
        This is what most financial websites show as 'adjusted close'
"""
df = data.copy()
        
        # First adjust for splits
        df = self.adjust_prices_for_splits(df)
        
        # Then adjust for dividends
        df = self.adjust_prices_for_dividends(df, include_dividends = True)
        
        return df

# Real - world example: Apple stock
def demonstrate_apple_splits():
"""
    Apple has had multiple stock splits:
- June 16, 2014: 7 -for-1 split
    - August 31, 2020: 4 -for-1 split
    
    Without adjustments, historical charts would show massive drops
"""
    
    # Create sample data
dates = pd.date_range('2013-01-01', '2024-01-01', freq = 'D')
prices = pd.DataFrame({
    'Close': 100 * (1 + np.random.normal(0.001, 0.02, len(dates))).cumprod()
}, index = dates)
    
    # Simulate splits
handler = CorporateActionsHandler('AAPL')
handler.add_split(datetime(2014, 6, 16), 7.0)  # 7 -for-1
    handler.add_split(datetime(2020, 8, 31), 4.0)  # 4 -for-1
    
    # Before adjustment
price_before_splits = prices.loc['2013-12-31', 'Close']
price_after_splits = prices.loc['2023-12-31', 'Close']
    
    # Adjust
adjusted = handler.adjust_prices_for_splits(prices)

adjusted_price_before = adjusted.loc['2013-12-31', 'Close']
adjusted_price_after = adjusted.loc['2023-12-31', 'Close']

print("Without split adjustment:")
print(f"  2013: ${price_before_splits:.2f}
}")
print(f"  2023: ${price_after_splits:.2f}")
print(f"  Ratio: {price_before_splits/price_after_splits:.2f}x")
print("  → Looks like massive price drop!\\n")

print("With split adjustment:")
print(f"  2013: ${adjusted_price_before:.2f}")
print(f"  2023: ${adjusted_price_after:.2f}")
print(f"  Ratio: {adjusted_price_before/adjusted_price_after:.2f}x")
print("  → Now shows true economic performance")

return adjusted

# Handling edge cases
class SplitDetector:
"""
    Automatically detect likely stock splits in data
"""

@staticmethod
    def detect_splits(data: pd.DataFrame,
    threshold: float = 0.4) -> List[Tuple[datetime, float]]:
"""
        Detect potential stock splits

Args:
data: Price data
threshold: Minimum price change to flag(0.4 = 40 %)

Returns:
            List of(date, suspected_ratio) tuples
"""
pct_change = data['Close'].pct_change()
        
        # Large negative moves might be splits
potential_splits = []

for date, change in pct_change.items():
    if change < -threshold:
                # Estimate split ratio
ratio = 1 / (1 + change)

potential_splits.append((date, ratio))

return potential_splits

@staticmethod
    def verify_split(data: pd.DataFrame,
    date: datetime,
    window: int = 5) -> bool:
"""
        Verify if a price drop is actually a split
        
        Real splits have these characteristics:
- Volume spike
    - Clean price ratio(2: 1, 3: 1, etc.)
        - No news of company problems
"""
try:
idx = data.index.get_loc(date)
        except KeyError:
return False
        
        # Check volume spike
pre_volume = data.iloc[max(0, idx - window):idx]['Volume'].mean()
post_volume = data.iloc[idx: idx + window]['Volume'].mean()

volume_increase = post_volume / pre_volume if pre_volume > 0 else 0
        
        # Splits often have 2 - 5x volume
return volume_increase > 1.5
\`\`\`

---

## Survivorship Bias

**The Problem:** Using only stocks that exist today creates massive bias.

\`\`\`python
class SurvivorshipBiasHandler:
    """
    Handle survivorship bias in historical data
    """
    
    def __init__(self):
        self.constituents_history: Dict[datetime, List[str]] = {}
    
    def load_point_in_time_constituents(self, 
                                       index_name: str = 'SP500'):
        """
        Load historical index constituents
        
        For each date, know which stocks were in the index at that time
        
        Data sources:
        - Bloomberg Terminal (expensive but accurate)
        - Norgate Data (~$500/year, includes delisted stocks)
        - Compustat Point-in-Time Database
        - DIY: Scrape SEC filings and reconstruct
        """
        # Example: S&P 500 constituents over time
        # In production, load from database
        
        self.constituents_history = {
            datetime(2010, 1, 1): ['AAPL', 'MSFT', 'LEH', 'WM', ...],  # Includes Lehman
            datetime(2015, 1, 1): ['AAPL', 'MSFT', 'FB', 'GOOGL', ...],
            datetime(2020, 1, 1): ['AAPL', 'MSFT', 'TSLA', 'AMZN', ...],
        }
    
    def get_universe_at_date(self, date: datetime) -> List[str]:
        """
        Get list of stocks that should be in universe at given date
        
        Critical for avoiding survivorship bias
        """
        # Find most recent constituent list before date
        available_dates = [d for d in self.constituents_history.keys() if d <= date]
        
        if not available_dates:
            raise ValueError(f"No constituent data available for {date}")
        
        most_recent = max(available_dates)
        return self.constituents_history[most_recent]
    
    def backtest_with_point_in_time_universe(self,
                                             strategy,
                                             start_date: datetime,
                                             end_date: datetime):
        """
        Run backtest using point-in-time universe
        
        Each month, refresh the universe to match what was actually
        available at that time
        """
        results = []
        
        current_date = start_date
        while current_date <= end_date:
            # Get universe for this month
            universe = self.get_universe_at_date(current_date)
            
            # Run strategy on this universe
            monthly_results = strategy.run(universe, current_date)
            results.append(monthly_results)
            
            # Move to next month
            current_date += timedelta(days=30)
        
        return pd.concat(results)

# Impact analysis
def measure_survivorship_bias_impact(strategy):
    """
    Measure impact of survivorship bias
    """
    # Test 1: Current constituents only (biased)
    current_sp500 = load_current_sp500()
    biased_result = backtest(strategy, current_sp500, '2010-01-01', '2023-12-31')
    
    # Test 2: Point-in-time constituents (unbiased)
    unbiased_result = backtest_with_point_in_time(strategy, 'SP500', '2010-01-01', '2023-12-31')
    
    return {
        'biased_return': biased_result.annual_return,
        'unbiased_return': unbiased_result.annual_return,
        'bias_impact': biased_result.annual_return - unbiased_result.annual_return,
        'bias_multiplier': biased_result.annual_return / unbiased_result.annual_return
    }

# Typical results
survivorship_bias_impact = {
    'Long-only equity strategy': {
        'biased_return': 14.2,  # %
        'unbiased_return': 9.1,  # %
        'overstatement': '56% higher!'
    },
    'Short-term momentum': {
        'biased_return': 8.5,
        'unbiased_return': 6.2,
        'overstatement': '37% higher'
    },
    'Conclusion': 'Survivorship bias typically inflates returns by 30-60%'
}
\`\`\`

---

## Data Storage and Retrieval

\`\`\`python
import sqlite3
from pathlib import Path

class MarketDataStore:
    """
    Efficient storage and retrieval of historical market data
    """
    
    def __init__(self, db_path: str = 'market_data.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
    
    def create_tables(self):
        """
        Create database schema
        
        Design for fast queries:
        - Partition by ticker
        - Index on date
        - Separate table for corporate actions
        """
        cursor = self.conn.cursor()
        
        # OHLCV data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                ticker TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                adjusted_close REAL,
                PRIMARY KEY (ticker, date)
            )
        """)
        
        # Indexes for fast queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticker_date 
            ON ohlcv(ticker, date)
        """)
        
        # Corporate actions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS corporate_actions (
                ticker TEXT NOT NULL,
                date DATE NOT NULL,
                action_type TEXT NOT NULL,  -- 'split', 'dividend', 'merger'
                value REAL,
                details TEXT,
                PRIMARY KEY (ticker, date, action_type)
            )
        """)
        
        self.conn.commit()
    
    def store_ohlcv(self, ticker: str, data: pd.DataFrame):
        """Store OHLCV data"""
        data_to_store = data.copy()
        data_to_store['ticker'] = ticker
        data_to_store.to_sql('ohlcv', self.conn, if_exists='append', index=True)
    
    def get_ohlcv(self, ticker: str, 
                  start_date: str, 
                  end_date: str) -> pd.DataFrame:
        """
        Retrieve OHLCV data
        
        Fast queries using indexes
        """
        query = """
            SELECT date, open, high, low, close, volume, adjusted_close
            FROM ohlcv
            WHERE ticker = ?
              AND date >= ?
              AND date <= ?
            ORDER BY date
        """
        
        df = pd.read_sql_query(
            query,
            self.conn,
            params=[ticker, start_date, end_date],
            index_col='date',
            parse_dates=['date']
        )
        
        return df
    
    def get_multiple_tickers(self, 
                            tickers: List[str],
                            start_date: str,
                            end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Efficiently load multiple tickers
        """
        result = {}
        
        for ticker in tickers:
            result[ticker] = self.get_ohlcv(ticker, start_date, end_date)
        
        return result

# Storage cost comparison
storage_comparison = {
    'CSV files': {
        '500 tickers × 10 years': '~5 GB',
        'query_speed': 'Slow (need to load full files)',
        'cost': 'Free (local storage)'
    },
    'SQLite': {
        '500 tickers × 10 years': '~2 GB (compressed)',
        'query_speed': 'Fast (indexed queries)',
        'cost': 'Free (local storage)'
    },
    'PostgreSQL + TimescaleDB': {
        '500 tickers × 10 years': '~1.5 GB (columnar compression)',
        'query_speed': 'Very fast (optimized for time-series)',
        'cost': '$60/month (AWS RDS)'
    },
    'Parquet on S3': {
        '500 tickers × 10 years': '~1 GB (highly compressed)',
        'query_speed': 'Fast (columnar format, partitioned)',
        'cost': '$0.023/GB/month = $0.02/month + query costs'
    },
    'Recommendation': 'SQLite for single-user, TimescaleDB for production'
}
\`\`\`

---

## Common Pitfalls

1. **Using unadjusted prices**: Always adjust for splits and dividends
2. **Survivorship bias**: Use point-in-time constituent data
3. **Data vendor errors**: Validate data from multiple sources
4. **Time zone issues**: Always use exchange local time or UTC consistently
5. **Missing data**: Fill gaps or adjust strategy logic

---

## Production Checklist

- [ ] Data validated for quality issues
- [ ] Corporate actions (splits, dividends) properly adjusted
- [ ] Survivorship bias addressed with point-in-time data
- [ ] Multiple data sources cross-checked
- [ ] Automated data refresh pipeline
- [ ] Data stored efficiently with proper indexing
- [ ] Backup and disaster recovery plan
- [ ] Monitoring for data feed failures

---

## Summary

**Key Takeaways:**1. **Data quality** is critical - validate thoroughly
2. **Corporate actions** must be handled correctly
3. **Survivorship bias** can double apparent returns
4. **Point-in-time data** is essential for realistic backtests
5. **Storage strategy** matters for performance and cost

Production data pipelines are complex but essential. Invest the time upfront to get data right.
`,
    exercises: [
        {
            prompt:
                'Build a data validation pipeline that automatically checks new market data for quality issues and alerts when problems are detected. Include validation for splits, price anomalies, and missing data.',
            solution:
                '// Implementation includes: 1) DataValidator class with comprehensive checks, 2) Automated pipeline that runs daily, 3) Alert system for critical issues, 4) Dashboard showing data quality metrics, 5) Integration with data providers, 6) Logging and monitoring',
        },
        {
            prompt:
                'Implement a corporate actions handler that automatically detects and adjusts for stock splits and dividends. Compare adjusted vs unadjusted returns to show the impact.',
            solution:
                '// Implementation includes: 1) Split detection algorithm, 2) Dividend tracking, 3) Backward price adjustment, 4) Before/after comparison, 5) Visualization of impact, 6) Edge case handling (reverse splits, special dividends)',
        },
    ],
};

