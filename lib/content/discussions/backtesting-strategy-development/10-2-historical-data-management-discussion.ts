export const historicalDataManagementDiscussion = [
    {
        id: 1,
        question:
            "Your trading firm is switching data vendors from Provider A to Provider B to save costs. The new provider's historical data for Apple (AAPL) shows different prices than Provider A for the same dates in 2020. What process would you implement to investigate and resolve these discrepancies? How would you ensure data quality when migrating terabytes of historical data?",
        answer: `## Comprehensive Data Vendor Migration Strategy:

### The Problem

Data discrepancies between vendors are common and can be caused by:
1. Different corporate action adjustment methodologies
2. Different data sources (exchanges, market makers)
3. Timing differences (exchange local vs UTC)
4. Errors in one or both datasets

### Investigation Process

\`\`\`python
class DataVendorComparison:
    """
    Compare data from two vendors to identify and resolve discrepancies
    """
    
    def __init__(self, vendor_a_data: pd.DataFrame, vendor_b_data: pd.DataFrame):
        self.vendor_a = vendor_a_data
        self.vendor_b = vendor_b_data
        self.discrepancies = []
    
    def run_comprehensive_comparison(self, ticker: str) -> dict:
        """
        Compare all aspects of the data
        """
        print(f"Comparing {ticker} data from both vendors...\\n")
        
        # 1. Compare date ranges
        date_comparison = self.compare_date_coverage()
        
        # 2. Compare prices
        price_comparison = self.compare_prices()
        
        # 3. Compare volumes
        volume_comparison = self.compare_volumes()
        
        # 4. Compare corporate actions
        corporate_actions_comparison = self.compare_corporate_actions(ticker)
        
        # 5. Generate recommendations
        recommendations = self.generate_recommendations()
        
        return {
            'ticker': ticker,
            'date_comparison': date_comparison,
            'price_comparison': price_comparison,
            'volume_comparison': volume_comparison,
            'corporate_actions': corporate_actions_comparison,
            'recommendations': recommendations
        }
    
    def compare_prices(self) -> dict:
        """
        Compare close prices between vendors
        """
        # Merge on date
        merged = pd.merge(
            self.vendor_a[['Close']],
            self.vendor_b[['Close']],
            left_index=True,
            right_index=True,
            suffixes=('_A', '_B'),
            how='inner'
        )
        
        # Calculate differences
        merged['Diff'] = merged['Close_A'] - merged['Close_B']
        merged['Pct_Diff'] = (merged['Diff'] / merged['Close_A']) * 100
        
        # Find significant discrepancies (> 1%)
        significant = merged[abs(merged['Pct_Diff']) > 1.0]
        
        return {
            'total_dates_compared': len(merged),
            'matching_dates': len(merged[abs(merged['Pct_Diff']) < 0.01]),
            'small_differences': len(merged[(abs(merged['Pct_Diff']) >= 0.01) & (abs(merged['Pct_Diff']) < 1.0)]),
            'significant_differences': len(significant),
            'max_difference_pct': abs(merged['Pct_Diff']).max(),
            'significant_dates': significant.index.tolist()[:10],  # First 10
            'avg_difference_pct': abs(merged['Pct_Diff']).mean()
        }
    
    def compare_corporate_actions(self, ticker: str) -> dict:
        """
        Compare corporate actions (splits, dividends) from both vendors
        
        This is often the source of price discrepancies
        """
        # In production, fetch corporate actions from both vendors
        actions_a = self.fetch_corporate_actions_from_vendor_a(ticker)
        actions_b = self.fetch_corporate_actions_from_vendor_b(ticker)
        
        # Find differences
        only_in_a = set(actions_a.keys()) - set(actions_b.keys())
        only_in_b = set(actions_b.keys()) - set(actions_a.keys())
        
        common_dates = set(actions_a.keys()) & set(actions_b.keys())
        different_values = [
            date for date in common_dates 
            if actions_a[date] != actions_b[date]
        ]
        
        return {
            'actions_only_in_vendor_a': len(only_in_a),
            'actions_only_in_vendor_b': len(only_in_b),
            'actions_with_different_values': len(different_values),
            'details': {
                'vendor_a_unique': list(only_in_a)[:5],
                'vendor_b_unique': list(only_in_b)[:5],
                'different_values': list(different_values)[:5]
            }
        }
    
    def fetch_corporate_actions_from_vendor_a(self, ticker: str) -> dict:
        """Fetch from Vendor A"""
        # Implementation would call Vendor A API
        return {}
    
    def fetch_corporate_actions_from_vendor_b(self, ticker: str) -> dict:
        """Fetch from Vendor B"""
        # Implementation would call Vendor B API
        return {}
    
    def generate_recommendations(self) -> list:
        """
        Generate actionable recommendations based on comparison
        """
        recommendations = []
        
        price_comp = self.compare_prices()
        
        if price_comp['significant_differences'] > 10:
            recommendations.append({
                'severity': 'HIGH',
                'issue': f"{price_comp['significant_differences']} dates with >1% price differences",
                'action': 'Cross-check with a third data source (Bloomberg, Reuters)',
                'affected_backtest_impact': 'Potentially significant - returns could differ by 2-5%'
            })
        
        if price_comp['significant_differences'] > 0:
            recommendations.append({
                'severity': 'MEDIUM',
                'issue': 'Price discrepancies found',
                'action': 'Investigate corporate actions handling between vendors',
                'affected_backtest_impact': 'Moderate - may affect specific time periods'
            })
        
        return recommendations

# Migration Process
class DataMigrationPipeline:
    """
    Safe migration from one vendor to another
    """
    
    def execute_migration(self,
                         tickers: list,
                         old_vendor: str,
                         new_vendor: str,
                         validation_threshold: float = 0.01):
        """
        Migrate with validation
        
        Args:
            tickers: List of tickers to migrate
            old_vendor: Current vendor
            new_vendor: New vendor
            validation_threshold: Maximum acceptable price difference (1%)
        """
        results = []
        failed_tickers = []
        
        for ticker in tickers:
            print(f"Processing {ticker}...")
            
            # 1. Load data from both vendors
            old_data = self.load_from_vendor(ticker, old_vendor)
            new_data = self.load_from_vendor(ticker, new_vendor)
            
            # 2. Run comparison
            comparison = DataVendorComparison(old_data, new_data)
            result = comparison.run_comprehensive_comparison(ticker)
            
            # 3. Validate quality
            if result['price_comparison']['max_difference_pct'] > validation_threshold:
                print(f"  ⚠️  WARNING: {ticker} has significant differences")
                failed_tickers.append(ticker)
            else:
                print(f"  ✓ {ticker} passed validation")
            
            results.append(result)
        
        # 4. Generate migration report
        report = self.generate_migration_report(results, failed_tickers)
        
        return report
    
    def generate_migration_report(self, results: list, failed: list) -> dict:
        """Generate comprehensive migration report"""
        total = len(results)
        passed = total - len(failed)
        
        return {
            'total_tickers': total,
            'passed_validation': passed,
            'failed_validation': len(failed),
            'pass_rate': (passed / total) * 100 if total > 0 else 0,
            'failed_tickers': failed,
            'recommendation': (
                'Proceed with migration' if len(failed) == 0
                else f'Investigate {len(failed)} tickers before migrating'
            )
        }
    
    def load_from_vendor(self, ticker: str, vendor: str) -> pd.DataFrame:
        """Load data from specified vendor"""
        # Implementation would load from vendor API/database
        pass
\`\`\`

### Data Quality Assurance Process

**Phase 1: Sampling (Week 1)**
1. Select 50 representative tickers (large-cap, mid-cap, small-cap)
2. Compare 10 years of data for each
3. Identify patterns in discrepancies

**Phase 2: Root Cause Analysis (Week 2)**

\`\`\`python
def investigate_discrepancy_patterns(comparison_results: list) -> dict:
    """
    Analyze patterns in discrepancies to find root cause
    """
    patterns = {
        'split_adjustments': 0,
        'dividend_adjustments': 0,
        'timezone_issues': 0,
        'data_source_differences': 0,
        'vendor_errors': 0
    }
    
    for result in comparison_results:
        # Check if discrepancies cluster around corporate action dates
        # Check if discrepancies are proportional(suggesting adjustment methodology)
        # Check if discrepancies are at market open / close(suggesting timezone)
pass

return patterns
    \`\`\`

**Phase 3: Validation (Week 3)**
1. Cross-check with third authoritative source (Bloomberg, exchange data)
2. Test backtests on both datasets
3. Compare results

**Phase 4: Migration (Week 4)**
1. Migrate data for validated tickers
2. Set up dual-source validation for first month
3. Monitor for issues

### Ensuring Quality at Scale

**For Terabytes of Data:**

\`\`\`python
class ScalableDataValidation:
    """
    Validate large datasets efficiently
    """
    
    def __init__(self):
        self.batch_size = 100  # Process 100 tickers at a time
    
    def validate_at_scale(self, all_tickers: list):
        """
        Validate thousands of tickers efficiently
        """
        # 1. Statistical sampling
        sample_size = max(100, int(len(all_tickers) * 0.05))  # 5% or min 100
        sample = random.sample(all_tickers, sample_size)
        
        # 2. Validate sample thoroughly
        sample_results = [self.validate_ticker(t) for t in sample]
        
        # 3. If sample passes, validate rest with lighter checks
        if self.sample_quality_acceptable(sample_results):
            # Parallel validation of remaining tickers
            remaining = [t for t in all_tickers if t not in sample]
            self.parallel_validation(remaining, light=True)
        else:
            # Sample failed - full validation needed
            print("Sample validation failed - running full validation")
            self.parallel_validation(all_tickers, light=False)
    
    def parallel_validation(self, tickers: list, light: bool = False):
        """
        Validate in parallel using multiprocessing
        """
        from multiprocessing import Pool
        
        with Pool(processes=8) as pool:
            validation_func = (
                self.light_validation if light 
                else self.full_validation
            )
            results = pool.map(validation_func, tickers)
        
        return results
    
    def light_validation(self, ticker: str) -> dict:
        """
        Quick validation (price continuity, no nulls)
        """
        # Fast checks only
        pass
    
    def full_validation(self, ticker: str) -> dict:
        """
        Comprehensive validation
        """
        # All checks
        pass
\`\`\`

### Recommendations for Production

**1. Maintain Dual Sources Initially**
- Keep both vendors active for 1-3 months
- Run all backtests on both datasets
- Compare results before committing to new vendor

**2. Automated Monitoring**

\`\`\`python
class DataQualityMonitor:
    """
    Continuous monitoring after migration
"""
    
    def daily_quality_check(self):
"""
        Run daily after market close
"""
        # 1. Check for missing data
        # 2. Validate corporate actions
        # 3. Compare against previous day's close
        # 4. Alert on anomalies
pass
    ```

            ** 3. Fallback Plan**
    - Keep old vendor data archived
    - Document all discrepancies
    - Have process to roll back if needed

### Expected Outcome

    ** Typical findings:**
        - 95 % of tickers match within 0.1 %
            - 4 % have minor differences(<1%) due to rounding
                - 1 % have significant differences requiring investigation

                    ** Common root causes:**
                        1. Different split adjustment methodologies(most common)
2. Different dividend adjustment approaches
3. Time zone handling for international stocks
4. Data source differences(some vendors use last trade, others use mid - price)

### Cost - Benefit Analysis

    ** Vendor A(expensive but trusted):**
        - Cost: $50K / year
            - Data quality: Excellent
                - Historical depth: 30 years

                    ** Vendor B(cheaper):**
                        - Cost: $15K / year
                            - Data quality: Good(after validation)
                                - Historical depth: 20 years

                                    ** Migration cost:**
                                        - Engineering time: 4 weeks
                                            - Risk of backtest disruption: Medium
                                                - Savings: $35K / year

                                                    ** Recommendation:**
                                                        Migration is worth it IF:
1. Validation shows < 1 % of tickers have significant differences
2. Differences are understood and documented
3. You have resources to handle migration properly
4. You're not in middle of critical strategy development

    ** Timeline:**
        - Urgent migration: 4 weeks(risky)
            - Careful migration: 8 - 12 weeks(recommended)
                - Parallel operation: 3 - 6 months(safest)
`,
  },
{
    id: 2,
        question:
    "Design a point-in-time database schema that tracks S&P 500 constituent changes over 20 years, ensuring your backtests never use future information about which stocks were/weren't in the index. Include how you'd handle additions, deletions, and sector reclassifications. Provide SQL schema and explain query patterns.",
        answer: `## Point -in -Time Database Design:

### The Challenge

When backtesting on an index like the S & P 500, you need to know which stocks were in the index ** at each point in time **.Using today's constituents creates survivorship bias.

    ** Example Problem:**
        - Today(2024): Tesla is in S & P 500
            - Backtest 2015 - 2020: Should NOT include Tesla(added Dec 2020)
                - But naive backtest using "current S&P 500" would include it

### Database Schema

\`\`\`sql
-- Core tables for point-in-time index tracking

-- 1. Universe of all stocks (ever tracked)
CREATE TABLE stocks (
    ticker VARCHAR(10) PRIMARY KEY,
    company_name VARCHAR(200),
    exchange VARCHAR(20),
    first_traded_date DATE,
    last_traded_date DATE,  -- NULL if still trading
    delisting_reason VARCHAR(100),  -- bankruptcy, merger, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_stocks_trading_dates ON stocks (first_traded_date, last_traded_date);

-- 2. Index definitions
CREATE TABLE indices (
    index_id SERIAL PRIMARY KEY,
    index_name VARCHAR(50) UNIQUE NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert S&P 500
INSERT INTO indices (index_name, description) 
VALUES ('SP500', 'S&P 500 Index');

-- 3. Point-in-time constituent tracking
CREATE TABLE index_constituents (
    id SERIAL PRIMARY KEY,
    index_id INTEGER REFERENCES indices(index_id),
    ticker VARCHAR(10) REFERENCES stocks(ticker),
    start_date DATE NOT NULL,  -- Date added to index
    end_date DATE,              -- Date removed from index (NULL if still in)
    reason_added VARCHAR(100),  -- IPO, size threshold, sector need
    reason_removed VARCHAR(100), -- delisted, size decline, sector rebal
    
    -- Metadata at time of addition
    market_cap_at_addition BIGINT,
    sector_at_addition VARCHAR(50),
    
    CONSTRAINT valid_date_range CHECK (end_date IS NULL OR end_date >= start_date)
);

-- Indexes for fast point-in-time queries
CREATE INDEX idx_constituents_ticker_dates ON index_constituents (ticker, start_date, end_date);
CREATE INDEX idx_constituents_index_dates ON index_constituents (index_id, start_date, end_date);
CREATE INDEX idx_constituents_active ON index_constituents (index_id, start_date, end_date) 
    WHERE end_date IS NULL;  -- Active constituents only

-- 4. Sector classifications (GICS)
CREATE TABLE sector_classifications (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) REFERENCES stocks(ticker),
    effective_date DATE NOT NULL,
    sector VARCHAR(50) NOT NULL,
    industry_group VARCHAR(50),
    industry VARCHAR(50),
    sub_industry VARCHAR(50),
    
    -- Tracks sector changes over time
    changed_from VARCHAR(50),
    change_reason VARCHAR(100)
);

CREATE INDEX idx_sector_ticker_date ON sector_classifications (ticker, effective_date);

-- 5. Market data with point-in-time flags
CREATE TABLE daily_prices (
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(10, 2),
    high DECIMAL(10, 2),
    low DECIMAL(10, 2),
    close DECIMAL(10, 2),
    volume BIGINT,
    adjusted_close DECIMAL(10, 2),
    
    -- Point-in-time flags
    in_sp500 BOOLEAN DEFAULT FALSE,  -- Was in S&P 500 this day?
    market_cap BIGINT,                -- Market cap as of this day
    
    PRIMARY KEY (ticker, date)
);

CREATE INDEX idx_prices_date ON daily_prices (date);
CREATE INDEX idx_prices_sp500 ON daily_prices (date, in_sp500) WHERE in_sp500 = TRUE;
\`\`\`

### Key Query Patterns

**Query 1: Get S&P 500 constituents at specific date**

\`\`\`sql
-- Get all stocks in S&P 500 on January 1, 2015
SELECT ic.ticker, s.company_name, ic.sector_at_addition
FROM index_constituents ic
JOIN stocks s ON ic.ticker = s.ticker
JOIN indices i ON ic.index_id = i.index_id
WHERE i.index_name = 'SP500'
  AND ic.start_date <= '2015-01-01'
  AND (ic.end_date IS NULL OR ic.end_date > '2015-01-01');

-- Result includes stocks that:
-- - Were added before or on 2015-01-01
-- - Were not removed before 2015-01-01
-- - This is POINT-IN-TIME correct
\`\`\`

**Query 2: Get constituent changes in date range**

\`\`\`sql
-- Find all additions and deletions between 2015-2020
SELECT 
    ticker,
    start_date as event_date,
    'ADDED' as event_type,
    reason_added as reason,
    market_cap_at_addition as market_cap
FROM index_constituents
WHERE index_id = (SELECT index_id FROM indices WHERE index_name = 'SP500')
  AND start_date BETWEEN '2015-01-01' AND '2020-12-31'

UNION ALL

SELECT 
    ticker,
    end_date as event_date,
    'REMOVED' as event_type,
    reason_removed as reason,
    NULL as market_cap
FROM index_constituents
WHERE index_id = (SELECT index_id FROM indices WHERE index_name = 'SP500')
  AND end_date BETWEEN '2015-01-01' AND '2020-12-31'

ORDER BY event_date;
\`\`\`

**Query 3: Backtest-friendly query**

\`\`\`sql
-- For backtesting: Get all data for stocks that were in index
-- during the backtest period (even if not in index entire time)

WITH backtest_period AS (
    SELECT '2015-01-01'::DATE as start_date, '2020-12-31'::DATE as end_date
),
relevant_constituents AS (
    -- Get all stocks that were in S&P 500 at ANY point during backtest
    SELECT DISTINCT ticker
    FROM index_constituents ic
    WHERE index_id = (SELECT index_id FROM indices WHERE index_name = 'SP500')
      AND start_date <= (SELECT end_date FROM backtest_period)
      AND (end_date IS NULL OR end_date >= (SELECT start_date FROM backtest_period))
)
SELECT 
    dp.ticker,
    dp.date,
    dp.close,
    dp.volume,
    -- Point-in-time membership flag
    CASE 
        WHEN EXISTS (
            SELECT 1 FROM index_constituents ic2
            WHERE ic2.ticker = dp.ticker
              AND ic2.index_id = (SELECT index_id FROM indices WHERE index_name = 'SP500')
              AND ic2.start_date <= dp.date
              AND (ic2.end_date IS NULL OR ic2.end_date > dp.date)
        ) THEN TRUE
        ELSE FALSE
    END as was_in_sp500
FROM daily_prices dp
WHERE dp.ticker IN (SELECT ticker FROM relevant_constituents)
  AND dp.date BETWEEN (SELECT start_date FROM backtest_period) 
                  AND (SELECT end_date FROM backtest_period)
ORDER BY dp.date, dp.ticker;
\`\`\`

### Python Integration

\`\`\`python
class PointInTimeIndex:
    """
    Interface for point-in-time index queries
    """
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def get_constituents_at_date(self, 
                                index_name: str, 
                                date: datetime) -> List[str]:
        """
        Get index constituents as of specific date
        
        This is the core function for avoiding survivorship bias
        """
        query = """
            SELECT ic.ticker
            FROM index_constituents ic
            JOIN indices i ON ic.index_id = i.index_id
            WHERE i.index_name = %s
              AND ic.start_date <= %s
              AND (ic.end_date IS NULL OR ic.end_date > %s)
        """
        
        result = pd.read_sql_query(
            query,
            self.db,
            params=[index_name, date, date]
        )
        
        return result['ticker'].tolist()
    
    def backtest_with_rolling_universe(self,
                                       strategy,
                                       index_name: str,
                                       start_date: datetime,
                                       end_date: datetime,
                                       rebalance_frequency: str = 'monthly'):
        """
        Run backtest with universe that updates at each rebalance
        
        This prevents survivorship bias by only including stocks
        that were actually available at each point in time
        """
        current_date = start_date
        results = []
        
        while current_date <= end_date:
            # Get universe as of this date (point-in-time)
            universe = self.get_constituents_at_date(index_name, current_date)
            
            # Run strategy on this universe
            period_result = strategy.run(
                universe=universe,
                start=current_date,
                end=current_date + self.get_period_delta(rebalance_frequency)
            )
            
            results.append(period_result)
            
            # Move to next rebalance date
            current_date += self.get_period_delta(rebalance_frequency)
        
        return pd.concat(results)
    
    def get_period_delta(self, frequency: str) -> timedelta:
        """Convert frequency to timedelta"""
        if frequency == 'daily':
            return timedelta(days=1)
        elif frequency == 'weekly':
            return timedelta(days=7)
        elif frequency == 'monthly':
            return timedelta(days=30)
        elif frequency == 'quarterly':
            return timedelta(days=90)
        else:
            raise ValueError(f"Unknown frequency: {frequency}")
    
    def get_sector_at_date(self, 
                          ticker: str, 
                          date: datetime) -> str:
        """
        Get stock's sector classification as of date
        
        Important: Sectors change over time (e.g., Google moved
        from Tech to Communication Services in 2018)
        """
        query = """
            SELECT sector
            FROM sector_classifications
            WHERE ticker = %s
              AND effective_date <= %s
            ORDER BY effective_date DESC
            LIMIT 1
        """
        
        result = pd.read_sql_query(
            query,
            self.db,
            params=[ticker, date]
        )
        
        return result['sector'].iloc[0] if len(result) > 0 else None
\`\`\`

### Handling Edge Cases

**1. Same-Day Addition and Removal**
```sql
--Stock added and removed same day(rare but possible)
--Use start_date <= date AND end_date > date(not >=)
--This excludes stocks removed on that day
    ```

**2. Sector Reclassifications**
```sql
--Google moved from Technology to Communication Services on Sept 24, 2018
INSERT INTO sector_classifications(ticker, effective_date, sector, changed_from)
VALUES('GOOGL', '2018-09-24', 'Communication Services', 'Technology');

--Query automatically uses correct sector for each date
    ```

**3. Corporate Actions (Mergers)**
```sql
--When Stock A acquires Stock B
--Stock B is removed from index
UPDATE index_constituents
SET end_date = '2020-06-15',
    reason_removed = 'Acquired by Stock A'
WHERE ticker = 'STOCKB';

--Stock A may increase weight but stays in index
    ```

### Performance Optimization

**For large backtests:**

1. **Pre-materialize membership flags**
```sql
--Run nightly job to update in_sp500 flag in daily_prices
UPDATE daily_prices dp
SET in_sp500 = EXISTS(
    SELECT 1 FROM index_constituents ic
    WHERE ic.ticker = dp.ticker
      AND ic.index_id = 1  -- S & P 500
      AND ic.start_date <= dp.date
      AND(ic.end_date IS NULL OR ic.end_date > dp.date)
);

--Now queries are much faster(no JOIN needed)
    ```

2. **Partition tables by year**
```sql
--For TimescaleDB or native PostgreSQL partitioning
CREATE TABLE daily_prices_2020 PARTITION OF daily_prices
FOR VALUES FROM('2020-01-01') TO('2021-01-01');
```

### Data Sources

**Where to get constituent history:**
1. **Bloomberg Terminal**: Most accurate, $24K/year
2. **S&P Dow Jones Indices**: Official source, expensive
3. **Norgate Data**: ~$500/year, includes delisted stocks
4. **Wikipedia + Manual**: Free but requires maintenance
5. **SEC Form N-CSR**: Mutual fund holdings (indirect)

### Conclusion

This schema enables:
- ✅ Point-in-time correct backtests
- ✅ No survivorship bias
- ✅ Sector rotation strategies
- ✅ Index rebalancing studies
- ✅ Historical constituent analysis

**Critical for production trading systems.**
`,
  },
{
    id: 3,
        question:
    "You're building a market data pipeline that ingests data from multiple sources (primary vendor, backup vendor, exchange direct feeds). How would you design a system to automatically detect and handle data quality issues in real-time? Include your approach to handling conflicts when sources disagree, data gaps, and corporate actions that aren't immediately reported. What monitoring and alerting would you implement?",
        answer: `## Real-Time Data Quality System Design:

### Architecture Overview

\`\`\`
┌─────────────────────────────────────────────────────────┐
│                Input Layer (Data Sources)                │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐         │
│  │Primary   │  │ Backup   │  │  Exchange    │         │
│  │ Vendor   │  │ Vendor   │  │ Direct Feed  │         │
│  └──────────┘  └──────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────┘
               │         │              │
               ▼         ▼              ▼
┌─────────────────────────────────────────────────────────┐
│            Normalization & Validation Layer              │
│  ┌──────────────────────────────────────────────────┐  │
│  │ • Timestamp standardization (UTC)                │  │
│  │ • Price format normalization                     │  │
│  │ • Symbol mapping (AAPL vs AAPL.O)              │  │
│  │ • Real-time validation checks                    │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│            Conflict Resolution Layer                     │
│  ┌──────────────────────────────────────────────────┐  │
│  │ • Compare prices from multiple sources           │  │
│  │ • Resolve conflicts with voting logic            │  │
│  │ • Flag suspicious data                           │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Storage & Distribution Layer                │
│  ┌────────────┐  ┌──────────────┐  ┌────────────┐     │
│  │TimescaleDB │  │    Redis     │  │  Kafka     │     │
│  │(Historical)│  │   (Cache)    │  │ (Stream)   │     │
│  └────────────┘  └──────────────┘  └────────────┘     │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│           Monitoring & Alerting Layer                    │
│  ┌────────────┐  ┌──────────────┐  ┌────────────┐     │
│  │Prometheus  │  │ PagerDuty    │  │ Dashboard  │     │
│  │(Metrics)   │  │  (Alerts)    │  │ (Grafana)  │     │
│  └────────────┘  └──────────────┘  └────────────┘     │
└─────────────────────────────────────────────────────────┘
\`\`\`

### Core Components

\`\`\`python
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd

class DataSource(Enum):
    """Data source types"""
    PRIMARY_VENDOR = "primary"
    BACKUP_VENDOR = "backup"
    EXCHANGE_DIRECT = "exchange"

class DataQualityIssueType(Enum):
    """Types of data quality issues"""
    MISSING_DATA = "missing"
    PRICE_DISCREPANCY = "price_discrepancy"
    STALE_DATA = "stale"
    INVALID_VALUE = "invalid"
    LARGE_MOVE = "large_move"
    VOLUME_ANOMALY = "volume_anomaly"

@dataclass
class MarketDataPoint:
    """Single market data point from a source"""
    source: DataSource
    ticker: str
    timestamp: datetime
    bid: Optional[float]
    ask: Optional[float]
    last: float
    volume: int
    exchange: str

@dataclass
class ConsolidatedQuote:
    """Consolidated quote from multiple sources"""
    ticker: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: int
    sources_used: List[DataSource]
    confidence_score: float  # 0-1, based on source agreement
    flags: List[str]  # Quality warnings

class RealTimeDataQualityMonitor:
    """
    Real-time monitoring and validation of market data
    """
    
    def __init__(self):
        self.price_tolerance = 0.001  # 0.1% tolerance
        self.latency_threshold = 1.0  # 1 second max latency
        self.alert_system = AlertSystem()
        self.metrics = MetricsCollector()
        
        # Cache recent data for validation
        self.recent_quotes: Dict[str, List[MarketDataPoint]] = {}
        self.last_valid_price: Dict[str, float] = {}
    
    def process_incoming_quote(self, quote: MarketDataPoint) -> Optional[ConsolidatedQuote]:
        """
        Process incoming quote with validation
        
        Returns:
            Consolidated quote if valid, None if rejected
        """
        ticker = quote.ticker
        
        # 1. Basic validation
        if not self.validate_quote_basic(quote):
            self.metrics.increment('quotes_rejected_invalid')
            return None
        
        # 2. Check for staleness
        if self.is_stale(quote):
            self.alert_system.alert(
                severity='WARNING',
                issue=DataQualityIssueType.STALE_DATA,
                details=f"{ticker} data is stale from {quote.source.value}"
            )
        
        # 3. Store in recent quotes buffer
        if ticker not in self.recent_quotes:
            self.recent_quotes[ticker] = []
        self.recent_quotes[ticker].append(quote)
        
        # Keep only last 10 seconds of data
        cutoff = datetime.now() - timedelta(seconds=10)
        self.recent_quotes[ticker] = [
            q for q in self.recent_quotes[ticker]
            if q.timestamp >= cutoff
        ]
        
        # 4. Get quotes from all sources for this ticker
        all_sources = self.recent_quotes[ticker]
        
        # 5. Consolidate and resolve conflicts
        consolidated = self.consolidate_quotes(ticker, all_sources)
        
        # 6. Validate consolidated quote
        if self.validate_consolidated(ticker, consolidated):
            self.last_valid_price[ticker] = consolidated.last
            return consolidated
        else:
            self.metrics.increment('quotes_rejected_consolidation')
            return None
    
    def validate_quote_basic(self, quote: MarketDataPoint) -> bool:
        """
        Basic validation of single quote
        """
        # Check for null/negative prices
        if quote.last <= 0:
            return False
        
        if quote.bid and quote.ask:
            # Bid must be less than ask
            if quote.bid >= quote.ask:
                return False
            
            # Spread must be reasonable (< 5%)
            spread_pct = (quote.ask - quote.bid) / quote.bid
            if spread_pct > 0.05:
                self.alert_system.alert(
                    severity='WARNING',
                    issue='WIDE_SPREAD',
                    details=f"{quote.ticker} spread {spread_pct*100:.2f}%"
                )
        
        # Check for unrealistic prices
        if self.last_valid_price.get(quote.ticker):
            last_price = self.last_valid_price[quote.ticker]
            change_pct = abs(quote.last - last_price) / last_price
            
            # Flag moves > 10% in one tick
            if change_pct > 0.10:
                self.alert_system.alert(
                    severity='HIGH',
                    issue=DataQualityIssueType.LARGE_MOVE,
                    details=f"{quote.ticker} moved {change_pct*100:.1f}% in one tick"
                )
                # Don't automatically reject - could be legit
        
        return True
    
    def consolidate_quotes(self, 
                          ticker: str,
                          quotes: List[MarketDataPoint]) -> ConsolidatedQuote:
        """
        Consolidate quotes from multiple sources
        
        Strategy:
        1. If all sources agree (within tolerance), use any
        2. If 2/3 agree, use majority
        3. If all disagree, use most reliable source (exchange > primary > backup)
        """
        if not quotes:
            return None
        
        # Group by source
        by_source = {}
        for q in quotes:
            if q.source not in by_source or q.timestamp > by_source[q.source].timestamp:
                by_source[q.source] = q
        
        if len(by_source) == 1:
            # Only one source available
            quote = list(by_source.values())[0]
            return ConsolidatedQuote(
                ticker=ticker,
                timestamp=quote.timestamp,
                bid=quote.bid or 0,
                ask=quote.ask or 0,
                last=quote.last,
                volume=quote.volume,
                sources_used=[quote.source],
                confidence_score=0.7,  # Lower confidence with single source
                flags=['single_source']
            )
        
        # Multiple sources - check for agreement
        prices = [q.last for q in by_source.values()]
        price_range = max(prices) - min(prices)
        avg_price = sum(prices) / len(prices)
        disagreement = price_range / avg_price if avg_price > 0 else 0
        
        if disagreement < self.price_tolerance:
            # Sources agree - use most recent
            most_recent = max(by_source.values(), key=lambda q: q.timestamp)
            return ConsolidatedQuote(
                ticker=ticker,
                timestamp=most_recent.timestamp,
                bid=most_recent.bid or 0,
                ask=most_recent.ask or 0,
                last=most_recent.last,
                volume=most_recent.volume,
                sources_used=list(by_source.keys()),
                confidence_score=1.0,
                flags=[]
            )
        else:
            # Sources disagree - resolve conflict
            resolved = self.resolve_price_conflict(ticker, by_source)
            
            self.alert_system.alert(
                severity='MEDIUM',
                issue=DataQualityIssueType.PRICE_DISCREPANCY,
                details=f"{ticker} sources disagree by {disagreement*100:.2f}%"
            )
            
            return resolved
    
    def resolve_price_conflict(self,
                               ticker: str,
                               sources: Dict[DataSource, MarketDataPoint]) -> ConsolidatedQuote:
        """
        Resolve conflicting prices from multiple sources
        
        Priority: Exchange Direct > Primary Vendor > Backup Vendor
        """
        # Priority order
        priority = [
            DataSource.EXCHANGE_DIRECT,
            DataSource.PRIMARY_VENDOR,
            DataSource.BACKUP_VENDOR
        ]
        
        # Use highest priority source available
        for source_type in priority:
            if source_type in sources:
                quote = sources[source_type]
                return ConsolidatedQuote(
                    ticker=ticker,
                    timestamp=quote.timestamp,
                    bid=quote.bid or 0,
                    ask=quote.ask or 0,
                    last=quote.last,
                    volume=quote.volume,
                    sources_used=[source_type],
                    confidence_score=0.8,  # Medium confidence due to conflict
                    flags=['price_conflict_resolved']
                )
        
        # Fallback: use average (should never happen)
        avg_price = sum(q.last for q in sources.values()) / len(sources)
        return ConsolidatedQuote(
            ticker=ticker,
            timestamp=datetime.now(),
            bid=0,
            ask=0,
            last=avg_price,
            volume=0,
            sources_used=list(sources.keys()),
            confidence_score=0.5,
            flags=['conflict_averaged']
        )
    
    def is_stale(self, quote: MarketDataPoint) -> bool:
        """Check if data is stale"""
        age = (datetime.now() - quote.timestamp).total_seconds()
        return age > self.latency_threshold
    
    def validate_consolidated(self, 
                             ticker: str,
                             quote: ConsolidatedQuote) -> bool:
        """
        Final validation of consolidated quote
        """
        if not quote:
            return False
        
        # Check confidence score
        if quote.confidence_score < 0.5:
            return False
        
        # Additional checks based on flags
        if 'conflict_averaged' in quote.flags:
            # Extra scrutiny for averaged prices
            pass
        
        return True

class CorporateActionDetector:
    """
    Detect corporate actions from market data patterns
    """
    
    def detect_likely_split(self, 
                           ticker: str,
                           price_history: pd.Series) -> Optional[dict]:
        """
        Detect likely stock split from price pattern
        
        Splits show:
        - Clean price ratio (2:1, 3:1, etc.)
        - Volume spike
        - Overnight gap (not intraday)
        """
        # Look for large overnight gaps
        returns = price_history.pct_change()
        large_gaps = returns[abs(returns) > 0.3]
        
        for date, return_val in large_gaps.items():
            # Check if ratio is clean (2:1, 3:1, etc.)
            ratio = 1 / (1 + return_val) if return_val < 0 else 1 + return_val
            
            # Check if close to common split ratios
            common_ratios = [2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 0.5, 0.33, 0.25]
            for common in common_ratios:
                if abs(ratio - common) < 0.05:  # Within 5% of common ratio
                    return {
                        'ticker': ticker,
                        'date': date,
                        'suspected_ratio': common,
                        'confidence': 'high',
                        'price_before': price_history[date - timedelta(days=1)],
                        'price_after': price_history[date]
                    }
        
        return None

class AlertSystem:
    """
    Alert system for data quality issues
    """
    
    def __init__(self):
        self.alert_thresholds = {
            'CRITICAL': 0,  # Alert immediately
            'HIGH': 5,      # Alert after 5 occurrences
            'MEDIUM': 10,   # Alert after 10 occurrences
            'WARNING': 50   # Alert after 50 occurrences
        }
        self.alert_counts: Dict[str, int] = {}
    
    def alert(self, severity: str, issue: str, details: str):
        """
        Send alert based on severity
        """
        key = f"{severity}:{issue}"
        
        self.alert_counts[key] = self.alert_counts.get(key, 0) + 1
        
        if self.alert_counts[key] >= self.alert_thresholds.get(severity, 1):
            # Send alert
            self.send_alert(severity, issue, details)
            
            # Reset counter
            self.alert_counts[key] = 0
    
    def send_alert(self, severity: str, issue: str, details: str):
        """
        Actually send alert (PagerDuty, Slack, email, etc.)
        """
        # Implementation would send to PagerDuty, Slack, etc.
        print(f"[{severity}] {issue}: {details}")

class MetricsCollector:
    """
    Collect metrics for monitoring
    """
    
    def __init__(self):
        self.metrics = {}
    
    def increment(self, metric: str, value: int = 1):
        """Increment counter metric"""
        self.metrics[metric] = self.metrics.get(metric, 0) + value
    
    def gauge(self, metric: str, value: float):
        """Set gauge metric"""
        self.metrics[metric] = value
    
    def histogram(self, metric: str, value: float):
        """Record histogram value"""
        if metric not in self.metrics:
            self.metrics[metric] = []
        self.metrics[metric].append(value)
\`\`\`

### Monitoring Dashboard

**Key Metrics to Track:**

1. **Data Quality Score** (0-100)
   - % of quotes passing validation
   - % of successful consolidations
   - Source agreement rate

2. **Latency Metrics**
   - P50, P95, P99 latency per source
   - Stale data incidents per hour

3. **Conflict Resolution**
   - Price discrepancies per ticker
   - Conflicts resolved successfully
   - Alerts triggered

4. **Source Health**
   - Uptime per source
   - Data quality per source
   - Failover events

### Handling Specific Scenarios

**1. Data Gap Detection and Fill**
```python
    class DataGapHandler:
    """
    Detect and handle data gaps
    """
    
    def detect_gap(self, ticker: str, expected_frequency: int = 1) -> bool:
    """
        Detect if data is missing

    Args:
    expected_frequency: Expected quotes per second
    """
        # Check last quote time
    last_quote_time = self.get_last_quote_time(ticker)
    gap_duration = (datetime.now() - last_quote_time).total_seconds()

    if gap_duration > expected_frequency * 2:
        return True
    return False
    
    def fill_gap(self, ticker: str):
    """
        Fill data gap from backup source or last known value
    """
        # Try backup sources
        # If not available, use last known value with flag
        pass
        ```

**2. Corporate Action Delayed Reporting**
```python
    class CorporateActionBuffer:
    """
    Buffer for handling delayed corporate action reporting
    """
    
    def __init__(self):
    self.pending_actions = {}
    
    def flag_suspicious_price_move(self, ticker: str, date: datetime, magnitude: float):
    """
        Flag potential corporate action that hasn't been reported yet
    """
    self.pending_actions[ticker] = {
        'date': date,
        'suspected_action': 'split',
        'magnitude': magnitude,
        'status': 'pending_confirmation',
        'flagged_at': datetime.now()
    }
        
        # Alert operations team
    self.alert_ops_team(ticker, date, magnitude)
    
    def confirm_corporate_action(self, ticker: str, action: dict):
    """
        Confirm and apply corporate action retroactively
    """
        # Adjust historical prices
        # Notify systems
        # Update database
    pass
        ```

### Production Checklist

- [ ] Multiple data sources configured
- [ ] Real-time validation rules implemented
- [ ] Conflict resolution logic tested
- [ ] Alert thresholds configured
- [ ] Monitoring dashboard deployed
- [ ] Runbook for common issues
- [ ] Automated failover tested
- [ ] Corporate action detection active
- [ ] Data quality SLA defined (e.g., 99.9% accuracy)

### Conclusion

This system provides:
- ✅ Real-time data quality validation
- ✅ Automatic conflict resolution
- ✅ Multi-source redundancy
- ✅ Comprehensive monitoring
- ✅ Automated corporate action detection

**Critical for production trading** where bad data = lost money.
`,
  },
];

