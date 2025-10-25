export const section10 = {
  title: 'Building a Financial Data Pipeline',
  content: `
# Building a Financial Data Pipeline

Build production-ready systems to automate financial statement analysis at scale - process hundreds of companies daily with zero manual intervention.

## Section 1: Complete ETL Architecture

\`\`\`python
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
from apscheduler.schedulers.background import BackgroundScheduler
import logging
from typing import Dict, List, Optional
import redis
from celery import Celery
import requests
from datetime import datetime, timedelta
import json

class ProductionFinancialPipeline:
    """
    Production-grade financial data pipeline.
    
    Handles 500+ companies with:
    - Parallel processing
    - Error recovery
    - Data validation
    - Alerting
    - Monitoring
    """
    
    def __init__(self, config: Dict):
        self.db_url = config['database_url']
        self.engine = create_engine (self.db_url, pool_size=20, max_overflow=40)
        self.redis_client = redis.Redis (host=config['redis_host'], port=6379)
        self.logger = self._setup_logging()
        
        # Celery for distributed task processing
        self.celery_app = Celery('financial_pipeline',
                                broker=config['celery_broker'],
                                backend=config['celery_backend'])
    
    def _setup_logging (self) -> logging.Logger:
        """Configure production logging."""
        logger = logging.getLogger('FinancialPipeline')
        logger.setLevel (logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel (logging.INFO)
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        fh = RotatingFileHandler('pipeline.log', 
                                 maxBytes=10*1024*1024,  # 10MB
                                 backupCount=5)
        fh.setLevel (logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch.setFormatter (formatter)
        fh.setFormatter (formatter)
        
        logger.addHandler (ch)
        logger.addHandler (fh)
        
        return logger
    
    def extract_edgar_data (self, ticker: str, cik: str) -> Dict:
        """
        Extract financial data from SEC EDGAR.
        
        Handles:
        - Rate limiting (10 requests/second SEC limit)
        - Retry logic
        - Data validation
        """
        import time
        from sec_edgar_downloader import Downloader
        
        # Rate limiting check (Redis)
        rate_limit_key = f"edgar_rate_limit:{datetime.now().second}"
        request_count = self.redis_client.incr (rate_limit_key)
        self.redis_client.expire (rate_limit_key, 1)
        
        if request_count > 8:  # Stay under 10/sec limit
            time.sleep(0.2)
        
        try:
            # Get company facts from SEC API
            url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            headers = {'User-Agent': 'MyCompany contact@company.com'}
            
            response = requests.get (url, headers=headers, timeout=30)
            response.raise_for_status()
            
            facts = response.json()
            
            self.logger.info (f"Successfully extracted data for {ticker}")
            
            return {
                'ticker': ticker,
                'cik': cik,
                'facts': facts,
                'extracted_at': datetime.utcnow().isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            self.logger.error (f"EDGAR extraction failed for {ticker}: {e}")
            raise
    
    def transform_financials (self, raw_data: Dict) -> pd.DataFrame:
        """
        Transform raw XBRL data into structured format.
        
        Returns DataFrame with:
        - Standardized columns
        - Validated data types
        - Cleaned values
        """
        facts = raw_data['facts']['us-gaap']
        
        # Extract key metrics
        metrics_map = {
            'Revenues': 'revenue',
            'CostOfRevenue': 'cogs',
            'GrossProfit': 'gross_profit',
            'OperatingIncomeLoss': 'operating_income',
            'NetIncomeLoss': 'net_income',
            'Assets': 'total_assets',
            'Liabilities': 'total_liabilities',
            'StockholdersEquity': 'shareholders_equity',
            'NetCashProvidedByUsedInOperatingActivities': 'operating_cash_flow',
            'PaymentsToAcquirePropertyPlantAndEquipment': 'capex'
        }
        
        records = []
        
        for xbrl_tag, column_name in metrics_map.items():
            if xbrl_tag not in facts:
                continue
            
            for unit_type in facts[xbrl_tag]['units']:
                for entry in facts[xbrl_tag]['units'][unit_type]:
                    if entry.get('form') in ['10-K', '10-Q']:
                        records.append({
                            'ticker': raw_data['ticker'],
                            'metric': column_name,
                            'period_end': entry['end'],
                            'fiscal_year': entry.get('fy'),
                            'fiscal_quarter': entry.get('fp'),
                            'value': entry['val'],
                            'form': entry['form'],
                            'filed_date': entry.get('filed'),
                            'extracted_at': raw_data['extracted_at']
                        })
        
        df = pd.DataFrame (records)
        
        # Data validation
        df = self._validate_data (df)
        
        return df
    
    def _validate_data (self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data quality and handle anomalies."""
        
        # Remove duplicates
        df = df.drop_duplicates (subset=['ticker', 'metric', 'period_end', 'fiscal_quarter'])
        
        # Ensure proper data types
        df['value'] = pd.to_numeric (df['value'], errors='coerce')
        df['period_end'] = pd.to_datetime (df['period_end'])
        
        # Remove nulls in critical fields
        df = df.dropna (subset=['ticker', 'metric', 'value'])
        
        # Flag outliers (values changing >10x quarter-over-quarter)
        df = df.sort_values(['ticker', 'metric', 'period_end'])
        df['value_change'] = df.groupby(['ticker', 'metric'])['value'].pct_change()
        df['outlier'] = df['value_change'].abs() > 10.0
        
        outlier_count = df['outlier'].sum()
        if outlier_count > 0:
            self.logger.warning (f"Found {outlier_count} potential outliers")
        
        return df
    
    def load_to_database (self, df: pd.DataFrame, table: str):
        """
        Load data to PostgreSQL with conflict handling.
        
        Uses UPSERT (INSERT ... ON CONFLICT) for idempotency.
        """
        from sqlalchemy.dialects.postgresql import insert
        
        if df.empty:
            self.logger.warning (f"Empty DataFrame, skipping load to {table}")
            return
        
        try:
            # Convert DataFrame to list of dicts
            records = df.to_dict('records')
            
            # Batch insert with conflict handling
            with self.engine.begin() as conn:
                for i in range(0, len (records), 1000):  # Batch size 1000
                    batch = records[i:i+1000]
                    
                    stmt = insert (table).values (batch)
                    
                    # On conflict, update the record
                    update_dict = {c.name: c for c in stmt.excluded if c.name != 'id'}
                    stmt = stmt.on_conflict_do_update(
                        index_elements=['ticker', 'period_end', 'metric'],
                        set_=update_dict
                    )
                    
                    conn.execute (stmt)
            
            self.logger.info (f"Loaded {len (df)} records to {table}")
            
        except Exception as e:
            self.logger.error (f"Database load failed: {e}")
            raise
    
    def calculate_financial_metrics (self, ticker: str) -> pd.DataFrame:
        """Calculate all financial ratios from base financials."""
        
        query = """
        SELECT 
            period_end,
            fiscal_year,
            fiscal_quarter,
            MAX(CASE WHEN metric = 'revenue' THEN value END) as revenue,
            MAX(CASE WHEN metric = 'cogs' THEN value END) as cogs,
            MAX(CASE WHEN metric = 'gross_profit' THEN value END) as gross_profit,
            MAX(CASE WHEN metric = 'operating_income' THEN value END) as operating_income,
            MAX(CASE WHEN metric = 'net_income' THEN value END) as net_income,
            MAX(CASE WHEN metric = 'total_assets' THEN value END) as total_assets,
            MAX(CASE WHEN metric = 'total_liabilities' THEN value END) as total_liabilities,
            MAX(CASE WHEN metric = 'shareholders_equity' THEN value END) as shareholders_equity,
            MAX(CASE WHEN metric = 'operating_cash_flow' THEN value END) as cfo,
            MAX(CASE WHEN metric = 'capex' THEN value END) as capex
        FROM financials
        WHERE ticker = %s
        GROUP BY period_end, fiscal_year, fiscal_quarter
        ORDER BY period_end DESC
        LIMIT 20
        """
        
        df = pd.read_sql (query, self.engine, params=(ticker,))
        
        # Calculate ratios
        df['gross_margin'] = df['gross_profit'] / df['revenue']
        df['operating_margin'] = df['operating_income'] / df['revenue']
        df['net_margin'] = df['net_income'] / df['revenue']
        df['roe'] = df['net_income'] / df['shareholders_equity']
        df['roa'] = df['net_income'] / df['total_assets']
        df['debt_to_equity'] = (df['total_liabilities'] - df['shareholders_equity']) / df['shareholders_equity']
        df['fcf'] = df['cfo'] + df['capex']  # capex is negative
        df['cfo_to_ni_ratio'] = df['cfo'] / df['net_income']
        
        # Add ticker for insert
        df['ticker'] = ticker
        df['calculated_at'] = datetime.utcnow()
        
        return df
    
    def detect_red_flags (self, ticker: str) -> List[Dict]:
        """Run all red flag detection models."""
        
        alerts = []
        
        # Get latest metrics
        query = """
        SELECT * FROM financial_metrics
        WHERE ticker = %s
        ORDER BY period_end DESC
        LIMIT 8
        """
        
        df = pd.read_sql (query, self.engine, params=(ticker,))
        
        if len (df) < 2:
            return alerts
        
        latest = df.iloc[0]
        prior = df.iloc[1]
        
        # Check 1: Declining CFO/NI ratio
        if latest['cfo_to_ni_ratio'] < 0.7 and prior['cfo_to_ni_ratio'] > 0.9:
            alerts.append({
                'ticker': ticker,
                'alert_type': 'EARNINGS_QUALITY',
                'severity': 'HIGH',
                'metric': 'CFO/NI Ratio',
                'current': latest['cfo_to_ni_ratio'],
                'prior': prior['cfo_to_ni_ratio'],
                'message': f"CFO/NI declined from {prior['cfo_to_ni_ratio']:.2f} to {latest['cfo_to_ni_ratio']:.2f}",
                'detected_at': datetime.utcnow()
            })
        
        # Check 2: Negative FCF
        if latest['fcf'] < 0 and df['fcf'].head(4).sum() < 0:
            alerts.append({
                'ticker': ticker,
                'alert_type': 'CASH_BURN',
                'severity': 'MEDIUM',
                'metric': 'Free Cash Flow',
                'current': latest['fcf'],
                'message': f"Negative FCF for 4 consecutive quarters",
                'detected_at': datetime.utcnow()
            })
        
        # Check 3: ROE declining
        if latest['roe'] < 0.05 and prior['roe'] > 0.10:
            alerts.append({
                'ticker': ticker,
                'alert_type': 'PROFITABILITY_DECLINE',
                'severity': 'MEDIUM',
                'metric': 'ROE',
                'current': latest['roe'],
                'prior': prior['roe'],
                'message': f"ROE declined from {prior['roe']:.1%} to {latest['roe']:.1%}",
                'detected_at': datetime.utcnow()
            })
        
        return alerts
    
    def send_alerts (self, alerts: List[Dict]):
        """Send alerts via multiple channels."""
        
        for alert in alerts:
            # Store in database
            self._store_alert (alert)
            
            # Send email for HIGH severity
            if alert['severity'] == 'HIGH':
                self._send_email_alert (alert)
            
            # Post to Slack
            self._post_to_slack (alert)
            
            self.logger.info (f"Alert sent for {alert['ticker']}: {alert['alert_type']}")
    
    def _send_email_alert (self, alert: Dict):
        """Send email alert using SMTP."""
        import smtplib
        from email.mime.text import MIMEText
        
        msg = MIMEText (f"""
        Alert: {alert['alert_type']}
        Company: {alert['ticker']}
        Severity: {alert['severity']}
        Message: {alert['message']}
        
        Detected at: {alert['detected_at']}
        """)
        
        msg['Subject'] = f"ALERT: {alert['ticker']} - {alert['alert_type']}"
        msg['From'] = 'alerts@company.com'
        msg['To'] = 'analyst@company.com'
        
        # Send via SMTP (configure your SMTP server)
        # smtp = smtplib.SMTP('smtp.gmail.com', 587)
        # smtp.send_message (msg)
        pass
    
    def _post_to_slack (self, alert: Dict):
        """Post alert to Slack channel."""
        webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
        
        color = 'danger' if alert['severity'] == 'HIGH' else 'warning'
        
        payload = {
            "attachments": [{
                "color": color,
                "title": f"{alert['ticker']} - {alert['alert_type']}",
                "text": alert['message'],
                "fields": [
                    {"title": "Severity", "value": alert['severity'], "short": True},
                    {"title": "Metric", "value": alert['metric'], "short": True}
                ]
            }]
        }
        
        # requests.post (webhook_url, json=payload)
        pass
    
    def run_full_pipeline (self, ticker: str, cik: str) -> Dict:
        """Execute complete ETL pipeline for one company."""
        
        pipeline_start = datetime.utcnow()
        
        try:
            self.logger.info (f"Starting pipeline for {ticker}")
            
            # Extract
            raw_data = self.extract_edgar_data (ticker, cik)
            
            # Transform
            clean_data = self.transform_financials (raw_data)
            
            # Load
            self.load_to_database (clean_data, 'financials')
            
            # Calculate metrics
            metrics = self.calculate_financial_metrics (ticker)
            self.load_to_database (metrics, 'financial_metrics')
            
            # Red flags
            alerts = self.detect_red_flags (ticker)
            if alerts:
                self.send_alerts (alerts)
            
            duration = (datetime.utcnow() - pipeline_start).total_seconds()
            
            result = {
                'ticker': ticker,
                'status': 'SUCCESS',
                'records_loaded': len (clean_data),
                'alerts_generated': len (alerts),
                'duration_seconds': duration,
                'completed_at': datetime.utcnow().isoformat()
            }
            
            self.logger.info (f"Pipeline completed for {ticker} in {duration:.1f}s")
            
            return result
            
        except Exception as e:
            self.logger.error (f"Pipeline failed for {ticker}: {e}", exc_info=True)
            
            return {
                'ticker': ticker,
                'status': 'FAILED',
                'error': str (e),
                'completed_at': datetime.utcnow().isoformat()
            }
    
    def run_for_all_companies (self):
        """Run pipeline for all monitored companies."""
        
        # Get list of companies from database
        query = "SELECT ticker, cik FROM companies WHERE active = TRUE"
        companies = pd.read_sql (query, self.engine)
        
        self.logger.info (f"Processing {len (companies)} companies")
        
        results = []
        
        # Process with Celery for parallelization
        from celery import group
        
        job = group (self.process_company_async.s (row['ticker'], row['cik']) 
                   for _, row in companies.iterrows())
        
        result = job.apply_async()
        results = result.get (timeout=3600)  # 1 hour timeout
        
        # Summary
        success = sum(1 for r in results if r['status'] == 'SUCCESS')
        failed = len (results) - success
        
        self.logger.info (f"Pipeline complete: {success} succeeded, {failed} failed")
        
        return results
    
    @celery_app.task
    def process_company_async (self, ticker: str, cik: str) -> Dict:
        """Celery task for async processing."""
        return self.run_full_pipeline (ticker, cik)

# Usage example
config = {
    'database_url': 'postgresql://user:pass@localhost/financial_data',
    'redis_host': 'localhost',
    'celery_broker': 'redis://localhost:6379/0',
    'celery_backend': 'redis://localhost:6379/1'
}

pipeline = ProductionFinancialPipeline (config)
\`\`\`

## Section 2: Database Schema Design

\`\`\`sql
-- Complete production schema

-- Companies master table
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) UNIQUE NOT NULL,
    cik VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    sector VARCHAR(100),
    industry VARCHAR(100),
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_companies_ticker ON companies (ticker);
CREATE INDEX idx_companies_cik ON companies (cik);
CREATE INDEX idx_companies_sector ON companies (sector);

-- Raw financials (long format)
CREATE TABLE financials (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    metric VARCHAR(100) NOT NULL,
    period_end DATE NOT NULL,
    fiscal_year INTEGER,
    fiscal_quarter VARCHAR(2),
    value NUMERIC(20, 2),
    form VARCHAR(10),
    filed_date DATE,
    extracted_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(ticker, metric, period_end, fiscal_quarter)
);

CREATE INDEX idx_financials_ticker_period ON financials (ticker, period_end);
CREATE INDEX idx_financials_metric ON financials (metric);
CREATE INDEX idx_financials_fiscal ON financials (fiscal_year, fiscal_quarter);

-- Computed metrics
CREATE TABLE financial_metrics (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    period_end DATE NOT NULL,
    fiscal_year INTEGER,
    fiscal_quarter VARCHAR(2),
    
    -- Profitability
    gross_margin NUMERIC(10, 4),
    operating_margin NUMERIC(10, 4),
    net_margin NUMERIC(10, 4),
    roe NUMERIC(10, 4),
    roa NUMERIC(10, 4),
    roic NUMERIC(10, 4),
    
    -- Liquidity
    current_ratio NUMERIC(10, 2),
    quick_ratio NUMERIC(10, 2),
    cash_ratio NUMERIC(10, 2),
    
    -- Leverage
    debt_to_equity NUMERIC(10, 2),
    debt_to_assets NUMERIC(10, 2),
    interest_coverage NUMERIC(10, 2),
    
    -- Efficiency
    asset_turnover NUMERIC(10, 2),
    inventory_turnover NUMERIC(10, 2),
    receivables_turnover NUMERIC(10, 2),
    
    -- Cash flow
    fcf NUMERIC(20, 2),
    cfo_to_ni_ratio NUMERIC(10, 2),
    fcf_margin NUMERIC(10, 4),
    
    calculated_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(ticker, period_end, fiscal_quarter)
);

CREATE INDEX idx_metrics_ticker_period ON financial_metrics (ticker, period_end);

-- Quality scores
CREATE TABLE quality_scores (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    period_end DATE NOT NULL,
    
    beneish_mscore NUMERIC(10, 4),
    altman_zscore NUMERIC(10, 4),
    piotroski_fscore INTEGER,
    
    earnings_quality_score INTEGER,  -- 0-100
    
    calculated_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(ticker, period_end)
);

-- Alerts
CREATE TABLE alerts (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    metric VARCHAR(50),
    current_value NUMERIC(20, 2),
    prior_value NUMERIC(20, 2),
    threshold_value NUMERIC(20, 2),
    message TEXT,
    detected_at TIMESTAMP NOT NULL,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMP,
    acknowledged_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_alerts_ticker ON alerts (ticker);
CREATE INDEX idx_alerts_detected ON alerts (detected_at);
CREATE INDEX idx_alerts_severity ON alerts (severity);
CREATE INDEX idx_alerts_unacknowledged ON alerts (acknowledged) WHERE NOT acknowledged;

-- Pipeline execution log
CREATE TABLE pipeline_runs (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    status VARCHAR(20) NOT NULL,
    records_loaded INTEGER,
    alerts_generated INTEGER,
    duration_seconds NUMERIC(10, 2),
    error_message TEXT,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_runs_ticker ON pipeline_runs (ticker);
CREATE INDEX idx_runs_status ON pipeline_runs (status);
CREATE INDEX idx_runs_started ON pipeline_runs (started_at);
\`\`\`

## Section 3: Monitoring & Observability

\`\`\`python
import prometheus_client as prom
from datadog import statsd
import structlog

class PipelineMonitoring:
    """Comprehensive pipeline monitoring."""
    
    def __init__(self):
        # Prometheus metrics
        self.pipeline_duration = prom.Histogram(
            'pipeline_duration_seconds',
            'Time spent processing company',
            ['ticker', 'status']
        )
        
        self.records_processed = prom.Counter(
            'records_processed_total',
            'Total records processed',
            ['table']
        )
        
        self.alerts_generated = prom.Counter(
            'alerts_generated_total',
            'Total alerts generated',
            ['severity', 'type']
        )
        
        self.pipeline_errors = prom.Counter(
            'pipeline_errors_total',
            'Total pipeline errors',
            ['error_type']
        )
        
        # Structured logging
        self.logger = structlog.get_logger()
    
    def track_pipeline_run (self, ticker: str, duration: float, status: str):
        """Track pipeline execution."""
        self.pipeline_duration.labels (ticker=ticker, status=status).observe (duration)
        
        statsd.histogram('pipeline.duration', duration, tags=[f'ticker:{ticker}', f'status:{status}'])
        
        self.logger.info('pipeline_completed', 
                        ticker=ticker,
                        duration=duration,
                        status=status)
    
    def track_data_quality (self, ticker: str, metrics: Dict):
        """Track data quality metrics."""
        
        # Missing data percentage
        missing_pct = metrics.get('missing_percentage', 0)
        statsd.gauge('data.missing_percentage', missing_pct, tags=[f'ticker:{ticker}'])
        
        # Outlier count
        outliers = metrics.get('outlier_count', 0)
        statsd.gauge('data.outliers', outliers, tags=[f'ticker:{ticker}'])
        
        if missing_pct > 0.10:  # >10% missing
            self.logger.warning('high_missing_data',
                              ticker=ticker,
                              missing_pct=missing_pct)
    
    def generate_health_dashboard (self) -> Dict:
        """Generate pipeline health summary."""
        
        query = """
        SELECT 
            DATE_TRUNC('day', started_at) as date,
            status,
            COUNT(*) as run_count,
            AVG(duration_seconds) as avg_duration,
            SUM(alerts_generated) as total_alerts
        FROM pipeline_runs
        WHERE started_at >= NOW() - INTERVAL '7 days'
        GROUP BY DATE_TRUNC('day', started_at), status
        ORDER BY date DESC
        """
        
        # Execute and return dashboard data
        pass
\`\`\`

## Key Takeaways

1. **Automate everything** - Manual analysis doesn't scale beyond 10-20 companies
2. **Use queues** - Celery + Redis for parallel processing (10x speedup)
3. **Handle failures gracefully** - One bad company shouldn't stop pipeline
4. **Monitor actively** - Prometheus + Datadog for real-time visibility
5. **Version data** - Track historical changes and restatements
6. **Batch operations** - Load 1000 records at once, not row-by-row
7. **Rate limit** - Respect SEC's 10 requests/second limit
8. **Alert intelligently** - Multi-channel (email, Slack, database)

Master data pipelines and you can analyze 1000s of companies with zero manual work!
`,
  discussionQuestions: [],
  multipleChoiceQuestions: [],
};
