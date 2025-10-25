export const section12 = {
  title: 'Module 3 Capstone Project',
  content: `
# Module 3 Capstone Project: Complete Financial Analysis Platform

Build an end-to-end financial analysis system that downloads, analyzes, and monitors public companies automatically - a portfolio piece that demonstrates production engineering skills.

## Project Overview

Create a production-ready platform that:
1. **Downloads** financial statements from SEC EDGAR (10-K/10-Q)
2. **Parses** XBRL and extracts structured data
3. **Analyzes** financial health using all ratios and models
4. **Detects** red flags (Beneish M-Score, channel stuffing, earnings manipulation)
5. **Monitors** covenant compliance and credit risk
6. **Compares** to industry peers with percentile rankings
7. **Applies NLP** to earnings calls and MD&A sections
8. **Generates** automated reports and real-time alerts
9. **Visualizes** trends and insights in interactive dashboard
10. **Backtests** predictive models on historical data

## Technical Stack

- **Backend**: Python 3.11+, FastAPI, SQLAlchemy
- **Database**: PostgreSQL 15+ with TimescaleDB extension
- **Cache/Queue**: Redis, Celery for async tasks
- **Data**: SEC EDGAR API, XBRL parsing (sec-edgar-downloader)
- **Analysis**: pandas, numpy, scikit-learn, scipy
- **NLP**: transformers (FinBERT), spaCy, textstat
- **Frontend**: React 18+, TypeScript, Tailwind CSS
- **Charts**: Plotly, Recharts
- **Infrastructure**: Docker, Docker Compose, AWS (EC2, RDS, S3)
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus, Grafana

## Architecture Diagram

\`\`\`
┌─────────────┐
│  SEC EDGAR  │
└──────┬──────┘
       │ Extract (daily cron)
       v
┌─────────────────────────────────────────────┐
│           Data Pipeline (Celery)             │
│  - Download filings (10-K/10-Q)             │
│  - Parse XBRL → structured data             │
│  - Validate & clean                          │
└──────────────────┬──────────────────────────┘
                   │
                   v
         ┌──────────────────┐
         │   PostgreSQL     │
         │   - financials   │
         │   - metrics      │
         │   - alerts       │
         └────────┬─────────┘
                  │
                  v
┌─────────────────────────────────────────────┐
│        Analysis Engine (FastAPI)            │
│  - Financial ratios                         │
│  - Quality scores (Beneish, Altman)         │
│  - Credit analysis                          │
│  - Peer comparison                          │
│  - NLP sentiment                            │
└──────────────────┬──────────────────────────┘
                   │
                   v
         ┌──────────────────┐
         │   React Dashboard│
         │   - Company view │
         │   - Peer comp    │
         │   - Alerts       │
         │   - Charts       │
         └──────────────────┘
\`\`\`

## Part 1: FastAPI Backend Implementation

\`\`\`python
# backend/main.py

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import pandas as pd
from datetime import datetime

from .database import get_db, engine
from .models import Company, Financial, FinancialMetric, Alert
from .edgar import EDGARDownloader
from .analyzer import FinancialAnalyzer
from .nlp import SentimentAnalyzer

app = FastAPI(
    title="Financial Analysis Platform",
    description="Automated financial statement analysis and monitoring",
    version="1.0.0"
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
edgar = EDGARDownloader()
analyzer = FinancialAnalyzer()
sentiment_analyzer = SentimentAnalyzer()

@app.get("/")
def read_root():
    return {
        "message": "Financial Analysis Platform API",
        "version": "1.0.0",
        "endpoints": {
            "companies": "/companies",
            "analysis": "/analysis/{ticker}",
            "peers": "/peers/{ticker}",
            "alerts": "/alerts"
        }
    }

@app.get("/companies", response_model=List[dict])
def get_companies(
    sector: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends (get_db)
):
    """Get list of companies, optionally filtered by sector."""
    
    query = db.query(Company)
    
    if sector:
        query = query.filter(Company.sector == sector)
    
    companies = query.limit (limit).all()
    
    return [
        {
            "ticker": c.ticker,
            "name": c.name,
            "sector": c.sector,
            "industry": c.industry
        }
        for c in companies
    ]

@app.post("/companies/{ticker}/refresh")
async def refresh_company_data(
    ticker: str,
    db: Session = Depends (get_db)
):
    """
    Trigger data refresh for company.
    
    This endpoint:
    1. Downloads latest filings from SEC
    2. Parses XBRL data
    3. Calculates all metrics
    4. Checks for red flags
    5. Returns updated analysis
    """
    
    company = db.query(Company).filter(Company.ticker == ticker).first()
    if not company:
        raise HTTPException (status_code=404, detail="Company not found")
    
    try:
        # Download latest filings
        filings = edgar.download_filings (ticker, company.cik)
        
        # Parse and store
        for filing in filings:
            financials = edgar.parse_xbrl (filing)
            
            # Store to database
            for metric, value in financials.items():
                db_financial = Financial(
                    ticker=ticker,
                    metric=metric,
                    value=value,
                    period_end=filing['period_end'],
                    fiscal_year=filing['fiscal_year'],
                    fiscal_quarter=filing['fiscal_quarter']
                )
                db.add (db_financial)
        
        db.commit()
        
        # Calculate metrics
        metrics = analyzer.calculate_all_metrics (ticker, db)
        
        return {
            "status": "success",
            "ticker": ticker,
            "filings_processed": len (filings),
            "metrics_calculated": len (metrics)
        }
        
    except Exception as e:
        raise HTTPException (status_code=500, detail=str (e))

@app.get("/analysis/{ticker}")
def get_company_analysis(
    ticker: str,
    db: Session = Depends (get_db)
):
    """
    Get complete financial analysis for company.
    
    Returns:
    - Latest financial ratios
    - Quality scores (Beneish, Altman, Piotroski)
    - Credit metrics
    - Trend analysis
    - Red flags
    """
    
    # Get latest metrics
    latest_metrics = db.query(FinancialMetric).filter(
        FinancialMetric.ticker == ticker
    ).order_by(FinancialMetric.period_end.desc()).first()
    
    if not latest_metrics:
        raise HTTPException (status_code=404, detail="No data available for company")
    
    # Calculate quality scores
    financials = analyzer.get_financials_df (ticker, db)
    
    beneish = analyzer.calculate_beneish_mscore (financials)
    altman = analyzer.calculate_altman_zscore (financials)
    piotroski = analyzer.calculate_piotroski_fscore (financials)
    
    # Get recent alerts
    alerts = db.query(Alert).filter(
        Alert.ticker == ticker,
        Alert.acknowledged == False
    ).order_by(Alert.detected_at.desc()).limit(10).all()
    
    return {
        "ticker": ticker,
        "as_of_date": latest_metrics.period_end.isoformat(),
        "profitability": {
            "roe": latest_metrics.roe,
            "roa": latest_metrics.roa,
            "roic": latest_metrics.roic,
            "gross_margin": latest_metrics.gross_margin,
            "operating_margin": latest_metrics.operating_margin,
            "net_margin": latest_metrics.net_margin
        },
        "liquidity": {
            "current_ratio": latest_metrics.current_ratio,
            "quick_ratio": latest_metrics.quick_ratio,
            "cash_ratio": latest_metrics.cash_ratio
        },
        "leverage": {
            "debt_to_equity": latest_metrics.debt_to_equity,
            "debt_to_assets": latest_metrics.debt_to_assets,
            "interest_coverage": latest_metrics.interest_coverage
        },
        "efficiency": {
            "asset_turnover": latest_metrics.asset_turnover,
            "inventory_turnover": latest_metrics.inventory_turnover,
            "receivables_turnover": latest_metrics.receivables_turnover
        },
        "quality_scores": {
            "beneish_mscore": float (beneish),
            "beneish_flag": beneish > -1.78,  # Potential manipulator
            "altman_zscore": float (altman),
            "altman_zone": "Safe" if altman > 2.99 else "Grey" if altman > 1.81 else "Distress",
            "piotroski_fscore": int (piotroski),
            "piotroski_grade": "Strong" if piotroski >= 7 else "Average" if piotroski >= 4 else "Weak"
        },
        "alerts": [
            {
                "type": a.alert_type,
                "severity": a.severity,
                "message": a.message,
                "detected_at": a.detected_at.isoformat()
            }
            for a in alerts
        ]
    }

@app.get("/peers/{ticker}")
def get_peer_comparison(
    ticker: str,
    db: Session = Depends (get_db)
):
    """
    Compare company to industry peers.
    
    Returns percentile rankings for key metrics.
    """
    
    # Get company's industry
    company = db.query(Company).filter(Company.ticker == ticker).first()
    if not company:
        raise HTTPException (status_code=404, detail="Company not found")
    
    # Get all peers in same industry
    peers = db.query(Company).filter(
        Company.industry == company.industry,
        Company.ticker != ticker
    ).all()
    
    peer_tickers = [p.ticker for p in peers]
    
    # Get latest metrics for all
    all_metrics = db.query(FinancialMetric).filter(
        FinancialMetric.ticker.in_([ticker] + peer_tickers)
    ).order_by(FinancialMetric.period_end.desc()).all()
    
    # Convert to DataFrame for percentile calculation
    df = pd.DataFrame([
        {
            'ticker': m.ticker,
            'roe': m.roe,
            'roa': m.roa,
            'gross_margin': m.gross_margin,
            'operating_margin': m.operating_margin,
            'net_margin': m.net_margin,
            'current_ratio': m.current_ratio,
            'debt_to_equity': m.debt_to_equity
        }
        for m in all_metrics
    ])
    
    # Get company's metrics
    company_metrics = df[df['ticker'] == ticker].iloc[0]
    
    # Calculate percentiles
    percentiles = {}
    for col in df.columns:
        if col != 'ticker':
            percentiles[col] = (df[col] < company_metrics[col]).mean() * 100
    
    return {
        "ticker": ticker,
        "industry": company.industry,
        "peer_count": len (peer_tickers),
        "percentile_rankings": percentiles,
        "interpretation": {
            "roe": "Top quartile" if percentiles['roe'] > 75 else "Bottom quartile" if percentiles['roe'] < 25 else "Average",
            "profitability": "Above average" if percentiles['operating_margin'] > 50 else "Below average"
        }
    }

@app.get("/alerts")
def get_alerts(
    severity: Optional[str] = None,
    acknowledged: bool = False,
    limit: int = 50,
    db: Session = Depends (get_db)
):
    """Get recent alerts, optionally filtered."""
    
    query = db.query(Alert).filter(Alert.acknowledged == acknowledged)
    
    if severity:
        query = query.filter(Alert.severity == severity)
    
    alerts = query.order_by(Alert.detected_at.desc()).limit (limit).all()
    
    return [
        {
            "id": a.id,
            "ticker": a.ticker,
            "type": a.alert_type,
            "severity": a.severity,
            "metric": a.metric,
            "message": a.message,
            "detected_at": a.detected_at.isoformat()
        }
        for a in alerts
    ]

@app.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(
    alert_id: int,
    user: str,
    db: Session = Depends (get_db)
):
    """Mark alert as acknowledged."""
    
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException (status_code=404, detail="Alert not found")
    
    alert.acknowledged = True
    alert.acknowledged_at = datetime.utcnow()
    alert.acknowledged_by = user
    
    db.commit()
    
    return {"status": "success"}

@app.get("/sentiment/{ticker}")
def get_sentiment_analysis(
    ticker: str,
    db: Session = Depends (get_db)
):
    """
    Get NLP sentiment analysis for latest earnings call.
    
    Note: This assumes earnings call transcripts are stored.
    In production, you'd integrate with services like AlphaSense or Seeking Alpha.
    """
    
    # Placeholder - in production, fetch actual transcript
    transcript = "Sample earnings call transcript..."
    
    sentiment = sentiment_analyzer.analyze_earnings_call (transcript)
    
    return {
        "ticker": ticker,
        "prepared_remarks_sentiment": sentiment['prepared_remarks']['overall_sentiment'],
        "qa_sentiment": sentiment['qa_section']['overall_sentiment'],
        "divergence": sentiment['divergence'],
        "red_flag": sentiment['red_flag']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run (app, host="0.0.0.0", port=8000)
\`\`\`

## Part 2: React Dashboard Implementation

\`\`\`typescript
// frontend/src/components/CompanyDashboard.tsx

import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from 'recharts';

interface AnalysisData {
  ticker: string;
  as_of_date: string;
  profitability: Record<string, number>;
  liquidity: Record<string, number>;
  leverage: Record<string, number>;
  quality_scores: {
    beneish_mscore: number;
    beneish_flag: boolean;
    altman_zscore: number;
    altman_zone: string;
    piotroski_fscore: number;
    piotroski_grade: string;
  };
  alerts: Array<{
    type: string;
    severity: string;
    message: string;
    detected_at: string;
  }>;
}

const CompanyDashboard: React.FC = () => {
  const { ticker } = useParams<{ ticker: string }>();
  const [data, setData] = useState<AnalysisData | null>(null);
  const [loading, setLoading] = useState (true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchAnalysis();
  }, [ticker]);

  const fetchAnalysis = async () => {
    try {
      setLoading (true);
      const response = await axios.get(\`http://localhost:8000/analysis/\${ticker}\`);
      setData (response.data);
      setError (null);
    } catch (err) {
      setError('Failed to fetch company data');
      console.error (err);
    } finally {
      setLoading (false);
    }
  };

  const refreshData = async () => {
    try {
      await axios.post(\`http://localhost:8000/companies/\${ticker}/refresh\`);
      await fetchAnalysis();
    } catch (err) {
      console.error('Failed to refresh data:', err);
    }
  };

  if (loading) {
    return <div className="flex items-center justify-center h-screen">
      <div className="text-xl">Loading analysis for {ticker}...</div>
    </div>;
  }

  if (error || !data) {
    return <div className="text-red-600 p-4">{error}</div>;
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'HIGH': return 'bg-red-100 text-red-800 border-red-300';
      case 'MEDIUM': return 'bg-yellow-100 text-yellow-800 border-yellow-300';
      case 'LOW': return 'bg-blue-100 text-blue-800 border-blue-300';
      default: return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  const getScoreColor = (score: number, metric: string) => {
    if (metric === 'beneish') {
      return score > -1.78 ? 'text-red-600' : 'text-green-600';
    } else if (metric === 'altman') {
      return score > 2.99 ? 'text-green-600' : score > 1.81 ? 'text-yellow-600' : 'text-red-600';
    } else if (metric === 'piotroski') {
      return score >= 7 ? 'text-green-600' : score >= 4 ? 'text-yellow-600' : 'text-red-600';
    }
    return 'text-gray-600';
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      {/* Header */}
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">{ticker}</h1>
          <p className="text-gray-600">As of {new Date (data.as_of_date).toLocaleDateString()}</p>
        </div>
        <button
          onClick={refreshData}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
        >
          Refresh Data
        </button>
      </div>

      {/* Quality Scores - Prominent Display */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-white rounded-lg shadow p-6 border-l-4 border-blue-500">
          <h3 className="text-sm font-medium text-gray-500 mb-2">Beneish M-Score</h3>
          <p className={\`text-3xl font-bold \${getScoreColor (data.quality_scores.beneish_mscore, 'beneish')}\`}>
            {data.quality_scores.beneish_mscore.toFixed(2)}
          </p>
          <p className="text-sm mt-2">
            {data.quality_scores.beneish_flag ? 
              <span className="text-red-600 font-semibold">⚠️ Potential Manipulator</span> :
              <span className="text-green-600">✓ Low Manipulation Risk</span>
            }
          </p>
        </div>

        <div className="bg-white rounded-lg shadow p-6 border-l-4 border-purple-500">
          <h3 className="text-sm font-medium text-gray-500 mb-2">Altman Z-Score</h3>
          <p className={\`text-3xl font-bold \${getScoreColor (data.quality_scores.altman_zscore, 'altman')}\`}>
            {data.quality_scores.altman_zscore.toFixed(2)}
          </p>
          <p className="text-sm mt-2 font-semibold">{data.quality_scores.altman_zone} Zone</p>
        </div>

        <div className="bg-white rounded-lg shadow p-6 border-l-4 border-green-500">
          <h3 className="text-sm font-medium text-gray-500 mb-2">Piotroski F-Score</h3>
          <p className={\`text-3xl font-bold \${getScoreColor (data.quality_scores.piotroski_fscore, 'piotroski')}\`}>
            {data.quality_scores.piotroski_fscore}/9
          </p>
          <p className="text-sm mt-2 font-semibold">{data.quality_scores.piotroski_grade} Fundamentals</p>
        </div>
      </div>

      {/* Alerts */}
      {data.alerts.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-bold mb-4">🚨 Active Alerts ({data.alerts.length})</h2>
          <div className="space-y-3">
            {data.alerts.map((alert, idx) => (
              <div 
                key={idx}
                className={\`p-4 border rounded-lg \${getSeverityColor (alert.severity)}\`}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <p className="font-semibold">{alert.type}</p>
                    <p className="text-sm mt-1">{alert.message}</p>
                  </div>
                  <span className="text-xs">{new Date (alert.detected_at).toLocaleDateString()}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Financial Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {/* Profitability */}
        <MetricCard title="ROE" value={(data.profitability.roe * 100).toFixed(1) + '%'} />
        <MetricCard title="ROA" value={(data.profitability.roa * 100).toFixed(1) + '%'} />
        <MetricCard title="Operating Margin" value={(data.profitability.operating_margin * 100).toFixed(1) + '%'} />
        <MetricCard title="Net Margin" value={(data.profitability.net_margin * 100).toFixed(1) + '%'} />
        
        {/* Liquidity */}
        <MetricCard title="Current Ratio" value={data.liquidity.current_ratio?.toFixed(2) || 'N/A'} />
        <MetricCard title="Quick Ratio" value={data.liquidity.quick_ratio?.toFixed(2) || 'N/A'} />
        
        {/* Leverage */}
        <MetricCard title="Debt/Equity" value={data.leverage.debt_to_equity?.toFixed(2) || 'N/A'} />
        <MetricCard title="Interest Coverage" value={data.leverage.interest_coverage?.toFixed(1) + 'x' || 'N/A'} />
      </div>
    </div>
  );
};

const MetricCard: React.FC<{ title: string; value: string }> = ({ title, value }) => (
  <div className="bg-white rounded-lg shadow p-4">
    <p className="text-sm text-gray-500 mb-1">{title}</p>
    <p className="text-2xl font-bold">{value}</p>
  </div>
);

export default CompanyDashboard;
\`\`\`

## Part 3: Docker Compose Setup

\`\`\`yaml
# docker-compose.yml

version: '3.8'

services:
  # PostgreSQL database
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: financial_data
      POSTGRES_USER: finuser
      POSTGRES_PASSWORD: finpass123
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U finuser"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis for caching and Celery
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # FastAPI backend
  backend:
    build: ./backend
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://finuser:finpass123@postgres:5432/financial_data
      REDIS_URL: redis://redis:6379/0
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
    volumes:
      - ./backend:/app

  # Celery worker for background tasks
  celery:
    build: ./backend
    command: celery -A tasks worker --loglevel=info
    environment:
      DATABASE_URL: postgresql://finuser:finpass123@postgres:5432/financial_data
      REDIS_URL: redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    volumes:
      - ./backend:/app

  # React frontend
  frontend:
    build: ./frontend
    command: npm start
    ports:
      - "3000:3000"
    environment:
      REACT_APP_API_URL: http://localhost:8000
    volumes:
      - ./frontend:/app
      - /app/node_modules
    depends_on:
      - backend

volumes:
  postgres_data:
  redis_data:
\`\`\`

## Part 4: CI/CD Pipeline (GitHub Actions)

\`\`\`yaml
# .github/workflows/ci.yml

name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test-backend:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_DB: test_db
          POSTGRES_USER: test_user
          POSTGRES_PASSWORD: test_pass
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        cd backend
        pip install -r requirements.txt
    
    - name: Run tests
      env:
        DATABASE_URL: postgresql://test_user:test_pass@localhost:5432/test_db
      run: |
        cd backend
        pytest tests/ -v --cov=. --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./backend/coverage.xml

  test-frontend:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Node
      uses: actions/setup-node@v3
      with:
        node-version: '18'
    
    - name: Install dependencies
      run: |
        cd frontend
        npm ci
    
    - name: Run tests
      run: |
        cd frontend
        npm test -- --coverage
    
    - name: Build
      run: |
        cd frontend
        npm run build

  deploy:
    needs: [test-backend, test-frontend]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to AWS
      env:
        AWS_ACCESS_KEY_ID: \${{ secrets.AWS_ACCESS_KEY_ID }}
        AWS_SECRET_ACCESS_KEY: \${{ secrets.AWS_SECRET_ACCESS_KEY }}
      run: |
        # Deploy script here
        echo "Deploying to production..."
\`\`\`

## Evaluation Criteria

### Required Features (80 points)
- **Data Pipeline** (20 pts): Automated SEC EDGAR download & parsing
- **Analysis Engine** (20 pts): All ratios, quality scores, credit analysis
- **Database** (10 pts): Proper schema, indexes, migrations
- **API** (15 pts): RESTful endpoints, proper error handling
- **Dashboard** (15 pts): Functional UI displaying all metrics

### Code Quality (10 points)
- Clean, modular code with proper separation of concerns
- Type hints in Python, TypeScript for React
- Comprehensive docstrings
- Error handling throughout

### Innovation (10 points)
- Creative features beyond requirements
- Performance optimizations
- Advanced visualizations
- ML predictions

**Total**: 100 points

## Timeline

**Week 1**: Backend Foundation
- Day 1-2: Database schema, models, migrations
- Day 3-4: EDGAR integration, XBRL parsing
- Day 5-7: Analysis engine (ratios, quality scores)

**Week 2**: API & Advanced Features
- Day 8-9: FastAPI endpoints
- Day 10-11: NLP integration
- Day 12-14: Alert system, background tasks

**Week 3**: Frontend
- Day 15-16: React setup, routing
- Day 17-18: Dashboard components
- Day 19-21: Charts, styling, responsive design

**Week 4**: Polish & Deploy
- Day 22-23: Testing, bug fixes
- Day 24-25: Docker setup, CI/CD
- Day 26-27: AWS deployment
- Day 28: Documentation, demo video

**Total**: 40-60 hours

## Resources

- **SEC EDGAR**: https://www.sec.gov/edgar
- **XBRL Parsing**: \`pip install sec-edgar-downloader\`
- **FinBERT**: \`transformers\` library from Hugging Face
- **FastAPI Docs**: https://fastapi.tiangolo.com
- **React + TypeScript**: https://react-typescript-cheatsheet.netlify.app
- **Plotly React**: https://plotly.com/javascript/react/
- **AWS Deployment**: EC2 (compute), RDS (database), S3 (static files)

## Submission

1. **GitHub Repository**
   - Clean commit history
   - Comprehensive README
   - Setup instructions
   - Architecture documentation

2. **Demo Video** (5-10 minutes)
   - Architecture overview
   - Live demo of features
   - Code walkthrough
   - Deployment process

3. **Written Report** (3-5 pages)
   - Design decisions
   - Technical challenges
   - Performance optimizations
   - Future enhancements

4. **Live Demo** (Optional but impressive)
   - Deployed on AWS with public URL
   - Sample data for 50-100 companies

## What This Demonstrates

Build this project and you'll have proven skills in:
- **Financial Analysis**: Deep understanding of financial statements, ratios, fraud detection
- **Data Engineering**: ETL pipelines, data validation, time-series databases
- **Backend Development**: Python, FastAPI, async processing, database design
- **Frontend Development**: React, TypeScript, data visualization
- **DevOps**: Docker, CI/CD, cloud deployment
- **Production Engineering**: Error handling, monitoring, scalability

This portfolio piece will impress employers at:
- Hedge funds (Citadel, Two Sigma, DE Shaw)
- Fintech companies (Stripe, Robinhood, Plaid)
- Investment banks (Goldman, Morgan Stanley)
- Credit rating agencies (Moody's, S&P)
- Big Tech finance teams (Google, Amazon, Meta)

**Build it. Ship it. Get hired. 🚀**
`,
  discussionQuestions: [],
  multipleChoiceQuestions: [],
};
