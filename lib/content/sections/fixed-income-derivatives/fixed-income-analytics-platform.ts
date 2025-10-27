export const fixedIncomeAnalyticsPlatform = {
  title: 'Project: Fixed Income Analytics Platform',
  id: 'fixed-income-analytics-platform',
  content: `
# Project: Fixed Income Analytics Platform

## Project Overview

Build a comprehensive fixed income analytics platform integrating all concepts from this module.

**Components**:
1. Bond pricing engine
2. Yield curve construction
3. Risk analytics (duration, convexity, Greeks)
4. Portfolio management
5. Derivative valuation
6. Risk management (VaR, stress testing)

---

## Architecture

### Microservices Design

\`\`\`
Services:
- pricing-service: Bond and derivative pricing
- risk-service: Greeks, VaR, stress tests
- portfolio-service: Portfolio construction and optimization
- market-data-service: Real-time and historical data
- reporting-service: Reports and dashboards

Communication: REST APIs + WebSocket (real-time)
Storage: PostgreSQL (relational), Redis (cache), S3 (historical)
\`\`\`

---

## Core Features

### 1. Bond Pricing Engine

**Functionality**:
- Price any bond (corporate, Treasury, municipal)
- Calculate yield measures (YTM, YTW, current yield)
- Handle day count conventions
- Support callable, putable, convertible bonds

**API**:
\`\`\`python
POST /pricing/bond
{
  "cusip": "912828ZG8",
  "settlement_date": "2024-01-15",
  "price": 98.50  # Optional, calculate yield
}

Response:
{
  "clean_price": 98.50,
  "dirty_price": 99.20,
  "ytm": 4.35,
  "duration": 7.2,
  "convexity": 68.5,
  "dv01": 7100
}
\`\`\`

### 2. Yield Curve Construction

**Functionality**:
- Bootstrap spot curve from par yields
- Multiple interpolation methods (linear, cubic spline, Nelson-Siegel)
- Generate forward curves
- Historical curve analysis

**API**:
\`\`\`python
GET /curves/treasury?date=2024-01-15

Response:
{
  "curve_date": "2024-01-15",
  "points": [
    {"maturity": 0.25, "rate": 5.25},
    {"maturity": 0.5, "rate": 5.30},
    {"maturity": 1.0, "rate": 5.15},
    ...
  ],
  "interpolation": "cubic_spline"
}
\`\`\`

### 3. Derivative Valuation

**Instruments**:
- Interest rate swaps
- Swaptions
- Credit default swaps
- Bond options
- Futures and forwards

**API**:
\`\`\`python
POST /pricing/swap
{
  "notional": 100000000,
  "fixed_rate": 4.50,
  "tenor": 5,
  "frequency": "semiannual"
}

Response:
{
  "npv": -125000,  # Mark-to-market
  "dv01": 45000,
  "par_rate": 4.51
}
\`\`\`

### 4. Portfolio Analytics

**Features**:
- Portfolio construction with constraints
- Risk metrics (tracking error, VaR, Greeks)
- Performance attribution
- Scenario analysis
- Optimization (minimize TE, maximize Sharpe)

**API**:
\`\`\`python
POST /portfolio/analyze
{
  "positions": [
    {"cusip": "...", "quantity": 1000, "price": 98.5},
    ...
  ]
}

Response:
{
  "total_value": 50000000,
  "duration": 6.8,
  "convexity": 52.3,
  "ytm": 4.25,
  "var_95_1d": 750000,
  "tracking_error": 0.85
}
\`\`\`

### 5. Risk Management

**Features**:
- Real-time VaR calculation (historical, parametric, Monte Carlo)
- Stress testing (predefined + custom scenarios)
- Risk limits monitoring
- Breach alerts
- Backtesting

**API**:
\`\`\`python
POST /risk/var
{
  "portfolio_id": "PF-12345",
  "method": "historical",
  "confidence": 0.95,
  "horizon_days": 1
}

Response:
{
  "var": 850000,
  "component_var": {
    "bonds": 600000,
    "swaps": 200000,
    "options": 50000
  }
}
\`\`\`

---

## Implementation Guidelines

### Technology Stack

**Backend**:
- Python 3.11+ (FastAPI)
- NumPy, SciPy (calculations)
- QuantLib (pricing library)
- Pandas (data manipulation)
- PostgreSQL (database)
- Redis (caching)

**Frontend**:
- React + TypeScript
- Recharts (visualizations)
- AG Grid (data tables)
- WebSocket (real-time updates)

**Infrastructure**:
- Docker + Kubernetes
- AWS/GCP (cloud)
- Prometheus + Grafana (monitoring)

### Data Sources

**Market Data**:
- Bloomberg API (bonds, rates, spreads)
- Refinitiv (alternative)
- Treasury Direct (government securities)
- TRACE (corporate bond trades)

**Reference Data**:
- CUSIP database
- Credit ratings (S&P, Moody's, Fitch)
- Corporate actions

### Production Considerations

**Performance**:
- Caching (Redis for market data)
- Batch processing (overnight risk reports)
- Async APIs (FastAPI async/await)
- Database indexing (CUSIP, date)

**Reliability**:
- Rate limiting
- Circuit breakers
- Retry logic
- Health checks

**Security**:
- OAuth 2.0 authentication
- Role-based access control
- API keys for services
- Audit logging

**Monitoring**:
- Prometheus metrics
- Distributed tracing (Jaeger)
- Error tracking (Sentry)
- Performance profiling

---

## Key Deliverables

1. **Pricing Engine**: Bond and derivative valuation with all conventions
2. **Risk Calculator**: Greeks, VaR, stress tests with multiple methodologies
3. **Portfolio Optimizer**: Construction and rebalancing with constraints
4. **Dashboard**: Real-time monitoring with charts and alerts
5. **API Documentation**: OpenAPI/Swagger with examples
6. **Test Suite**: Unit, integration, and load tests (>90% coverage)

---

## Success Metrics

**Accuracy**:
- Pricing within 1bp of Bloomberg
- Greeks within 5% of market consensus

**Performance**:
- Bond pricing <10ms
- VaR calculation <1s for 1000 positions
- API latency p95 <100ms

**Reliability**:
- 99.9% uptime
- Zero data loss
- <1 min recovery time

---

## Extensions

**Advanced Features**:
1. Machine learning for spread prediction
2. Natural language processing for news analysis
3. Automated trading (execution algorithms)
4. Multi-currency support
5. ESG analytics integration

**Integration**:
- Trading systems (FIX protocol)
- Risk systems (Aladdin, RiskMetrics)
- Accounting systems (GL integration)
- Regulatory reporting (EMIR, MiFID II)

---

## Conclusion

This project synthesizes all concepts from the module into a production-grade system. Focus on:
- Clean, maintainable code
- Comprehensive testing
- Production-ready patterns
- Real-world edge cases

**Key Skills Demonstrated**:
- Fixed income mathematics
- Software engineering best practices
- System design and architecture
- Performance optimization
- Risk management implementation
`,
};
