export const riskManagementPlatformProject = {
  id: 'risk-management-platform-project',
  title: 'Risk Management Platform - Capstone Project',
  content: `
# Risk Management Platform - Capstone Project

## Introduction

**Congratulations on reaching the capstone project!**

You've learned risk management from fundamentals to advanced techniques. Now it's time to integrate everything into a production-grade risk management platform.

This capstone will test your ability to:
- Design scalable system architecture
- Implement real-time risk calculations
- Build comprehensive limit frameworks
- Create automated reporting
- Handle regulatory requirements

## Project Overview

Build **RiskOS** - a comprehensive risk management platform that handles:
- Multi-asset portfolio tracking
- Real-time risk calculations (VaR, CVaR, Greeks)
- Automated limit monitoring
- Stress testing and scenario analysis
- Risk reporting and dashboards
- Regulatory compliance

**Scale**: Handle $1B+ AUM, 10,000+ positions, real-time updates

## Architecture Requirements

\`\`\`python
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

class ModuleType(Enum):
    """System modules"""
    DATA_INGESTION = "Data Ingestion"
    PORTFOLIO_MGMT = "Portfolio Management"
    RISK_ENGINE = "Risk Engine"
    LIMIT_MONITOR = "Limit Monitor"
    REPORTING = "Reporting"
    API = "API Layer"

@dataclass
class SystemRequirement:
    """Single system requirement"""
    module: ModuleType
    requirement: str
    priority: str  # CRITICAL, HIGH, MEDIUM, LOW
    acceptance_criteria: List[str]

class RiskOSRequirements:
    """
    Complete requirements for RiskOS platform
    """
    
    @staticmethod
    def get_all_requirements() -> List[SystemRequirement]:
        """Get all system requirements"""
        return [
            # Data Ingestion
            SystemRequirement(
                module=ModuleType.DATA_INGESTION,
                requirement="Real-time market data ingestion",
                priority="CRITICAL",
                acceptance_criteria=[
                    "Ingest price updates < 100ms latency",
                    "Support multiple data sources (CSV, API, WebSocket)",
                    "Handle 10,000+ securities",
                    "Data validation and quality checks",
                    "Store historical data"
                ]
            ),
            SystemRequirement(
                module=ModuleType.DATA_INGESTION,
                requirement="Position data ingestion",
                priority="CRITICAL",
                acceptance_criteria=[
                    "Import positions from CSV/Excel",
                    "Real-time position updates",
                    "Support multi-asset (equity, fixed income, derivatives)",
                    "Reconciliation with source systems"
                ]
            ),
            
            # Portfolio Management
            SystemRequirement(
                module=ModuleType.PORTFOLIO_MGMT,
                requirement="Position tracking",
                priority="CRITICAL",
                acceptance_criteria=[
                    "Track 10,000+ positions",
                    "Real-time P&L calculation",
                    "Position aggregation (by asset class, sector, geography)",
                    "Historical position tracking",
                    "Support cash, equity, fixed income, derivatives"
                ]
            ),
            SystemRequirement(
                module=ModuleType.PORTFOLIO_MGMT,
                requirement="Trade management",
                priority="HIGH",
                acceptance_criteria=[
                    "Record trades",
                    "Update positions automatically",
                    "Calculate trade P&L",
                    "Support buy, sell, short, cover"
                ]
            ),
            
            # Risk Engine
            SystemRequirement(
                module=ModuleType.RISK_ENGINE,
                requirement="VaR calculation",
                priority="CRITICAL",
                acceptance_criteria=[
                    "Historical VaR (95%, 99%)",
                    "Parametric VaR",
                    "Monte Carlo VaR (100K+ simulations)",
                    "Calculate portfolio and position-level VaR",
                    "Update in < 60 seconds for full portfolio"
                ]
            ),
            SystemRequirement(
                module=ModuleType.RISK_ENGINE,
                requirement="CVaR calculation",
                priority="HIGH",
                acceptance_criteria=[
                    "99% CVaR (Expected Shortfall)",
                    "Portfolio and position-level",
                    "All three methods (Historical, Parametric, Monte Carlo)"
                ]
            ),
            SystemRequirement(
                module=ModuleType.RISK_ENGINE,
                requirement="Greeks calculation",
                priority="HIGH",
                acceptance_criteria=[
                    "Delta, Gamma, Vega, Theta, Rho",
                    "Portfolio-level Greeks",
                    "Update on price changes"
                ]
            ),
            SystemRequirement(
                module=ModuleType.RISK_ENGINE,
                requirement="Stress testing",
                priority="CRITICAL",
                acceptance_criteria=[
                    "Historical scenarios (2008, COVID, etc.)",
                    "Hypothetical scenarios",
                    "Custom scenario builder",
                    "Scenario P&L calculation"
                ]
            ),
            
            # Limit Monitor
            SystemRequirement(
                module=ModuleType.LIMIT_MONITOR,
                requirement="Risk limit framework",
                priority="CRITICAL",
                acceptance_criteria=[
                    "Position limits by security",
                    "VaR limits (95%, 99%)",
                    "Loss limits (daily, monthly)",
                    "Concentration limits",
                    "Hard vs soft limits",
                    "Real-time monitoring"
                ]
            ),
            SystemRequirement(
                module=ModuleType.LIMIT_MONITOR,
                requirement="Pre-trade risk checks",
                priority="CRITICAL",
                acceptance_criteria=[
                    "Check limits before trade execution",
                    "Reject trades that breach hard limits",
                    "Warn on soft limit breaches",
                    "< 10ms check latency"
                ]
            ),
            SystemRequirement(
                module=ModuleType.LIMIT_MONITOR,
                requirement="Alerting",
                priority="HIGH",
                acceptance_criteria=[
                    "Email alerts on limit breaches",
                    "Dashboard color-coding",
                    "Alert history/audit log",
                    "Escalation procedures"
                ]
            ),
            
            # Reporting
            SystemRequirement(
                module=ModuleType.REPORTING,
                requirement="Daily risk report",
                priority="CRITICAL",
                acceptance_criteria=[
                    "Automated generation",
                    "PDF/HTML/Excel export",
                    "Include P&L, VaR, CVaR, exposures",
                    "Position breakdown",
                    "Limit status",
                    "Email distribution"
                ]
            ),
            SystemRequirement(
                module=ModuleType.REPORTING,
                requirement="Risk dashboard",
                priority="HIGH",
                acceptance_criteria=[
                    "Web-based dashboard",
                    "Real-time updates",
                    "Color-coded risk status",
                    "Charts and visualizations",
                    "Drill-down capability"
                ]
            ),
            SystemRequirement(
                module=ModuleType.REPORTING,
                requirement="Regulatory reports",
                priority="HIGH",
                acceptance_criteria=[
                    "Basel III capital report",
                    "Stress test report",
                    "Export in required formats"
                ]
            ),
            
            # API
            SystemRequirement(
                module=ModuleType.API,
                requirement="REST API",
                priority="HIGH",
                acceptance_criteria=[
                    "RESTful endpoints for all operations",
                    "Authentication and authorization",
                    "Rate limiting",
                    "API documentation",
                    "< 100ms response time"
                ]
            ),
            SystemRequirement(
                module=ModuleType.API,
                requirement="WebSocket API",
                priority="MEDIUM",
                acceptance_criteria=[
                    "Real-time risk metric streaming",
                    "Position updates",
                    "Alert notifications"
                ]
            )
        ]
    
    @staticmethod
    def print_requirements():
        """Print all requirements"""
        requirements = RiskOSRequirements.get_all_requirements()
        
        # Group by module
        by_module = {}
        for req in requirements:
            if req.module not in by_module:
                by_module[req.module] = []
            by_module[req.module].append(req)
        
        print("RiskOS System Requirements")
        print("="*80)
        print()
        
        for module, reqs in by_module.items():
            print(f"{module.value}:")
            print(f"  Requirements: {len(reqs)}")
            for req in reqs:
                print(f"  • {req.requirement} [{req.priority}]")
                print(f"    Acceptance Criteria:")
                for criterion in req.acceptance_criteria:
                    print(f"      - {criterion}")
            print()

# Example
if __name__ == "__main__":
    RiskOSRequirements.print_requirements()
\`\`\`

## Phase 1: Core Infrastructure (Week 1-2)

### Task 1.1: Database Schema

Design and implement database schema:

\`\`\`python
# Example schema structure

class DatabaseSchema:
    """
    RiskOS database schema
    """
    
    tables = {
        'securities': {
            'columns': [
                'symbol VARCHAR(20) PRIMARY KEY',
                'name VARCHAR(200)',
                'asset_class VARCHAR(50)',
                'sector VARCHAR(100)',
                'country VARCHAR(50)',
                'currency VARCHAR(3)',
                'created_at TIMESTAMP',
                'updated_at TIMESTAMP'
            ],
            'indexes': ['asset_class', 'sector', 'country']
        },
        
        'positions': {
            'columns': [
                'position_id SERIAL PRIMARY KEY',
                'portfolio_id INTEGER',
                'symbol VARCHAR(20) REFERENCES securities(symbol)',
                'quantity DECIMAL(20,4)',
                'cost_basis DECIMAL(20,4)',
                'market_value DECIMAL(20,4)',
                'pnl DECIMAL(20,4)',
                'date DATE',
                'created_at TIMESTAMP',
                'updated_at TIMESTAMP'
            ],
            'indexes': ['portfolio_id', 'symbol', 'date']
        },
        
        'market_data': {
            'columns': [
                'data_id SERIAL PRIMARY KEY',
                'symbol VARCHAR(20) REFERENCES securities(symbol)',
                'timestamp TIMESTAMP',
                'price DECIMAL(20,4)',
                'volume BIGINT',
                'bid DECIMAL(20,4)',
                'ask DECIMAL(20,4)'
            ],
            'indexes': ['symbol', 'timestamp'],
            'partitioning': 'BY RANGE (timestamp)'
        },
        
        'risk_metrics': {
            'columns': [
                'metric_id SERIAL PRIMARY KEY',
                'portfolio_id INTEGER',
                'date DATE',
                'var_95 DECIMAL(20,4)',
                'var_99 DECIMAL(20,4)',
                'cvar_99 DECIMAL(20,4)',
                'max_drawdown DECIMAL(10,6)',
                'sharpe_ratio DECIMAL(10,4)',
                'net_exposure DECIMAL(20,4)',
                'gross_exposure DECIMAL(20,4)',
                'created_at TIMESTAMP'
            ],
            'indexes': ['portfolio_id', 'date']
        },
        
        'trades': {
            'columns': [
                'trade_id SERIAL PRIMARY KEY',
                'portfolio_id INTEGER',
                'symbol VARCHAR(20) REFERENCES securities(symbol)',
                'side VARCHAR(10)',  # BUY, SELL, SHORT, COVER
                'quantity DECIMAL(20,4)',
                'price DECIMAL(20,4)',
                'trade_value DECIMAL(20,4)',
                'commission DECIMAL(20,4)',
                'trade_date DATE',
                'settlement_date DATE',
                'created_at TIMESTAMP'
            ],
            'indexes': ['portfolio_id', 'symbol', 'trade_date']
        },
        
        'risk_limits': {
            'columns': [
                'limit_id SERIAL PRIMARY KEY',
                'portfolio_id INTEGER',
                'limit_type VARCHAR(50)',  # VAR, POSITION, LOSS, etc.
                'limit_name VARCHAR(200)',
                'threshold DECIMAL(20,4)',
                'current_value DECIMAL(20,4)',
                'is_hard_limit BOOLEAN',
                'created_at TIMESTAMP',
                'updated_at TIMESTAMP'
            ],
            'indexes': ['portfolio_id', 'limit_type']
        },
        
        'limit_breaches': {
            'columns': [
                'breach_id SERIAL PRIMARY KEY',
                'limit_id INTEGER REFERENCES risk_limits(limit_id)',
                'breach_time TIMESTAMP',
                'threshold DECIMAL(20,4)',
                'actual_value DECIMAL(20,4)',
                'excess DECIMAL(20,4)',
                'action_taken VARCHAR(200)',
                'resolved_at TIMESTAMP'
            ],
            'indexes': ['limit_id', 'breach_time']
        }
    }
\`\`\`

### Task 1.2: Data Ingestion Pipeline

Implement data ingestion:

\`\`\`python
import pandas as pd
from typing import Dict
from datetime import datetime

class DataIngestionPipeline:
    """
    Ingest market data and positions
    """
    
    def ingest_market_data_csv(self, filepath: str) -> pd.DataFrame:
        """
        Ingest market data from CSV
        
        Expected format:
        timestamp,symbol,price,volume,bid,ask
        """
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Validate
        required_cols = ['timestamp', 'symbol', 'price']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        # Quality checks
        df = df.dropna(subset=['price'])
        df = df[df['price'] > 0]
        
        return df
    
    def ingest_positions_csv(self, filepath: str) -> pd.DataFrame:
        """
        Ingest positions from CSV
        
        Expected format:
        portfolio_id,symbol,quantity,cost_basis,date
        """
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        
        # Validate
        required_cols = ['portfolio_id', 'symbol', 'quantity']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        return df

# TODO: Implement full pipeline with:
# - API ingestion
# - WebSocket streaming
# - Data validation
# - Error handling
# - Logging
\`\`\`

## Phase 2: Risk Engine (Week 3-4)

### Task 2.1: VaR Calculator

Implement all three VaR methods:

\`\`\`python
# TODO: Implement comprehensive VaR calculator
# Requirements:
# - Historical VaR (1 day, 10 days)
# - Parametric VaR with covariance matrix
# - Monte Carlo VaR with 100K+ simulations
# - Portfolio and position-level VaR
# - Component VaR and Marginal VaR
# - Backtesting with traffic light system
# - Performance: < 60s for 10,000 position portfolio
\`\`\`

### Task 2.2: Stress Testing Engine

Implement stress testing:

\`\`\`python
# TODO: Implement stress testing engine
# Requirements:
# - Historical scenarios (2008 crisis, COVID, etc.)
# - Hypothetical scenarios
# - Custom scenario builder
# - Factor-based stress testing
# - P&L calculation under stress
# - Report generation
\`\`\`

## Phase 3: Limit Framework (Week 5-6)

### Task 3.1: Limit Monitoring System

\`\`\`python
# TODO: Implement comprehensive limit monitoring
# Requirements:
# - Multiple limit types (position, VaR, loss, concentration)
# - Hard vs soft limits
# - Real-time monitoring
# - Pre-trade checks (< 10ms)
# - Automatic alerts
# - Breach history
# - Kill switch for emergencies
\`\`\`

### Task 3.2: Pre-Trade Risk Check

\`\`\`python
# TODO: Implement pre-trade risk check
# Requirements:
# - Check all applicable limits
# - Reject hard limit breaches
# - Warn on soft limit breaches
# - Calculate impact on portfolio metrics
# - < 10ms latency
# - Detailed rejection reasons
\`\`\`

## Phase 4: Reporting & Dashboard (Week 7-8)

### Task 4.1: Daily Risk Report

\`\`\`python
# TODO: Implement automated daily risk report
# Requirements:
# - HTML/PDF/Excel output
# - Executive summary
# - P&L by strategy/sector/geography
# - VaR, CVaR, Greeks
# - Top positions
# - Limit status
# - Scenario analysis results
# - Automated email distribution
\`\`\`

### Task 4.2: Web Dashboard

\`\`\`python
# TODO: Implement web dashboard
# Technology suggestions:
# - Backend: FastAPI or Flask
# - Frontend: React or Vue.js
# - Real-time: WebSocket
# - Charts: Plotly or Chart.js
# 
# Requirements:
# - Real-time risk metrics
# - Position breakdown
# - Limit monitoring
# - P&L charts
# - Drill-down capability
# - Color-coded status indicators
# - Mobile responsive
\`\`\`

## Phase 5: Integration & Testing (Week 9-10)

### Task 5.1: Integration Testing

Test complete system:

\`\`\`python
# TODO: Implement comprehensive tests
# Requirements:
# - Unit tests for all modules (>80% coverage)
# - Integration tests for workflows
# - Load testing (10,000 positions)
# - Stress testing (market crashes)
# - Latency testing (< 100ms APIs)
# - Accuracy testing (compare to known values)
\`\`\`

### Task 5.2: Documentation

\`\`\`python
# TODO: Create complete documentation
# - System architecture diagram
# - API documentation
# - User guide
# - Administrator guide
# - Deployment guide
# - Troubleshooting guide
\`\`\`

## Evaluation Criteria

Your project will be evaluated on:

### Functionality (40%)
- [ ] All modules implemented
- [ ] Meets acceptance criteria
- [ ] Handles edge cases
- [ ] Error handling

### Code Quality (20%)
- [ ] Clean, readable code
- [ ] Proper abstractions
- [ ] DRY principle
- [ ] Type hints
- [ ] Docstrings

### Performance (15%)
- [ ] Meets latency requirements
- [ ] Scales to 10,000 positions
- [ ] Efficient algorithms
- [ ] Optimized queries

### Testing (15%)
- [ ] >80% test coverage
- [ ] Integration tests
- [ ] Edge case tests
- [ ] Performance tests

### Documentation (10%)
- [ ] Architecture documented
- [ ] API documentation
- [ ] User guide
- [ ] Code comments

## Bonus Challenges (+10% each)

1. **Machine Learning Integration**: Predict VaR breaches
2. **Multi-Currency Support**: Handle FX risk
3. **Options Pricing**: Full Black-Scholes with Greeks
4. **Blockchain Integration**: Track crypto positions
5. **Mobile App**: Native iOS/Android dashboard

## Sample Data

Use the following for testing:

**Portfolios**:
- Long-only equity portfolio ($500M, 200 stocks)
- Long/short hedge fund ($1B, 500 positions)
- Multi-asset balanced fund ($2B, 1000 positions)

**Historical Scenarios**:
- Black Monday (1987)
- Dot-com bubble (2000)
- Financial crisis (2008)
- Flash crash (2010)
- COVID crash (2020)

## Deliverables

1. **Source Code**: Complete, working implementation
2. **Documentation**: Architecture, API, user guide
3. **Demo Video**: 10-minute walkthrough
4. **Test Results**: Coverage report, performance metrics
5. **Deployment**: Docker containers or cloud deployment

## Timeline

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1-2 | Infrastructure | Database, ingestion pipeline |
| 3-4 | Risk Engine | VaR, CVaR, stress testing |
| 5-6 | Limits | Monitoring, pre-trade checks |
| 7-8 | Reporting | Daily reports, dashboard |
| 9-10 | Testing | Integration tests, docs |

## Resources

- **Code Templates**: Provided in previous sections
- **Sample Data**: Financial datasets available
- **Libraries**: pandas, numpy, scipy, FastAPI, React
- **Infrastructure**: Docker, PostgreSQL, Redis

## Success Stories

Students who complete this capstone have gone on to:
- Quantitative Risk Analyst at major banks
- Risk Platform Engineer at fintech startups
- Portfolio Manager at hedge funds
- Risk Consultant at big-four firms

## Final Advice

1. **Start Simple**: Get basic version working first
2. **Iterate**: Add features incrementally
3. **Test Constantly**: Don't wait until end
4. **Ask for Help**: Use office hours, forums
5. **Document**: Write docs as you build
6. **Think Production**: Build like it's going to prod

## Conclusion

This capstone integrates everything from Module 15:
- Risk management fundamentals
- VaR and CVaR calculations
- Stress testing
- Market, credit, operational risk
- Limit frameworks
- Real-time monitoring
- Reporting and dashboards
- Aladdin architecture principles

Building RiskOS will give you practical experience building production-grade risk systems. This is the type of project that impresses employers and demonstrates real capability.

**Good luck! You've got this. 🚀**

## Submission

When complete, submit:
1. GitHub repository with code
2. README with setup instructions
3. Demo video (upload to YouTube/Vimeo)
4. Documentation (PDF or website)
5. Test results and performance metrics

**Estimated Time**: 100-150 hours over 10 weeks

**You've reached the end of Module 15. Congratulations! 🎉**
`,
};
