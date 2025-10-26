export const portfolioOptimizationProject = {
    title: 'Module Project: Portfolio Optimization Platform',
    id: 'portfolio-optimization-project',
    content: `
# Module Project: Portfolio Optimization Platform

## Project Overview

Build a **production-grade portfolio optimization platform** that integrates all concepts from this module:

**Core Features**:
1. Mean-variance optimization (Markowitz, Sharpe ratio maximization)
2. Risk parity and risk budgeting
3. Factor analysis (Fama-French)
4. Constrained optimization (sector limits, ESG screening, turnover)
5. Rebalancing engine (calendar, threshold, adaptive)
6. Professional backtesting with realistic costs
7. Performance analytics and reporting
8. Web dashboard for visualization

**Technology Stack**:
- **Backend**: Python (FastAPI)
- **Optimization**: CVXPY, scipy.optimize
- **Data**: yfinance, pandas, numpy
- **Factor Models**: pandas-datareader (Kenneth French data)
- **Testing**: pytest
- **Visualization**: Plotly
- **Frontend** (optional): React + Recharts

**What Makes This Production-Grade**:
- Error handling and validation
- Logging and monitoring
- Unit tests and integration tests
- Configuration management
- API endpoints for integration
- Scalable architecture
- Documentation

**Time Estimate**: 10-15 hours for full implementation.

**Learning Outcomes**:
1. Integrate multiple portfolio optimization techniques
2. Build production-quality financial software
3. Handle real-world complexities (costs, constraints, data quality)
4. Create reusable, modular codebase
5. Develop API for portfolio management system

---

## System Architecture

### High-Level Design

\`\`\`
┌─────────────────────────────────────────────────────────┐
│                    Web Dashboard                       │
│                (React/Streamlit)                       │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   REST API (FastAPI)                   │
│  /optimize, /backtest, /rebalance, /performance       │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Optimization │  │  Backtesting │  │   Analytics  │
│    Engine    │  │    Engine    │  │    Engine    │
└──────────────┘  └──────────────┘  └──────────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
        ┌─────────────────────────────────┐
        │      Data Management Layer      │
        │  (Cache, Validation, Storage)   │
        └─────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────┐
        │       External Data Sources     │
        │  (Yahoo Finance, Ken French,    │
        │   Alternative Data APIs)        │
        └─────────────────────────────────┘
\`\`\`

### Module Structure

\`\`\`
portfolio_platform/
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI app
│   │   ├── routes/
│   │   │   ├── optimization.py     # Optimization endpoints
│   │   │   ├── backtesting.py      # Backtest endpoints
│   │   │   ├── analytics.py        # Analytics endpoints
│   │   │   └── portfolio.py        # Portfolio management
│   │   └── models.py               # Pydantic models
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── optimizer.py            # Main optimization engine
│   │   ├── risk_models.py          # Risk parity, budgeting
│   │   ├── factor_models.py        # Fama-French analysis
│   │   ├── constraints.py          # Constraint builders
│   │   └── rebalancer.py           # Rebalancing logic
│   │
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── engine.py               # Backtest engine
│   │   ├── execution.py            # Trade execution simulator
│   │   └── performance.py          # Performance metrics
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── providers.py            # Data source adapters
│   │   ├── cache.py                # Caching layer
│   │   └── validation.py           # Data validation
│   │
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── metrics.py              # Performance metrics
│   │   ├── attribution.py          # Return attribution
│   │   └── reporting.py            # Report generation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py               # Configuration
│       ├── logging.py              # Logging setup
│       └── exceptions.py           # Custom exceptions
│
├── tests/
│   ├── __init__.py
│   ├── test_optimizer.py
│   ├── test_backtest.py
│   └── test_api.py
│
├── config/
│   ├── default.yaml                # Default configuration
│   └── production.yaml             # Production config
│
├── notebooks/
│   ├── examples.ipynb              # Usage examples
│   └── analysis.ipynb              # Portfolio analysis
│
├── requirements.txt
├── setup.py
├── README.md
└── docker-compose.yml
\`\`\`

---

## Core Implementation

### 1. Unified Optimizer Engine

\`\`\`python
"""
src/core/optimizer.py - Unified Portfolio Optimization Engine
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    """Supported optimization objectives."""
    MAX_SHARPE = "max_sharpe"
    MIN_RISK = "min_risk"
    MAX_RETURN = "max_return"
    RISK_PARITY = "risk_parity"
    CUSTOM_RISK_BUDGET = "custom_risk_budget"

@dataclass
class OptimizationResult:
    """Container for optimization results."""
    weights: np.ndarray
    tickers: List[str]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    risk_contributions: Optional[np.ndarray] = None
    factor_exposures: Optional[Dict[str, float]] = None
    status: str = "optimal"
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'weights': {ticker: float(w) for ticker, w in zip(self.tickers, self.weights)},
            'expected_return': float(self.expected_return),
            'volatility': float(self.volatility),
            'sharpe_ratio': float(self.sharpe_ratio),
            'risk_contributions': self.risk_contributions.tolist() if self.risk_contributions is not None else None,
            'factor_exposures': self.factor_exposures,
            'status': self.status,
            'metadata': self.metadata or {}
        }

class PortfolioOptimizer:
    """
    Unified portfolio optimization engine supporting multiple objectives and constraints.
    """
    
    def __init__(self,
                 mean_returns: pd.Series,
                 cov_matrix: pd.DataFrame,
                 risk_free_rate: float = 0.04):
        """
        Initialize optimizer.
        
        Args:
            mean_returns: Expected returns (annualized)
            cov_matrix: Covariance matrix (annualized)
            risk_free_rate: Risk-free rate
        """
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.tickers = list(mean_returns.index)
        self.n_assets = len(self.tickers)
        self.risk_free_rate = risk_free_rate
        
        # Validate inputs
        self._validate_inputs()
        
        logger.info(f"Initialized optimizer with {self.n_assets} assets")
    
    def _validate_inputs(self):
        """Validate input data."""
        if len(self.mean_returns) != len(self.cov_matrix):
            raise ValueError("Mean returns and covariance matrix dimensions don't match")
        
        if not np.allclose(self.cov_matrix, self.cov_matrix.T):
            raise ValueError("Covariance matrix must be symmetric")
        
        # Check positive semi-definite
        eigenvalues = np.linalg.eigvalsh(self.cov_matrix)
        if np.any(eigenvalues < -1e-8):
            logger.warning("Covariance matrix is not positive semi-definite, adjusting...")
            # Add small ridge to diagonal
            self.cov_matrix = self.cov_matrix + np.eye(self.n_assets) * 1e-6
    
    def optimize(self,
                 objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
                 constraints: Optional[Dict] = None,
                 **kwargs) -> OptimizationResult:
        """
        Optimize portfolio.
        
        Args:
            objective: Optimization objective
            constraints: Dictionary of constraints
            **kwargs: Additional parameters (target_return, target_risk, risk_budget)
        
        Returns:
            OptimizationResult with optimal weights and metrics
        """
        logger.info(f"Running optimization: {objective.value}")
        
        try:
            if objective == OptimizationObjective.MAX_SHARPE:
                weights = self._optimize_max_sharpe(constraints)
            elif objective == OptimizationObjective.MIN_RISK:
                weights = self._optimize_min_risk(constraints, kwargs.get('target_return'))
            elif objective == OptimizationObjective.MAX_RETURN:
                weights = self._optimize_max_return(constraints, kwargs.get('target_risk'))
            elif objective == OptimizationObjective.RISK_PARITY:
                weights = self._optimize_risk_parity(constraints)
            elif objective == OptimizationObjective.CUSTOM_RISK_BUDGET:
                risk_budget = kwargs.get('risk_budget')
                if risk_budget is None:
                    raise ValueError("risk_budget required for CUSTOM_RISK_BUDGET objective")
                weights = self._optimize_custom_risk_budget(risk_budget, constraints)
            else:
                raise ValueError(f"Unknown objective: {objective}")
            
            # Calculate metrics
            result = self._build_result(weights, objective)
            
            logger.info(f"Optimization successful: Sharpe={result.sharpe_ratio:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
    
    def _optimize_max_sharpe(self, constraints: Optional[Dict] = None) -> np.ndarray:
        """Maximize Sharpe ratio."""
        w = cp.Variable(self.n_assets)
        
        ret = self.mean_returns.values @ w
        risk = cp.quad_form(w, self.cov_matrix.values)
        
        # Convert to convex problem: maximize return subject to risk = 1
        constraint_list = [
            cp.sum(w) == 1,
            risk <= 1
        ]
        
        constraint_list.extend(self._build_constraints(w, constraints))
        
        prob = cp.Problem(
            cp.Maximize(ret - self.risk_free_rate),
            constraint_list
        )
        
        prob.solve(solver=cp.ECOS, verbose=False)
        
        if prob.status not in ['optimal', 'optimal_inaccurate']:
            raise RuntimeError(f"Optimization failed with status: {prob.status}")
        
        # Normalize weights
        weights = w.value / np.sum(w.value)
        return weights
    
    def _optimize_min_risk(self, 
                          constraints: Optional[Dict] = None,
                          target_return: Optional[float] = None) -> np.ndarray:
        """Minimize risk (with optional target return)."""
        w = cp.Variable(self.n_assets)
        
        risk = cp.quad_form(w, self.cov_matrix.values)
        
        constraint_list = [cp.sum(w) == 1]
        
        if target_return is not None:
            constraint_list.append(self.mean_returns.values @ w >= target_return)
        
        constraint_list.extend(self._build_constraints(w, constraints))
        
        prob = cp.Problem(cp.Minimize(risk), constraint_list)
        prob.solve(solver=cp.ECOS, verbose=False)
        
        if prob.status not in ['optimal', 'optimal_inaccurate']:
            raise RuntimeError(f"Optimization failed with status: {prob.status}")
        
        return w.value
    
    def _optimize_max_return(self,
                            constraints: Optional[Dict] = None,
                            target_risk: Optional[float] = None) -> np.ndarray:
        """Maximize return (with optional risk constraint)."""
        w = cp.Variable(self.n_assets)
        
        ret = self.mean_returns.values @ w
        
        constraint_list = [cp.sum(w) == 1]
        
        if target_risk is not None:
            risk = cp.quad_form(w, self.cov_matrix.values)
            constraint_list.append(risk <= target_risk ** 2)
        
        constraint_list.extend(self._build_constraints(w, constraints))
        
        prob = cp.Problem(cp.Maximize(ret), constraint_list)
        prob.solve(solver=cp.ECOS, verbose=False)
        
        if prob.status not in ['optimal', 'optimal_inaccurate']:
            raise RuntimeError(f"Optimization failed with status: {prob.status}")
        
        return w.value
    
    def _optimize_risk_parity(self, constraints: Optional[Dict] = None) -> np.ndarray:
        """Optimize for risk parity (equal risk contribution)."""
        # Start with inverse volatility weights
        volatilities = np.sqrt(np.diag(self.cov_matrix.values))
        init_weights = 1 / volatilities
        init_weights /= init_weights.sum()
        
        def objective(w):
            # Calculate risk contributions
            portfolio_risk = np.sqrt(w @ self.cov_matrix.values @ w)
            marginal_contrib = (self.cov_matrix.values @ w) / portfolio_risk
            risk_contrib = w * marginal_contrib
            
            # Minimize variance of risk contributions
            target = portfolio_risk / self.n_assets
            return np.sum((risk_contrib - target) ** 2)
        
        from scipy.optimize import minimize
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # Bounds
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        result = minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        if not result.success:
            logger.warning(f"Risk parity optimization warning: {result.message}")
        
        return result.x
    
    def _optimize_custom_risk_budget(self,
                                     risk_budget: np.ndarray,
                                     constraints: Optional[Dict] = None) -> np.ndarray:
        """Optimize for custom risk budget."""
        if len(risk_budget) != self.n_assets:
            raise ValueError("Risk budget length must match number of assets")
        
        if not np.isclose(risk_budget.sum(), 1.0):
            raise ValueError("Risk budget must sum to 1")
        
        # Similar to risk parity but with custom targets
        volatilities = np.sqrt(np.diag(self.cov_matrix.values))
        init_weights = risk_budget / volatilities
        init_weights /= init_weights.sum()
        
        def objective(w):
            portfolio_risk = np.sqrt(w @ self.cov_matrix.values @ w)
            marginal_contrib = (self.cov_matrix.values @ w) / portfolio_risk
            risk_contrib = w * marginal_contrib
            risk_contrib_pct = risk_contrib / portfolio_risk
            return np.sum((risk_contrib_pct - risk_budget) ** 2)
        
        from scipy.optimize import minimize
        
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        result = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=cons)
        
        return result.x
    
    def _build_constraints(self, w: cp.Variable, constraints: Optional[Dict]) -> List:
        """Build CVXPY constraints from dictionary."""
        if constraints is None:
            constraints = {}
        
        constraint_list = []
        
        # Long-only (default)
        if constraints.get('long_only', True):
            constraint_list.append(w >= 0)
        
        # Position limits
        if 'max_weight' in constraints:
            max_w = constraints['max_weight']
            if isinstance(max_w, (int, float)):
                constraint_list.append(w <= max_w)
            else:
                for i, mw in enumerate(max_w):
                    constraint_list.append(w[i] <= mw)
        
        if 'min_weight' in constraints:
            min_w = constraints['min_weight']
            if isinstance(min_w, (int, float)):
                constraint_list.append(w >= min_w)
            else:
                for i, mw in enumerate(min_w):
                    constraint_list.append(w[i] >= mw)
        
        # Sector constraints
        if 'sector_exposure' in constraints:
            for sector, (assets, min_w, max_w) in constraints['sector_exposure'].items():
                sector_weight = cp.sum([w[i] for i in assets])
                if min_w is not None:
                    constraint_list.append(sector_weight >= min_w)
                if max_w is not None:
                    constraint_list.append(sector_weight <= max_w)
        
        # Risk limit
        if 'max_risk' in constraints:
            max_risk = constraints['max_risk']
            constraint_list.append(
                cp.quad_form(w, self.cov_matrix.values) <= max_risk ** 2
            )
        
        # More constraints can be added here...
        
        return constraint_list
    
    def _calculate_risk_contributions(self, weights: np.ndarray) -> np.ndarray:
        """Calculate each asset's contribution to portfolio risk."""
        portfolio_risk = np.sqrt(weights @ self.cov_matrix.values @ weights)
        marginal_contrib = (self.cov_matrix.values @ weights) / portfolio_risk
        risk_contrib = weights * marginal_contrib
        return risk_contrib
    
    def _build_result(self, 
                     weights: np.ndarray,
                     objective: OptimizationObjective) -> OptimizationResult:
        """Build OptimizationResult from weights."""
        expected_return = float(self.mean_returns.values @ weights)
        volatility = float(np.sqrt(weights @ self.cov_matrix.values @ weights))
        sharpe = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        risk_contrib = self._calculate_risk_contributions(weights)
        
        return OptimizationResult(
            weights=weights,
            tickers=self.tickers,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            risk_contributions=risk_contrib,
            status="optimal",
            metadata={'objective': objective.value}
        )
    
    def efficient_frontier(self,
                          n_points: int = 50,
                          constraints: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculate efficient frontier.
        
        Args:
            n_points: Number of points on frontier
            constraints: Optimization constraints
        
        Returns:
            DataFrame with returns, risks, and weights
        """
        logger.info(f"Calculating efficient frontier with {n_points} points")
        
        min_return = self.mean_returns.min()
        max_return = self.mean_returns.max()
        target_returns = np.linspace(min_return, max_return, n_points)
        
        results = []
        
        for target_ret in target_returns:
            try:
                result = self.optimize(
                    OptimizationObjective.MIN_RISK,
                    constraints=constraints,
                    target_return=target_ret
                )
                
                results.append({
                    'return': result.expected_return,
                    'risk': result.volatility,
                    'sharpe': result.sharpe_ratio,
                    'weights': result.weights
                })
            except Exception as e:
                logger.debug(f"Failed to optimize for return {target_ret:.3f}: {e}")
                continue
        
        df = pd.DataFrame(results)
        logger.info(f"✓ Calculated {len(df)} efficient frontier points")
        
        return df

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Simulate data
    np.random.seed(42)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    mean_returns = pd.Series(
        np.random.uniform(0.08, 0.15, len(tickers)),
        index=tickers
    )
    
    # Correlation matrix
    corr = np.array([
        [1.00, 0.70, 0.65, 0.60, 0.55],
        [0.70, 1.00, 0.75, 0.65, 0.60],
        [0.65, 0.75, 1.00, 0.70, 0.65],
        [0.60, 0.65, 0.70, 1.00, 0.70],
        [0.55, 0.60, 0.65, 0.70, 1.00]
    ])
    
    volatilities = np.random.uniform(0.20, 0.35, len(tickers))
    cov_matrix = pd.DataFrame(
        np.outer(volatilities, volatilities) * corr,
        index=tickers,
        columns=tickers
    )
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(mean_returns, cov_matrix)
    
    # Max Sharpe
    print("\\n1. Max Sharpe Portfolio:")
    result = optimizer.optimize(OptimizationObjective.MAX_SHARPE)
    print(f"Expected Return: {result.expected_return:.2%}")
    print(f"Volatility: {result.volatility:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
    print("\\nWeights:")
    for ticker, weight in zip(result.tickers, result.weights):
        print(f"  {ticker}: {weight:.2%}")
    
    # Risk Parity
    print("\\n2. Risk Parity Portfolio:")
    result_rp = optimizer.optimize(OptimizationObjective.RISK_PARITY)
    print(f"Expected Return: {result_rp.expected_return:.2%}")
    print(f"Volatility: {result_rp.volatility:.2%}")
    print(f"Sharpe Ratio: {result_rp.sharpe_ratio:.3f}")
    print("\\nWeights:")
    for ticker, weight in zip(result_rp.tickers, result_rp.weights):
        print(f"  {ticker}: {weight:.2%}")
    
    # With constraints
    print("\\n3. Max Sharpe with Constraints (max 30% per stock):")
    result_constrained = optimizer.optimize(
        OptimizationObjective.MAX_SHARPE,
        constraints={'max_weight': 0.30}
    )
    print(f"Expected Return: {result_constrained.expected_return:.2%}")
    print(f"Volatility: {result_constrained.volatility:.2%}")
    print(f"Sharpe Ratio: {result_constrained.sharpe_ratio:.3f}")
    print("\\nWeights:")
    for ticker, weight in zip(result_constrained.tickers, result_constrained.weights):
        print(f"  {ticker}: {weight:.2%}")
\`\`\`

---

## API Implementation

### FastAPI Endpoints

\`\`\`python
"""
src/api/main.py - FastAPI Application
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

from src.core.optimizer import PortfolioOptimizer, OptimizationObjective, OptimizationResult
from src.data.providers import DataProvider
from src.backtesting.engine import BacktestEngine, BacktestConfig

app = FastAPI(
    title="Portfolio Optimization Platform",
    description="Professional portfolio optimization and backtesting API",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class OptimizeRequest(BaseModel):
    tickers: List[str] = Field(..., example=["AAPL", "MSFT", "GOOGL"])
    objective: str = Field("max_sharpe", example="max_sharpe")
    start_date: str = Field(..., example="2020-01-01")
    end_date: str = Field(..., example="2023-12-31")
    constraints: Optional[Dict] = None
    risk_free_rate: float = 0.04

class BacktestRequest(BaseModel):
    tickers: List[str]
    strategy: str = Field("equal_weight", example="equal_weight")
    start_date: str
    end_date: str
    initial_capital: float = 10000
    rebalance_frequency: str = "monthly"
    
# Initialize data provider
data_provider = DataProvider()

@app.get("/")
async def root():
    return {
        "name": "Portfolio Optimization Platform",
        "version": "1.0.0",
        "endpoints": [
            "/optimize",
            "/backtest",
            "/efficient-frontier",
            "/performance"
        ]
    }

@app.post("/optimize")
async def optimize_portfolio(request: OptimizeRequest):
    """
    Optimize portfolio given tickers and objective.
    """
    try:
        # Fetch data
        returns_data = data_provider.get_returns(
            request.tickers,
            request.start_date,
            request.end_date
        )
        
        mean_returns = returns_data.mean() * 252
        cov_matrix = returns_data.cov() * 252
        
        # Initialize optimizer
        optimizer = PortfolioOptimizer(
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            risk_free_rate=request.risk_free_rate
        )
        
        # Run optimization
        objective = OptimizationObjective(request.objective)
        result = optimizer.optimize(
            objective=objective,
            constraints=request.constraints
        )
        
        return result.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/backtest")
async def backtest_strategy(request: BacktestRequest):
    """
    Backtest a portfolio strategy.
    """
    try:
        config = BacktestConfig(
            initial_capital=request.initial_capital,
            rebalance_frequency=request.rebalance_frequency
        )
        
        engine = BacktestEngine(
            tickers=request.tickers,
            start_date=request.start_date,
            end_date=request.end_date,
            config=config
        )
        
        # Define strategy function based on request
        if request.strategy == "equal_weight":
            def strategy(date, hist_prices):
                n = len(request.tickers)
                return {ticker: 1/n for ticker in request.tickers}
        
        # Run backtest
        results = engine.run(strategy)
        metrics = engine.get_performance_metrics(results)
        
        return {
            "performance": metrics,
            "portfolio_values": results['Portfolio Value'].tolist(),
            "dates": [d.strftime('%Y-%m-%d') for d in results.index]
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/efficient-frontier")
async def calculate_efficient_frontier(request: OptimizeRequest):
    """
    Calculate efficient frontier for given assets.
    """
    try:
        # Fetch data
        returns_data = data_provider.get_returns(
            request.tickers,
            request.start_date,
            request.end_date
        )
        
        mean_returns = returns_data.mean() * 252
        cov_matrix = returns_data.cov() * 252
        
        # Initialize optimizer
        optimizer = PortfolioOptimizer(
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            risk_free_rate=request.risk_free_rate
        )
        
        # Calculate frontier
        frontier = optimizer.efficient_frontier(
            n_points=50,
            constraints=request.constraints
        )
        
        return {
            "returns": frontier['return'].tolist(),
            "risks": frontier['risk'].tolist(),
            "sharpes": frontier['sharpe'].tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
\`\`\`

---

## Testing Strategy

### Unit Tests

\`\`\`python
"""
tests/test_optimizer.py
"""

import pytest
import numpy as np
import pandas as pd
from src.core.optimizer import PortfolioOptimizer, OptimizationObjective

@pytest.fixture
def sample_data():
    """Create sample returns and covariance."""
    tickers = ['A', 'B', 'C']
    mean_returns = pd.Series([0.10, 0.12, 0.08], index=tickers)
    
    corr = np.array([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.4],
        [0.3, 0.4, 1.0]
    ])
    
    volatilities = np.array([0.15, 0.20, 0.12])
    cov_matrix = pd.DataFrame(
        np.outer(volatilities, volatilities) * corr,
        index=tickers,
        columns=tickers
    )
    
    return mean_returns, cov_matrix

def test_optimizer_initialization(sample_data):
    """Test optimizer initialization."""
    mean_returns, cov_matrix = sample_data
    optimizer = PortfolioOptimizer(mean_returns, cov_matrix)
    
    assert optimizer.n_assets == 3
    assert len(optimizer.tickers) == 3

def test_max_sharpe_optimization(sample_data):
    """Test max Sharpe optimization."""
    mean_returns, cov_matrix = sample_data
    optimizer = PortfolioOptimizer(mean_returns, cov_matrix)
    
    result = optimizer.optimize(OptimizationObjective.MAX_SHARPE)
    
    assert result.status == "optimal"
    assert np.isclose(result.weights.sum(), 1.0)
    assert np.all(result.weights >= 0)
    assert result.sharpe_ratio > 0

def test_risk_parity_optimization(sample_data):
    """Test risk parity optimization."""
    mean_returns, cov_matrix = sample_data
    optimizer = PortfolioOptimizer(mean_returns, cov_matrix)
    
    result = optimizer.optimize(OptimizationObjective.RISK_PARITY)
    
    assert np.isclose(result.weights.sum(), 1.0)
    
    # Check risk contributions are approximately equal
    risk_contrib = result.risk_contributions
    assert np.std(risk_contrib) < np.mean(risk_contrib) * 0.1  # Within 10%

def test_constrained_optimization(sample_data):
    """Test optimization with constraints."""
    mean_returns, cov_matrix = sample_data
    optimizer = PortfolioOptimizer(mean_returns, cov_matrix)
    
    constraints = {'max_weight': 0.40}
    result = optimizer.optimize(
        OptimizationObjective.MAX_SHARPE,
        constraints=constraints
    )
    
    assert np.all(result.weights <= 0.41)  # Small tolerance
    assert np.isclose(result.weights.sum(), 1.0)

def test_efficient_frontier(sample_data):
    """Test efficient frontier calculation."""
    mean_returns, cov_matrix = sample_data
    optimizer = PortfolioOptimizer(mean_returns, cov_matrix)
    
    frontier = optimizer.efficient_frontier(n_points=10)
    
    assert len(frontier) > 0
    assert 'return' in frontier.columns
    assert 'risk' in frontier.columns
    assert 'sharpe' in frontier.columns
    
    # Risk should be monotonically increasing (approximately)
    assert np.all(np.diff(frontier['risk'].values) >= -1e-6)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
\`\`\`

---

## Deployment

### Docker Configuration

\`\`\`dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY src/ ./src/
COPY config/ ./config/

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
\`\`\`

\`\`\`yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
      - CACHE_ENABLED=true
    volumes:
      - ./config:/app/config
      - ./data:/app/data
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
\`\`\`

---

## Project Extensions

### Extension Ideas

1. **Real-Time Monitoring**: Track live portfolio performance
2. **Alert System**: Notify when rebalancing needed or risk limits breached
3. **Machine Learning Integration**: ML-based return predictions
4. **Multi-Currency Support**: Handle FX risk
5. **Tax Optimization**: Tax-loss harvesting, asset location
6. **Scenario Analysis**: Stress testing, Monte Carlo simulation
7. **Client Portal**: Individual client portfolios
8. **Reporting**: Automated PDF reports

---

## Key Takeaways

1. **Production-Grade Platform**: Integrated optimization, backtesting, rebalancing, and analytics in modular architecture.

2. **Core Components**:
   - Unified optimizer supporting multiple objectives (Sharpe, risk parity, custom)
   - Professional backtesting with realistic costs and slippage
   - REST API for integration
   - Comprehensive testing

3. **Technologies**: Python, CVXPY, FastAPI, pandas, numpy, pytest, Docker.

4. **Best Practices**:
   - Error handling and validation
   - Logging and monitoring
   - Unit and integration tests
   - Configuration management
   - API documentation
   - Containerization

5. **Real-World Features**:
   - Multiple optimization objectives
   - Flexible constraints system
   - Transaction costs and slippage
   - Performance attribution
   - Efficient frontier calculation

**Congratulations!** You've completed Module 6: Portfolio Theory & Asset Allocation. You now have the skills to build production-grade portfolio optimization systems used by hedge funds, asset managers, and fintech companies. This platform can serve as the foundation for a robo-advisor, investment management tool, or quantitative research system.
`,
};

