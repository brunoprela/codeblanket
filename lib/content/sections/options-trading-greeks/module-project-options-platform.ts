export const moduleProjectOptionsPlatform = {
    title: 'Module Project: Options Pricing & Trading Platform',
    id: 'module-project-options-platform',
    content: `
# Module Project: Options Pricing & Trading Platform

## Project Overview

Build a **comprehensive options trading platform** that incorporates all concepts from Module 8.

**Objective:** Create a production-ready system for:
- Options pricing (Black-Scholes + Greeks)
- Strategy analysis and recommendation
- Portfolio management
- Risk monitoring and alerts
- Backtesting framework

**Technologies:**
- **Python** for core logic
- **NumPy/SciPy** for numerical computations
- **Pandas** for data management
- **Matplotlib/Plotly** for visualization
- **Streamlit or Flask** for web interface (optional)

---

## Phase 1: Options Pricing Engine

### Requirements

Implement **Black-Scholes pricing** with full Greeks calculation.

\`\`\`python
"""
Phase 1: Options Pricing Engine
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Literal

@dataclass
class OptionPricing:
    """Complete option pricing with Greeks"""
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    implied_vol: float = None


class BlackScholesEngine:
    """
    Production-grade Black-Scholes pricing engine
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.r = risk_free_rate
    
    def price_option(self, S: float, K: float, T: float, sigma: float,
                    option_type: Literal['call', 'put']) -> OptionPricing:
        """
        Price option and calculate all Greeks
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            sigma: Volatility (annualized)
            option_type: 'call' or 'put'
            
        Returns:
            OptionPricing object with price and Greeks
        """
        # Handle edge cases
        if T <= 0:
            if option_type == 'call':
                return OptionPricing(
                    price=max(S - K, 0),
                    delta=1.0 if S > K else 0.0,
                    gamma=0, theta=0, vega=0, rho=0
                )
            else:
                return OptionPricing(
                    price=max(K - S, 0),
                    delta=-1.0 if S < K else 0.0,
                    gamma=0, theta=0, vega=0, rho=0
                )
        
        # Calculate d1 and d2
        d1 = (np.log(S/K) + (self.r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        # Calculate price
        if option_type == 'call':
            price = S*norm.cdf(d1) - K*np.exp(-self.r*T)*norm.cdf(d2)
            delta = norm.cdf(d1)
            theta = (-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) 
                    - self.r*K*np.exp(-self.r*T)*norm.cdf(d2)) / 365
            rho = K*T*np.exp(-self.r*T)*norm.cdf(d2) / 100
        else:  # put
            price = K*np.exp(-self.r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            delta = -norm.cdf(-d1)
            theta = (-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) 
                    + self.r*K*np.exp(-self.r*T)*norm.cdf(-d2)) / 365
            rho = -K*T*np.exp(-self.r*T)*norm.cdf(-d2) / 100
        
        # Greeks (same for call and put)
        gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
        vega = S*norm.pdf(d1)*np.sqrt(T) / 100  # Per 1% change
        
        return OptionPricing(
            price=price,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho
        )
    
    def calculate_implied_volatility(self, market_price: float, S: float, K: float,
                                    T: float, option_type: str,
                                    initial_guess: float = 0.25) -> float:
        """Calculate IV using Newton-Raphson"""
        sigma = initial_guess
        max_iterations = 100
        tolerance = 1e-6
        
        for _ in range(max_iterations):
            option = self.price_option(S, K, T, sigma, option_type)
            
            diff = option.price - market_price
            if abs(diff) < tolerance:
                return sigma
            
            # Newton step
            if option.vega < 1e-10:
                return None
            
            sigma = sigma - diff / (option.vega * 100)  # Vega is per 1%
            sigma = max(0.01, min(sigma, 5.0))  # Keep reasonable
        
        return None  # Did not converge


# Testing
if __name__ == "__main__":
    engine = BlackScholesEngine(risk_free_rate=0.05)
    
    # Price ATM call
    option = engine.price_option(
        S=100, K=100, T=30/365, sigma=0.25, option_type='call'
    )
    
    print("=" * 70)
    print("OPTIONS PRICING ENGINE - TEST")
    print("=" * 70)
    print(f"\\nATM Call (S=$100, K=$100, T=30d, σ=25%):")
    print(f"  Price: ${option.price: .4f
}")
print(f"  Delta: {option.delta:.4f}")
print(f"  Gamma: {option.gamma:.4f}")
print(f"  Theta: ${option.theta:.4f}/day")
print(f"  Vega: ${option.vega:.4f} per 1% IV")
print(f"  Rho: ${option.rho:.4f} per 1% rate")
    
    # Calculate IV
market_price = 3.50
iv = engine.calculate_implied_volatility(
    market_price = market_price, S = 100, K = 100, T = 30 / 365, option_type = 'call'
)
print(f"\\nImplied Volatility for ${market_price} market price: {iv*100:.2f}%")
\`\`\`

---

## Phase 2: Strategy Builder

### Requirements

Create strategy templates for all major strategies.

\`\`\`python
"""
Phase 2: Strategy Builder
"""

from typing import List, Dict
from enum import Enum

class StrategyType(Enum):
    LONG_CALL = "Long Call"
    LONG_PUT = "Long Put"
    COVERED_CALL = "Covered Call"
    PROTECTIVE_PUT = "Protective Put"
    BULL_CALL_SPREAD = "Bull Call Spread"
    BEAR_PUT_SPREAD = "Bear Put Spread"
    IRON_CONDOR = "Iron Condor"
    LONG_STRADDLE = "Long Straddle"
    SHORT_STRANGLE = "Short Strangle"
    # Add more...


@dataclass
class OptionLeg:
    """Single option leg"""
    option_type: Literal['call', 'put']
    strike: float
    expiration_days: int
    quantity: int  # Positive = long, negative = short
    pricing: OptionPricing = None


class Strategy:
    """Options strategy container"""
    
    def __init__(self, name: str, legs: List[OptionLeg], stock_position: int = 0):
        self.name = name
        self.legs = legs
        self.stock_position = stock_position  # For covered strategies
        
    def calculate_payoff(self, stock_prices: np.ndarray, 
                        current_stock: float) -> np.ndarray:
        """Calculate P&L at expiration"""
        payoff = np.zeros_like(stock_prices)
        
        # Stock position P&L
        if self.stock_position != 0:
            payoff += self.stock_position * (stock_prices - current_stock)
        
        # Options P&L
        for leg in self.legs:
            if leg.option_type == 'call':
                option_payoff = np.maximum(stock_prices - leg.strike, 0)
            else:
                option_payoff = np.maximum(leg.strike - stock_prices, 0)
            
            # Adjust for long/short and quantity
            if leg.pricing:
                # Include premium paid/received
                payoff += leg.quantity * (option_payoff - leg.pricing.price) * 100
            else:
                payoff += leg.quantity * option_payoff * 100
        
        return payoff
    
    def calculate_greeks(self, engine: BlackScholesEngine, 
                        current_stock: float, current_vol: float) -> Dict:
        """Calculate aggregate Greeks"""
        total_delta = self.stock_position  # Stock delta = 1 per share
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        total_rho = 0
        
        for leg in self.legs:
            T = leg.expiration_days / 365
            pricing = engine.price_option(
                S=current_stock,
                K=leg.strike,
                T=T,
                sigma=current_vol,
                option_type=leg.option_type
            )
            
            leg.pricing = pricing
            
            # Aggregate Greeks (multiply by quantity × 100)
            multiplier = leg.quantity * 100
            total_delta += pricing.delta * multiplier
            total_gamma += pricing.gamma * multiplier
            total_theta += pricing.theta * multiplier
            total_vega += pricing.vega * multiplier
            total_rho += pricing.rho * multiplier
        
        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'theta': total_theta,
            'vega': total_vega,
            'rho': total_rho
        }


class StrategyBuilder:
    """Factory for creating common strategies"""
    
    @staticmethod
    def create_iron_condor(stock_price: float, wing_width: float = 10,
                          expiration: int = 30) -> Strategy:
        """Create iron condor strategy"""
        put_short_strike = stock_price * 0.90
        put_long_strike = put_short_strike - wing_width
        call_short_strike = stock_price * 1.10
        call_long_strike = call_short_strike + wing_width
        
        legs = [
            OptionLeg('put', put_long_strike, expiration, 1),   # Buy OTM put
            OptionLeg('put', put_short_strike, expiration, -1),  # Sell put
            OptionLeg('call', call_short_strike, expiration, -1), # Sell call
            OptionLeg('call', call_long_strike, expiration, 1),  # Buy OTM call
        ]
        
        return Strategy("Iron Condor", legs)
    
    @staticmethod
    def create_long_straddle(stock_price: float, expiration: int = 30) -> Strategy:
        """Create long straddle"""
        legs = [
            OptionLeg('call', stock_price, expiration, 1),
            OptionLeg('put', stock_price, expiration, 1),
        ]
        return Strategy("Long Straddle", legs)
    
    # Add more factory methods...


# Testing
if __name__ == "__main__":
    engine = BlackScholesEngine()
    
    # Create iron condor
    strategy = StrategyBuilder.create_iron_condor(stock_price=100)
    
    # Calculate Greeks
    greeks = strategy.calculate_greeks(engine, current_stock=100, current_vol=0.25)
    
    print("=" * 70)
    print("STRATEGY BUILDER - IRON CONDOR TEST")
    print("=" * 70)
    print(f"\\nStrategy: {strategy.name}")
    print(f"Legs: {len(strategy.legs)}")
    print(f"\\nAggregate Greeks:")
    for greek, value in greeks.items():
        print(f"  {greek.capitalize()}: {value:.2f}")
    
    # Calculate payoff
    stock_prices = np.linspace(70, 130, 200)
    payoff = strategy.calculate_payoff(stock_prices, current_stock=100)
    
    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(stock_prices, payoff, linewidth=2)
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.xlabel('Stock Price at Expiration')
    plt.ylabel('Profit / Loss')
    plt.title('Iron Condor Payoff')
    plt.grid(True, alpha=0.3)
    plt.show()
\`\`\`

---

## Phase 3: Portfolio Manager

### Requirements

Manage multiple positions with risk monitoring.

\`\`\`python
"""
Phase 3: Portfolio Manager
"""

class Portfolio:
    """Portfolio management system"""
    
    def __init__(self, capital: float):
        self.capital = capital
        self.positions: List[Strategy] = []
        self.engine = BlackScholesEngine()
        
        # Risk limits
        self.max_delta = 10000
        self.max_gamma = 500
        self.max_vega = 50000
    
    def add_position(self, strategy: Strategy):
        """Add strategy to portfolio"""
        self.positions.append(strategy)
    
    def calculate_portfolio_greeks(self, stock_price: float, 
                                   vol: float) -> Dict:
        """Aggregate Greeks across all positions"""
        totals = {
            'delta': 0,
            'gamma': 0,
            'theta': 0,
            'vega': 0,
            'rho': 0
        }
        
        for position in self.positions:
            greeks = position.calculate_greeks(self.engine, stock_price, vol)
            for key in totals:
                totals[key] += greeks[key]
        
        return totals
    
    def check_risk_limits(self, greeks: Dict) -> List[str]:
        """Check if any risk limits breached"""
        alerts = []
        
        if abs(greeks['delta']) > self.max_delta:
            alerts.append(f"⚠️  Delta limit breached: {greeks['delta']:.0f}")
        
        if abs(greeks['gamma']) > self.max_gamma:
            alerts.append(f"⚠️  Gamma limit breached: {greeks['gamma']:.0f}")
        
        if abs(greeks['vega']) > self.max_vega:
            alerts.append(f"⚠️  Vega limit breached: {greeks['vega']:.0f}")
        
        return alerts
    
    def stress_test(self, current_price: float, current_vol: float) -> pd.DataFrame:
        """Run stress test scenarios"""
        scenarios = [
            {'name': 'Current', 'price_change': 0, 'iv_change': 0},
            {'name': 'Up 5%', 'price_change': 0.05, 'iv_change': -0.03},
            {'name': 'Down 5%', 'price_change': -0.05, 'iv_change': 0.03},
            {'name': 'Down 10%', 'price_change': -0.10, 'iv_change': 0.10},
            {'name': 'Crash -20%', 'price_change': -0.20, 'iv_change': 0.20},
        ]
        
        results = []
        current_greeks = self.calculate_portfolio_greeks(current_price, current_vol)
        
        for scenario in scenarios:
            new_price = current_price * (1 + scenario['price_change'])
            new_vol = current_vol + scenario['iv_change']
            price_move = new_price - current_price
            
            # Calculate P&L
            delta_pnl = current_greeks['delta'] * price_move
            gamma_pnl = 0.5 * current_greeks['gamma'] * (price_move ** 2)
            vega_pnl = current_greeks['vega'] * scenario['iv_change'] * 100
            theta_pnl = current_greeks['theta']  # 1 day
            
            total_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
            
            results.append({
                'Scenario': scenario['name'],
                'Price': f"${new_price: .2f}",
'Delta P&L': f"${delta_pnl:,.0f}",
    'Gamma P&L': f"${gamma_pnl:,.0f}",
        'Vega P&L': f"${vega_pnl:,.0f}",
            'Total P&L': f"${total_pnl:,.0f}",
                'Portfolio %': f"{(total_pnl/self.capital*100):.2f}%"
            })

return pd.DataFrame(results)


# Testing
portfolio = Portfolio(capital = 1_000_000)

# Add positions
portfolio.add_position(StrategyBuilder.create_iron_condor(100))
portfolio.add_position(StrategyBuilder.create_long_straddle(100))

# Check risk
greeks = portfolio.calculate_portfolio_greeks(stock_price = 100, vol = 0.25)
alerts = portfolio.check_risk_limits(greeks)

print("=" * 70)
print("PORTFOLIO MANAGER TEST")
print("=" * 70)
print(f"\\nPortfolio Capital: ${portfolio.capital:,.0f}")
print(f"Positions: {len(portfolio.positions)}")
print(f"\\nPortfolio Greeks:")
for greek, value in greeks.items():
    print(f"  {greek.capitalize()}: {value:,.2f}")

if alerts:
    print(f"\\n⚠️  RISK ALERTS:")
for alert in alerts:
    print(f"  {alert}")
else:
print(f"\\n✓ All risk limits within acceptable range")

# Stress test
stress_results = portfolio.stress_test(current_price = 100, current_vol = 0.25)
print(f"\\nStress Test Results:")
print(stress_results.to_string(index = False))
\`\`\`

---

## Phase 4: Backtesting Framework

### Requirements

Test strategies on historical data.

\`\`\`python
"""
Phase 4: Backtesting Framework
"""

class Backtester:
    """Backtest options strategies"""
    
    def __init__(self, historical_data: pd.DataFrame):
        """
        historical_data: DataFrame with columns:
        - date, open, high, low, close, volume, implied_vol
        """
        self.data = historical_data
        self.engine = BlackScholesEngine()
    
    def run_strategy_backtest(self, strategy_func, **params):
        """
        Backtest a strategy over historical data
        
        Args:
            strategy_func: Function that returns a Strategy
            params: Parameters for strategy
        """
        results = []
        
        for i in range(len(self.data) - params.get('holding_period', 30)):
            entry_date = self.data.index[i]
            entry_price = self.data.loc[entry_date, 'close']
            entry_vol = self.data.loc[entry_date, 'implied_vol']
            
            # Create strategy
            strategy = strategy_func(entry_price, **params)
            
            # Hold for period
            exit_idx = i + params.get('holding_period', 30)
            exit_date = self.data.index[exit_idx]
            exit_price = self.data.loc[exit_date, 'close']
            
            # Calculate P&L
            pnl = self.calculate_strategy_pnl(
                strategy, entry_price, exit_price, entry_vol
            )
            
            results.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'return_pct': (pnl / strategy_cost(strategy)) * 100
            })
        
        return pd.DataFrame(results)
\`\`\`

---

## Deliverables

**1. Core Engine** (pricing + Greeks)
**2. Strategy Library** (10+ strategies)
**3. Portfolio Manager** (risk monitoring)
**4. Backtester** (historical testing)
**5. Web Interface** (optional: Streamlit dashboard)
**6. Documentation** (README, API docs)

**Evaluation Criteria:**
- Code quality and organization
- Accuracy of pricing and Greeks
- Risk management implementation
- Backtest methodology
- User interface (if applicable)

**Bonus Features:**
- Real-time data integration (Yahoo Finance API)
- Machine learning for IV prediction
- Advanced Greeks (volga, vanna, charm)
- Monte Carlo simulation
- Options chain analyzer

This is your capstone project - demonstrate mastery of all Module 8 concepts!
`,
};

