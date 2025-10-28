export const marketRiskManagement = {
  id: 'market-risk-management',
  title: 'Market Risk Management',
  content: `
# Market Risk Management

## Introduction

Market risk - the risk of losses from adverse price movements - is the most visible and actively managed risk in financial institutions. Every second, trillions of dollars of market risk are being measured, monitored, hedged, and traded globally.

Unlike credit risk (which materializes slowly as defaults) or operational risk (which comes from failures), market risk is **real-time and continuous**. Prices move constantly, and so does your risk profile.

This section covers how professional trading firms manage market risk across equities, fixed income, FX, and derivatives.

## Types of Market Risk

### 1. Equity Risk

Exposure to stock price movements:

**Delta Risk**: Directional exposure to stock prices
- Long position: Profit from price increase
- Short position: Profit from price decrease
- Delta-neutral: No directional exposure

**Specific Risk vs Systematic Risk**:
- Specific (idiosyncratic): Company-specific news
- Systematic (market): Overall market movements

\`\`\`python
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass

class EquityRiskManager:
    """
    Manage equity portfolio market risk
    """
    def __init__(self, positions: Dict[str, float], 
                 betas: Dict[str, float],
                 correlations: pd.DataFrame):
        """
        Args:
            positions: Dict of {ticker: shares}
            betas: Dict of {ticker: beta}
            correlations: Correlation matrix of returns
        """
        self.positions = positions
        self.betas = betas
        self.correlations = correlations
        
    def calculate_portfolio_beta(self, prices: Dict[str, float]) -> float:
        """
        Portfolio beta = weighted average of individual betas
        """
        total_value = sum(shares * prices[ticker] 
                         for ticker, shares in self.positions.items())
        
        weighted_beta = sum(
            (shares * prices[ticker] / total_value) * self.betas[ticker]
            for ticker, shares in self.positions.items()
        )
        
        return weighted_beta
    
    def calculate_portfolio_delta(self, prices: Dict[str, float]) -> float:
        """
        Dollar delta: How much P&L changes for $1 market move
        
        Delta = Portfolio Value × Beta
        """
        portfolio_value = sum(shares * prices[ticker] 
                             for ticker, shares in self.positions.items())
        portfolio_beta = self.calculate_portfolio_beta(prices)
        
        return portfolio_value * portfolio_beta
    
    def calculate_var_decomposition(self, 
                                   prices: Dict[str, float],
                                   market_volatility: float = 0.01) -> Dict:
        """
        Decompose VaR into systematic and specific components
        """
        # Systematic VaR (from market exposure)
        portfolio_beta = self.calculate_portfolio_beta(prices)
        portfolio_value = sum(shares * prices[ticker] 
                             for ticker, shares in self.positions.items())
        
        systematic_var = portfolio_value * portfolio_beta * market_volatility * 1.65  # 95% confidence
        
        # Specific VaR (from individual stock volatility)
        specific_vars = {}
        for ticker, shares in self.positions.items():
            position_value = shares * prices[ticker]
            # Specific vol = Total vol - Beta × Market vol
            specific_vol = 0.015  # Simplified - should calculate from residuals
            specific_vars[ticker] = position_value * specific_vol * 1.65
        
        # Total specific VaR (diversification reduces)
        total_specific_var = np.sqrt(sum(v**2 for v in specific_vars.values())) * 0.5  # Diversification benefit
        
        return {
            'systematic_var': abs(systematic_var),
            'specific_var': total_specific_var,
            'total_var': abs(systematic_var) + total_specific_var,
            'systematic_percentage': abs(systematic_var) / (abs(systematic_var) + total_specific_var)
        }
    
    def hedge_market_exposure(self, 
                             prices: Dict[str, float],
                             spy_price: float,
                             spy_multiplier: int = 1) -> int:
        """
        Calculate SPY futures contracts needed to hedge beta
        
        Returns:
            Number of contracts to short (negative for short)
        """
        portfolio_delta = self.calculate_portfolio_delta(prices)
        
        # Each SPY futures contract has delta = SPY price × multiplier
        spy_delta_per_contract = spy_price * spy_multiplier
        
        # Number of contracts to short
        contracts_to_hedge = -int(portfolio_delta / spy_delta_per_contract)
        
        return contracts_to_hedge

# Example
if __name__ == "__main__":
    # Portfolio
    positions = {
        'AAPL': 10000,   # 10k shares
        'GOOGL': 5000,
        'MSFT': 8000,
        'TSLA': 3000,
        'NVDA': 4000
    }
    
    # Current prices
    prices = {
        'AAPL': 180.0,
        'GOOGL': 140.0,
        'MSFT': 380.0,
        'TSLA': 250.0,
        'NVDA': 500.0
    }
    
    # Betas
    betas = {
        'AAPL': 1.2,
        'GOOGL': 1.1,
        'MSFT': 1.0,
        'TSLA': 2.0,
        'NVDA': 1.8
    }
    
    # Create correlations (simplified)
    tickers = list(positions.keys())
    correlations = pd.DataFrame(
        np.eye(len(tickers)),
        index=tickers,
        columns=tickers
    )
    
    # Risk manager
    risk_mgr = EquityRiskManager(positions, betas, correlations)
    
    # Calculate metrics
    portfolio_value = sum(shares * prices[ticker] for ticker, shares in positions.items())
    portfolio_beta = risk_mgr.calculate_portfolio_beta(prices)
    portfolio_delta = risk_mgr.calculate_portfolio_delta(prices)
    
    print("Equity Risk Analysis")
    print("="*60)
    print(f"Portfolio Value: \${portfolio_value:,.0f}")
print(f"Portfolio Beta: {portfolio_beta:.2f}")
print(f"Portfolio Delta: \${portfolio_delta:,.0f}")
print()
    
    # VaR decomposition
var_decomp = risk_mgr.calculate_var_decomposition(prices)
print("VaR Decomposition:")
print(f"  Systematic VaR: \${var_decomp['systematic_var']:,.0f} ({var_decomp['systematic_percentage']*100:.0f}%)")
print(f"  Specific VaR: \${var_decomp['specific_var']:,.0f}")
print(f"  Total VaR: \${var_decomp['total_var']:,.0f}")
print()
    
    # Hedge calculation
spy_price = 450.0
hedge_contracts = risk_mgr.hedge_market_exposure(prices, spy_price, multiplier = 50)
print(f"Hedge: Short {abs(hedge_contracts)} SPY futures contracts")
\`\`\`

### 2. Interest Rate Risk

Risk from yield curve movements:

**DV01 (Dollar Value of 01)**: P&L change for 1bp yield move
**Duration**: Sensitivity to parallel yield shift
**Key Rate Duration**: Sensitivity to specific tenor moves
**Convexity**: Second-order effect (curvature)

\`\`\`python
class InterestRateRiskManager:
    """
    Manage fixed income portfolio interest rate risk
    """
    def __init__(self, bond_positions: List[Dict]):
        """
        Args:
            bond_positions: List of bonds with characteristics
        """
        self.positions = bond_positions
        
    def calculate_dv01(self) -> float:
        """
        DV01 = Sum of (Duration × Position Value) / 10,000
        
        Represents dollar P&L for 1 basis point yield move
        """
        total_dv01 = 0
        
        for position in self.positions:
            position_value = position['face_value'] * position['price'] / 100
            duration = position['duration']
            
            # DV01 for this position
            position_dv01 = (duration * position_value) / 10000
            total_dv01 += position_dv01
        
        return total_dv01
    
    def calculate_portfolio_duration(self) -> float:
        """
        Portfolio duration = weighted average duration
        """
        total_value = sum(
            pos['face_value'] * pos['price'] / 100 
            for pos in self.positions
        )
        
        weighted_duration = sum(
            (pos['face_value'] * pos['price'] / 100) * pos['duration']
            for pos in self.positions
        ) / total_value
        
        return weighted_duration
    
    def calculate_key_rate_durations(self, tenors: List[int]) -> Dict[int, float]:
        """
        Key rate durations: sensitivity to specific tenor moves
        
        Args:
            tenors: List of key tenors (e.g., [2, 5, 10, 30] years)
        """
        key_rate_dv01s = {tenor: 0 for tenor in tenors}
        
        for position in self.positions:
            maturity = position['maturity_years']
            position_dv01 = self.calculate_position_dv01(position)
            
            # Allocate to nearest tenor (simplified)
            nearest_tenor = min(tenors, key=lambda x: abs(x - maturity))
            key_rate_dv01s[nearest_tenor] += position_dv01
        
        return key_rate_dv01s
    
    def calculate_position_dv01(self, position: Dict) -> float:
        """DV01 for single position"""
        position_value = position['face_value'] * position['price'] / 100
        return (position['duration'] * position_value) / 10000
    
    def estimate_pnl_for_yield_shift(self, 
                                     yield_changes: Dict[int, float]) -> float:
        """
        Estimate P&L for yield curve shift
        
        Args:
            yield_changes: Dict of {tenor: basis_point_change}
        """
        tenors = list(yield_changes.keys())
        key_rate_dv01s = self.calculate_key_rate_durations(tenors)
        
        total_pnl = 0
        for tenor, bp_change in yield_changes.items():
            # P&L = -DV01 × bp_change (negative because bond prices fall when yields rise)
            pnl = -key_rate_dv01s[tenor] * bp_change
            total_pnl += pnl
        
        return total_pnl
    
    def calculate_convexity(self) -> float:
        """
        Convexity: second-order sensitivity
        
        Price change ≈ -Duration × Δy + 0.5 × Convexity × (Δy)²
        """
        total_value = sum(
            pos['face_value'] * pos['price'] / 100 
            for pos in self.positions
        )
        
        weighted_convexity = sum(
            (pos['face_value'] * pos['price'] / 100) * pos.get('convexity', 0)
            for pos in self.positions
        ) / total_value
        
        return weighted_convexity
    
    def hedge_duration(self, target_duration: float = 0) -> Dict:
        """
        Calculate hedge to achieve target duration
        """
        current_duration = self.calculate_portfolio_duration()
        current_value = sum(
            pos['face_value'] * pos['price'] / 100 
            for pos in self.positions
        )
        
        duration_gap = current_duration - target_duration
        
        # Use 10-year treasury futures (duration ~9) to hedge
        futures_duration = 9.0
        futures_price = 115  # Per $100 face value
        futures_notional = 100000  # $100k per contract
        
        # Contracts needed
        hedge_value = (duration_gap * current_value) / futures_duration
        contracts_needed = hedge_value / (futures_notional * futures_price / 100)
        
        return {
            'current_duration': current_duration,
            'target_duration': target_duration,
            'duration_gap': duration_gap,
            'hedge_contracts': -int(contracts_needed),  # Negative to hedge
            'hedge_direction': 'SHORT' if contracts_needed > 0 else 'LONG'
        }

# Example
if __name__ == "__main__":
    # Bond portfolio
    bond_portfolio = [
        {
            'name': '2-Year Treasury',
            'face_value': 10000000,
            'price': 98.5,
            'duration': 1.95,
            'convexity': 0.05,
            'maturity_years': 2
        },
        {
            'name': '10-Year Treasury',
            'face_value': 20000000,
            'price': 102.3,
            'duration': 8.7,
            'convexity': 0.85,
            'maturity_years': 10
        },
        {
            'name': '30-Year Treasury',
            'face_value': 5000000,
            'price': 95.0,
            'duration': 19.5,
            'convexity': 3.2,
            'maturity_years': 30
        }
    ]
    
    ir_risk = InterestRateRiskManager(bond_portfolio)
    
    print("Interest Rate Risk Analysis")
    print("="*60)
    
    # DV01
    dv01 = ir_risk.calculate_dv01()
    print(f"Portfolio DV01: \${dv01:, .2f}")
print(f"  (P&L for 1bp parallel shift)")
print()
    
    # Duration
duration = ir_risk.calculate_portfolio_duration()
print(f"Portfolio Duration: {duration:.2f} years")
print()
    
    # Key rate durations
key_rates = ir_risk.calculate_key_rate_durations([2, 10, 30])
print("Key Rate DV01s:")
for tenor, dv01_val in key_rates.items():
    print(f"  {tenor}Y: \${dv01_val:,.2f}")
print()
    
    # Scenario: yield curve steepening
scenario = {
    2: -10,   # 2Y yields down 10bp
        10: 0,    # 10Y unchanged
        30: 20    # 30Y yields up 20bp
}
pnl = ir_risk.estimate_pnl_for_yield_shift(scenario)
print(f"P&L for yield curve steepening: \${pnl:,.0f}")
print()
    
    # Hedge to duration - neutral
hedge_info = ir_risk.hedge_duration(target_duration = 0)
print("Duration Hedge:")
print(f"  Current duration: {hedge_info['current_duration']:.2f}")
print(f"  {hedge_info['hedge_direction']} {abs(hedge_info['hedge_contracts'])} futures contracts")
\`\`\`

### 3. Foreign Exchange Risk

Currency exposure:

\`\`\`python
class FXRiskManager:
    """
    Manage FX exposure
    """
    def __init__(self, cash_positions: Dict[str, float]):
        """
        Args:
            cash_positions: Dict of {currency: amount}
        """
        self.positions = cash_positions
        self.base_currency = 'USD'
        
    def calculate_fx_exposure(self, fx_rates: Dict[str, float]) -> Dict:
        """
        Calculate exposure to each currency
        
        Args:
            fx_rates: Dict of {currency_pair: rate} (vs USD)
        """
        exposures = {}
        
        for currency, amount in self.positions.items():
            if currency == self.base_currency:
                exposures[currency] = amount
            else:
                # Convert to USD
                fx_pair = f"{currency}{self.base_currency}"
                rate = fx_rates.get(fx_pair, 1.0)
                usd_value = amount * rate
                exposures[currency] = usd_value
        
        return exposures
    
    def calculate_fx_delta(self, fx_rates: Dict[str, float]) -> Dict[str, float]:
        """
        Delta: how much P&L changes for 1% FX move
        """
        deltas = {}
        
        for currency, amount in self.positions.items():
            if currency == self.base_currency:
                continue
            
            fx_pair = f"{currency}{self.base_currency}"
            rate = fx_rates.get(fx_pair, 1.0)
            usd_value = amount * rate
            
            # Delta = position × current rate × 1%
            deltas[currency] = usd_value * 0.01
        
        return deltas
    
    def calculate_fx_var(self, 
                        fx_rates: Dict[str, float],
                        fx_volatilities: Dict[str, float],
                        correlations: pd.DataFrame,
                        confidence: float = 0.99) -> float:
        """
        VaR from FX exposure
        """
        # Get currencies (excluding base)
        currencies = [c for c in self.positions.keys() if c != self.base_currency]
        
        # Position values in USD
        position_values = []
        volatilities = []
        
        for currency in currencies:
            fx_pair = f"{currency}{self.base_currency}"
            rate = fx_rates.get(fx_pair, 1.0)
            usd_value = self.positions[currency] * rate
            position_values.append(usd_value)
            volatilities.append(fx_volatilities.get(currency, 0.10))
        
        # Portfolio variance
        position_array = np.array(position_values)
        vol_array = np.array(volatilities)
        
        # Covariance matrix
        cov_matrix = np.outer(vol_array, vol_array) * correlations.values
        
        # Portfolio variance
        portfolio_variance = position_array @ cov_matrix @ position_array
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # VaR (95% or 99%)
        from scipy import stats
        z_score = stats.norm.ppf(confidence)
        var = portfolio_vol * z_score
        
        return abs(var)
    
    def hedge_fx_exposure(self, 
                         fx_rates: Dict[str, float],
                         currencies_to_hedge: List[str]) -> Dict:
        """
        Calculate FX forwards/swaps needed to hedge
        """
        hedges = {}
        
        for currency in currencies_to_hedge:
            if currency == self.base_currency:
                continue
            
            amount = self.positions.get(currency, 0)
            if amount == 0:
                continue
            
            fx_pair = f"{currency}{self.base_currency}"
            spot_rate = fx_rates.get(fx_pair, 1.0)
            
            # Hedge: opposite position in FX forward
            # If long EUR, sell EUR forward
            hedges[currency] = {
                'exposure': amount,
                'spot_rate': spot_rate,
                'hedge_direction': 'SELL' if amount > 0 else 'BUY',
                'hedge_amount': abs(amount),
                'usd_equivalent': abs(amount * spot_rate)
            }
        
        return hedges

# Example
if __name__ == "__main__":
    # Multi-currency portfolio
    cash_positions = {
        'USD': 5000000,
        'EUR': 2000000,
        'GBP': 1000000,
        'JPY': 300000000,
        'CHF': 500000
    }
    
    # FX rates (vs USD)
    fx_rates = {
        'EURUSD': 1.10,
        'GBPUSD': 1.27,
        'JPYUSD': 0.0067,
        'CHFUSD': 1.15
    }
    
    fx_risk = FXRiskManager(cash_positions)
    
    print("FX Risk Analysis")
    print("="*60)
    
    # Exposure
    exposures = fx_risk.calculate_fx_exposure(fx_rates)
    print("FX Exposure (USD equivalent):")
    total_exposure = sum(exposures.values())
    for currency, value in exposures.items():
        pct = (value / total_exposure) * 100
        print(f"  {currency}: \${value:, .0f} ({ pct: .1f } %)")
print()
    
    # Delta
deltas = fx_risk.calculate_fx_delta(fx_rates)
print("FX Delta (P&L for 1% move):")
for currency, delta in deltas.items():
    print(f"  {currency}: \${delta:,.0f}")
print()
    
    # Hedge recommendations
hedges = fx_risk.hedge_fx_exposure(fx_rates, ['EUR', 'GBP', 'JPY'])
print("FX Hedge Recommendations:")
for currency, hedge_info in hedges.items():
    print(f"  {currency}: {hedge_info['hedge_direction']} {hedge_info['hedge_amount']:,.0f} "
              f"(\${hedge_info['usd_equivalent']:,.0f})")
\`\`\`

### 4. Options Greeks Risk

For portfolios with options:

\`\`\`python
class GreeksRiskManager:
    """
    Manage options Greeks risk
    """
    def __init__(self, options_positions: List[Dict]):
        """
        Args:
            options_positions: List of option positions with Greeks
        """
        self.positions = options_positions
    
    def calculate_portfolio_greeks(self) -> Dict[str, float]:
        """
        Aggregate portfolio Greeks
        """
        portfolio_greeks = {
            'delta': 0,
            'gamma': 0,
            'vega': 0,
            'theta': 0,
            'rho': 0
        }
        
        for position in self.positions:
            contracts = position['contracts']
            multiplier = position.get('multiplier', 100)
            
            # Aggregate each Greek
            for greek in portfolio_greeks.keys():
                greek_value = position.get(greek, 0)
                portfolio_greeks[greek] += greek_value * contracts * multiplier
        
        return portfolio_greeks
    
    def estimate_pnl_scenario(self,
                             stock_move: float,
                             vol_change: float,
                             time_decay_days: int = 1) -> Dict:
        """
        Estimate P&L for market scenario
        
        Args:
            stock_move: Dollar move in underlying
            vol_change: Change in implied volatility (e.g., 0.01 = 1 vol point)
            time_decay_days: Number of days passed
        """
        greeks = self.calculate_portfolio_greeks()
        
        # P&L components
        delta_pnl = greeks['delta'] * stock_move
        gamma_pnl = 0.5 * greeks['gamma'] * (stock_move ** 2)
        vega_pnl = greeks['vega'] * vol_change
        theta_pnl = greeks['theta'] * time_decay_days
        
        total_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
        
        return {
            'total_pnl': total_pnl,
            'delta_pnl': delta_pnl,
            'gamma_pnl': gamma_pnl,
            'vega_pnl': vega_pnl,
            'theta_pnl': theta_pnl,
            'greeks': greeks
        }
    
    def check_greek_limits(self, limits: Dict[str, float]) -> List[Dict]:
        """
        Check if Greeks exceed risk limits
        """
        current_greeks = self.calculate_portfolio_greeks()
        violations = []
        
        for greek, limit in limits.items():
            current = abs(current_greeks.get(greek, 0))
            if current > limit:
                violations.append({
                    'greek': greek,
                    'current': current,
                    'limit': limit,
                    'excess': current - limit,
                    'excess_percentage': ((current - limit) / limit) * 100
                })
        
        return violations
    
    def suggest_delta_hedge(self, underlying_price: float) -> Dict:
        """
        Calculate stock hedge to neutralize delta
        """
        portfolio_delta = self.calculate_portfolio_greeks()['delta']
        
        # Shares to hedge = -delta (opposite sign)
        shares_to_hedge = -portfolio_delta
        hedge_value = shares_to_hedge * underlying_price
        
        return {
            'current_delta': portfolio_delta,
            'shares_to_hedge': int(shares_to_hedge),
            'hedge_direction': 'BUY' if shares_to_hedge > 0 else 'SELL',
            'hedge_value': abs(hedge_value),
            'post_hedge_delta': 0
        }

# Example
if __name__ == "__main__":
    # Options portfolio
    options_portfolio = [
        {
            'symbol': 'SPY_CALL_450',
            'contracts': 100,  # Long 100 calls
            'multiplier': 100,
            'delta': 0.65,
            'gamma': 0.015,
            'vega': 0.45,
            'theta': -0.12,
            'rho': 0.25
        },
        {
            'symbol': 'SPY_PUT_430',
            'contracts': -50,  # Short 50 puts
            'multiplier': 100,
            'delta': -0.35,
            'gamma': 0.018,
            'vega': 0.40,
            'theta': -0.10,
            'rho': -0.20
        }
    ]
    
    greeks_risk = GreeksRiskManager(options_portfolio)
    
    print("Options Greeks Risk Analysis")
    print("="*60)
    
    # Portfolio Greeks
    portfolio_greeks = greeks_risk.calculate_portfolio_greeks()
    print("Portfolio Greeks:")
    for greek, value in portfolio_greeks.items():
        print(f"  {greek.capitalize()}: {value:,.2f}")
    print()
    
    # Scenario analysis
    scenario_result = greeks_risk.estimate_pnl_scenario(
        stock_move=5.0,      # $5 up
        vol_change=0.02,     # 2 vol points up
        time_decay_days=1
    )
    
    print("Scenario: SPY +$5, Vol +2, 1 day:")
    print(f"  Total P&L: \${scenario_result['total_pnl']:, .0f}")
print(f"    Delta P&L: \${scenario_result['delta_pnl']:,.0f}")
print(f"    Gamma P&L: \${scenario_result['gamma_pnl']:,.0f}")
print(f"    Vega P&L: \${scenario_result['vega_pnl']:,.0f}")
print(f"    Theta P&L: \${scenario_result['theta_pnl']:,.0f}")
print()
    
    # Greek limits
limits = {
    'delta': 5000,
    'gamma': 100,
    'vega': 3000,
    'theta': 2000
}

violations = greeks_risk.check_greek_limits(limits)
if violations:
    print("Greek Limit Violations:")
for v in violations:
    print(f"  {v['greek'].capitalize()}: {v['current']:,.0f} > {v['limit']:,.0f} "
                  f"({v['excess_percentage']:.0f}% over)")
    else:
print("All Greeks within limits ✓")
print()
    
    # Delta hedge
spy_price = 445.0
hedge = greeks_risk.suggest_delta_hedge(spy_price)
print("Delta Hedge:")
print(f"  Current delta: {hedge['current_delta']:,.0f}")
print(f"  {hedge['hedge_direction']} {abs(hedge['shares_to_hedge'])} shares")
print(f"  Hedge value: \${hedge['hedge_value']:,.0f}")
\`\`\`

## Risk Limits Framework

### Position Limits

\`\`\`python
class MarketRiskLimits:
    """
    Enforce market risk limits
    """
    def __init__(self, limits_config: Dict):
        self.limits = limits_config
        
    def check_all_limits(self, portfolio_metrics: Dict) -> Dict:
        """
        Check all risk limits
        """
        violations = []
        warnings = []
        
        # Check each limit
        for limit_name, limit_config in self.limits.items():
            current_value = portfolio_metrics.get(limit_name, 0)
            hard_limit = limit_config['hard_limit']
            soft_limit = limit_config.get('soft_limit', hard_limit * 0.8)
            
            if abs(current_value) > hard_limit:
                violations.append({
                    'limit': limit_name,
                    'current': current_value,
                    'limit': hard_limit,
                    'severity': 'HARD_LIMIT_BREACH'
                })
            elif abs(current_value) > soft_limit:
                warnings.append({
                    'limit': limit_name,
                    'current': current_value,
                    'limit': soft_limit,
                    'severity': 'SOFT_LIMIT_BREACH'
                })
        
        return {
            'violations': violations,
            'warnings': warnings,
            'status': 'VIOLATED' if violations else ('WARNING' if warnings else 'OK')
        }

# Example limits
risk_limits = {
    'portfolio_delta': {
        'hard_limit': 10000,
        'soft_limit': 8000,
        'description': 'Total portfolio delta'
    },
    'dv01': {
        'hard_limit': 50000,
        'soft_limit': 40000,
        'description': 'Interest rate sensitivity'
    },
    'vega': {
        'hard_limit': 5000,
        'soft_limit': 4000,
        'description': 'Volatility sensitivity'
    },
    'var_99': {
        'hard_limit': 5000000,
        'soft_limit': 4000000,
        'description': '99% daily VaR'
    }
}
\`\`\`

## Real-Time Risk Monitoring

\`\`\`python
class RealTimeMarketRiskMonitor:
    """
    Monitor market risk in real-time
    """
    def __init__(self, risk_limits):
        self.risk_limits = risk_limits
        self.alerts = []
        
    def update_risk_metrics(self, positions, market_data):
        """
        Calculate current risk metrics
        """
        metrics = {
            'timestamp': pd.Timestamp.now(),
            'portfolio_value': self.calculate_portfolio_value(positions, market_data),
            'portfolio_delta': self.calculate_delta(positions, market_data),
            'dv01': self.calculate_dv01(positions),
            'var_99': self.calculate_var(positions, market_data),
            # ... other metrics
        }
        
        # Check limits
        limit_check = self.risk_limits.check_all_limits(metrics)
        
        if limit_check['status'] != 'OK':
            self.generate_alerts(limit_check)
        
        return metrics, limit_check
    
    def generate_alerts(self, limit_check):
        """
        Generate alerts for limit breaches
        """
        for violation in limit_check['violations']:
            alert = {
                'timestamp': pd.Timestamp.now(),
                'severity': 'HIGH',
                'type': 'HARD_LIMIT_BREACH',
                'limit': violation['limit'],
                'current': violation['current'],
                'limit_value': violation['limit']
            }
            self.alerts.append(alert)
            self.send_alert(alert)
    
    def send_alert(self, alert):
        """
        Send alert via email/slack/PagerDuty
        """
        print(f"🚨 ALERT: {alert['type']} - {alert['limit']}")
        print(f"   Current: {alert['current']:,.0f}, Limit: {alert['limit_value']:,.0f}")
        # In production: send email, Slack message, PagerDuty, etc.
\`\`\`

## Key Takeaways

1. **Multiple Risk Types**: Equity, rates, FX, options all need different approaches
2. **Real-Time Monitoring**: Market risk changes constantly
3. **Greeks Management**: Essential for options portfolios
4. **Hedging**: Use futures, forwards, swaps to manage exposures
5. **Risk Limits**: Multi-layered (soft and hard limits)
6. **Aggregation**: Portfolio-level Greeks and sensitivities
7. **Scenario Analysis**: Stress test market moves

## Common Pitfalls

❌ **Ignoring Gamma**: Delta hedge works until gamma flips  
❌ **Static Hedges**: Market risk requires dynamic hedging  
❌ **Missing Correlations**: Diversification benefits can disappear  
❌ **Wrong Horizon**: 1-day VaR ≠ 10-day VaR / √10  
❌ **Forgetting Second-Order**: Gamma and convexity matter

## Conclusion

Market risk management is about understanding and controlling your sensitivities to market moves. Whether it's delta, DV01, or FX exposure - you need to measure it, monitor it, limit it, and hedge it.

Professional firms monitor market risk in real-time, with automatic alerts and pre-trade risk checks. The goal isn't to eliminate market risk (that's impossible) - it's to take calculated risks within acceptable boundaries.

Next: Credit Risk Management - managing counterparty default risk.
`,
};
