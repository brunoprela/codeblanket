export const financeTerminologyDevelopers = {
    title: 'Finance Terminology for Developers',
    id: 'finance-terminology-developers',
    content: `
# Finance Terminology for Developers

## Introduction

Finance has its own language. As an engineer, you'll hear terms like "alpha," "Sharpe ratio," "P&L," "volatility" constantly. This section translates finance jargon into concepts developers understand, with code examples demonstrating how to calculate key metrics.

---

## Performance Metrics

### Alpha (α)

**Definition**: Excess return above benchmark, adjusted for risk.

**Formula**: α = Portfolio Return - (Risk-Free Rate + β × (Market Return - Risk-Free Rate))

**In dev terms**: "Performance above what you'd expect given the risk you took"

\`\`\`python
"""
Calculate Alpha
"""
import numpy as np

def calculate_alpha(portfolio_returns: np.array, 
                   market_returns: np.array,
                   risk_free_rate: float = 0.02) -> float:
    """
    Calculate alpha (excess return above expected)
    
    Alpha = actual return - expected return
    Expected return = risk_free_rate + beta * (market_return - risk_free_rate)
    """
    # Calculate beta first
    covariance = np.cov(portfolio_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    beta = covariance / market_variance
    
    # Calculate returns
    portfolio_return = np.mean(portfolio_returns)
    market_return = np.mean(market_returns)
    
    # Expected return (CAPM)
    expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
    
    # Alpha = actual - expected
    alpha = portfolio_return - expected_return
    
    return alpha, beta


# Example: Calculate alpha for a trading strategy
portfolio_returns = np.array([0.15, 0.08, -0.03, 0.12, 0.18, 0.05, 0.09, 0.14, -0.02, 0.11])
market_returns = np.array([0.10, 0.06, -0.05, 0.09, 0.12, 0.04, 0.07, 0.10, -0.04, 0.08])

alpha, beta = calculate_alpha(portfolio_returns, market_returns)

print(f"Alpha: {alpha:.4f} ({alpha*100:.2f}%)")
print(f"Beta: {beta:.4f}")

if alpha > 0:
    print(f"✓ Positive alpha! Strategy outperformed market by {alpha*100:.2f}% after risk adjustment")
else:
    print(f"✗ Negative alpha. Strategy underperformed market by {abs(alpha)*100:.2f}%")
\`\`\`

### Beta (β)

**Definition**: Sensitivity to market movements.

**Interpretation**:
- β = 1: Moves with market (SPY)
- β > 1: Amplifies market (TSLA β ≈ 2)
- β < 1: Less volatile (utilities β ≈ 0.5)
- β < 0: Inverse (gold, VIX)

**In dev terms**: "Coefficient measuring correlation strength"

### Sharpe Ratio

**Definition**: Return per unit of risk.

**Formula**: Sharpe = (Return - Risk-Free Rate) / Volatility

**In dev terms**: "ROI adjusted for variance"

\`\`\`python
"""
Calculate Sharpe Ratio
"""

def calculate_sharpe_ratio(returns: np.array, risk_free_rate: float = 0.02) -> float:
    """
    Sharpe ratio = (mean return - risk-free rate) / std dev of returns
    
    Higher = better (more return per unit of risk)
    - Sharpe > 2: Excellent
    - Sharpe > 1: Good
    - Sharpe < 1: Poor (better to buy index)
    """
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    
    # Annualize (√252 trading days)
    sharpe_annualized = sharpe * np.sqrt(252)
    
    return sharpe_annualized


# Example: Compare two strategies
strategy_a_returns = np.random.normal(0.0005, 0.01, 252)  # Mean 0.05%, std 1%
strategy_b_returns = np.random.normal(0.0008, 0.02, 252)  # Mean 0.08%, std 2%

sharpe_a = calculate_sharpe_ratio(strategy_a_returns)
sharpe_b = calculate_sharpe_ratio(strategy_b_returns)

print(f"\\n=== Sharpe Ratio Comparison ===")
print(f"Strategy A: {sharpe_a:.2f}")
print(f"Strategy B: {sharpe_b:.2f}")

if sharpe_a > sharpe_b:
    print("✓ Strategy A is better (higher risk-adjusted returns)")
else:
    print("✓ Strategy B is better")
\`\`\`

---

## Risk Metrics

### Volatility (σ)

**Definition**: Standard deviation of returns.

**In dev terms**: "How much returns bounce around the mean"

**Calculation**:
\`\`\`python
"""
Calculate Volatility
"""

def calculate_volatility(returns: np.array, annualize: bool = True) -> float:
    """
    Volatility = standard deviation of returns
    
    Typically annualized: daily_vol * sqrt(252)
    """
    volatility = np.std(returns)
    
    if annualize:
        volatility *= np.sqrt(252)  # 252 trading days
    
    return volatility


# Example: Compare volatility
aapl_daily_returns = np.random.normal(0.001, 0.02, 252)  # 2% daily vol
spy_daily_returns = np.random.normal(0.0005, 0.01, 252)  # 1% daily vol

aapl_vol = calculate_volatility(aapl_daily_returns)
spy_vol = calculate_volatility(spy_daily_returns)

print(f"\\n=== Volatility ===")
print(f"AAPL: {aapl_vol*100:.1f}% annualized")
print(f"SPY: {spy_vol*100:.1f}% annualized")
print(f"AAPL is {aapl_vol/spy_vol:.1f}x more volatile than SPY")
\`\`\`

### Maximum Drawdown (MDD)

**Definition**: Largest peak-to-trough decline.

**In dev terms**: "Worst case loss from any previous high"

\`\`\`python
"""
Calculate Maximum Drawdown
"""

def calculate_max_drawdown(returns: np.array) -> tuple:
    """
    Max drawdown = largest percentage decline from peak to trough
    """
    # Calculate cumulative returns
    cumulative = np.cumprod(1 + returns)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative)
    
    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max
    
    max_dd = np.min(drawdown)
    max_dd_idx = np.argmin(drawdown)
    
    return max_dd, max_dd_idx


# Example: Calculate drawdown
returns = np.array([0.10, 0.05, -0.08, -0.12, 0.03, 0.15, -0.05, 0.08])

max_dd, dd_idx = calculate_max_drawdown(returns)

print(f"\\n=== Maximum Drawdown ===")
print(f"Max drawdown: {max_dd*100:.2f}%")
print(f"Occurred at period: {dd_idx}")
print(f"\\nInterpretation: Portfolio declined {abs(max_dd)*100:.1f}% from peak")
\`\`\`

---

## Trading Terms

### P&L (Profit & Loss)

**Definition**: Profit or loss on position.

**Types**:
- **Realized P&L**: Closed positions (cash in hand)
- **Unrealized P&L**: Open positions (paper gain/loss)

\`\`\`python
"""
Calculate P&L
"""

class Position:
    """Track position P&L"""
    
    def __init__(self, ticker: str, quantity: int, entry_price: float):
        self.ticker = ticker
        self.quantity = quantity
        self.entry_price = entry_price
        self.current_price = entry_price
    
    def update_price(self, new_price: float):
        """Update current price"""
        self.current_price = new_price
    
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L"""
        return (self.current_price - self.entry_price) * self.quantity
    
    def realized_pnl(self, exit_price: float, quantity: int) -> float:
        """Calculate realized P&L when closing position"""
        return (exit_price - self.entry_price) * quantity


# Example: Track P&L
position = Position('AAPL', 100, 150.00)

# Price moves
position.update_price(155.00)
print(f"Unrealized P&L: ${position.unrealized_pnl():, .2f
}")

# Close half position
realized = position.realized_pnl(155.00, 50)
print(f"Realized P&L (50 shares): ${realized:,.2f}")
\`\`\`

### Bid-Ask Spread

**Definition**: Difference between buy price (ask) and sell price (bid).

**In dev terms**: "Transaction cost - you lose half the spread buying, half selling"

### Slippage

**Definition**: Difference between expected price and execution price.

**Causes**:
- Market orders in volatile markets
- Large orders (eat through order book)
- Low liquidity

**In dev terms**: "Runtime performance worse than benchmark"

---

## Portfolio Terms

### NAV (Net Asset Value)

**Definition**: Total asset value minus liabilities, divided by shares.

**Formula**: NAV = (Assets - Liabilities) / Shares Outstanding

**In dev terms**: "Book value per share"

### AUM (Assets Under Management)

**Definition**: Total market value of assets managed.

**Example**: BlackRock has $10 trillion AUM (largest asset manager)

### MTM (Mark-to-Market)

**Definition**: Valuing assets at current market price.

**In dev terms**: "Real-time pricing vs. historical cost"

\`\`\`python
"""
Mark-to-Market Valuation
"""

class Portfolio:
    """Portfolio with MTM valuation"""
    
    def __init__(self):
        self.positions = {}  # {ticker: {'quantity': int, 'cost_basis': float}}
        self.cash = 0
    
    def add_position(self, ticker: str, quantity: int, price: float):
        """Add position at cost"""
        if ticker not in self.positions:
            self.positions[ticker] = {'quantity': 0, 'cost_basis': 0}
        
        # Update weighted average cost
        current_qty = self.positions[ticker]['quantity']
        current_cost = self.positions[ticker]['cost_basis']
        
        new_cost = (current_cost * current_qty + price * quantity) / (current_qty + quantity)
        
        self.positions[ticker]['quantity'] += quantity
        self.positions[ticker]['cost_basis'] = new_cost
        
        self.cash -= price * quantity
    
    def mark_to_market(self, prices: dict) -> dict:
        """
        Calculate MTM valuation
        
        prices: {ticker: current_price}
        """
        market_value = self.cash
        cost_value = self.cash
        
        for ticker, position in self.positions.items():
            qty = position['quantity']
            cost = position['cost_basis']
            market_price = prices.get(ticker, cost)
            
            market_value += qty * market_price
            cost_value += qty * cost
        
        unrealized_pnl = market_value - cost_value
        
        return {
            'market_value': market_value,
            'cost_basis': cost_value,
            'unrealized_pnl': unrealized_pnl,
            'return_pct': (unrealized_pnl / cost_value * 100) if cost_value != 0 else 0
        }


# Example: Portfolio MTM
portfolio = Portfolio()
portfolio.cash = 100_000

portfolio.add_position('AAPL', 100, 150.00)
portfolio.add_position('MSFT', 50, 300.00)

# Prices move
current_prices = {'AAPL': 155.00, 'MSFT': 310.00}

mtm = portfolio.mark_to_market(current_prices)

print(f"\\n=== Mark-to-Market ===")
print(f"Market value: ${mtm['market_value']:, .2f}")
print(f"Cost basis: ${mtm['cost_basis']:,.2f}")
print(f"Unrealized P&L: ${mtm['unrealized_pnl']:,.2f}")
print(f"Return: {mtm['return_pct']:.2f}%")
\`\`\`

---

## Communication Protocols

### FIX Protocol

**Definition**: Financial Information eXchange - standard messaging for trades.

**In dev terms**: "HTTPS for trading - binary protocol for order routing"

**Message types**:
- New Order (type 'D')
- Execution Report (type '8')
- Order Cancel Request (type 'F')

**Example FIX message**:
\`\`\`
8=FIX.4.2|9=178|35=D|49=SENDER|56=TARGET|34=1|52=20240115-10:30:00|
11=ORDER123|21=1|55=AAPL|54=1|60=20240115-10:30:00|38=100|40=2|44=150.00|10=123|
\`\`\`

**Parsing**:
\`\`\`python
"""
Simple FIX Message Parser
"""

class FIXMessage:
    """Parse FIX protocol messages"""
    
    # FIX tag definitions
    TAGS = {
        '8': 'BeginString',
        '9': 'BodyLength',
        '35': 'MsgType',
        '49': 'SenderCompID',
        '56': 'TargetCompID',
        '11': 'ClOrdID',
        '55': 'Symbol',
        '54': 'Side',
        '38': 'OrderQty',
        '40': 'OrdType',
        '44': 'Price',
    }
    
    MSG_TYPES = {
        'D': 'New Order Single',
        '8': 'Execution Report',
        'F': 'Order Cancel Request',
    }
    
    @staticmethod
    def parse(fix_message: str) -> dict:
        """Parse FIX message into dictionary"""
        fields = fix_message.split('|')
        parsed = {}
        
        for field in fields:
            if '=' in field:
                tag, value = field.split('=', 1)
                field_name = FIXMessage.TAGS.get(tag, f'Tag{tag}')
                parsed[field_name] = value
        
        # Add human-readable message type
        if 'MsgType' in parsed:
            parsed['MsgType_Name'] = FIXMessage.MSG_TYPES.get(parsed['MsgType'], 'Unknown')
        
        return parsed


# Example: Parse FIX order
fix_msg = "8=FIX.4.2|35=D|49=CLIENT|56=BROKER|11=ORD001|55=AAPL|54=1|38=100|40=2|44=150.00"

parsed = FIXMessage.parse(fix_msg)

print("\\n=== FIX Message ===")
for field, value in parsed.items():
    print(f"{field}: {value}")
\`\`\`

---

## Key Takeaways

1. **Alpha** = outperformance above expected (actual return - CAPM expected return)
2. **Beta** = sensitivity to market (β=1 means moves with market)
3. **Sharpe ratio** = return per unit of risk (higher = better)
4. **Volatility** = standard deviation of returns (risk measure)
5. **P&L** = profit/loss (realized = closed, unrealized = open positions)
6. **NAV** = net asset value per share (assets - liabilities / shares)
7. **FIX protocol** = standard for trading messages (like HTTP for finance)

**Next section**: Mathematics for Finance - the math you need for quant finance.
`,
};

