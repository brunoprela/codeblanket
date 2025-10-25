export const moduleProjectPersonalFinanceDashboard = {
  title: 'Module Project: Personal Finance Dashboard',
  id: 'module-project-personal-finance-dashboard',
  content: `
# Module Project: Personal Finance Dashboard

## Introduction

**Build a complete personal finance dashboard** that consolidates your financial life in one place. This capstone project applies everything from Module 0:

1. Portfolio tracking (stocks, bonds, crypto)
2. Performance analytics (returns, Sharpe ratio, drawdowns)
3. Budget tracking (income, expenses, savings rate)
4. Net worth monitoring (assets, liabilities, trends)
5. Financial goal tracking (retirement, house down payment)

By the end, you'll have a production-ready app you can actually use!

---

## Project Requirements

### Core Features

1. **Portfolio Management**
   - Add accounts (brokerage, 401k, IRA, crypto wallet)
   - Track positions (ticker, quantity, cost basis)
   - Real-time pricing (via Yahoo Finance API)
   - Historical performance (returns, volatility)

2. **Performance Analytics**
   - Total return (dollar and percentage)
   - Time-weighted return (accounts for deposits/withdrawals)
   - Asset allocation (pie chart: stocks/bonds/cash/crypto)
   - Sharpe ratio, max drawdown

3. **Budget Tracking**
   - Income sources (salary, side hustle, dividends)
   - Expense categories (housing, food, entertainment)
   - Monthly trends (spending patterns)
   - Savings rate ((Income - Expenses) / Income)

4. **Net Worth Tracking**
   - Assets (investments, home equity, cash)
   - Liabilities (mortgage, student loans, credit cards)
   - Net worth trend (monthly snapshots)
   - Debt-to-income ratio

5. **Financial Goals**
   - Retirement planning (target age, needed savings)
   - Short-term goals (emergency fund, vacation)
   - Progress tracking (% complete, time remaining)
   - Projections (will you hit goals?)

### Technical Stack

- **Backend**: Python + Flask/FastAPI
- **Database**: SQLite (local) or PostgreSQL (production)
- **Data**: yfinance for market prices
- **Frontend**: Streamlit (easiest) or React (advanced)
- **Deployment**: Local or Streamlit Cloud (free)

---

## Implementation

### Database Schema

\`\`\`python
"""
Database schema for personal finance dashboard
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Account(Base):
    """Financial accounts (brokerage, 401k, etc.)"""
    __tablename__ = 'accounts'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)  # e.g., "Vanguard Brokerage"
    account_type = Column(String)  # 'brokerage', '401k', 'ira', 'crypto', 'checking'
    institution = Column(String)  # e.g., "Vanguard"
    created_at = Column(DateTime, default=datetime.now)
    
    positions = relationship("Position", back_populates="account")


class Position(Base):
    """Holdings in accounts"""
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True)
    account_id = Column(Integer, ForeignKey('accounts.id'))
    ticker = Column(String, nullable=False)  # e.g., "AAPL"
    quantity = Column(Float, nullable=False)
    cost_basis = Column(Float, nullable=False)  # Purchase price per share
    purchase_date = Column(DateTime)
    
    account = relationship("Account", back_populates="positions")


class Transaction(Base):
    """Income and expenses"""
    __tablename__ = 'transactions'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)
    type = Column(String, nullable=False)  # 'income' or 'expense'
    category = Column(String)  # 'salary', 'rent', 'food', etc.
    amount = Column(Float, nullable=False)
    description = Column(String)


class NetWorthSnapshot(Base):
    """Monthly net worth snapshots"""
    __tablename__ = 'networth_snapshots'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)
    total_assets = Column(Float)
    total_liabilities = Column(Float)
    net_worth = Column(Float)


class Goal(Base):
    """Financial goals"""
    __tablename__ = 'goals'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)  # e.g., "Retirement"
    target_amount = Column(Float, nullable=False)
    current_amount = Column(Float, default=0)
    target_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.now)


# Create database
engine = create_engine('sqlite:///finance_dashboard.db')
Base.metadata.create_all (engine)
\`\`\`

### Portfolio Tracker

\`\`\`python
"""
Portfolio tracking module
"""
import yfinance as yf
import pandas as pd
import numpy as np
from sqlalchemy.orm import sessionmaker

class PortfolioTracker:
    """Track portfolio performance"""
    
    def __init__(self, engine):
        self.engine = engine
        self.Session = sessionmaker (bind=engine)
    
    def get_portfolio_value (self) -> pd.DataFrame:
        """
        Calculate current portfolio value across all accounts
        """
        session = self.Session()
        
        # Fetch all positions
        positions = session.query(Position).all()
        
        portfolio_data = []
        
        for pos in positions:
            # Get current price
            ticker_obj = yf.Ticker (pos.ticker)
            current_price = ticker_obj.history (period='1d')['Close'].iloc[-1]
            
            # Calculate values
            market_value = pos.quantity * current_price
            cost_basis = pos.quantity * pos.cost_basis
            gain_loss = market_value - cost_basis
            gain_loss_pct = (gain_loss / cost_basis) * 100 if cost_basis > 0 else 0
            
            portfolio_data.append({
                'ticker': pos.ticker,
                'quantity': pos.quantity,
                'cost_basis': pos.cost_basis,
                'current_price': current_price,
                'market_value': market_value,
                'gain_loss': gain_loss,
                'gain_loss_pct': gain_loss_pct
            })
        
        session.close()
        
        return pd.DataFrame (portfolio_data)
    
    def calculate_performance (self, start_date: str, end_date: str) -> dict:
        """
        Calculate portfolio performance metrics
        """
        session = self.Session()
        positions = session.query(Position).all()
        
        # Get historical prices
        tickers = [pos.ticker for pos in positions]
        weights = [pos.quantity * pos.cost_basis for pos in positions]
        total_invested = sum (weights)
        weights = [w / total_invested for w in weights]
        
        # Download historical data
        data = yf.download (tickers, start=start_date, end=end_date)['Adj Close']
        
        # Calculate portfolio returns
        returns = data.pct_change().dropna()
        portfolio_returns = (returns * weights).sum (axis=1)
        
        # Performance metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len (portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / volatility  # Assuming 2% risk-free rate
        
        # Max drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        session.close()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }


# Example usage
# engine = create_engine('sqlite:///finance_dashboard.db')
# tracker = PortfolioTracker (engine)
# portfolio = tracker.get_portfolio_value()
# print(portfolio)
\`\`\`

### Budget Tracker

\`\`\`python
"""
Budget tracking module
"""

class BudgetTracker:
    """Track income and expenses"""
    
    def __init__(self, engine):
        self.engine = engine
        self.Session = sessionmaker (bind=engine)
    
    def add_transaction (self, date: datetime, type: str, category: str, 
                       amount: float, description: str = ""):
        """Add income or expense"""
        session = self.Session()
        
        transaction = Transaction(
            date=date,
            type=type,
            category=category,
            amount=amount,
            description=description
        )
        
        session.add (transaction)
        session.commit()
        session.close()
    
    def get_monthly_summary (self, year: int, month: int) -> dict:
        """Get income/expense summary for a month"""
        session = self.Session()
        
        start_date = datetime (year, month, 1)
        if month == 12:
            end_date = datetime (year + 1, 1, 1)
        else:
            end_date = datetime (year, month + 1, 1)
        
        transactions = session.query(Transaction).filter(
            Transaction.date >= start_date,
            Transaction.date < end_date
        ).all()
        
        income = sum (t.amount for t in transactions if t.type == 'income')
        expenses = sum (t.amount for t in transactions if t.type == 'expense')
        savings = income - expenses
        savings_rate = (savings / income * 100) if income > 0 else 0
        
        # Category breakdown
        expense_by_category = {}
        for t in transactions:
            if t.type == 'expense':
                if t.category not in expense_by_category:
                    expense_by_category[t.category] = 0
                expense_by_category[t.category] += t.amount
        
        session.close()
        
        return {
            'income': income,
            'expenses': expenses,
            'savings': savings,
            'savings_rate': savings_rate,
            'expense_breakdown': expense_by_category
        }
    
    def get_trend (self, num_months: int = 12) -> pd.DataFrame:
        """Get income/expense trend over time"""
        session = self.Session()
        
        transactions = session.query(Transaction).all()
        df = pd.DataFrame([{
            'date': t.date,
            'type': t.type,
            'amount': t.amount
        } for t in transactions])
        
        if df.empty:
            return pd.DataFrame()
        
        df['month'] = pd.to_datetime (df['date']).dt.to_period('M')
        
        trend = df.groupby(['month', 'type'])['amount'].sum().unstack (fill_value=0)
        
        if 'income' in trend.columns and 'expense' in trend.columns:
            trend['savings'] = trend['income'] - trend['expense']
            trend['savings_rate'] = (trend['savings'] / trend['income'] * 100).round(1)
        
        session.close()
        
        return trend.tail (num_months)
\`\`\`

### Streamlit Dashboard

\`\`\`python
"""
Streamlit dashboard (app.py)
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config (page_title="Personal Finance Dashboard", layout="wide")

# Title
st.title("📊 Personal Finance Dashboard")

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", ["Portfolio", "Budget", "Net Worth", "Goals"])

# Initialize trackers
# engine = create_engine('sqlite:///finance_dashboard.db')
# portfolio_tracker = PortfolioTracker (engine)
# budget_tracker = BudgetTracker (engine)

if page == "Portfolio":
    st.header("💼 Portfolio")
    
    # Get portfolio
    portfolio = portfolio_tracker.get_portfolio_value()
    
    if not portfolio.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_value = portfolio['market_value'].sum()
        total_cost = (portfolio['quantity'] * portfolio['cost_basis']).sum()
        total_gain = total_value - total_cost
        total_gain_pct = (total_gain / total_cost * 100) if total_cost > 0 else 0
        
        col1.metric("Total Value", f"\${total_value:,.0f}")
col2.metric("Total Cost", f"\${total_cost:,.0f}")
col3.metric("Gain/Loss", f"\${total_gain:,.0f}", f"{total_gain_pct:.1f}%")
col4.metric("Positions", len (portfolio))
        
        # Portfolio table
st.subheader("Holdings")
st.dataframe (portfolio, use_container_width = True)
        
        # Allocation pie chart
fig = px.pie (portfolio, values = 'market_value', names = 'ticker',
    title = 'Portfolio Allocation')
st.plotly_chart (fig, use_container_width = True)
        
        # Performance metrics
st.subheader("Performance (Last Year)")
perf = portfolio_tracker.calculate_performance(
    start_date = (datetime.now() - timedelta (days = 365)).strftime('%Y-%m-%d'),
    end_date = datetime.now().strftime('%Y-%m-%d')
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Return", f"{perf['total_return']*100:.2f}%")
col2.metric("Annual Return", f"{perf['annualized_return']*100:.2f}%")
col3.metric("Sharpe Ratio", f"{perf['sharpe_ratio']:.2f}")
col4.metric("Max Drawdown", f"{perf['max_drawdown']*100:.2f}%")
    else:
st.info("No positions found. Add positions to see portfolio analysis.")

elif page == "Budget":
st.header("💰 Budget")
    
    # Month selector
col1, col2 = st.columns(2)
year = col1.number_input("Year", min_value = 2020, max_value = 2030, value = datetime.now().year)
month = col2.number_input("Month", min_value = 1, max_value = 12, value = datetime.now().month)
    
    # Get monthly summary
summary = budget_tracker.get_monthly_summary (year, month)
    
    # Summary metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Income", f"\${summary['income']:,.0f}")
col2.metric("Expenses", f"\${summary['expenses']:,.0f}")
col3.metric("Savings", f"\${summary['savings']:,.0f}")
col4.metric("Savings Rate", f"{summary['savings_rate']:.1f}%")
    
    # Expense breakdown
if summary['expense_breakdown']:
    st.subheader("Expense Breakdown")
df = pd.DataFrame (list (summary['expense_breakdown'].items()),
    columns = ['Category', 'Amount'])
fig = px.bar (df, x = 'Category', y = 'Amount', title = 'Expenses by Category')
st.plotly_chart (fig, use_container_width = True)
    
    # Trend over time
st.subheader("12-Month Trend")
trend = budget_tracker.get_trend (num_months = 12)
if not trend.empty:
fig = go.Figure()
fig.add_trace (go.Scatter (x = trend.index.astype (str), y = trend['income'],
    name = 'Income', mode = 'lines+markers'))
fig.add_trace (go.Scatter (x = trend.index.astype (str), y = trend['expense'],
    name = 'Expenses', mode = 'lines+markers'))
fig.add_trace (go.Scatter (x = trend.index.astype (str), y = trend['savings'],
    name = 'Savings', mode = 'lines+markers'))
st.plotly_chart (fig, use_container_width = True)

# Add similar pages for Net Worth and Goals...
\`\`\`

---

## Deployment

### Run Locally

\`\`\`bash
# Install dependencies
pip install streamlit yfinance sqlalchemy pandas numpy plotly

# Run app
streamlit run app.py

# Access at: http://localhost:8501
\`\`\`

### Deploy to Streamlit Cloud (Free)

1. Push code to GitHub
2. Go to streamlit.io/cloud
3. Connect GitHub repo
4. Deploy!

**URL**: your-app-name.streamlit.app

---

## Extensions (Advanced)

1. **Plaid Integration**: Auto-sync bank/brokerage accounts
2. **Tax Optimization**: Tax-loss harvesting suggestions
3. **Monte Carlo**: Retirement probability simulations
4. **Alerts**: Email when portfolio drops >5%
5. **Mobile App**: React Native version

---

## Key Takeaways

1. Built complete full-stack finance app
2. Applied Module 0 concepts: markets, data APIs, performance metrics
3. Real-world database design and ORM usage
4. Data visualization with Plotly
5. Production deployment with Streamlit Cloud

**Congratulations on completing Module 0!** You now understand finance foundations and have a working personal finance dashboard. Ready for Module 1: Market Microstructure!
`,
};
