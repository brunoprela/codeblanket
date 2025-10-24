import { optionsFundamentals } from '@/lib/content/sections/quantitative-finance/options-fundamentals';
import { blackScholesModel } from '@/lib/content/sections/quantitative-finance/black-scholes-model';
import { theGreeks } from '@/lib/content/sections/quantitative-finance/the-greeks';
import { portfolioTheory } from '@/lib/content/sections/quantitative-finance/portfolio-theory';
import { factorModels } from '@/lib/content/sections/quantitative-finance/factor-models';

export const quantitativeFinance = {
  id: 'quantitative-finance',
  title: 'Quantitative Finance Fundamentals',
  description:
    'Master options pricing, derivatives, portfolio theory, and quantitative finance for professional trading. Learn Black-Scholes, Greeks, factor models, risk measures, and build a complete quantitative trading foundation.',
  icon: '💰',
  sections: [
    optionsFundamentals,
    blackScholesModel,
    theGreeks,
    portfolioTheory,
    factorModels,
    // Additional sections will be added as they're completed
  ],
  keyTakeaways: [
    'Master options fundamentals: calls, puts, strategies, and payoff diagrams',
    'Understand Black-Scholes pricing model, implied volatility, and volatility smile/skew',
    'Calculate and interpret Greeks (Delta, Gamma, Theta, Vega, Rho) for risk management',
    'Apply modern portfolio theory, CAPM, and efficient frontier optimization',
    'Analyze factor models (Fama-French) and smart beta strategies',
    'Price fixed income securities: bonds, duration, convexity, yield curve',
    'Value derivatives: forwards, futures, swaps, and exotic options',
    'Implement risk measures: VaR, CVaR, volatility estimation, stress testing',
    'Understand market microstructure: bid-ask spread, order flow, liquidity',
    'Build statistical arbitrage strategies: pairs trading, cointegration, mean reversion',
    'Develop quantitative trading strategies with proper risk management',
    'Diversify with alternative investments: hedge funds, private equity, commodities',
  ],
  learningObjectives: [
    'Price options using Black-Scholes and understand its assumptions and limitations',
    'Calculate implied volatility and interpret volatility smile/skew patterns',
    'Manage option positions using Greeks: delta hedging, gamma scalping, vega trading',
    'Construct optimal portfolios using mean-variance optimization and factor models',
    'Analyze and price fixed income securities with duration and convexity',
    'Value derivatives contracts: forwards, futures, swaps, and structured products',
    'Implement professional risk management: VaR, stress testing, scenario analysis',
    'Understand market microstructure and its impact on trading strategies',
    'Build pairs trading and statistical arbitrage strategies',
    'Develop factor-based trading strategies (momentum, value, quality)',
    'Evaluate alternative investments and their role in portfolio diversification',
    'Apply quantitative methods to real-world trading with Python implementations',
  ],
  prerequisites: [
    'Calculus fundamentals (derivatives, optimization)',
    'Linear algebra (matrices, eigenvalues)',
    'Probability and statistics (distributions, hypothesis testing)',
    'Python programming with NumPy, Pandas, Matplotlib',
    'Understanding of financial markets and basic trading concepts',
  ],
  practicalProjects: [
    {
      title: 'Black-Scholes Option Pricer with Greeks Calculator',
      description:
        'Build a complete option pricing tool with: (1) Black-Scholes pricing for calls/puts, (2) All Greeks calculation (Delta, Gamma, Theta, Vega, Rho), (3) Implied volatility solver using Newton-Raphson, (4) Volatility surface visualization, (5) Strategy payoff diagrams (straddles, spreads, condors), (6) Real-time data integration with yfinance',
      technologies: ['Python', 'NumPy', 'SciPy', 'Matplotlib', 'yfinance'],
      difficulty: 'Intermediate',
    },
    {
      title: 'Portfolio Optimization Engine',
      description:
        'Implement Modern Portfolio Theory: (1) Mean-variance optimization with efficient frontier, (2) Sharpe ratio maximization, (3) Factor model integration (Fama-French), (4) Black-Litterman with investor views, (5) Risk parity and minimum variance portfolios, (6) Backtesting with realistic constraints (transaction costs, position limits)',
      technologies: ['Python', 'PyPortfolioOpt', 'pandas', 'cvxpy'],
      difficulty: 'Advanced',
    },
    {
      title: 'Pairs Trading System',
      description:
        'Build statistical arbitrage strategy: (1) Cointegration testing (Engle-Granger, Johansen), (2) Pairs selection from stock universe, (3) Mean-reversion signal generation, (4) Position sizing and risk management, (5) Walk-forward backtesting, (6) Live paper trading with real-time execution',
      technologies: ['Python', 'statsmodels', 'QuantLib', 'Alpaca API'],
      difficulty: 'Advanced',
    },
    {
      title: 'Options Greeks Dashboard & Risk Management',
      description:
        'Build portfolio risk management tool: (1) Multi-position Greeks aggregation, (2) Scenario analysis (stock moves, IV changes, time decay), (3) Delta hedging recommendations, (4) P&L attribution (gamma, vega, theta breakdown), (5) Real-time Greeks monitoring, (6) Alerts for risk limit breaches',
      technologies: ['Python', 'Dash/Streamlit', 'WebSocket', 'MongoDB'],
      difficulty: 'Advanced',
    },
    {
      title: 'Factor-Based Trading Strategy',
      description:
        'Implement multi-factor quantitative strategy: (1) Factor construction (momentum, value, quality, size), (2) Factor neutralization and orthogonalization, (3) Factor timing and regime detection, (4) Portfolio construction with factor tilts, (5) Performance attribution to factors, (6) Risk management with factor exposures',
      technologies: ['Python', 'pandas', 'scikit-learn', 'Zipline'],
      difficulty: 'Expert',
    },
  ],
};
