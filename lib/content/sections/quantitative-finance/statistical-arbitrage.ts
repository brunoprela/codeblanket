import type { ContentSection } from '@/lib/types';

export const statisticalArbitrage: ContentSection = {
  id: 'statistical-arbitrage',
  title: 'Statistical Arbitrage',
  content: `
# Statistical Arbitrage

## Introduction

Statistical arbitrage (stat arb) exploits temporary mispricings between related securities using quantitative models. Unlike pure arbitrage, stat arb involves statistical risk and requires rigorous backtesting.

## Core Strategies

### Pairs Trading
- **Concept**: Trade two cointegrated stocks (long undervalued, short overvalued)
- **Cointegration**: Long-run equilibrium relationship despite short-term divergence
- **Entry signal**: Spread widens beyond threshold (e.g., 2 standard deviations)
- **Exit signal**: Spread reverts to mean

### Mean Reversion
- **Premise**: Prices/spreads revert to historical averages
- **Metrics**: Z-score, Bollinger Bands, residuals from regression
- **Risk**: Fundamental regime changes (mean shifts permanently)

### Factor Models for Stat Arb
- **Market-neutral**: Beta-hedged to eliminate systematic risk
- **Factor exposures**: Control for size, value, momentum factors
- **Alpha generation**: Residual returns after factor adjustment

## Cointegration Testing
- **Engle-Granger test**: Tests for cointegration between two series
- **Johansen test**: Tests multiple series simultaneously
- **Half-life**: Speed of mean reversion (typical: 5-20 days for daily data)

## Position Sizing & Risk Management
- **Kelly criterion**: Optimal leverage based on win rate and payoff ratio
- **Stop losses**: Exit if spread exceeds 3-4 standard deviations
- **Correlation risk**: Pairs may decorrelate during crises

## Python Implementation

- Cointegration testing (statsmodels)
- Spread construction and z-score calculation
- Backtesting frameworks (backtrader, zipline)
- Portfolio construction with multiple pairs
- Risk management (position limits, correlation monitoring)

## Key Takeaways

1. Pairs trading requires cointegration, not just correlation
2. Mean reversion strategies fail when fundamentals change permanently
3. Transaction costs significantly impact stat arb profitability (need tight spreads)
4. Capacity limits: Strategies degrade as AUM grows
5. Risk management is critical—use stop losses and correlation monitoring
`,
};

