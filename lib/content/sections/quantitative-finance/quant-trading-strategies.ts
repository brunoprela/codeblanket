import type { ContentSection } from '@/lib/types';

export const quantTradingStrategies: ContentSection = {
  id: 'quant-trading-strategies',
  title: 'Quantitative Trading Strategies',
  content: `
# Quantitative Trading Strategies

## Introduction

Quantitative trading applies mathematical models and systematic rules to identify and exploit market inefficiencies. This section covers momentum, mean reversion, trend following, and volatility strategies.

## Momentum Strategies

### Time-Series Momentum
- **Concept**: Assets with positive past returns continue rising (persistence)
- **Lookback**: 1-12 months (skip most recent month to avoid reversal)
- **Signal**: Binary (long if past return > 0) or continuous (proportional to return)

### Cross-Sectional Momentum
- **Concept**: Buy winners, sell losers relative to universe
- **Construction**: Rank stocks by past returns, long top decile, short bottom decile
- **Rebalancing**: Monthly (balance turnover costs vs signal decay)

### Momentum Crashes
- **Risk**: Severe losses during market reversals (e.g., 2009: -73%)
- **Mitigation**: Volatility scaling, trend filters, stop losses

## Mean Reversion Strategies

### Short-Term Reversal
- **Concept**: Stocks with extreme short-term returns (1 day-1 week) reverse
- **Mechanism**: Liquidity provision, overreaction correction
- **Horizon**: 1-5 days (signals decay quickly)

### Statistical Arbitrage
- **Pairs trading**: Long-short cointegrated pairs
- **Index arbitrage**: ETF vs basket of stocks
- **Cross-asset**: Bonds vs equities, currencies vs commodities

## Trend Following

### Moving Average Crossovers
- **Golden cross**: 50-day MA crosses above 200-day MA (bullish)
- **Death cross**: 50-day MA crosses below 200-day MA (bearish)
- **Execution**: Enter trend, exit on reversal signal

### Breakout Strategies
- **Concept**: Enter when price breaks support/resistance levels
- **Donchian channels**: Buy 20-day high, sell 20-day low
- **Risk**: False breakouts (whipsaws)

## Volatility Strategies

### Volatility Arbitrage
- **Concept**: Trade realized vs implied volatility
- **Long vol**: Buy options when implied vol < expected realized
- **Short vol**: Sell options when implied vol > expected realized

### VIX Trading
- **VIX futures**: Trade volatility expectations
- **Contango trade**: Short VIX futures (roll yield)
- **Risk**: Volatility spikes (unlimited loss potential)

## Strategy Implementation

### Backtesting Framework
- Data quality and survivorship bias
- Transaction costs modeling
- Slippage and market impact
- Out-of-sample testing

### Risk Management
- Position sizing (volatility scaling)
- Stop losses and profit targets
- Drawdown limits
- Correlation monitoring

## Python Applications

- Signal generation and backtesting
- Portfolio optimization
- Risk management (VaR, CVaR)
- Performance attribution
- MLflow for experiment tracking

## Key Takeaways

1. Momentum and mean reversion are complementary (different timeframes)
2. Transaction costs are critical—high-frequency strategies need tight spreads
3. Volatility scaling improves Sharpe ratios by 20-30%
4. Trend following excels in trending markets, fails in choppy markets
5. Combine multiple strategies for diversification (momentum + mean reversion)
`,
};

