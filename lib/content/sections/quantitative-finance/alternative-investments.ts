import type { ContentSection } from '@/lib/types';

export const alternativeInvestments: ContentSection = {
  id: 'alternative-investments',
  title: 'Alternative Investments',
  content: `
# Alternative Investments

## Introduction

Alternative investments include hedge funds, private equity, real estate, commodities, and cryptocurrencies—assets beyond traditional stocks and bonds. These provide diversification, inflation hedging, and access to unique return sources.

## Hedge Funds

### Strategies
- **Long-short equity**: Beta-neutral stock picking
- **Global macro**: Top-down bets on currencies, rates, commodities
- **Event-driven**: Merger arbitrage, distressed debt
- **Relative value**: Convertible arbitrage, fixed income arbitrage

### Performance Characteristics
- **Absolute return focus**: Target positive returns regardless of market
- **Low correlation**: 0.3-0.6 correlation with equities
- **Fee structure**: 2% management + 20% performance (2-and-20)

## Private Equity

### Types
- **Buyouts**: Acquire companies, improve operations, sell (5-7 year horizon)
- **Venture capital**: Early-stage tech/biotech (high risk, high return)
- **Growth equity**: Minority stakes in growth companies

### Characteristics
- **Illiquidity**: 10+ year lock-ups
- **J-curve**: Negative returns initially, positive after 3-5 years
- **IRR focus**: Target 15-25% net IRR

## Real Estate

### Investment Vehicles
- **REITs**: Publicly traded real estate portfolios
- **Direct ownership**: Physical property investment
- **Real estate funds**: Pooled private real estate

### Return Drivers
- **Income**: Rental yields (4-6% annually)
- **Appreciation**: Property value growth
- **Leverage**: Amplifies returns via mortgages

## Commodities

### Categories
- **Energy**: Crude oil, natural gas
- **Metals**: Gold, silver, copper
- **Agriculture**: Corn, wheat, soybeans

### Investment Methods
- **Futures**: Direct commodity exposure
- **ETFs/ETNs**: Commodity index funds
- **Stocks**: Mining/energy companies

### Characteristics
- **Inflation hedge**: Commodities rise with inflation
- **Contango drag**: Futures roll costs
- **Low correlation**: 0.0-0.3 with equities

## Cryptocurrencies

### Major Assets
- **Bitcoin**: Digital gold, store of value
- **Ethereum**: Smart contract platform
- **Stablecoins**: Pegged to fiat currencies

### Characteristics
- **High volatility**: 50-100% annual volatility
- **24/7 trading**: No market close
- **Custody risk**: Exchange hacks, wallet security

## Portfolio Allocation

### Diversification Benefits
- **Low correlation**: Alternatives reduce portfolio volatility
- **Risk-return optimization**: Efficient frontier shifts outward
- **Tail risk hedging**: Some alternatives (gold, managed futures) protect in crashes

### Typical Allocations
- **Institutional**: 20-40% alternatives (endowments, pensions)
- **Retail**: 5-15% alternatives (via liquid alts, REITs)

## Python Applications

- Portfolio optimization with alternatives
- Correlation analysis
- Liquidity-adjusted returns
- Private equity cash flow modeling
- Cryptocurrency market analysis (ccxt library)

## Key Takeaways

1. Alternatives provide diversification—low 0.3-0.6 correlation with equities
2. Illiquidity premium: Private equity/real estate offer 2-5% premium vs public markets
3. Fees matter: 2-and-20 structure requires significant outperformance to justify
4. Commodities hedge inflation but suffer contango drag (negative roll yield)
5. Cryptocurrencies offer high return potential but extreme volatility (50%+ annually)
`,
};

