import { MultipleChoiceQuestion } from '@/lib/types';

export const financialMarketsExplainedMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'fme-mc-1',
      question:
        'A 10-year bond with 5% annual coupon is trading at $1,050 (par = $1,000). If market yields rise from 4.5% to 5.5%, approximately how much will the bond price change?',
      options: [
        'Increase by ~$50 (yields up, prices up)',
        'Decrease by ~$50 (proportional to yield change)',
        'Decrease by ~$75 (duration effect)',
        'No change (coupon rate fixed)',
      ],
      correctAnswer: 2,
      explanation:
        'Bond prices move INVERSELY to yields. Duration approximates price change: ΔP/P ≈ -Duration × Δy. For a 10-year bond, Modified Duration ≈ 8.5 years. Yield change: 5.5% - 4.5% = 1% = 0.01. Price change: -8.5 × 0.01 = -0.085 = -8.5%. On $1,050: $1,050 × -0.085 ≈ -$89. Closest answer is ~$75 (duration varies by exact cash flows). Key insight: Longer duration = higher interest rate sensitivity. A 30-year bond would fall ~-20% for same 1% yield rise. This is why bond investors fear rising rates: The Fed raising rates from 0% to 5% (2022-2023) caused 20-year Treasury bonds to fall 40%!',
    },
    {
      id: 'fme-mc-2',
      question:
        'In the forex market, if EUR/USD = 1.1000 and you believe the Euro will strengthen, which position should you take?',
      options: [
        'Buy USD, sell EUR (short EUR/USD)',
        'Buy EUR, sell USD (long EUR/USD)',
        'Buy both EUR and USD',
        'Short both EUR and USD',
      ],
      correctAnswer: 1,
      explanation:
        'EUR/USD = 1.1000 means 1 Euro = 1.10 USD. If Euro strengthens, EUR/USD rises (e.g., to 1.1500). To profit, buy EUR/USD (go long). You\'re buying Euros with USD. Example: Buy EUR/USD at 1.1000, sell at 1.1500 → Profit = 1.1500 - 1.1000 = 0.0500 = 5% gain. FX pairs quote base currency / quote currency. Long = bullish on base currency (EUR). Short = bearish on base currency. Key: FX is ALWAYS a pair trade (buy one currency, sell another simultaneously). You can\'t just "buy dollars" - you must specify vs what (EUR, JPY, GBP, etc.).',
    },
    {
      id: 'fme-mc-3',
      question:
        'Bitcoin is trading at $50,000 with 80% annualized volatility. Assuming normal distribution, what is the approximate 1-day 95% confidence interval?',
      options: [
        '$49,000 - $51,000 (±$1,000)',
        '$47,500 - $52,500 (±$2,500)',
        '$45,000 - $55,000 (±$5,000)',
        '$40,000 - $60,000 (±$10,000)',
      ],
      correctAnswer: 1,
      explanation:
        'Daily volatility = Annual volatility / sqrt(365) = 80% / sqrt(365) ≈ 4.2%. For 95% confidence (±2 standard deviations): ±2 × 4.2% = ±8.4%. Price range: $50,000 × 0.916 to $50,000 × 1.084 = $45,800 to $54,200. Closest: $47,500 - $52,500. Key insight: High volatility = wide expected range. 80% volatility is EXTREME (S&P 500 typically 15-20%, Bitcoin often 50-100%). This means: (1) Bitcoin can easily move ±5% daily (normal!), (2) 5%+ moves happen ~30% of days, (3) 10%+ moves happen ~5% of days. This is why Bitcoin is risky: $50K today could be $45K tomorrow (still within normal range). Compare to Apple stock (20% vol): 1-day 95% CI = ±2% ($180 → $176-184, much tighter).',
    },
    {
      id: 'fme-mc-4',
      question:
        'What is the primary advantage of trading commodity futures instead of physical commodities?',
      options: [
        'No delivery required (cash-settled)',
        'Higher returns than physical',
        'Zero transaction costs',
        'No price risk',
      ],
      correctAnswer: 0,
      explanation:
        "Futures contracts are cash-settled (most traders never take delivery). Advantages vs physical: (1) No storage costs (oil tanks expensive!), (2) No transportation (don't need tanker ships), (3) High liquidity (trade millions instantly), (4) Leverage (control $100K oil with $10K margin). Physical commodity challenges: If you buy 1,000 barrels of oil, where do you store it? How do you transport it? What if it spoils (agriculture)? Futures solve this: Trade oil futures, close position before expiration, never touch physical oil. Only commercial users (airlines hedging jet fuel, farmers hedging wheat) take delivery. 98% of futures traders close positions before expiration. Famous exception: 2020 oil futures went NEGATIVE (-$37/barrel) because no storage space + contract expiring + nobody wanted delivery!",
    },
    {
      id: 'fme-mc-5',
      question:
        'The S&P 500 index is at 4,500 and you want to get broad market exposure with $100,000. Which is the most cost-effective approach?',
      options: [
        'Buy all 500 stocks proportionally',
        'Buy SPY ETF (S&P 500 ETF)',
        'Buy S&P 500 futures contracts',
        'Buy mutual fund tracking S&P 500',
      ],
      correctAnswer: 1,
      explanation:
        'SPY ETF (SPDR S&P 500 ETF Trust) is most cost-effective: Expense ratio 0.09% annually, trade like stock (instant execution), highly liquid ($100B+ daily volume), tax-efficient. Compare alternatives: (1) Buying 500 stocks: $100K / 500 = $200 per stock, but many stocks >$200 (Google $150, Amazon $150), rebalancing costs, lots of trades. (2) Futures: Leverage risk, margin requirements, rollover costs, complex for retail. (3) Mutual fund: Higher fees (0.5-1%+), NAV pricing (end of day only), less tax-efficient. SPY advantages: Tiny fee (0.09% = $90/year on $100K), instant trades during market hours, 1 share ≈ $450 (fractional shares possible). VOO (Vanguard S&P 500 ETF) even cheaper: 0.03% expense ratio. This is why passive ETFs revolutionized investing: $100K in SPY costs $30-90/year vs $500-1000/year for active mutual funds.',
    },
  ];
