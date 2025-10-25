export const etfsMutualFundsQuiz = {
  title: 'ETFs & Mutual Funds - Quiz',
  questions: [
    {
      id: 1,
      question:
        "An investor has $10,000 invested in an active mutual fund with a 0.75% expense ratio that matches the market's 10% annual return. A passive index fund with a 0.03% expense ratio also tracks the market. After 10 years, what is the approximate difference in final values?",
      options: ['$720', '$1,100', '$1,480', '$2,000'],
      correctAnswer: 1,
      explanation:
        'Active fund net return: 10% - 0.75% = 9.25%. Passive fund: 10% - 0.03% = 9.97%. After 10 years: Active = $10K × (1.0925)^10 = $24,185. Passive = $10K × (1.0997)^10 = $25,286. Difference = $1,101. The 0.72% annual fee difference compounds to ~$1,100 lost over 10 years. This is why fees matter!',
    },
    {
      id: 2,
      question:
        'SPY (S&P 500 ETF) has a NAV of $500.00 but is trading at $500.40. An Authorized Participant (AP) can create a creation unit of 50,000 shares. Ignoring transaction costs, what is the arbitrage profit per creation unit?',
      options: ['$2,000', '$10,000', '$20,000', '$40,000'],
      correctAnswer: 2,
      explanation:
        'Premium = $500.40 - $500.00 = $0.40 per share. The AP: 1) Buys underlying 500 stocks ($500.00 per SPY equivalent), 2) Delivers to SPY for 50,000 new shares, 3) Sells SPY shares at $500.40. Profit = $0.40 × 50,000 shares = $20,000 per creation unit. This arbitrage mechanism keeps ETF prices close to NAV.',
    },
    {
      id: 3,
      question:
        'A mutual fund with $100M AUM and $20M unrealized capital gains has a $10M redemption (10% of AUM). Using cash redemption, approximately how much in capital gains will be distributed to remaining shareholders? (Assume 20% capital gains tax)',
      options: ['$0', '$400,000', '$2,000,000', '$4,000,000'],
      correctAnswer: 1,
      explanation:
        "Redemption % = $10M / $100M = 10%. Gains realized = $20M × 10% = $2M. Tax bill = $2M × 20% = $400,000. This $400K tax burden hits ALL remaining shareholders, even those who didn't redeem! This is why mutual funds are tax-inefficient vs ETFs (which use in-kind redemptions, realizing zero gains).",
    },
    {
      id: 4,
      question:
        'Factor investing data shows that from 1970-2000, the Value factor (low P/E stocks) outperformed Growth by ~3% annually. However, from 2010-2020, Value underperformed Growth by ~5% annually. What is the most likely explanation?',
      options: [
        'Value factor is permanently dead and will never work again',
        'Crowding effect: Once discovered, too much capital exploited the factor',
        'Value companies went bankrupt during this period',
        'Factor data before 2010 was incorrectly calculated',
      ],
      correctAnswer: 1,
      explanation:
        "The crowding effect: When Fama-French published the Value factor (1993), capital rushed into value stocks, driving up prices and eliminating the 'cheap' premium. Plus, 2010-2020 was unique: low rates favored growth, tech disruption hurt value (banks, energy, retail). This doesn't mean value is dead - it's cyclical. Value outperformed 2020-2022 after the drought.",
    },
    {
      id: 5,
      question:
        'During the March 2020 COVID crash, some corporate bond ETFs (like LQD) traded at 5-10% discounts to NAV despite the creation/redemption mechanism that normally keeps premiums/discounts under 0.5%. Why did the arbitrage mechanism fail?',
      options: [
        'Authorized Participants stopped working during the pandemic',
        'ETF issuers suspended creations to protect themselves',
        'Underlying bonds stopped trading, making NAV unreliable and arbitrage too risky',
        'The SEC banned arbitrage trading during the crisis',
      ],
      correctAnswer: 2,
      explanation:
        "In March 2020's liquidity crisis, corporate bonds largely stopped trading. Without bond trades, NAV calculations used stale prices. APs couldn't price their arbitrage risk (what if bonds are really worth 10% less than NAV shows?). Plus, APs hit risk limits, funding dried up, and extreme volatility made settlement risk too high. The mechanism works in normal markets but can break during liquidity crises, especially for less-liquid assets like bonds.",
    },
  ],
};
