export const equityMarketsMultipleChoiceQuestions = [
  {
    id: 1,
    question: "A stock is trading with a bid of $50.00 (size: 200 shares) and an ask of $50.05 (size: 300 shares). If you place a market order to buy 500 shares, what is the MOST likely outcome?",
    options: [
      "You'll buy all 500 shares at $50.05",
      "You'll buy 300 shares at $50.05, and the remaining 200 at the next ask price level (likely higher than $50.05)",
      "You'll buy all 500 shares at the mid-price of $50.025",
      "You'll buy 200 shares at $50.00 (the bid price) and 300 at $50.05",
      "Your order will be rejected because it exceeds available liquidity"
    ],
    correctAnswer: 1,
    explanation: "A market BUY order takes liquidity from the ask (sell) side of the order book. You'll first match with the 300 shares available at $50.05, then you'll need to 'walk up the book' to fill the remaining 200 shares at the next price level, which will be higher. This is called **price slippage** and is why large market orders can be expensive on illiquid stocks.",
    difficulty: "intermediate"
  },
  {
    id: 2,
    question: "Company A's stock trades at $200 per share with 1 billion shares outstanding. Company B's stock trades at $50 per share with 6 billion shares outstanding. Which statement is TRUE?",
    options: [
      "Company A is more valuable because its stock price is 4x higher",
      "Company B is more valuable with a market cap of $300B vs Company A's $200B",
      "They have equal value since 200 × 1 = 50 × 4",
      "Cannot determine value without knowing their revenue",
      "Company A is more valuable because higher stock prices always mean better companies"
    ],
    correctAnswer: 1,
    explanation: "Market capitalization = Stock Price × Shares Outstanding. Company A: $200 × 1B = $200B. Company B: $50 × 6B = $300B. Company B is 50% more valuable despite having a lower stock price. **Stock price alone is meaningless** - market cap is what matters. This is why Amazon at $3,000/share wasn't necessarily 'more expensive' than Apple at $150/share (after splits).",
    difficulty: "beginner"
  },
  {
    id: 3,
    question: "An engineer is backtesting a trading strategy and finds it would have returned 15% annually from 2018-2023, with a Sharpe ratio of 2.1. However, the strategy generated 250 trades per year, and the backtest used **closing prices** without considering bid-ask spreads. The stock's average spread is 0.10%. What is the most likely ACTUAL return after accounting for realistic trading costs?",
    options: [
      "Still around 15% because spreads are too small to matter",
      "Around 12-13% after accounting for spreads",
      "Around 8-10% after spreads and other hidden costs (slippage, market impact)",
      "Around 3-5% after all realistic costs make the strategy barely profitable",
      "Negative returns - the strategy would lose money in reality"
    ],
    correctAnswer: 3,
    explanation: "With 250 trades/year and 0.10% spread cost per trade (actually 0.10% × 2 = 0.20% per round-trip), you lose 250 × 0.20% = **50% cumulative drag** over the year. Additional costs include: slippage (0.05-0.10%), market impact on larger orders (0.05-0.10%), commissions, and timing delays. A 15% gross return can easily become 3-5% net, or even negative. This is why **most backtested strategies fail in live trading** - transaction costs are often underestimated.",
    difficulty: "advanced"
  },
  {
    id: 4,
    question: "The weak-form efficient market hypothesis states that:",
    options: [
      "Stock prices reflect all information, including insider information, so no one can beat the market",
      "Stock prices reflect all publicly available information, so fundamental analysis doesn't work",
      "Stock prices reflect all historical price information, so technical analysis doesn't provide an edge",
      "Stock markets are weakly correlated with economic fundamentals",
      "Smaller stocks are less efficient than large-cap stocks"
    ],
    correctAnswer: 2,
    explanation: "Weak-form EMH states that current prices already incorporate all **historical price and volume data**. If true, patterns in past prices (technical analysis) cannot predict future prices better than chance. **Semi-strong form** says public information is priced in (fundamental analysis won't help). **Strong form** says even insider info is priced in (illegal to trade on anyway). In reality, large-cap stocks are mostly weak-form efficient, but small-caps and international markets show more anomalies.",
    difficulty: "intermediate"
  },
  {
    id: 5,
    question: "You're designing a real-time stock quote display system for a trading app. Users will see the data with a 15-minute delay (free tier). A user places an order at 2:30 PM, but the quote they see shows data from 2:15 PM. During those 15 minutes, the stock moved from $100.00 to $102.00. The user places a market order to buy thinking they'll get ~$100, but gets filled at $102. This is a violation of:",
    options: [
      "SEC Rule 10b-5 (insider trading regulations)",
      "Pattern Day Trader (PDT) rules",
      "Best execution requirements - the broker must get the best available price",
      "Nothing - the user should have paid for real-time data, and the order was executed properly at current market price",
      "Reg NMS Order Protection Rule"
    ],
    correctAnswer: 3,
    explanation: "This is NOT a violation. The user is seeing **delayed data** (clearly disclosed as 15-minute delay), and the broker executed the market order at the **current best available price** ($102), which IS best execution. The broker has no obligation to fill at the delayed price the user saw. **Key lesson for engineers:** Display warnings when users are viewing delayed data, show estimated current price, and educate users that market orders execute at current market price, not the stale price they see. Consider blocking market orders for users on delayed data, or requiring limit orders.",
    difficulty: "advanced"
  }
];

