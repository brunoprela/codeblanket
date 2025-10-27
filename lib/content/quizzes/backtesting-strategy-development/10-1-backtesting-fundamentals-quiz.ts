export const backtestingFundamentalsQuiz = [
  {
    id: 1,
    question:
      'You backtest a trading strategy on S&P 500 stocks from 2010-2023 using a dataset of the current 500 constituents. The strategy shows a 15% annual return. What is the MOST likely problem with this backtest?',
    options: [
      'The backtest period is too long and should be shortened to 5 years',
      'Survivorship bias - the dataset excludes stocks that were delisted or went bankrupt, artificially inflating returns',
      'The strategy is overfitted because 15% is too high to be realistic',
      'Transaction costs were not included, but they would only reduce returns by 1-2%',
      'Nothing is wrong - this is a properly conducted backtest',
    ],
    correctAnswer: 1,
    explanation:
      "This is a classic case of **survivorship bias**. The current S&P 500 constituents are the 'winners' - companies that survived and thrived. The dataset excludes companies that failed (like Lehman Brothers in 2008, Enron, etc.) or were removed from the index. Studies show survivorship bias can inflate returns by 2-5% annually or more. To fix this, you need point-in-time constituent data - on Jan 1 2010, use the S&P 500 members as of that date, not as of today. **Real impact**: A 15% return with survivorship bias might actually be 8-10% with proper data.",
    difficulty: 'intermediate',
  },
  {
    id: 2,
    question: 'Which of the following code snippets contains look-ahead bias?',
    options: [
      "df['Signal'] = np.where(df['Close'] > df['Close'].shift(1), 1, -1)",
      "df['SMA_50'] = df['Close'].rolling(50).mean(); df['Signal'] = np.where(df['Close'] > df['SMA_50'], 1, -1)",
      "df['Tomorrow_Close'] = df['Close'].shift(-1); df['Signal'] = np.where(df['Tomorrow_Close'] > df['Close'], 1, -1)",
      "df['Signal'] = np.where(df['Close'] > df['Open'], 1, -1)",
      'None of the above contain look-ahead bias',
    ],
    correctAnswer: 2,
    explanation:
      "Option C contains **look-ahead bias**. Using `shift(-1)` pulls tomorrow's closing price into today's row, allowing the strategy to 'peek into the future'. The signal is based on tomorrow's price, which wouldn't be known at the time of trading. This will show amazing backtest results but fail immediately in live trading. **Option A** is correct (uses yesterday's close with shift(1)). **Option B** is also correct (SMA uses only past data). **Option D** is debatable - using same-bar close to generate a signal executed at that close is technically look-ahead, but if you're using the signal for next bar, it's fine. The rule: Never use `shift(-1)` or any future data.",
    difficulty: 'intermediate',
  },
  {
    id: 3,
    question:
      'A backtest shows your strategy generates 200 trades per year with an average return of 0.5% per trade. However, you forgot to include transaction costs. The average bid-ask spread is 0.05% and commissions are $1 per trade. For a typical trade size of $10,000, what is the ACTUAL expected return per trade after costs?',
    options: [
      'Still around 0.5% - costs are negligible',
      'About 0.4% after costs',
      'About 0.2% after costs',
      'Close to 0% or slightly negative',
      'Significantly negative - the strategy loses money',
    ],
    correctAnswer: 3,
    explanation:
      "Let's calculate the actual costs:\n\n**Costs per round-trip trade (buy + sell):**\n- Bid-ask spread: 0.05% × 2 = 0.10% per round trip\n- Commissions: $1 × 2 = $2 per round trip\n- On $10,000: $2/$10,000 = 0.02%\n- **Total cost: 0.10% + 0.02% = 0.12% per trade**\n\n**Net return:** 0.5% - 0.12% = **0.38%** per trade\n\nHowever, we haven't included slippage (execution at worse prices than expected), which typically adds another 0.05-0.10%. Adding 0.08% slippage: 0.38% - 0.08% = **0.30%** net.\n\nBut with 200 trades/year, costs compound. A strategy showing 0.5% per trade (100% annual) might actually deliver 60-70% after costs, or the strategy might be close to breakeven. Option D is most realistic. **Key lesson:** High-frequency strategies are destroyed by transaction costs.",
    difficulty: 'advanced',
  },
  {
    id: 4,
    question:
      'What is the main advantage of event-driven backtesting over vectorized backtesting?',
    options: [
      'Event-driven backtesting is much faster and more efficient',
      'Event-driven backtesting can more realistically simulate order execution, partial fills, and complex portfolio logic',
      'Event-driven backtesting is easier to implement and requires less code',
      'Event-driven backtesting automatically eliminates look-ahead bias',
      'Vectorized backtesting is always better - there is no advantage to event-driven',
    ],
    correctAnswer: 1,
    explanation:
      "**Event-driven backtesting** processes data point-by-point (like live trading) and can realistically model:\n- Order execution delays\n- Partial fills (order only partially executed)\n- Market impact (your orders moving prices)\n- Complex portfolio rebalancing logic\n- Stop losses and dynamic risk management\n\nThis makes it much more realistic for production systems. **Vectorized backtesting** (processing entire dataset at once with numpy/pandas) is **faster** and simpler but trades off realism. It's good for quick prototyping but can't easily model complex order logic.\n\n**Production systems** (Renaissance Technologies, Citadel, Two Sigma) all use event-driven architectures because the code closely resembles live trading systems, making the transition smoother. The trade-off: event-driven is 10-100x slower, but accuracy matters more than speed.",
    difficulty: 'beginner',
  },
  {
    id: 5,
    question:
      'Your strategy passes backtesting with a Sharpe ratio of 2.5 and shows consistent profits across different time periods. What should you do BEFORE deploying it with real money?',
    options: [
      'Deploy immediately - a Sharpe of 2.5 is excellent and rare',
      'Run paper trading for 30 days, then go live with full capital',
      'Run paper trading for 3-6 months with real-time data, then start live trading with small capital (1-5% of intended size)',
      'Skip paper trading but start with small capital and scale up weekly',
      'Get more historical data and re-run the backtest',
    ],
    correctAnswer: 2,
    explanation:
      'Even with excellent backtest results, you should **never deploy directly to live trading**. The proper progression is:\n\n**Stage 1: Backtest** (✓ Done - Sharpe 2.5 is excellent)\n**Stage 2: Paper Trading** - Run for **3-6 months** with real-time data and simulated execution. This reveals:\n- How strategy behaves with live market conditions\n- Execution challenges (slippage, partial fills)\n- Data feed issues\n- Operational problems in your code\n- Whether backtest results were realistic\n\n**Typical degradation**: Expect 20-50% worse performance in paper trading vs backtest.\n\n**Stage 3: Live Trading** - Start with **small capital** (1-5% of intended size):\n- Monitor for 30-60 days\n- Scale up gradually if performance matches expectations\n- Be ready to shut down if performance degrades\n\n**Real-world**: Renaissance Technologies and Two Sigma run paper trading for 6-12 months. Most strategies that pass backtest **fail in paper trading**. Option C is the professional approach.',
    difficulty: 'beginner',
  },
];
