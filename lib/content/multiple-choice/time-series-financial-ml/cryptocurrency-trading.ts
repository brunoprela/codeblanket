import { MultipleChoiceQuestion } from '@/lib/types';

export const cryptocurrencyTradingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'ct-mc-1',
    question: 'What is unique about crypto trading hours?',
    options: [
      '9:30am-4pm like stocks',
      '24/7/365 no market close',
      'Only weekdays',
      'Random hours',
    ],
    correctAnswer: 1,
    explanation:
      '24/7/365: Crypto never closes. Trade anytime. No weekend gaps (unlike stocks). Pros: More opportunities, flexible. Cons: Need constant monitoring, no "safe" overnight. Solution: Automated systems or shift trading. Can\'t relax on weekends.',
  },
  {
    id: 'ct-mc-2',
    question: 'What is typical crypto volatility vs stocks?',
    options: [
      'Same as stocks',
      '2-5x higher (40-60% annual vs 15-20%)',
      'Lower',
      'No volatility',
    ],
    correctAnswer: 1,
    explanation:
      'Crypto: 40-60% annual volatility. Stocks: 15-20%. Bitcoin 2-5x more volatile than S&P 500. Altcoins even higher (100%+). Pros: More profit potential. Cons: Larger drawdowns, harder to hold. Requires tighter risk management, smaller positions.',
  },
  {
    id: 'ct-mc-3',
    question: 'What are on-chain metrics?',
    options: [
      'Random data',
      'Blockchain data: transactions, addresses, exchange flows',
      'Stock data',
      'Nothing',
    ],
    correctAnswer: 1,
    explanation:
      'On-chain: Data from blockchain. Examples: Exchange inflows (bearish), outflows (bullish), active addresses, transaction volume, MVRV ratio. Unique to crypto. Edge over pure technical analysis. Sources: Glassnode, CryptoQuant. Institutional traders use heavily.',
  },
  {
    id: 'ct-mc-4',
    question: 'What is funding rate in perpetual futures?',
    options: [
      'Random fee',
      'Periodic payment long/short traders exchange to balance market',
      'Stock dividend',
      'No such thing',
    ],
    correctAnswer: 1,
    explanation:
      'Funding rate: Payment every 8 hours in perpetual futures. Positive rate → Longs pay shorts (bullish market). Negative → Shorts pay longs (bearish). High positive funding (>0.1%) = overheated longs → bearish signal. Use as contrarian indicator.',
  },
  {
    id: 'ct-mc-5',
    question: 'What is safer position size for crypto vs stocks?',
    options: [
      'Same as stocks (2%)',
      '1% per trade (vs 2% stocks) due to higher volatility',
      '10% per trade',
      'All-in',
    ],
    correctAnswer: 1,
    explanation:
      '1% risk per trade for crypto vs 2% for stocks. Why: 2-5x higher volatility = larger swings. 2% crypto risk = equivalent to 5-10% stock risk. Prevents blown account. With 1%, need 100 consecutive losses to wipe out (unlikely). Conservative sizing essential for survival.',
  },
];
