import { MultipleChoiceQuestion } from '@/lib/types';

export const optionsFundamentalsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'options-fundamentals-mc-1',
    question:
      'You buy 1 AAPL 150 call option for $5.00 when AAPL is trading at $148. At expiration, AAPL is at $160. What is your total profit?',
    options: [
      '$500 (10 points × 100 shares - $500 premium)',
      '$1000 (10 points × 100 shares)',
      '$1500 (15 points × 100 shares - $500 premium)',
      '$5 (per share profit)',
    ],
    correctAnswer: 0,
    explanation:
      'Calculate: Payoff at expiration = max(160 - 150, 0) = $10 per share. Profit = Payoff - Premium = $10 - $5 = $5 per share. Total profit = $5 × 100 shares per contract = $500. Common mistake: Forgetting to subtract the premium paid ($500), which would give payoff ($1000) but not profit. Another mistake: Not multiplying by contract size (100 shares). The option gives you the right to buy 100 shares at $150 when market is $160, netting $10/share intrinsic, minus $5 premium = $5/share profit × 100 = $500.',
  },
  {
    id: 'options-fundamentals-mc-2',
    question:
      'An ATM call option is trading at $6 when the stock is at $100. The intrinsic value is $0. Two weeks later, the stock is still at $100, but the option is now worth $4. What happened?',
    options: [
      'Volatility decreased',
      'Time decay (theta)',
      'Interest rates changed',
      'All of the above could contribute',
    ],
    correctAnswer: 3,
    explanation:
      'The option lost value despite no stock price movement. Since it\'s ATM, intrinsic value is still $0, so the entire $2 loss came from time value. Primary cause: Time decay (theta) - as expiration approaches, time value decreases. The option went from $6 to $4 over two weeks, losing ~33% of its time value. However, volatility decrease could also reduce option price (lower vega), and interest rate changes affect option pricing (rho), though typically minor. The most significant factor for short-term ATM options is theta decay. This demonstrates why selling ATM options can be profitable if the stock stays flat - you collect premium that decays over time.',
  },
  {
    id: 'options-fundamentals-mc-3',
    question:
      'Which statement about moneyness is TRUE?',
    options: [
      'Deep ITM options have the highest time value',
      'ATM options have the highest gamma and theta',
      'OTM options have the highest delta',
      'ITM calls have negative intrinsic value',
    ],
    correctAnswer: 1,
    explanation:
      'ATM (at-the-money) options have the highest gamma (sensitivity to stock price changes) and highest theta (time decay). This is because they have maximum uncertainty about expiring ITM or OTM. Deep ITM options have LOW time value (mostly intrinsic); they behave like stock with high delta. OTM options have LOW delta (0.1-0.4), not high - they have low probability of expiring ITM. ITM calls have POSITIVE intrinsic value (stock price > strike). The gamma and theta relationship at ATM is critical: gamma means small stock moves create large option value changes, theta means time decay is fastest here.',
  },
  {
    id: 'options-fundamentals-mc-4',
    question:
      'You own an American-style AAPL 100 call (50 days to expiration) trading at $12 when AAPL is at $110. The intrinsic value is $10. Should you exercise the option now?',
    options: [
      'Yes, lock in the $10 profit immediately',
      'No, the option has $2 of time value you would lose',
      'Yes, if a dividend is about to be paid',
      'Both B and C are valid considerations',
    ],
    correctAnswer: 3,
    explanation:
      'The option is worth $12 (intrinsic $10 + time value $2). If you exercise now, you get only the $10 intrinsic value, LOSING the $2 time value. Better to SELL the option for $12 and capture both components. However, American options CAN be optimally exercised early in one scenario: if AAPL pays a dividend soon. If dividend > time value, you should exercise before ex-dividend date to capture the dividend. Example: If AAPL pays $1 dividend tomorrow and time value is only $0.50, exercise today to get shares and dividend. Otherwise, hold or sell the option. This is why American options are worth slightly more than European options - the flexibility to capture dividends.',
  },
  {
    id: 'options-fundamentals-mc-5',
    question:
      'A trader sells (writes) a naked call option on TSLA with a 200 strike for $8 premium when TSLA is at $195. At expiration, TSLA is at $250. What is the trader\'s profit/loss?',
    options: [
      'Profit of $800 (collected premium)',
      'Loss of $4,200 ($50 move - $8 premium)',
      'Loss of $5,000 ($50 intrinsic value)',
      'Loss of $42 per share',
    ],
    correctAnswer: 1,
    explanation:
      'Selling a naked call means you collected $8 premium but must deliver shares at $200 if assigned. At expiration, TSLA at $250 means the option is $50 ITM - you\'re assigned. Loss calculation: You must buy TSLA at $250 (market) and sell at $200 (strike) = -$50 per share. You collected $8 premium. Net loss = -$50 + $8 = -$42 per share × 100 shares = -$4,200 total. This demonstrates the UNLIMITED RISK of naked calls - if TSLA went to $400, your loss would be $192 per share (minus premium) = -$18,400. This is why brokers require high margin and approval levels for naked call selling. Defined-risk alternatives like call spreads are safer for most traders.',
  },
];

