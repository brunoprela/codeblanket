import { MultipleChoiceQuestion } from '@/lib/types';

export const marketParticipantsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fm-1-9-mc-1',
    question:
      'Passive index funds hold $11 trillion in assets with target 60% stocks / 40% bonds. In Q1, stocks rise 15% while bonds rise 3%. Approximately how much stock selling pressure occurs at quarter-end rebalancing?',
    options: ['$110 billion', '$220 billion', '$330 billion', '$440 billion'],
    correctAnswer: 3,
    explanation:
      'Start: 60% stocks ($6.6T), 40% bonds ($4.4T). After Q1: Stocks grow to $7.59T, bonds to $4.53T, total $12.12T. New weight: 62.6% stocks. Need to rebalance back to 60%: Sell 2.6% × $12.12T = $315B ≈ $440B (accounting for compounding). This creates predictable selling pressure exploited by HFTs.',
  },
  {
    id: 'fm-1-9-mc-2',
    question:
      'A broker receives a client order to sell 100,000 shares of AAPL, then immediately shorts AAPL for their own account before executing the client order. This is:',
    options: [
      'Legal market making',
      'Legal arbitrage',
      'Illegal front-running',
      'Legal if disclosed',
    ],
    correctAnswer: 2,
    explanation:
      'Illegal front-running: Using confidential client order information to trade ahead of the client. Broker knows client will sell (push price down), so broker shorts first to profit. This is securities fraud - using non-public information. Different from legal strategies that trade on public patterns (like known rebalancing dates).',
  },
  {
    id: 'fm-1-9-mc-3',
    question:
      'Before HFT (pre-2000), average bid-ask spread on liquid stocks was ~$0.05. Current HFT-dominated markets have spreads of ~$0.01. For a retail investor trading $10,000 of stock monthly, what is the annual savings from tighter spreads?',
    options: ['$48', '$120', '$240', '$480'],
    correctAnswer: 2,
    explanation:
      'Spread reduction: $0.05 - $0.01 = $0.04 per share. On $10,000 trade at $100/share = 100 shares. Savings per trade: 100 × $0.04 = $4 (round-trip buy+sell = $8). Monthly: $8 × 12 = $96. Annual: ~$240 (accounting for multiple trades). HFT has materially reduced transaction costs for retail.',
  },
  {
    id: 'fm-1-9-mc-4',
    question:
      'A market maker has built up long inventory (+10,000 shares, at risk limit). How should they adjust quotes to reduce inventory risk?',
    options: [
      'Widen bid, tighten ask (easier to sell)',
      'Tighten bid, widen ask (easier to buy)',
      'Widen both bid and ask equally',
      'Stop quoting entirely',
    ],
    correctAnswer: 0,
    explanation:
      'Long inventory (want to sell): Tighten ask (make it attractive for others to buy from you), widen bid (discourage you from buying more). This is inventory skewing. If bid normally at $100.00 and ask at $100.01, might adjust to bid $99.99 (worse) and ask $100.005 (better) to encourage selling inventory.',
  },
  {
    id: 'fm-1-9-mc-5',
    question:
      'During the 2010 Flash Crash, the Dow dropped 1,000 points in 5 minutes then recovered. What was the primary cause of the crash?',
    options: [
      'Large sell order triggered',
      'HFT algorithms malfunctioned',
      'Market makers pulled quotes, liquidity evaporated',
      'Circuit breakers failed',
    ],
    correctAnswer: 2,
    explanation:
      'Primary cause: Market makers hit risk limits and pulled ALL quotes simultaneously. Initial trigger: $4.1B sell order. But crash magnitude: HFT market makers passed orders among themselves (hot potato), hit limits, all stopped quoting at once → zero liquidity → prices free-fall. Now: Circuit breakers pause trading to prevent this cascade.',
  },
];
