import { MultipleChoiceQuestion } from '@/lib/types';

export const marketDataPriceDiscoveryMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'fm-1-12-mc-1',
      question:
        'An order book shows: Bid 10,000 shares at $100.00, Ask 5,000 shares at $100.05. What is the order book imbalance ratio and what does it suggest?',
      options: [
        '0.5 (selling pressure)',
        '2.0 (buying pressure)',
        '15,000 (total size)',
        '0.05 (spread)',
      ],
      correctAnswer: 1,
      explanation:
        'Imbalance = Bid size / Ask size = 10,000 / 5,000 = 2.0. Interpretation: 2x more buying interest than selling → suggests upward price pressure. Imbalance >1.5 often precedes price rise. <0.67 suggests downward pressure. However, visible book is only part of story (hidden orders exist).',
    },
    {
      id: 'fm-1-12-mc-2',
      question:
        'A trader places a 10,000 share buy order at $100.00, then cancels it 100 milliseconds later without executing. This happens 50 times in an hour. This pattern suggests:',
      options: [
        'Legitimate market making',
        'Liquidity provision',
        'Spoofing (illegal manipulation)',
        'Normal trading behavior',
      ],
      correctAnswer: 2,
      explanation:
        'Spoofing: Placing large fake orders to manipulate perception, then cancelling before execution. 50 large orders, 0 fills, >95% cancel rate = clear spoof. Purpose: Create false buy pressure (push price up), then sell real shares at inflated price. Illegal under Dodd-Frank. Regulators detect via high cancel/trade ratio.',
    },
    {
      id: 'fm-1-12-mc-3',
      question:
        'Professional market data (Bloomberg Terminal) costs $2,000/month. A retail alternative using Polygon.io costs $200/month. What is the primary feature difference?',
      options: [
        'Bloomberg is 100ms faster',
        'Bloomberg has news/analytics/terminal features',
        'Polygon data is 15-min delayed',
        'Bloomberg covers more exchanges',
      ],
      correctAnswer: 1,
      explanation:
        'Both provide real-time price data with similar latency (adequate for non-HFT). Bloomberg advantage: Integrated news, analytics, company data, chat, complex terminals. For pure price data, Polygon.io sufficient. Bloomberg worth it for institutions needing integrated workflow. Retail traders: Polygon.io + free tools = 90% of Bloomberg for 10% cost.',
    },
    {
      id: 'fm-1-12-mc-4',
      question:
        'A stock normally trades with a $0.01 spread (1 cent). Today, spread widened to $0.10 (10 cents). What is the most likely cause?',
      options: [
        'More liquidity available',
        'Lower volatility',
        'Pending news or reduced liquidity',
        'Better price discovery',
      ],
      correctAnswer: 2,
      explanation:
        'Spread widening 10x: Market makers pulling back due to: 1) Pending news (earnings, FDA approval), 2) Informed traders detected (adverse selection), 3) Low liquidity (volume dried up), or 4) High volatility (more risk). Transaction cost increased 10x - wait for spread to normalize before trading.',
    },
    {
      id: 'fm-1-12-mc-5',
      question:
        'Level 1 market data shows best bid/ask. Level 2 shows full order book depth. For a $50,000 institutional order, why is Level 2 data valuable?',
      options: [
        'Level 2 is free, Level 1 costs money',
        'See full depth to estimate market impact',
        'Level 2 is faster than Level 1',
        'No difference for this order size',
      ],
      correctAnswer: 1,
      explanation:
        'Level 2 shows order book depth: Can see if $50K order will walk through multiple price levels (high impact) or fill at one level (low impact). Example: Level 1 shows $100.05 ask, but only 1,000 shares there. Level 2 reveals next levels: $100.10 (5K), $100.15 (10K) → estimate fill price ~$100.12. Essential for large orders.',
    },
  ];
