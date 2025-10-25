import { MultipleChoiceQuestion } from '@/lib/types';

export const commoditiesMarketsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fm-1-5-mc-1',
    question:
      'Crude oil futures show 1-month at $80, 12-month at $90. This curve structure is called:',
    options: ['Backwardation', 'Contango', 'Inversion', 'Flat'],
    correctAnswer: 1,
    explanation:
      'Contango: Near-term contracts cheaper than distant contracts. Caused by storage costs + financing costs > convenience yield. Normal state for most commodities. Roll cost: When rolling long positions, you sell $80 and buy $82 next month = negative roll yield. Annual drag can be 10-15%.',
  },
  {
    id: 'fm-1-5-mc-2',
    question:
      'An oil ETF (USO) holds front-month futures in contango (1M: $80, 2M: $82). Each month when rolling, what is the approximate roll loss per barrel?',
    options: ['$0 (no loss)', '$2 (2.5%)', '$10 (12.5%)', '$20 (25%)'],
    correctAnswer: 1,
    explanation:
      'Roll loss = next month - current month = $82 - $80 = $2 per barrel (2.5%). Over 12 months: ~2.5% × 12 = 30% erosion even if oil price unchanged. 2020 example: USO lost 70% while oil fell only 20%, due to extreme contango roll costs.',
  },
  {
    id: 'fm-1-5-mc-3',
    question:
      'Gold futures: 1-month at $2000, 12-month at $1980 (backwardation). What does this signal?',
    options: [
      'Ample supply',
      'Tight current supply / high demand',
      'Storage costs high',
      'Market expects gold to rise',
    ],
    correctAnswer: 1,
    explanation:
      'Backwardation signals tight current supply - buyers pay premium for immediate delivery vs future delivery. Convenience yield > cost of carry. Common during crises when everyone wants physical gold NOW. Creates positive roll yield: sell expiring at $2000, buy next at $1990 = profit.',
  },
  {
    id: 'fm-1-5-mc-4',
    question:
      'Cash-and-carry arbitrage: Spot oil = $79, 12-month futures = $90, storage + financing = $8/barrel. What is the arbitrage profit per barrel?',
    options: ['$0 (no arbitrage)', '$3', '$8', '$11'],
    correctAnswer: 1,
    explanation:
      'Fair futures price = Spot + Carry = $79 + $8 = $87. Actual futures = $90. Arbitrage profit = $90 - $87 = $3/barrel. Trade: Buy spot at $79, sell futures at $90, store for 12 months (cost $8), deliver at $90 → net $3 profit. Risk-free if costs are certain.',
  },
  {
    id: 'fm-1-5-mc-5',
    question:
      'A commodity curve was in contango, now flipping to backwardation. What likely happened?',
    options: [
      'Supply increased',
      'Storage costs fell',
      'Demand surge / supply disruption',
      'Interest rates dropped',
    ],
    correctAnswer: 2,
    explanation:
      'Flip from contango to backwardation indicates supply shortage or demand surge. Market now values immediate delivery more than future. Example: Oil 2022 Russia-Ukraine war → immediate supply concerns → backwardation. Creates positive roll yield for long positions (sell high near, buy low far).',
  },
];
