import { MultipleChoiceQuestion } from '@/lib/types';

export const foreignExchangeMarketsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fm-1-4-mc-1',
    question:
      'EUR/USD is quoted at 1.1000/1.1005. You want to buy €500,000. How much USD do you pay?',
    options: ['$550,000', '$550,250', '$552,500', '$555,000'],
    correctAnswer: 1,
    explanation:
      "When you buy EUR, you pay the ask price: 1.1005. EUR 500,000 × 1.1005 = $550,250. The bid (1.1000) is where the bank buys EUR from you. The 5-pip spread ($250 on this trade) is the bank's profit for providing liquidity.",
  },
  {
    id: 'fm-1-4-mc-2',
    question:
      'A carry trade borrows JPY at 0.1% and invests in AUD at 4.0%. If AUD/JPY is unchanged after 1 year, what is the approximate return?',
    options: ['0.1%', '3.9%', '4.0%', '4.1%'],
    correctAnswer: 1,
    explanation:
      'Carry return = invest rate - borrow rate = 4.0% - 0.1% = 3.9%. This assumes FX rate unchanged. If AUD strengthens, you gain more. If AUD weakens, you can lose more than the carry earned. 2008 crisis: AUD crashed -29%, wiping out years of 3-4% annual carry gains.',
  },
  {
    id: 'fm-1-4-mc-3',
    question:
      'An FX trading system receives quotes from 3 providers for EUR/USD: Provider A: 1.1000/1.1005, Provider B: 1.1001/1.1004, Provider C: 1.0999/1.1006. What is the best bid (to sell EUR)?',
    options: ['1.0999', '1.1000', '1.1001', '1.1004'],
    correctAnswer: 2,
    explanation:
      'Best bid = highest bid across all providers = 1.1001 (Provider B). When selling EUR, you want the highest bid (receive most USD). Smart order routing: Buy EUR at 1.1004 (Provider B, best ask), Sell EUR at 1.1001 (Provider B, best bid). This minimizes transaction costs.',
  },
  {
    id: 'fm-1-4-mc-4',
    question:
      'You have a portfolio with $1M USD and €500K EUR. Current EUR/USD = 1.10. What is your total portfolio value in USD?',
    options: ['$1.5M', '$1.55M', '$1.6M', '$2.1M'],
    correctAnswer: 1,
    explanation:
      'Total USD = $1M + (€500K × 1.10) = $1M + $550K = $1.55M. To calculate P&L, convert all positions to base currency. If EUR strengthens to 1.15, portfolio value = $1M + $575K = $1.575M (+$25K gain on EUR position).',
  },
  {
    id: 'fm-1-4-mc-5',
    question:
      'During the 2008 crisis, carry trades unwound violently. AUD/JPY fell from 100 to 60. What was the approximate loss on a ¥100M carry trade?',
    options: ['-10%', '-20%', '-40%', '-60%'],
    correctAnswer: 2,
    explanation:
      'AUD/JPY from 100 to 60 = -40% loss on AUD position. Initial: ¥100M → AUD 1M (at 100). Final: AUD 1M → ¥60M (at 60). Loss: ¥40M = 40%. Even with +3.9% carry earned, net loss ~36%. This shows carry trade risk: small gains in calm markets, large losses in crisis.',
  },
];
