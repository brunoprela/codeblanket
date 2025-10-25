import { MultipleChoiceQuestion } from '@/lib/types';

export const typesOfFinancialInstitutionsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'tfi-mc-1',
      question:
        'An investment bank underwrites a $2 billion IPO with a 7% spread. How much does the investment bank earn?',
      options: ['$14 million', '$70 million', '$140 million', '$700 million'],
      correctAnswer: 2,
      explanation:
        'Underwriting fee = $2B × 0.07 = $140M. The "spread" is the difference between what the bank pays the company and what it sells shares to the public for. For tech IPOs, 7% is standard (lower for larger, established companies, higher for risky ones). Example: Airbnb\'s $3.5B IPO → ~$245M in banking fees split among Goldman Sachs, Morgan Stanley, etc. Investment banking is a transaction business: Big deals = big fees.',
    },
    {
      id: 'tfi-mc-2',
      question:
        'A hedge fund manages $10 billion and returns 20% annually. With "2 and 20" fees, how much does the fund earn in fees?',
      options: [
        '$200 million (2% management fee only)',
        '$400 million (20% performance fee only)',
        '$600 million (both fees)',
        '$2.4 billion (20% of ending value)',
      ],
      correctAnswer: 2,
      explanation:
        'Management fee: $10B × 2% = $200M. Gross profit: $10B × 20% = $2B. Performance fee: $2B × 20% = $400M. Total fees: $200M + $400M = $600M. The fund takes 30% of the $2B profit (\$600M), investors keep $1.4B (14% net return). This is why hedge funds need to outperform: After fees, 20% gross → 14% net. Note: Modern funds often charge less (1.5 and 15 or 1 and 10) due to competition and underperformance. Renaissance Medallion famously charges 5% + 44%!',
    },
    {
      id: 'tfi-mc-3',
      question:
        'Which institution type typically offers the best work-life balance for engineers?',
      options: [
        'Market makers (Citadel Securities, Jane Street)',
        'Hedge funds (Two Sigma, Renaissance Technologies)',
        'Asset managers (BlackRock, Vanguard)',
        'Investment banks (Goldman Sachs, Morgan Stanley)',
      ],
      correctAnswer: 2,
      explanation:
        'Asset managers (BlackRock, Vanguard) typically have the best work-life balance: 45-55 hour weeks, predictable hours, less pressure. Investment banks: 60-80+ hours (worst). Hedge funds: 50-70 hours, high pressure. Market makers: 50-65 hours, market-hours stress. Fintech: 45-60 hours, startup pace. Trade-off: Asset managers pay less ($120-280K) vs market makers ($200-500K+), but better balance. Choose based on priorities: Max comp → market makers/quant funds. Max balance → asset managers. Modern tech + products → fintech.',
    },
    {
      id: 'tfi-mc-4',
      question:
        'A market maker quotes $100.00 bid / $100.02 ask on a stock with $50 million average daily volume. If the market maker trades 1% of daily volume (500K shares), earning the spread on each trade, what is the daily profit?',
      options: [
        '$1,000 (0.02 × 50,000 shares)',
        '$10,000 (0.02 × 500,000 shares)',
        '$100,000 (0.20 × 500,000 shares)',
        '$1,000,000 (2.00 × 500,000 shares)',
      ],
      correctAnswer: 1,
      explanation:
        'Spread = $0.02 per share. Volume = 500K shares. Profit = 500,000 × $0.02 = $10,000 daily. Annualized (252 trading days): $2.52M. Key insight: Market making is HIGH VOLUME, LOW MARGIN. Need to trade millions of shares daily to be profitable. At scale: If Citadel Securities trades 1% of $5 trillion daily U.S. equity volume = $50B, with average 0.5 bps spread = $25M daily profit = $6.3B annually. This explains why market makers invest heavily in speed: Microseconds matter when profit per trade is $0.02.',
    },
    {
      id: 'tfi-mc-5',
      question:
        "Stripe processes $300 billion in payment volume annually and charges 2.9% + $0.30 per transaction. If the average transaction is $50, what is Stripe\'s approximate annual revenue?",
      options: [
        '$3 billion (300B × 1%)',
        '$6.9 billion (300B × 2.3%)',
        '$8.7 billion (300B × 2.9%)',
        '$10.5 billion (fees + fixed)',
      ],
      correctAnswer: 3,
      explanation:
        'Number of transactions: $300B / $50 = 6 billion transactions. Percentage fee: $300B × 2.9% = $8.7B. Fixed fee: 6 billion × $0.30 = $1.8B. Total revenue: $8.7B + $1.8B = $10.5B. Actual Stripe revenue ~$10-12B (2023 estimated). This model scales beautifully: Fixed costs (infrastructure) ~$1B, variable costs ~$6B (interchange to banks), profit ~$3-4B. Key insight: Payment processing is a SCALE business. Need massive volume to justify infrastructure investment. Stripe competes by: (1) Best developer experience (7 lines of code), (2) Global reach (135+ countries), (3) Platform expansion (Stripe Capital, Treasury, Tax).',
    },
  ];
