import { MultipleChoiceQuestion } from '@/lib/types';

export const fixedIncomePortfolioManagementMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'fipm-mc-1',
      question:
        'What is the primary advantage of a barbell strategy over a bullet strategy in fixed income?',
      options: [
        'Higher convexity (greater price appreciation when rates fall)',
        'Higher yield',
        'Lower transaction costs',
        'Simpler to manage',
      ],
      correctAnswer: 0,
      explanation:
        'Barbell provides higher convexity than bullet. Barbell: 50% short-term + 50% long-term bonds. Bullet: All bonds at single maturity. Convexity advantage: Barbell gains more when rates fall (positive convexity). Example: Rates fall 100bp, Barbell +12%, Bullet +10%. Trade-off: Lower yield than bullet (pay for convexity).',
    },
    {
      id: 'fipm-mc-2',
      question: 'What does tracking error measure in portfolio management?',
      options: [
        'Standard deviation of portfolio returns vs benchmark returns',
        'Absolute returns',
        'Duration mismatch',
        'Yield difference',
      ],
      correctAnswer: 0,
      explanation:
        'Tracking error = StdDev(Portfolio - Benchmark returns). Measures consistency of outperformance/underperformance. Low TE (<1%): Index-like, High TE (>3%): Active bets. Target: <1% enhanced index, 2-4% active management.',
    },
    {
      id: 'fipm-mc-3',
      question:
        'In performance attribution analysis, what does "selection effect" measure?',
      options: [
        'Value added by picking specific bonds that outperform sector',
        'Duration positioning',
        'Sector allocation',
        'Total returns',
      ],
      correctAnswer: 0,
      explanation:
        'Selection effect: Bond-picking skill within sectors. Measures: Chosen bonds vs sector average performance. Example: Picked corporate bonds +2% vs corporate sector +1.5% = +0.5% selection alpha. Different from sector allocation (choosing which sectors).',
    },
    {
      id: 'fipm-mc-4',
      question:
        'What is the main disadvantage of passive index tracking compared to active management?',
      options: [
        'Cannot outperform benchmark (limited to market returns)',
        'Higher costs',
        'More complex',
        'Higher turnover',
      ],
      correctAnswer: 0,
      explanation:
        'Passive: Cannot outperform benchmark by design (tracks index). Active: Can outperform (security selection, timing). Trade-off: Passive low cost (0.05%), guaranteed tracking, Active high cost (0.50%), potential alpha. Investors choose based on manager skill belief.',
    },
    {
      id: 'fipm-mc-5',
      question:
        'What is a ladder strategy in fixed income portfolio management?',
      options: [
        'Equal weights across multiple maturities',
        'All short-term bonds',
        'All long-term bonds',
        'Concentrated at single maturity',
      ],
      correctAnswer: 0,
      explanation:
        'Ladder: Equal investments across maturity spectrum. Example: 10% each in 1, 2, 3...10 year bonds. Benefits: Diversification, Regular cash flows, Reinvestment flexibility. Use: Conservative investors, Cash flow matching. As bonds mature: Reinvest at long end (maintain ladder).',
    },
  ];
