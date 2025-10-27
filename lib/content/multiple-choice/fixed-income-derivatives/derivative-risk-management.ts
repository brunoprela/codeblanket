import { MultipleChoiceQuestion } from '@/lib/types';

export const derivativeRiskManagementMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'drm-mc-1',
      question: 'What does a 95% 1-day VaR of $5M mean?',
      options: [
        '95% confident loss will not exceed $5M tomorrow',
        'Expected loss is $5M',
        'Maximum possible loss is $5M',
        'Average loss over 20 days',
      ],
      correctAnswer: 0,
      explanation:
        "95% VaR: 95% probability loss ≤ $5M, 5% probability loss > $5M (1 in 20 days). Not expected loss (that's mean), not maximum (tail can be larger). Used for: Daily risk reporting, Risk limit setting, Regulatory capital.",
    },
    {
      id: 'drm-mc-2',
      question:
        'What is the main advantage of Monte Carlo simulation over parametric VaR?',
      options: [
        'Can handle non-linear instruments (options) accurately',
        'Faster computation',
        'Simpler to implement',
        'Always more accurate',
      ],
      correctAnswer: 0,
      explanation:
        'Monte Carlo: Reprices options under simulated paths (captures gamma, vega). Parametric: Linear approximation (delta only), inaccurate for options. Trade-off: MC slower but better for complex portfolios. Historical: Alternative (actual data, no assumptions).',
    },
    {
      id: 'drm-mc-3',
      question: 'What is a reverse stress test in risk management?',
      options: [
        'Start with failure outcome, work backwards to find scenario',
        'Test system under reverse scenarios',
        'Historical scenario in reverse order',
        'Opposite of forward stress test',
      ],
      correctAnswer: 0,
      explanation:
        'Reverse stress: Define failure ($100M loss, bankruptcy), Solve for scenario causing it (rates +X bp, spreads +Y bp), Identifies vulnerabilities. Example: "Lose $100M if rates +250bp AND spreads +200bp simultaneously". Regulatory: Required by Basel III.',
    },
    {
      id: 'drm-mc-4',
      question: 'What is the purpose of a DV01 limit in derivatives trading?',
      options: [
        'Limit interest rate risk exposure',
        'Limit notional size',
        'Limit credit risk',
        'Limit number of trades',
      ],
      correctAnswer: 0,
      explanation:
        'DV01: Dollar value of 1bp rate move. DV01 limit: Caps interest rate sensitivity. Example: DV01 = $1M limit, 1bp rate move = max $1M loss. Prevents: Excessive duration risk, Large rate exposure. Monitors: Total DV01 across all bond/swap positions.',
    },
    {
      id: 'drm-mc-5',
      question: 'What is model risk in derivatives valuation?',
      options: [
        'Risk that models are wrong or misused',
        'Risk of computer system failure',
        'Risk of data errors only',
        'Risk of regulatory changes',
      ],
      correctAnswer: 0,
      explanation:
        'Model risk: Models based on assumptions (normal distribution, constant volatility) that may be wrong. Examples: Black-Scholes assumes constant vol (actually varies), Correlations assumed stable (spike to 1 in crisis). Mitigation: Model validation, Backtesting, Reserves for uncertainty.',
    },
  ];
