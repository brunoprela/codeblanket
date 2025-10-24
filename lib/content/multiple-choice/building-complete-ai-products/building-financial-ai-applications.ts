import { MultipleChoiceQuestion } from '../../../types';

export const buildingFinancialAiApplicationsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'bcap-bfaa-mc-1',
      question:
        'What is the most important disclaimer for AI stock analysis applications?',
      options: [
        'No disclaimer needed',
        'Prominent "Not financial advice. For informational purposes only. Consult a licensed advisor."',
        'Small text at bottom of page',
        '"AI is always correct"',
      ],
      correctAnswer: 1,
      explanation:
        'Legal protection requires prominent disclaimers: (1) Every response includes: "This is not financial advice. For informational purposes only. Consult a licensed financial advisor.", (2) Visible before recommendations, (3) No guarantees or specific price targets, (4) No urgency ("buy now"), (5) Track record publicly (transparency about 55-60% directional accuracy). Failure to disclaim = potential liability for user losses.',
    },
    {
      id: 'bcap-bfaa-mc-2',
      question: 'How should AI portfolio rebalancing suggestions be executed?',
      options: [
        'Auto-execute all trades without approval',
        'Show proposed trades, tax impact, require user approval + 2FA, never auto-execute',
        'Execute randomly',
        'Only suggest, never calculate actual trades',
      ],
      correctAnswer: 1,
      explanation:
        'Human-in-the-loop execution: (1) Generate rebalancing plan (which trades, amounts), (2) Calculate tax impact (capital gains, wash sales), (3) Show complete plan to user with costs, (4) Require explicit approval + 2FA, (5) User executes manually or approves automation. NEVER auto-execute financial trades without approval - legal liability and user trust issue. Many jurisdictions require investment advisor license for automated trading.',
    },
    {
      id: 'bcap-bfaa-mc-3',
      question:
        'What is the SAR (Suspicious Activity Report) threshold for financial transactions?',
      options: ['$1,000', '$10,000', '$100,000', 'No threshold'],
      correctAnswer: 1,
      explanation:
        '$10,000 is the SAR threshold (US): (1) Any transaction >$10k must be auto-flagged, (2) File SAR with FinCEN within 30 days if suspicious, (3) Also flag: Multiple transactions just under $10k (structuring), unusual patterns, rapid buy/sell, (4) Failure to file = fines ($25k-100k), potential criminal charges, (5) Maintain 5+ year audit trail. This is legal requirement, not optional.',
    },
    {
      id: 'bcap-bfaa-mc-4',
      question:
        'How should different signals be weighted in AI stock analysis?',
      options: [
        'Equal weighting for all signals',
        'Ensemble: Technical (30%), Fundamentals (30%), Sentiment (20%), Insider (20%), adjust by market conditions',
        'Only use one signal',
        'Random weighting',
      ],
      correctAnswer: 1,
      explanation:
        'Multi-signal ensemble: (1) Technical indicators: 30% (RSI, MACD, moving averages), (2) Fundamentals: 30% (P/E, debt ratio, revenue growth), (3) Sentiment: 20% (news analysis, social media), (4) Insider trading: 20% (strong signal when execs buy/sell). Adjust weights by conditions: Bull market → more technical, Recession → more fundamental. Use LLM to synthesize all signals into narrative analysis with risks.',
    },
    {
      id: 'bcap-bfaa-mc-5',
      question:
        'What is realistic directional accuracy for AI stock predictions?',
      options: [
        '90-95% (nearly perfect)',
        '55-60% (better than random, but far from perfect)',
        '100% (AI is always right)',
        '10-20% (worse than random)',
      ],
      correctAnswer: 1,
      explanation:
        'Realistic expectations: 55-60% directional accuracy (predicting stock up/down). This is: (1) Better than random (50%), (2) Comparable to professional analysts, (3) NOT market-beating alpha, (4) Factors: Market efficiency, unpredictable events, signal noise. Be transparent about limitations, track record publicly, never guarantee returns. Focus on: Risk-adjusted insights, showing reasoning, helping users make informed decisions - not promising profits.',
    },
  ];
