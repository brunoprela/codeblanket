import { MultipleChoiceQuestion } from '@/lib/types';

export const leveragedBuyoutsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'leveraged-buyouts-mc-1',
    question:
      'In an LBO, if a PE firm invests $300M equity and exits 5 years later with $900M equity value, what is the MOIC and approximate IRR?',
    options: [
      'MOIC: 3.0×, IRR: ~25%',
      'MOIC: 3.0×, IRR: ~60%',
      'MOIC: 2.5×, IRR: ~20%',
      'MOIC: 3.5×, IRR: ~30%',
    ],
    correctAnswer: 0,
    explanation:
      'MOIC (Multiple of Invested Capital) = Exit value / Initial investment = $900M / $300M = 3.0×. For IRR: $300M × (1 + IRR)^5 = $900M. (1 + IRR)^5 = 3.0. 1 + IRR = 3.0^0.2 = 1.246. IRR = 24.6% ≈ 25%. Rule of thumb: 3× in 5 years ≈ 25% IRR. MOIC measures total return (money out / money in). IRR measures annualized return (time-adjusted).',
  },
  {
    id: 'leveraged-buyouts-mc-2',
    question:
      'Which of the following is NOT a typical source of value creation in an LBO?',
    options: [
      'EBITDA growth through operational improvements',
      'Multiple expansion from market timing',
      'Deleveraging through debt paydown',
      'Synergies from horizontal integration',
    ],
    correctAnswer: 3,
    explanation:
      'The three sources of LBO value creation are: (1) EBITDA Growth: Operational improvements, revenue growth, cost cutting. (2) Multiple Expansion: Buy at 8×, sell at 10× EBITDA. (3) Deleveraging: Pay down debt using cash flow, converting debt to equity value. Synergies from horizontal integration are NOT typical in LBOs because PE firms usually buy standalone companies, not integrate them with other portfolio companies. Synergies are the domain of strategic/corporate acquirers (M&A), not financial buyers (LBOs). LBOs are financial engineering, not strategic consolidation.',
  },
  {
    id: 'leveraged-buyouts-mc-3',
    question:
      'An ideal LBO candidate typically has all of the following characteristics EXCEPT:',
    options: [
      'Stable, predictable cash flows',
      'High ongoing CapEx requirements',
      'Strong EBITDA margins (15%+)',
      'Defensible market position',
    ],
    correctAnswer: 1,
    explanation:
      'Ideal LBO candidates have: Stable cash flows (can service debt), Low CapEx (maximizes FCF for debt paydown), Strong margins (profitability), Defensible position (competitive moat). HIGH ongoing CapEx is a red flag for LBOs because: Reduces FCF available for debt paydown. Increases risk (must keep investing to compete). Capital-intensive businesses are poor LBO targets. Examples of good targets: Software (low CapEx, recurring revenue). Consumer products (branded, stable). Examples of bad targets: Manufacturing (high CapEx). Semiconductors (constant R&D/CapEx).',
  },
  {
    id: 'leveraged-buyouts-mc-4',
    question:
      'PE firm buys company for $1B at 10× EBITDA (60% debt, 40% equity). At exit, EBITDA is $120M, exit multiple is 10×, debt is $300M. What is the equity value at exit?',
    options: ['$900M', '$1,200M', '$800M', '$1,500M'],
    correctAnswer: 0,
    explanation:
      'Exit enterprise value = Exit EBITDA × Exit multiple = $120M × 10 = $1,200M. Exit equity value = Exit EV - Exit debt = $1,200M - $300M = $900M. The equity increased from $400M (initial) to $900M (exit), a 2.25× MOIC. Value creation came from: EBITDA growth: $100M → $120M (+$20M → +$200M EV at 10×). Deleveraging: $600M → $300M (debt paydown of $300M converted to equity). No multiple expansion (10× entry, 10× exit).',
  },
  {
    id: 'leveraged-buyouts-mc-5',
    question:
      'What is the primary reason LBOs use high leverage (60-70% debt)?',
    options: [
      "To reduce the company's tax burden through interest deductions",
      'To amplify equity returns by using less equity capital',
      'To signal confidence to management',
      'To qualify for government subsidies',
    ],
    correctAnswer: 1,
    explanation:
      "LBOs use high leverage primarily to amplify equity returns. Example: Buy for $100M with 100% equity → Sell for $150M → 50% return. Buy for $100M with 40% equity (\$40M) + 60% debt (\$60M) → Sell for $150M, debt paid to $20M → Equity = $130M → (\$130M - $40M) / $40M = 225% return! Leverage amplifies returns when business performs well (but also amplifies losses if it underperforms). Tax shield (interest deductibility) is a secondary benefit, not the primary driver. The primary driver is financial engineering—use OPM (other people's money) to boost equity IRR.",
  },
];
