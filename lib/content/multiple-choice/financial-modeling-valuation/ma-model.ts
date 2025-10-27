import { MultipleChoiceQuestion } from '@/lib/types';

export const maModelMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'ma-mc-1',
    question:
      'An acquirer with 30x P/E buys a target with 20x P/E at a 25% premium using 100% stock. Assuming no synergies, the deal will most likely be:',
    options: [
      'Highly accretive (>5%)',
      'Modestly accretive (1-5%)',
      'Neutral (±1%)',
      'Dilutive',
    ],
    correctAnswer: 1,
    explanation:
      'Option 2 is correct: Modestly accretive. Analysis: Acquirer P/E (30x) > Target P/E (20x) = favorable P/E arbitrage. After 25% premium, effective P/E paid = 20x × 1.25 = 25x. Still below acquirer\'s 30x = accretive. Math: Acquirer paying 25x for earnings while own stock trades at 30x = getting earnings "cheaper" than own valuation. Each $1 of target earnings costs 25x, but acquirer\'s earnings valued at 30x = creates 20% arbitrage value (30/25 - 1). However, premium reduces benefit: Without premium (20x paid): Highly accretive. With 25% premium (25x paid): Modestly accretive (1-3%). If premium were 50%+ (30x+ paid): Would become dilutive. Key insight: P/E arbitrage minus premium determines accretion. High P/E buyers have advantage but can overpay with excessive premium.',
  },
  {
    id: 'ma-mc-2',
    question:
      'Which of the following would make an M&A deal MORE accretive to the acquirer?',
    options: [
      'Paying a higher premium for the target',
      'Financing with debt instead of stock',
      'Lower cost synergies realized',
      'Faster synergy realization timeline',
    ],
    correctAnswer: 3,
    explanation:
      'Option 4 is correct: Faster synergy realization = more accretive Year 1. Explanation: Accretion/dilution typically measured in Year 1 post-close. Synergies realized faster = higher Year 1 pro forma earnings = more accretive. Example: $100M annual synergies. If realized 50% Year 1: $50M benefit to Year 1 EPS. If realized 80% Year 1 (faster): $80M benefit to Year 1 EPS (+60% vs slow case). Option 1 (higher premium) makes deal LESS accretive—paying more for same earnings. Option 2 (debt vs stock) is ambiguous: Debt avoids share dilution (good) but adds interest expense (bad). Net effect depends on interest rate vs dilution impact. Often debt is less accretive for expensive debt. Option 3 (lower synergies) makes deal LESS accretive—less earnings benefit.',
  },
  {
    id: 'ma-mc-3',
    question:
      'An acquirer (200M shares, $440M net income) issues 40M new shares to acquire a target ($80M net income). If combined company generates $520M net income, what is the pro forma EPS?',
    options: ['$2.20', '$2.17', '$2.60', '$2.00'],
    correctAnswer: 1,
    explanation:
      'Option 2 is correct: $2.17. Calculation: Pro forma shares = 200M + 40M = 240M. Pro forma net income = $520M (given). Pro forma EPS = $520M / 240M = $2.167 ≈ $2.17. Note: Standalone acquirer EPS = $440M / 200M = $2.20. Pro forma EPS $2.17 < Standalone $2.20 = DILUTIVE by $0.03 or 1.4%. Why dilutive? Share dilution: 40M / 200M = 20% more shares. Earnings growth: ($520M - $440M) / $440M = 18.2% more earnings. Share dilution (20%) > Earnings growth (18.2%) = EPS dilution. Common mistake: Option 1 ($2.20) forgets about new shares issued. Option 3 ($2.60) incorrectly divides by only acquirer shares (200M). Option 4 ($2.00) uses wrong math.',
  },
  {
    id: 'ma-mc-4',
    question:
      'In an M&A model, which of the following is typically the LARGEST source of value creation?',
    options: [
      'EPS accretion from P/E arbitrage',
      'Cost synergies from eliminating redundancies',
      'Revenue synergies from cross-selling',
      'Tax benefits from deal structure',
    ],
    correctAnswer: 1,
    explanation:
      "Option 2 is correct: Cost synergies are largest and most reliable. Breakdown: (1) Cost synergies: 60-80% of synergy value in typical deal. Eliminate duplicate functions: 2 CFOs → 1, consolidate facilities, reduce vendors, IT system consolidation. Highly achievable (90%+ realization) because directly controllable. Typical: 10-20% of target's cost base. (2) Revenue synergies: 20-40% of synergy value. Cross-sell products, geographic expansion, combined offerings. Lower achievable (50-70% realization) because dependent on customer behavior, sales execution, market acceptance. Typical: 2-5% revenue lift. (3) EPS accretion from P/E arbitrage: Not value creation—it's accounting redistribution. Creates EPS lift but doesn't generate new cash flows. (4) Tax benefits: 5-15% of synergy value. Structural optimization, NOL utilization, international tax efficiency. Small relative to operational synergies. Reality: Most deals justified by cost synergies. Revenue synergies are \"upside case.\" Tax benefits are cherry on top.",
  },
  {
    id: 'ma-mc-5',
    question:
      'A company with 15x P/E considers two acquisition targets, both with $50M earnings. Target A trades at 12x P/E ($600M market cap). Target B trades at 18x P/E ($900M market cap). Assuming 20% premium and 100% stock deals, which is more accretive?',
    options: [
      'Target A (accretion from buying below own P/E)',
      'Target B (higher quality asset)',
      'Both equally accretive',
      'Neither will be accretive due to premium',
    ],
    correctAnswer: 0,
    explanation:
      "Option 1 is correct: Target A is more accretive. Analysis: Target A: Market cap $600M (12x), 20% premium = $720M purchase price. Effective P/E paid = $720M / $50M = 14.4x. Acquirer P/E = 15x > 14.4x paid = ACCRETIVE. Logic: Buying earnings at 14.4x while own earnings valued at 15x = accretive arbitrage. Target B: Market cap $900M (18x), 20% premium = $1.08B purchase price. Effective P/E paid = $1.08B / $50M = 21.6x. Acquirer P/E = 15x < 21.6x paid = DILUTIVE. Logic: Buying earnings at 21.6x while own earnings valued at 15x = paying more than own multiple = dilutive. Rule: Compare acquirer P/E to (target P/E × (1 + premium)). If acquirer P/E > effective paid = accretive. If acquirer P/E < effective paid = dilutive. Option 2 is wrong—quality doesn't determine accretion, relative P/E does (though quality affects long-term value creation). Option 3 is wrong—A is accretive, B is dilutive. Option 4 is wrong—A is accretive despite premium.",
  },
];
