import { MultipleChoiceQuestion } from '@/lib/types';

export const sensitivityScenarioAnalysisMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'ssa-mc-1',
      question:
        'In a two-way sensitivity table for DCF valuation, which pair of variables is MOST appropriate to test together?',
      options: [
        'Revenue growth × Tax rate',
        'WACC × Terminal growth rate',
        'CapEx × Depreciation',
        'Share count × Dividend yield',
      ],
      correctAnswer: 1,
      explanation:
        'Option 2 is correct: WACC × Terminal growth. These are: (1) Both highly uncertain (market-dependent, not controllable), (2) Both have large impact on valuation (terminal value is 60-80% of EV), (3) Potentially correlated (rising rates → higher WACC and lower terminal growth). Testing together shows combined effect. Option 1 (revenue × tax rate): Tax rate is usually fixed (21% corporate rate), not worth sensitivity testing. Option 3 (CapEx × D&A): Mechanically linked (CapEx drives future D&A), not independent variables. Option 4 (shares × dividend): Irrelevant for enterprise value DCF (equity-side variables).',
    },
    {
      id: 'ssa-mc-2',
      question:
        'A tornado chart shows: Variable A has $5B range, Variable B has $3B range, Variable C has $0.5B range. You have 10 hours for additional due diligence. How should you allocate time?',
      options: [
        'Equal time (3.3 hours each) for balanced analysis',
        '7 hours on A, 2.5 hours on B, 0.5 hours on C',
        '10 hours on A only (highest impact)',
        '5 hours on B and C (need more confidence on these)',
      ],
      correctAnswer: 1,
      explanation:
        "Option 2 is correct: Allocate time proportional to impact. Variable A ($5B range) drives 60%+ of valuation uncertainty—deserves most effort (7 hours). Variable B ($3B range) deserves moderate effort (2.5 hours). Variable C ($0.5B range) is immaterial—minimal time (0.5 hours, just validate it's reasonable). Rationale: Focus where you get most risk reduction per hour spent. Deep diving Variable C from $0.5B to $0.3B range (40% improvement!) only reduces total uncertainty by $0.2B. Improving Variable A from $5B to $4B range (20% improvement) reduces uncertainty by $1B (5x more valuable). Option 1 (equal time) wastes effort on low-impact variables. Option 3 (all on A) ignores B which is also meaningful. Option 4 (focus on B/C) is backwards—spend time where impact is highest.",
    },
    {
      id: 'ssa-mc-3',
      question:
        'Which of the following is an example of PROPER scenario analysis (internally consistent assumptions)?',
      options: [
        'Bear: 5% revenue growth, 30% EBITDA margin, 12% WACC',
        'Bull: 25% revenue growth, 15% EBITDA margin, 8% WACC',
        'Base: 15% revenue growth, 20% EBITDA margin, 10% WACC',
        'Bear: 30% revenue growth, 10% EBITDA margin, 8% WACC',
      ],
      correctAnswer: 2,
      explanation:
        'Option 3 is correct: Base case with moderate assumptions across all variables (15% growth, 20% margin, 10% WACC). Internally consistent—reasonable assumptions that align. Option 1 (Bear) is INCONSISTENT: Low growth (5%) doesn\'t align with high margins (30%). In bear case, low growth usually comes with margin pressure (competition, price cuts). Should be: 5% growth, 15% margin, 12% WACC (all negative). Option 2 (Bull) is INCONSISTENT: High growth (25%) doesn\'t align with low margins (15%). High-growth companies usually have expanding margins (scale). Should be: 25% growth, 30% margin, 8% WACC (all positive). Option 4 (Bear) is COMPLETELY BACKWARDS: Called "bear" but has bull assumptions (30% growth, 8% WACC). Scenarios must have coherent stories where all variables move in same direction.',
    },
    {
      id: 'ssa-mc-4',
      question:
        'Your DCF model is most sensitive to terminal growth rate (±1% terminal growth = ±$3B valuation). What is the best way to reduce this sensitivity?',
      options: [
        'Use Monte Carlo simulation with 10,000 iterations',
        'Extend projection period from 10 to 15 years',
        'Lower the WACC to reduce discount rate',
        'Increase near-term revenue growth assumptions',
      ],
      correctAnswer: 1,
      explanation:
        "Option 2 is correct: Extend projection period to 15 years. This reduces terminal value % of total EV from ~70% (10-year projection) to ~50% (15-year projection). More valuation weight on explicit projections (knowable) vs perpetuity (speculative). Math: With 10-year projection, sum of PV(FCF years 1-10) might be $2B, PV(terminal value) $4B (67% of $6B EV). With 15-year projection, sum of PV(FCF years 1-15) = $3.5B, PV(terminal value) $2.5B (42% of $6B EV). Now ±1% terminal growth has less impact. Option 1 (Monte Carlo): Quantifies uncertainty but doesn't reduce it. Option 3 (lower WACC): Arbitrary manipulation, doesn't address root cause. Option 4 (increase growth): Changes valuation but doesn't reduce terminal value sensitivity.",
    },
    {
      id: 'ssa-mc-5',
      question:
        'You run three scenarios: Bear ($5B, 20% probability), Base ($8B, 60% probability), Bull ($12B, 20% probability). What is the probability-weighted expected value?',
      options: [
        '$8.0B (the base case)',
        '$8.4B',
        '$8.3B (average of three scenarios)',
        '$7.4B',
      ],
      correctAnswer: 1,
      explanation:
        "Option 2 is correct: $8.4B. Calculation: Expected Value = Σ(Probability × Valuation). EV = (0.20 × $5B) + (0.60 × $8B) + (0.20 × $12B) = $1B + $4.8B + $2.4B = $8.4B. Option 1 ($8B) is wrong—that's just the base case, ignoring bear and bull. Option 3 ($8.3B) is wrong—simple average ($5B + $8B + $12B) / 3 = $8.3B ignores probabilities. If bear and bull were equally likely, simple average would work. But base case has 60% weight (should pull expected value toward it). Option 4 ($7.4B) is miscalculation. Key insight: Expected value is probability-weighted, not simple average. If confident in base case (60% weight), expected value stays close to base.",
    },
  ];
