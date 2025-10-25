export const peersMultipleChoiceQuestions = [
  {
    id: 1,
    question:
      'Peers trade at median EV/EBITDA of 12x. Target has EBITDA of $500M and enterprise value of $5B. Is target cheap or expensive?',
    options: [
      'Cheap - trading below fair value',
      'Expensive - trading above fair value',
      'Fairly valued; 10x EV/EBITDA (\$5B / $500M) vs 12x median suggests target is 17% cheaper than peers',
      'Cannot determine',
      'Needs more information',
    ],
    correctAnswer: 2,
    explanation: `Correct answer is C. Target trades at 10x EV/EBITDA (\$5B / $500M = 10x) while peers trade at median 12x. Target is CHEAPER by 17% ((12-10)/12 = 17%). If target should trade at peer median, fair EV = $500M × 12 = $6B. Current $5B implies 17% upside to fair value. This suggests potential buying opportunity if target's fundamentals (growth, margins, leverage) are similar to peers.`,
  },

  {
    id: 2,
    question:
      'Company has P/E of 30 and 30% earnings growth. What is PEG ratio?',
    options: [
      '0.5 - Undervalued',
      '1.0 - Fairly valued; PEG = P/E / Growth = 30 / 30 = 1.0 suggests valuation matches growth rate',
      '2.0 - Overvalued',
      '30 - Very expensive',
      'Cannot calculate',
    ],
    correctAnswer: 1,
    explanation: `Correct answer is B. PEG = P/E / (Growth × 100) = 30 / 30 = 1.0. PEG of 1.0 is "fairly valued" - the P/E multiple matches growth rate. Rule of thumb: PEG <1.0 = undervalued, PEG ~1.0 = fair, PEG >1.0 = overvalued. This 30x P/E might seem expensive in isolation, but 30% growth justifies it. Compare to low-growth company at 15x P/E with 5% growth (PEG = 3.0) - that's actually MORE expensive despite lower P/E.`,
  },

  {
    id: 3,
    question:
      'Why use EV/EBITDA instead of P/E for comparing companies with different capital structures?',
    options: [
      'P/E is always better',
      "EV/EBITDA adjusts for debt differences; EV includes debt so high-leverage companies aren't artificially cheaper, and EBITDA is pre-interest",
      "They're interchangeable",
      'EV/EBITDA is outdated',
      'P/E accounts for debt',
    ],
    correctAnswer: 1,
    explanation: `Correct answer is B. EV/EBITDA is capital structure neutral: (1) EV = Market Cap + Debt - Cash, so includes all investors, (2) EBITDA is before interest, unaffected by debt levels. Example: Company A (no debt) and Company B (high debt) can have same operations but different P/E because interest reduces B's earnings. EV/EBITDA compares apples-to-apples. Use P/E for equity investors, EV/EBITDA for enterprise value comparisons. This is why leveraged buyout firms use EV/EBITDA - they'll restructure capital anyway.`,
  },

  {
    id: 4,
    question:
      'Median peer P/E is 20x but range is 10x to 40x. How should you use this for valuation?',
    options: [
      'Use mean instead of median',
      'Use 20x median but investigate why range is wide; understand if high/low multiples justified by growth, margins, or quality differences',
      'Ignore outliers and recalculate',
      'Wide range makes peer comparison useless',
      'Average all multiples',
    ],
    correctAnswer: 1,
    explanation: `Correct answer is B. Wide range (10-40x) suggests peers aren't truly comparable or have vastly different characteristics. Actions: (1) Investigate what drives 40x vs 10x companies - likely growth/quality differences, (2) Adjust for differences (high-growth companies deserve premium multiples), (3) Consider sub-groups (fast-growth vs mature peers), (4) Use regression to model multiple as function of growth/margins. Blindly using 20x median without understanding dispersion leads to mispricing. The 40x company might be cheap if growing 50% vs 10x company declining.`,
  },

  {
    id: 5,
    question:
      'All sector peers trade at 15x P/E. Target also trades at 15x. Is target fairly valued?',
    options: [
      'Yes - matches peer median',
      'No - need absolute valuation too',
      'Maybe; relative to peers yes, but entire sector could be over/undervalued; combine with DCF, historical P/E, and FCF yield for complete picture',
      'Definitely overvalued',
      'Definitely undervalued',
    ],
    correctAnswer: 2,
    explanation: `Correct answer is C. Relative valuation shows target is in-line with peers (15x each), but doesn't confirm absolute fair value. Entire sector could be: (1) In bubble (all expensive together - tech in 2000), (2) Undervalued (all cheap - banks in 2009), (3) Fairly valued. To determine: (1) Compare 15x to historical sector average (if historically 10x, sector is expensive), (2) Run DCF for intrinsic value, (3) Check FCF yield (1/15 = 6.7% yield vs alternatives), (4) Consider macro factors. Relative valuation alone is insufficient - target is "expensive as peers" but are peers expensive?`,
  },
];
