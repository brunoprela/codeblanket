import { MultipleChoiceQuestion } from '@/lib/types';

export const automatedModelGenerationMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'amg-mc-1',
      question:
        'What is the PRIMARY advantage of automated financial modeling over manual Excel models?',
      options: [
        'Better accuracy (no human error)',
        'Scalability (value 100 companies as easily as 1)',
        'Prettier formatting',
        'Lower cost (no Excel licenses)',
      ],
      correctAnswer: 1,
      explanation:
        'Option 2 is correct: Scalability. Automation shines when: Valuing 100 companies (automated = 10 minutes, manual = 100 hours), Running 1000 Monte Carlo simulations (automated = instant, manual = impossible), Updating 50 models daily (automated = automated, manual = full-time job). Option 1 (accuracy) is false—automation can have bugs, manual can be accurate. Option 3 (formatting) is backwards—manual Excel often looks better. Option 4 (cost) is trivial—Excel costs $10/month.',
    },
    {
      id: 'amg-mc-2',
      question:
        'In automated valuation, which data source is typically MOST reliable for public company financials?',
      options: [
        'Yahoo Finance (free API)',
        'Bloomberg Terminal (paid, professional)',
        'Company website investor relations',
        'Wikipedia',
      ],
      correctAnswer: 1,
      explanation:
        'Option 2 is correct: Bloomberg Terminal. Reliability ranking: (1) Bloomberg/CapitalIQ/FactSet (paid, $20k+/year, professional-grade, verified data). (2) SEC filings direct (10-K, 10-Q—authoritative source, free). (3) Company IR website (accurate but may be delayed). (4) Yahoo Finance (free but frequent errors, especially for small caps). (5) Wikipedia (crowd-sourced, not reliable for financials). For production systems, pay for Bloomberg/CapitalIQ. For personal/learning, use SEC filings directly.',
    },
    {
      id: 'amg-mc-3',
      question:
        'You automate DCF models pulling beta from API. API returns beta = 3.5 for a utility company (typical beta = 0.5-0.8). What should the system do?',
      options: [
        'Use 3.5 (trust the API)',
        'Cap at 2.0 (reasonable maximum)',
        'Flag for manual review (outlier)',
        'Use industry average 0.6 (override)',
      ],
      correctAnswer: 2,
      explanation:
        "Option 3 is correct: Flag for manual review. Beta = 3.5 for utility is clearly wrong (utilities are low-risk, beta should be <1). System should: (1) Flag as outlier (beta > 2.0 for low-vol sector), (2) Route to analyst for manual review, (3) Provide alternative (industry avg beta = 0.6), (4) Don't automatically override—could be legitimate (e.g., distressed utility). Option 1 (trust API) blindly uses bad data. Option 2 (cap) is arbitrary. Option 4 (override) assumes error without investigation. Best: Automate detection, route for human judgment.",
    },
    {
      id: 'amg-mc-4',
      question:
        'For automated valuation platforms, which is the BEST practice for version control?',
      options: [
        'Save Excel files with v1, v2, v3 naming',
        'Email final version to team',
        'Use Git for code + database for results',
        'No version control needed (always use latest data)',
      ],
      correctAnswer: 2,
      explanation:
        "Option 3 is correct: Git for code + database for results. Version control best practices: Code (Python models): Git repository with commit history. Every change tracked: who, when, what, why. Results (valuations): Database with timestamps. Store: valuation date, inputs, outputs, model version. Reproduction: Checkout git commit from 6 months ago, rerun exact analysis. Option 1 (v1/v2/v3) doesn't scale—need proper version control. Option 2 (email) loses history, no collaboration. Option 4 (no control) can't reproduce historical analyses (regulatory/legal problem).",
    },
    {
      id: 'amg-mc-5',
      question:
        'An automated valuation platform generates PDF reports. Which is LEAST important to include?',
      options: [
        'Data sources and timestamps',
        'Key assumptions (growth rate, WACC)',
        'Sensitivity analysis (range of outcomes)',
        'Full Python source code',
      ],
      correctAnswer: 3,
      explanation:
        'Option 4 is correct: Full Python source code is least important in PDF report. Report priorities: (1) Data sources + timestamps (critical—when was data pulled? From where?). (2) Key assumptions (critical—growth, WACC, terminal—drive valuation). (3) Sensitivity analysis (critical—show range, not single point). (4) Model description (important—explain methodology). (5) Full source code (NOT needed—too technical, clutters report). Clients want insights, not code. Keep code in Git, deliver clean PDF with analysis.',
    },
  ];
