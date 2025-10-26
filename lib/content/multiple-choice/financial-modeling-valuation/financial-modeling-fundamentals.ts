import { MultipleChoiceQuestion } from '@/lib/types';

export const financialModelingFundamentalsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'fmf-mc-1',
    question:
      'In a professional financial model, which of the following is the WORST practice?',
    options: [
      'Using named ranges for key assumptions (e.g., "WACC" instead of cell reference "B15")',
      'Color-coding input cells blue and calculated cells black',
      'Including the formula "=Revenue * 0.65" to calculate Cost of Goods Sold',
      'Adding data validation dropdown lists to limit input choices',
    ],
    correctAnswer: 2,
    explanation:
      'Option 3 is the worst practice - hard-coding the percentage (0.65) directly in the formula. If the COGS margin changes, you must find and update every instance, creating error risk. Best practice: Create a named cell "COGS_Margin" with value 0.65, then use "=Revenue * COGS_Margin". This creates a single source of truth. Options 1, 2, and 4 are all good practices: named ranges improve formula readability and reduce errors, color coding distinguishes inputs from calculations visually, and data validation prevents invalid inputs (e.g., restricting growth rate to 0%-100%).',
  },
  {
    id: 'fmf-mc-2',
    question:
      'You\'re building a 10-year DCF model with quarterly projections (40 periods). The model has circular references because: cash balance affects interest income, which affects net income, which flows to cash balance. Which solution is most appropriate?',
    options: [
      'Delete the interest income calculation to break the circular reference',
      'Enable Excel\'s iterative calculation feature (File → Options → Formulas)',
      'Manually calculate and hard-code cash balance values for all 40 periods',
      'Ignore the circular reference error; Excel will handle it automatically',
    ],
    correctAnswer: 1,
    explanation:
      'Option 2 is correct - enabling iterative calculation allows Excel to solve circular references by repeatedly calculating until values converge (typically 100 iterations). This is the standard approach for models with unavoidable circular logic. Option 1 removes necessary functionality (interest income is real). Option 3 creates maintainability nightmare and defeats purpose of dynamic model - any assumption change requires recalculating all 40 periods manually. Option 4 is dangerous; unhandled circular references produce incorrect results or #VALUE! errors. Alternative solution (if iterative calculation causes performance issues): Make cash a "plug" or "balancing item" by using IF statements, though this is more complex. Professional models often have circular references in: cash/debt/interest, working capital with days-based calculations, consolidated financials with intercompany eliminations.',
  },
  {
    id: 'fmf-mc-3',
    question:
      'Which of the following validation checks would NOT typically be included in a professional financial model?',
    options: [
      'Balance sheet check: ABS(Assets - (Liabilities + Equity)) < $1',
      'Cash flow tie: Cash Flow Statement ending cash = Balance Sheet cash',
      'Revenue growth check: Growth rate between -50% and +200%',
      'Industry comparison: Company margins exactly match industry average',
    ],
    correctAnswer: 3,
    explanation:
      'Option 4 is NOT a valid validation check - company margins should not be forced to match industry average. Companies can legitimately have higher or lower margins than peers (better operations, different business mix, geography, etc.). Validation checks verify mathematical consistency and flag unreasonable assumptions, not enforce conformity. Option 1 checks fundamental accounting identity (must hold). Option 2 ensures cash flow statement ties to balance sheet (common error point). Option 3 flags suspicious growth rates (200%+ growth or 50%+ decline warrants investigation, though possible in special situations). Better approach for industry comparison: FLAG for review if margins deviate significantly (e.g., gross margin >20% above/below industry), but don\'t error out. Analyst should document rationale: "Company has 45% gross margin vs industry average 35% due to premium brand positioning and direct-to-consumer sales model."',
  },
  {
    id: 'fmf-mc-4',
    question:
      'For which use case would Python be LEAST preferable compared to Excel for financial modeling?',
    options: [
      'Running 10,000 Monte Carlo simulations to stress-test a valuation model',
      'Building a DCF model for client presentation with live assumption updates during partner meeting',
      'Automatically pulling and analyzing quarterly financials for 50 portfolio companies',
      'Performing regression analysis on 20 years of historical data to forecast revenue',
    ],
    correctAnswer: 1,
    explanation:
      'Option 2 favors Excel - live client meetings require immediate, visual updates as partners debate assumptions. Excel\'s spreadsheet interface allows real-time "what-if" analysis visible to all participants. Stakeholders can see formulas, trace calculations, and suggest changes intuitively. Python requires code edits → re-run → regenerate outputs, disrupting meeting flow. Options 1, 3, and 4 strongly favor Python: (1) Monte Carlo with 10K simulations: Python handles in seconds, Excel would take hours and likely crash. (2) Automated data pulls for 50 companies: Python can API-fetch and process in minutes; manual Excel entry takes days. (3) Historical regression: Python\'s statsmodels/sklearn provide robust statistical tools; Excel\'s regression limited and cumbersome. Best practice: Hybrid approach. Build core model in Python (calculations, simulations, automation), but export outputs to Excel dashboard for client presentation. For live meetings, create Excel front-end with key assumption inputs linked to Python backend, OR have pre-run scenarios ready in Excel. Investment banks often use Python for analysis, Excel for final client deliverable.',
  },
  {
    id: 'fmf-mc-5',
    question:
      'A financial model has the following structure: Tab 1 contains both inputs and calculations mixed together; Tab 2 has income statement with formulas referencing Tab 1; Tab 3 has balance sheet referencing Tab 2; outputs are scattered across all three tabs. What is the PRIMARY problem with this structure?',
    options: [
      'Using multiple tabs increases file size and slows down Excel performance',
      'Mixed inputs and calculations make scenario analysis difficult and error-prone',
      'The tab order (income statement before balance sheet) violates accounting standards',
      'Formulas that reference other tabs are inherently less accurate than single-tab formulas',
    ],
    correctAnswer: 1,
    explanation:
      'Option 2 is the primary problem - mixing inputs and calculations prevents efficient scenario analysis. Best practice: separate inputs (all assumptions in one clearly defined area), calculations (reference inputs), and outputs (summary of key results). When inputs are scattered, analysts must hunt through formulas to find what to change, increasing error risk (might miss one input) and time (inefficient). Professional models have dedicated "Assumptions" section/tab where all user-changeable inputs live, often color-coded blue. Option 1 is minor concern; multiple tabs don\'t significantly impact performance in modern Excel. Option 3 is incorrect; tab order doesn\'t matter for accounting standards (all statements are as of same period). Option 4 is false; cross-tab references are fine if properly structured. Benefit of separation: Scenario manager can save different input sets (Base/Bull/Bear), swap entire assumption sheets, run data tables varying specific inputs. Mixed structure requires manually finding/changing each assumption. Additional issue with described structure: scattered outputs make it hard to present results. Should have dedicated "Summary" tab with dashboard showing key metrics, valuation, sensitivities.',
  },
];

