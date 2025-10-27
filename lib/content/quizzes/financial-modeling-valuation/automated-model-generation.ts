export const automatedModelGenerationQuiz = [
  {
    id: 'amg-q-1',
    question:
      'You automate DCF models using Python pulling data from APIs. A colleague says "Manual Excel is better—you can see formulas, audit easily, and customize." Defend automation. What are advantages and when is manual Excel preferred?',
    sampleAnswer:
      'Automation vs manual trade-offs: Automation advantages: (1) Speed: Generate 100 DCFs in minutes vs 100 hours manual. (2) Consistency: Same structure, no formula errors. (3) Scalability: Screen entire S&P 500 automatically. (4) Version control: Git tracks every change. (5) Reproducibility: Rerun exact analysis months later. (6) Integration: Connect to live data APIs. Manual Excel advantages: (1) Transparency: See every formula, audit trail. (2) Flexibility: Custom adjustments per company. (3) Presentation: Client-ready formatting. (4) No coding required: Analysts without programming can build. Best practice: Automate data processing, projections, calculations. Export to Excel for final review, customization, and client delivery. Hybrid approach captures both benefits.',
    keyPoints: [
      'Automation: Speed (100 models in minutes), consistency (no formula errors), scalability (screen 500+ companies)',
      'Manual Excel: Transparency (visible formulas), flexibility (custom adjustments), presentation quality',
      'Best practice: Automate calculations + data, export to Excel for final review and client delivery (hybrid)',
    ],
  },
  {
    id: 'amg-q-2',
    question:
      'You build automated valuation platform pulling real-time data from APIs. Data is sometimes wrong (e.g., Yahoo Finance shows negative EBITDA when company is profitable). How do you handle data quality?',
    sampleAnswer:
      "Data quality framework: (1) Validation checks: Flag negative EBITDA if revenue > $1B (likely error). Flag margins >50% (investigate). (2) Multiple sources: Pull from Bloomberg, CapitalIQ, FactSet—triangulate if diverge. (3) Manual overrides: Allow analysts to override API data with verified values. (4) Audit trails: Log all data sources and changes. (5) Alerts: Email analyst if material data change (>20% quarter-over-quarter). (6) Fallback: Use LTM average if current period is anomalous. Best practice: Automate but don't blindly trust APIs. Build validation layer + manual review.",
    keyPoints: [
      'Data quality issues common in APIs; implement validation (negative EBITDA flags, margin reasonableness checks)',
      'Multiple data sources (Bloomberg, CapitalIQ, FactSet); triangulate if sources diverge >10%',
      "Allow manual overrides with audit trail; automate but don't blindly trust API data",
    ],
  },
  {
    id: 'amg-q-3',
    question:
      'Your automated platform values 500 companies daily. How do you identify which valuations to trust vs which need manual review?',
    sampleAnswer:
      'Automated quality scoring: Assign confidence score (0-100) based on: (1) Data completeness (100 if all fields populated, 0 if missing revenue). (2) Model fit (100 if R² > 0.8 in projections, lower if poor fit). (3) Valuation reasonableness (flag if 3x current price or <0.3x). (4) Consistency (flag if swings >50% day-over-day). Auto-route: Score >80: Auto-approve, publish. Score 60-80: Flag for junior analyst review. Score <60: Flag for senior analyst deep dive. Example: Score = (Data complete)×0.3 + (Model R²)×0.3 + (Valuation reasonable)×0.2 + (Consistency)×0.2. Only manually review 20% flagged, automate 80%.',
    keyPoints: [
      'Confidence scoring: Data completeness + model fit + valuation reasonableness + consistency = 0-100 score',
      'Auto-route: >80 score = approve, 60-80 = junior review, <60 = senior deep dive (review 20%, automate 80%)',
      'Flags: 3x current price (overvalued?), <0.3x (undervalued?), >50% day-over-day change (data error?)',
    ],
  },
];
