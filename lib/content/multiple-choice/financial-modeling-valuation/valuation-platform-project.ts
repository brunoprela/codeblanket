import { MultipleChoiceQuestion } from '@/lib/types';

export const valuationPlatformProjectMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'vpp-mc-1',
    question: 'In a production valuation platform, which is the MOST critical feature for user trust?',
    options: ['Fancy UI with animations', 'Transparency of assumptions and data sources', 'Fastest performance (<1 second response)', 'Most valuation methods (10+ models)'],
    correctAnswer: 1,
    explanation: 'Option 2 is correct: Transparency builds trust. Users trust platforms that show: (1) Data sources (Yahoo Finance vs Bloomberg vs SEC), (2) Timestamps (when was data fetched?), (3) Assumptions (growth rate, WACC, terminal growth—explicit), (4) Methodology (how did you calculate?), (5) Limitations (what this model doesn\'t capture). Without transparency, "black box" models are dismissed as unreliable. Option 1 (fancy UI) is nice but doesn\'t build credibility. Option 3 (speed) matters but secondary to accuracy/transparency. Option 4 (many methods) can confuse—better to do 3 methods well than 10 poorly.',
  },
  {
    id: 'vpp-mc-2',
    question: 'Your valuation platform generates PDF reports. Which section should appear FIRST?',
    options: ['Detailed DCF calculations (technical)', 'Executive summary (1-page overview)', 'Data sources and disclaimers', 'Full financial statement projections'],
    correctAnswer: 1,
    explanation: 'Option 2 is correct: Executive summary first. Report structure: (1) Executive Summary (1 page): Company overview, valuation range, recommendation, key risks. Busy executives read this ONLY. (2) Investment Thesis (2 pages): Why undervalued/overvalued, catalysts, competitive position. (3) Valuation Methods (5-7 pages): DCF, comps, transactions with details. (4) Financial Projections (3-5 pages): Revenue model, margin analysis, FCF projections. (5) Risk Factors (2 pages): Downside scenarios, sensitivity. (6) Appendix (5-10 pages): Data sources, detailed calcs, disclaimers. Lead with summary (for executives), put technical details later (for analysts).',
  },
  {
    id: 'vpp-mc-3',
    question: 'For a production valuation platform, which database is MOST appropriate for storing valuation results?',
    options: ['CSV files in folder', 'Excel spreadsheets', 'PostgreSQL (relational database)', 'Text files with JSON'],
    correctAnswer: 2,
    explanation: 'Option 3 is correct: PostgreSQL (or MySQL, SQL Server). Production needs: (1) ACID compliance (atomicity, consistency, isolation, durability—no data loss). (2) Concurrent access (multiple users read/write simultaneously). (3) Query performance (retrieve valuations by ticker, date, user). (4) Backup/restore (automated backups, point-in-time recovery). (5) Scalability (millions of valuations over years). PostgreSQL provides all. Options 1, 2, 4 (CSV, Excel, JSON files) don\'t scale, lack concurrency, prone to corruption. Use for personal projects, not production.',
  },
  {
    id: 'vpp-mc-4',
    question: 'Your valuation platform takes 45 seconds to generate a DCF. Users complain it\'s too slow. What is the BEST optimization?',
    options: ['Cache API data (don\'t fetch every time)', 'Use faster programming language (C++ instead of Python)', 'Simplify model (fewer projection years)', 'Buy faster servers'],
    correctAnswer: 0,
    explanation: 'Option 1 is correct: Cache API data. Performance bottleneck analysis: API data fetch: 30 seconds (network latency), Calculations: 10 seconds (DCF, Monte Carlo), PDF generation: 5 seconds. Optimization: Cache API data for 24 hours (unless user forces refresh). Reduces 45s to 15s (3x faster!). Other optimizations: Async/parallel processing (fetch multiple tickers simultaneously), Pre-calculate common results (S&P 500 valuations run nightly), Use CDN for static assets. Option 2 (C++) is overkill—Python is fast enough for finance calcs. Option 3 (simplify) reduces quality. Option 4 (faster servers) addresses symptom not root cause (API latency).',
  },
  {
    id: 'vpp-mc-5',
    question: 'Which feature is LEAST important for MVP (Minimum Viable Product) valuation platform?',
    options: ['DCF valuation (core feature)', 'Trading comps analysis', 'AI-generated investment thesis', 'PDF report export'],
    correctAnswer: 2,
    explanation: 'Option 3 is correct: AI-generated thesis is least important for MVP. MVP priorities: (1) DCF valuation (MUST have—core feature). (2) Trading comps (MUST have—cross-validation). (3) PDF export (MUST have—deliverable for clients). (4) Basic UI (MUST have—usability). (5) Data validation (MUST have—reliability). Nice-to-haves for v2: AI thesis generation, Portfolio tracking, Screening tools, Alerts, Collaboration features. Build MVP fast (4-8 weeks), get user feedback, iterate. Don\'t over-engineer v1 with AI before core valuation works perfectly.',
  },
];
