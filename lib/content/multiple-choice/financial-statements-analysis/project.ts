export const projectMultipleChoiceQuestions = [
  {
    id: 1,
    question:
      'For the capstone project, which architecture pattern is best for processing 500 companies daily?',
    options: [
      'Single-threaded sequential processing',
      'Multi-threaded parallel processing with task queue; allows concurrent processing of multiple companies while handling failures gracefully',
      'Manual processing as needed',
      'Batch process weekly instead of daily',
      'Cloud functions for each company',
    ],
    correctAnswer: 1,
    explanation: `Correct answer is B. Multi-threaded/multiprocessing with task queue (Celery + RabbitMQ or Redis) provides: (1) **Concurrency**: Process 10-20 companies simultaneously, reducing 8-hour sequential job to <1 hour, (2) **Fault tolerance**: If one company fails, others continue, (3) **Scalability**: Add workers as needed, (4) **Monitoring**: Track progress, retry failures, (5) **Priority queuing**: Process time-sensitive companies first. Architecture: FastAPI workers pull from Redis queue, process company, write to PostgreSQL. This is production-grade pattern used by hedge funds and fintech companies.`,
  },

  {
    id: 2,
    question:
      'Your capstone dashboard should prioritize which metric as primary health indicator?',
    options: [
      'Stock price',
      'Revenue growth only',
      'Composite score combining earnings quality (CFO/NI), credit risk (Interest Coverage), and profitability (ROE); single metric misses full picture',
      'P/E ratio',
      'Market cap',
    ],
    correctAnswer: 2,
    explanation: `Correct answer is C. Composite score provides holistic view: (1) **Earnings Quality** (CFO/NI >1.0): Ensures profits are real cash, (2) **Credit Risk** (Interest Coverage >5x): Company can service debt, (3) **Profitability** (ROE >15%): Efficient capital use. Example scoring: Score = (CFO/NI × 20) + (Interest Coverage × 5) + (ROE × 100). Companies scoring >80 are "healthy", <50 are "distressed". Single metrics mislead: High ROE with low CFO/NI = accounting games, High revenue with low profitability = unprofitable growth. Composite prevents false positives.`,
  },

  {
    id: 3,
    question:
      "How should the capstone project handle SEC filing delays (company hasn't filed 10-Q yet)?",
    options: [
      'Wait indefinitely for filing',
      'Use stale data and ignore delay',
      "Flag as 'Data Pending' in dashboard, show last available data with timestamp, set alert if filing is >5 days late (potential red flag)",
      'Delete company from system',
      'Estimate missing data',
    ],
    correctAnswer: 2,
    explanation: `Correct answer is C. Professional approach: (1) **Display last available data** with clear "As of Q2 2024" timestamp, (2) **Flag delays**: "10-Q due Oct 15, not filed yet" - late filings can signal problems (accounting issues, auditor disputes), (3) **Alert thresholds**: >5 days late = warning, >10 days = critical (may indicate serious issues), (4) **Historical context**: Check if company has history of late filings. Real example: Luckin Coffee delayed filings before fraud revelation. Never estimate missing data or use stale data without clear indicators - misleads users and creates liability.`,
  },

  {
    id: 4,
    question:
      "What\'s the best way to handle companies that restate earnings in your capstone project?",
    options: [
      'Delete old data and use restated only',
      'Keep only original reported data',
      'Temporal versioning: Store both as-reported and restated data with version numbers; allows analyzing both original reports and corrected figures',
      'Ignore restatements',
      'Average the two values',
    ],
    correctAnswer: 2,
    explanation: `Correct answer is C. Temporal versioning schema: (1) **version_id** column tracking each version, (2) **is_restated** boolean flag, (3) **valid_from / valid_to** dates, (4) **restatement_reason** text field. Benefits: (1) Research management credibility (how often do they restate?), (2) Backtest on as-reported data (avoid look-ahead bias), (3) Compare original vs restated (size of restatements indicates quality), (4) Regulatory compliance (auditors want original filings). Query examples: "Show as reported in 2020" vs "Show latest restated values". Professional systems maintain full audit trail - never delete historical data.`,
  },

  {
    id: 5,
    question:
      "For the capstone project presentation, what's the most impressive feature to demonstrate?",
    options: [
      'Basic data download functionality',
      'Simple charts of revenue',
      'Live red flag detection: Upload any ticker, system automatically runs Beneish M-Score, Altman Z-Score, channel stuffing checks, generates instant risk report with actionable alerts',
      'Company information display',
      'Database schema documentation',
    ],
    correctAnswer: 2,
    explanation: `Correct answer is C. Live automated red flag detection demonstrates: (1) **Integration**: Combines multiple modules (data extraction, ratio calculation, fraud models, credit analysis), (2) **Sophistication**: Implements advanced techniques (Beneish, Altman, accruals analysis), (3) **Practical value**: Produces actionable insights ("Avoid - High M-Score indicates manipulation risk"), (4) **Real-time**: Works on any ticker instantly. Demo scenario: Type "XYZ", system fetches latest 10-K, calculates 50+ metrics, runs 5 fraud models, compares to peers, generates "Investment Recommendation: SELL - 3 red flags detected" with detailed explanations. This showcases technical skills + financial domain expertise employers seek.`,
  },
];
