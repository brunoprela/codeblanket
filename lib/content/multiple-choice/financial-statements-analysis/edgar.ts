export const edgarMultipleChoiceQuestions = [
  {
    id: 1,
    question:
      'Which SEC filing would you monitor to detect a sudden management change at a public company?',
    options: [
      '10-K annual report',
      '10-Q quarterly report',
      '8-K current report (Item 5.02 - Departure of Directors or Certain Officers)',
      'DEF 14A proxy statement',
      'Form 4 insider transaction',
    ],
    correctAnswer: 2,
    explanation: `The correct answer is C: 8-K current report (Item 5.02). Companies must file an 8-K within 4 business days of material events. Item 5.02 specifically covers departure of directors or principal officers (CEO, CFO, etc.). This is filed immediately when it happens, unlike 10-K/Q which are periodic. Management changes, especially CFO/CEO, can be red flags for accounting problems or strategic issues, so monitoring 8-Ks is critical for real-time risk detection.`,
  },

  {
    id: 2,
    question:
      "You're parsing XBRL data and find a company uses the tag 'AdjustedRevenue' (custom extension) instead of standard 'Revenues' tag. What does this indicate?",
    options: [
      'The company has better data quality standards',
      'XBRL parsing error that should be ignored',
      'Potential red flag: company is using custom tag to report non-standard revenue definition, reducing comparability with peers and enabling potential manipulation',
      'Normal practice for all companies',
      'Required by SEC regulations',
    ],
    correctAnswer: 2,
    explanation: `The correct answer is C. While companies can create custom XBRL extensions, using a custom revenue tag instead of the standard 'Revenues' tag is concerning because: (1) Standard tags exist specifically for comparability, (2) Custom definitions allow company to report revenue differently than GAAP, (3) Reduces ability to compare with peers, (4) May indicate aggressive revenue recognition. Excessive custom tags (>15% of all tags) often correlates with lower earnings quality. Analysts should investigate what "Adjusted" means and why standard tags aren't used.`,
  },

  {
    id: 3,
    question:
      "When analyzing MD&A section using NLP, you notice the current 10-K has a Fog Index of 22 vs prior year's 18. What does this suggest?",
    options: [
      'The company improved disclosure quality',
      'Management is being more transparent',
      'Increased complexity; company may be obfuscating problems or hiding issues in dense prose - warrants deeper investigation',
      'Normal year-over-year variation with no significance',
      'The MD&A is easier to read this year',
    ],
    correctAnswer: 2,
    explanation: `The correct answer is C. Fog Index measures readability (higher = more complex). An increase from 18 to 22 represents significant added complexity. Research shows companies increase document complexity when: (1) Hiding poor performance, (2) Facing litigation/investigations, (3) Implementing aggressive accounting. This is called "obfuscation hypothesis" - deliberately making disclosures harder to understand. Combined with other red flags (declining metrics, qualified auditor opinion), this warrants investigation. Note: Fog Index >18 is considered "unreadable" by most standards.`,
  },

  {
    id: 4,
    question:
      "A company's XBRL balance sheet shows: Assets = $500M, Liabilities = $300M, Equity = $190M. What should you do?",
    options: [
      'Accept the data as filed with SEC',
      'Calculate ratios using these numbers',
      'Flag as data integrity issue: Assets ($500M) ≠ Liabilities + Equity ($490M); investigate before using data',
      'Ignore the $10M difference as immaterial',
      "Assume it's a rounding difference",
    ],
    correctAnswer: 2,
    explanation: `The correct answer is C. The accounting equation Assets = Liabilities + Equity must ALWAYS balance. $500M ≠ $490M indicates either: (1) XBRL parsing error, (2) Missing data element, (3) Company filing error, (4) Data provider error. This $10M difference (2% of assets) is material. Before using this data: (1) Check source HTML/PDF filing, (2) Verify all balance sheet line items were captured, (3) Look for minority interest or other equity components, (4) Contact data provider if persistent. Using unbalanced data will produce incorrect ratios and analysis.`,
  },

  {
    id: 5,
    question:
      "You're building an automated system to monitor Form 4 insider transactions. Which pattern is most concerning?",
    options: [
      'CEO sells $1M of stock through pre-arranged 10b5-1 plan',
      'CFO and CEO both sell large amounts ($5M+) within same week, shortly after 10-Q filing',
      'Multiple executives exercise options and immediately sell',
      'Director purchases $100K of stock on open market',
      'CEO sells small amount for tax withholding on vested RSUs',
    ],
    correctAnswer: 1,
    explanation: `The correct answer is B. Multiple senior executives (CFO + CEO) selling large amounts simultaneously, especially right after a filing, is highly concerning because: (1) Coordinated selling suggests shared negative information, (2) Post-filing timing may indicate issues not yet public, (3) $5M+ is material for most executives, (4) CFO involvement particularly concerning (financial knowledge). Options A, C, E are normal/routine. Option D is bullish (director buying). This pattern warrants: (1) Deep dive into recent 10-Q, (2) Check MD&A for risks, (3) Analyze if metrics are deteriorating, (4) Monitor next quarter closely for negative surprises.`,
  },
];
