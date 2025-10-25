export const quizQuestions = [
  {
    id: '2-3-q1',
    question: 'A quantitative hedge fund needs superior factor models, cleaner historical data, and strong Excel integration. They have a $15K/year budget per analyst. Which platform is most suitable?',
    options: [
      'Bloomberg Terminal - industry standard',
      'FactSet - best for quantitative research',
      'S&P Capital IQ - best for fundamentals',
      'Refinitiv Eikon - best news coverage',
      'YCharts - affordable alternative'
    ],
    correctAnswer: 1,
    explanation: 'FactSet is purpose-built for quantitative research with superior factor models, better historical data consistency, point-in-time data, and the most powerful Excel integration for systematic strategies. At $12-18K/year, it fits the budget and is specifically designed for this use case. Bloomberg is more expensive ($24K) and better for trading desks. Capital IQ excels at M&A/comps but lacks quant tools. YCharts doesn\'t have the depth needed for professional quant research.'
  },
  {
    id: '2-3-q2',
    question: 'An investment bank needs to research private company acquisition targets, build comparable company analyses, and create M&A pitch books. Which platform combination is ideal?',
    options: [
      'Bloomberg Terminal only',
      'FactSet + Morningstar Direct',
      'S&P Capital IQ + PitchBook',
      'YCharts + Koyfin Premium',
      'Refinitiv Eikon + Bloomberg'
    ],
    correctAnswer: 2,
    explanation: 'S&P Capital IQ + PitchBook is the ideal combination for M&A work. Capital IQ excels at public company comps, has excellent Excel templates for M&A models, and strong fundamental data. PitchBook provides comprehensive private company data including funding history, valuations, and M&A transactions. Together they cover both public comps and private target research. Bloomberg is useful but not optimized for M&A workflow. FactSet is better for quant research. YCharts/Koyfin lack the depth and private company data needed.'
  },
  {
    id: '2-3-q3',
    question: 'A startup quant fund with $100M AUM has only $6,000/year budget for data platforms. They need equity screening, portfolio analytics, and Python integration. What\'s the best approach?',
    options: [
      'Get one Bloomberg Terminal and share',
      'Subscribe to FactSet at minimum package',
      'YCharts Professional ($4,800) + Polygon.io API ($1,200) + free sources',
      'Koyfin Premium + TradingView Pro',
      'Use only free sources (yfinance, FRED) and build everything in Python'
    ],
    correctAnswer: 2,
    explanation: 'YCharts Professional ($4,800/year) + Polygon.io for real-time data ($1,200/year) + free sources provides comprehensive coverage within budget. YCharts offers professional screening, charting, and portfolio analytics. Polygon.io provides real-time market data via API for Python integration. Free sources (yfinance, FRED) supplement for historical data. This combo provides 80% of enterprise platform functionality at 25% of the cost. Bloomberg/FactSet are 2-4x over budget. Free sources alone lack professional screening and analytics needed for client-facing fund.'
  },
  {
    id: '2-3-q4',
    question: 'Which statement BEST describes the key differentiation between Bloomberg Terminal and its main competitors (FactSet, Refinitiv, Capital IQ)?',
    options: [
      'Bloomberg has more accurate data than competitors',
      'Bloomberg has the communication network (Messenger) creating network effects',
      'Bloomberg has better analytics and quantitative tools',
      'Bloomberg covers more markets and asset classes',
      'Bloomberg has superior customer support'
    ],
    correctAnswer: 1,
    explanation: 'Bloomberg\'s primary moat is its communication network (Bloomberg Messenger). With 325,000+ professionals using it, leaving Bloomberg means losing instant access to your entire professional network. The data quality across platforms is comparable. FactSet actually has better quantitative tools. Coverage is similar across major platforms. The network effect is Bloomberg\'s unbreakable competitive advantage - you can get equivalent data elsewhere, but you can\'t message the entire finance industry without Bloomberg.'
  },
  {
    id: '2-3-q5',
    question: 'A Python developer is building a multi-source data aggregation system. When should they prioritize premium platforms (Bloomberg/FactSet API) over free sources (yfinance)?',
    options: [
      'Always use premium platforms for better accuracy',
      'Use free sources for backtesting, premium for live trading and real-time requirements',
      'Free sources are sufficient for all use cases',
      'Only use premium if you need international data',
      'Premium platforms are required for regulatory compliance'
    ],
    correctAnswer: 1,
    explanation: 'Free sources (yfinance, FRED) are excellent for backtesting, research, and historical analysis - they have sufficient quality and coverage. Premium platforms become necessary for: (1) Real-time data feeds (sub-second latency), (2) Live trading infrastructure, (3) Complex derivatives pricing, (4) Fixed income analytics, (5) Enterprise-grade SLAs. The data quality from free sources is generally adequate; the premium is for speed, reliability, and specialized features. Start with free, upgrade to premium only when you hit clear limitations.'
  }
];

