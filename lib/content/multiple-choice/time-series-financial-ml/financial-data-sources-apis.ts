import { MultipleChoiceQuestion } from '@/lib/types';

export const financialDataSourcesAPIsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'fdsa-mc-1',
      question: 'What is the main limitation of yfinance for live trading?',
      options: [
        'It costs too much',
        'It has a 15-minute delay and occasional data gaps',
        'It only supports US stocks',
        'It requires complex authentication',
      ],
      correctAnswer: 1,
      explanation:
        "yfinance has a 15-minute delay (unusable for intraday trading) and occasional data gaps/errors, especially for volume and weekend data. It\'s free and great for backtesting/learning, but not reliable for live trading where you need real-time data and guaranteed uptime. For paper trading or live trading, use Polygon.io or broker APIs with <1 second latency.",
    },
    {
      id: 'fdsa-mc-2',
      question:
        'For a daily trading strategy with $5,000 capital, which data source is most cost-effective?',
      options: [
        'Bloomberg Terminal ($2000/month)',
        'yfinance (free)',
        'Polygon.io Professional ($499/month)',
        'Reuters ($1500/month)',
      ],
      correctAnswer: 1,
      explanation:
        "For daily strategies with small capital ($5k), yfinance is sufficient. Daily strategies don't need real-time data or millisecond latency. The improved data quality from Polygon ($99-499/month) won't generate enough additional profit to justify the cost with small capital. Upgrade to paid data when capital >$10k and strategy is consistently profitable. Daily return improvement from better data is typically 2-5%.",
    },
    {
      id: 'fdsa-mc-3',
      question:
        'Why use TimescaleDB instead of regular PostgreSQL for storing OHLCV data?',
      options: [
        'TimescaleDB is free, PostgreSQL costs money',
        'TimescaleDB is optimized for time series with automatic partitioning and faster range queries',
        'TimescaleDB supports more data types',
        'PostgreSQL cannot store financial data',
      ],
      correctAnswer: 1,
      explanation:
        'TimescaleDB extends PostgreSQL with time series optimizations: automatic partitioning by time (chunks), faster range queries (SELECT * WHERE timestamp > X), continuous aggregates (pre-computed rollups), compression (75% space savings). For OHLCV queries like "get all data for 2023", TimescaleDB is 10-100× faster than regular PostgreSQL. Both are free and open-source.',
    },
    {
      id: 'fdsa-mc-4',
      question:
        'What is the typical API rate limit for Alpha Vantage free tier?',
      options: [
        'Unlimited calls',
        '100 calls per minute',
        '5 calls per minute',
        '1 call per second',
      ],
      correctAnswer: 2,
      explanation:
        'Alpha Vantage free tier: 5 API calls per minute (12 seconds between calls), 500 calls per day. This limits how fast you can download historical data (100 stocks = 20 minutes). Paid tier: 500 calls/minute. For production systems, this rate limiting is problematic. Polygon.io unlimited plan better for bulk downloads. Use rate-limited downloads with time.sleep(12) between calls.',
    },
    {
      id: 'fdsa-mc-5',
      question:
        'When storing OHLCV data, which validation check is most critical?',
      options: [
        'Check that all prices are above $1',
        'Check that High >= max(Open, Close, Low) and Low <= min(Open, Close, High)',
        'Check that volume is exactly equal to average',
        'Check that Close equals Open',
      ],
      correctAnswer: 1,
      explanation:
        'OHLC relationships MUST hold by definition: High must be >= all other prices (Open, Close, Low), Low must be <= all other prices. Violations indicate data errors. Common causes: Incorrect split adjustments, API bugs, exchange reporting errors. Example valid: Open=100, High=105, Low=98, Close=102. Invalid: High=100, Low=105 (impossible). Always validate OHLC before trading on data.',
    },
  ];
