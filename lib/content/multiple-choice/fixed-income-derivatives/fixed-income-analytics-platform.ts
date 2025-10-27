import { MultipleChoiceQuestion } from '@/lib/types';

export const fixedIncomeAnalyticsPlatformMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'fiap-mc-1',
      question:
        'What is the main advantage of using microservices architecture for a fixed income platform?',
      options: [
        'Independent scaling and deployment of services',
        'Simpler codebase',
        'Faster development',
        'Lower infrastructure costs',
      ],
      correctAnswer: 0,
      explanation:
        'Microservices: Scale pricing service independently (high CPU) vs market data service (high I/O). Deploy risk service without affecting pricing. Technology flexibility (Python pricing, Go market-data). Trade-offs: Complexity (service discovery, monitoring), Network latency (inter-service calls). Best for: Large platforms with high scale.',
    },
    {
      id: 'fiap-mc-2',
      question:
        'Why is caching critical for a fixed income analytics platform?',
      options: [
        'Reduce expensive recalculations and API calls',
        'Reduce database size',
        'Simplify code',
        'Improve security',
      ],
      correctAnswer: 0,
      explanation:
        'Caching: Market data (avoid Bloomberg API calls, $$$), Pricing results (avoid recalculating same bond 100× per second), Implementation: Redis (in-memory, <1ms), TTL (5min for market data, 1min for prices), Invalidation: On market data update, expire related caches. Performance: 10× faster response times (10ms → 1ms).',
    },
    {
      id: 'fiap-mc-3',
      question:
        'What is the purpose of WebSocket in a real-time fixed income dashboard?',
      options: [
        'Bidirectional streaming of real-time updates (trades, risk metrics)',
        'Authentication only',
        'File uploads',
        'Static data delivery',
      ],
      correctAnswer: 0,
      explanation:
        'WebSocket: Persistent connection, server pushes updates to client (no polling). Use cases: Real-time Greeks updates (on every trade), Risk limit alerts (breach notification), Live market data (prices, yields). Alternative: HTTP polling (inefficient, 1s lag). Implementation: FastAPI WebSocket endpoints, React useWebSocket hook.',
    },
    {
      id: 'fiap-mc-4',
      question: 'Why compare platform pricing to Bloomberg in testing?',
      options: [
        'Validate accuracy against industry standard',
        'Bloomberg is always correct',
        'Regulatory requirement',
        'Faster than own tests',
      ],
      correctAnswer: 0,
      explanation:
        'Bloomberg: Industry standard, widely trusted (not perfect). Accuracy validation: Compare bond prices (tolerance ±1bp acceptable), Compare yields, duration, convexity. Discrepancies: Investigate (different day count? accrued interest calc? curve?). Tests: Automated daily (fetch Bloomberg prices, compare to platform). Builds confidence: Users trust platform (matches market standard).',
    },
    {
      id: 'fiap-mc-5',
      question:
        'What is the main challenge of calculating VaR for a large portfolio (10,000+ positions)?',
      options: [
        'Computational performance (must complete within reasonable time)',
        'Lack of data',
        'No standard methodology',
        'Regulatory restrictions',
      ],
      correctAnswer: 0,
      explanation:
        'Performance challenge: 10K positions, Reprice each under 1000 scenarios (Monte Carlo) = 10M calculations. Solutions: Parallel processing (multiprocessing, Dask), Approximate (representative positions, factor models), Cache (reuse calculations for unchanged positions), Batch (overnight for full revaluation, intraday incremental). Target: <60s for overnight VaR, <10s for intraday updates.',
    },
  ];
