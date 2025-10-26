export const fixedIncomeAnalyticsPlatformQuiz = [
    {
        id: 'fiap-q-1',
        question: 'Design the complete architecture for a fixed income analytics platform: (1) Microservices for pricing, risk, portfolio management, market data, (2) API design (REST + WebSocket for real-time), (3) Database schema (bonds, positions, trades, risk metrics), (4) Caching strategy (Redis for market data, calc results), (5) Monitoring and observability. Include: service communication patterns, data consistency, scalability, fault tolerance. How do you handle: market data updates (propagate to all services)? Large portfolio calculations (batch processing)? Regulatory reporting requirements?',
        sampleAnswer: 'Architecture: Microservices (pricing, risk, portfolio, market-data, reporting), Communication: REST for request/response, WebSocket for real-time updates, Message queue (Kafka) for events, Database: PostgreSQL (positions, trades), TimescaleDB (time-series market data), Redis (cache), Caching: Market data (5min TTL), Pricing results (1min TTL), Scalability: Horizontal scaling (K8s), Load balancer, Monitoring: Prometheus + Grafana, Jaeger tracing, Audit logs.',
        keyPoints: ['Microservices design', 'REST + WebSocket', 'Multi-tier caching', 'Horizontal scaling', 'Observability'],
    },
    {
        id: 'fiap-q-2',
        question: 'Implement a real-time portfolio risk dashboard that: (1) Displays portfolio Greeks (duration, convexity, DV01, vega), (2) Shows VaR breakdown by position, (3) Monitors risk limit utilization with visual alerts, (4) Updates on every trade (WebSocket streaming), (5) Historical risk metrics charts. Include: performance optimization (lazy loading, virtualization), responsive design. How do you handle: thousands of positions (pagination, aggregation)? Real-time calculation latency? User-specific views (filtering, sorting)?',
        sampleAnswer: 'Dashboard: WebSocket connection for real-time updates, Display: AG Grid (virtualized) for positions table, Recharts for time-series (VaR history), Greeks: Aggregate at top, drill-down to position level, VaR: Component VaR bar chart, position-level table, Limits: Progress bars with color coding (green <80%, yellow 80-95%, red >95%), Performance: Virtual scrolling (render only visible rows), Lazy load historical data (on scroll), Aggregate: Sector/desk level by default, expand for details, Updates: Incremental (send deltas, not full snapshot).',
        keyPoints: ['WebSocket real-time', 'Virtualized tables', 'Component VaR visualization', 'Limit alerts', 'Drill-down aggregation'],
    },
    {
        id: 'fiap-q-3',
        question: 'Build a comprehensive testing strategy for the fixed income platform: (1) Unit tests for pricing functions (edge cases, accuracy), (2) Integration tests for API endpoints (happy path + error cases), (3) Performance tests (load testing, stress testing), (4) Accuracy tests (compare to Bloomberg/market), (5) Regression tests (prevent breaking changes). Include: test data generation, mocking strategies. How do you handle: flaky tests? Test environment setup (market data, databases)? CI/CD integration?',
        sampleAnswer: 'Testing strategy: Unit tests: Pytest, test pricing functions (bonds, swaps, options), Mock market data (fixtures), Target >90% coverage, Integration tests: Test APIs end-to-end (FastAPI TestClient), Mock external services (Bloomberg API), Test error handling (400/500 responses), Performance tests: Locust (load testing), Target: 1000 req/s, <100ms p95 latency, Stress test: Increase load until failure (find breaking point), Accuracy tests: Compare calculated prices to Bloomberg (tolerance ±1bp), Regression: Snapshot testing (save expected outputs), CI/CD: GitHub Actions, run tests on every PR, block merge if tests fail.',
        keyPoints: ['Unit tests >90% coverage', 'Integration testing', 'Load testing (Locust)', 'Accuracy validation', 'CI/CD automation'],
    },
];

