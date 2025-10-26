export const portfolioOptimizationProjectQuiz = {
    id: 'portfolio-optimization-project',
    title: 'Module Project: Portfolio Optimization Platform',
    questions: [
        {
            id: 'pop-system-architecture',
            text: `Design the system architecture for a production portfolio optimization platform serving 50 institutional clients with $10B AUM total. Requirements: (1) Support multiple optimization engines (MVO, Black-Litterman, risk parity, robust optimization) with sub-second response times, (2) Handle 1000+ securities with daily updates, (3) Comply with regulatory requirements (audit trails, position limits, pre-trade compliance checks), (4) Scale to 100 concurrent users during market hours. Design: database schema for positions/returns/constraints, API endpoints, caching strategy, microservices architecture, and disaster recovery. Calculate: compute resources needed (CPU/memory for 1000-stock covariance estimation and optimization), data storage (5 years of daily returns for 1000 stocks), and estimated infrastructure costs.`,
            type: 'discussion' as const,
            sampleAnswer: `**Architecture: Microservices with (1) Data ingestion service (real-time prices, fundamentals), (2) Risk calculation engine (covariance matrices, factor models), (3) Optimization service (CVXPY/Gurobi backend), (4) Portfolio management service (positions, constraints), (5) API gateway (authentication, rate limiting). Database: PostgreSQL for transactions, Redis for caching, S3 for historical data. Covariance calc: 1000×1000 matrix = 1M elements, 5-year rolling daily = 1250 days, requires 8GB RAM, 2-5 seconds on 16-core CPU. Optimization: CVXPY with 1000 variables + 200 constraints = 5-10 seconds per solve. Caching: store computed covariances for 1 hour (90% hit rate), reduces compute 10x. Infrastructure: 4× c5.4xlarge instances ($1200/mo), RDS PostgreSQL ($800/mo), ElastiCache Redis ($300/mo), S3 storage 500GB ($12/mo), total ~$2,500/mo. DR: Multi-AZ deployment, daily S3 backups, 4-hour RTO. Scales to 100 concurrent with auto-scaling group.**`,
            keyPoints: [
                'Microservices architecture separates concerns: data, risk, optimization, portfolio management, API gateway',
                'Compute bottleneck: 1000×1000 covariance matrix estimation requires 8GB RAM, 16-core CPU, 2-5 seconds',
                'Optimization time: CVXPY 1000 variables with 200 constraints solves in 5-10 seconds per client',
                'Caching critical: compute covariance once, cache 1 hour, serve 100 clients → 100x efficiency gain',
                'Database design: PostgreSQL transactions, Redis cache, S3 historical data; separation by update frequency',
                'Infrastructure costs: ~$2,500/month for 50 clients = $50/client/month for compute/storage',
                'Scalability: auto-scaling group handles 100 concurrent users, horizontal scaling for more',
                'Compliance: audit trails in PostgreSQL, pre-trade checks in optimization service, regulatory reporting pipeline'
            ]
        },
        {
            id: 'pop-production-features',
            text: `Implement production-ready features for institutional portfolio optimization. Build: (1) Constraint validation engine checking 15+ constraint types (position limits, sector exposure, factor loadings, ESG scores, turnover, tracking error) before optimization, (2) Multi-objective optimization allowing clients to specify preferences: maximize return, minimize risk, maximize ESG score with weights, (3) Scenario analysis: stress test portfolio under 10 market scenarios (crash -40%, inflation surge, rate spike, etc.), (4) Performance attribution: decompose realized returns into factor contributions, asset allocation, security selection, interaction effects. Discuss implementation challenges, error handling, and how to present results to non-technical portfolio managers.`,
            type: 'discussion' as const,
            sampleAnswer: `**Constraint validation: Rule engine with hierarchy (1) Regulatory constraints (ERISA, UCITS), (2) Risk constraints (VaR, tracking error), (3) Client constraints (ESG, exclusions), (4) Operational constraints (liquidity, turnover). Pre-optimization validation prevents infeasible solutions, returns clear error messages. Multi-objective: Weighted sum approach: Minimize λ₁(risk) - λ₂(return) + λ₃(ESG_deviation), where λ weights set by client (e.g., 60% risk minimization, 30% ESG improvement, 10% return target). Pareto frontier shows trade-offs. Scenario analysis: Monte Carlo + historical stress tests, compute portfolio value under each, present as heatmap showing worst/best scenarios. Attribution: Brinson model decomposing into allocation effect (sector timing), selection effect (stock picking), interaction effect. Results dashboard: visual charts (efficient frontier, risk contribution pie chart, scenario fan chart), executive summary (1-page PDF), detailed Excel export. Error handling: graceful degradation (if optimization times out after 30s, return last feasible solution), clear error messages ("Tech sector constraint cannot be met with current portfolio - suggest relaxing from 30% to 35%"), logging for debugging.**`,
            keyPoints: [
                'Constraint validation hierarchy prevents infeasible optimizations: regulatory > risk > client > operational',
                'Multi-objective optimization uses weighted sum: λ₁(risk) - λ₂(return) + λ₃(ESG); Pareto frontier shows trade-offs',
                'Scenario analysis combines historical stress tests + Monte Carlo: compute portfolio under 10 scenarios (crash, inflation, etc.)',
                'Performance attribution: Brinson model decomposing return into allocation (sector timing) + selection (stock picking) + interaction',
                'Error handling critical: timeout fallback (return last feasible solution), clear messages ("relax constraint X"), audit logging',
                'Visualization for non-technical users: efficient frontier chart, risk contribution pie, scenario heatmap, 1-page PDF summary',
                'Production reliability: 99.9% uptime target, <5 second response time 95th percentile, graceful degradation under load',
                'Deployment: blue-green deployment for zero downtime, comprehensive integration tests, staging environment for validation'
            ]
        },
        {
            id: 'pop-integration-testing',
            text: `Design comprehensive testing strategy for portfolio optimization platform. Build test suite covering: (1) Unit tests for optimization engines (verify Sharpe maximization finds correct weights, constraint handling), (2) Integration tests across services (data ingestion → risk calculation → optimization → results), (3) Performance tests (10,000 optimization requests over 1 hour, measure p50/p95/p99 latency), (4) Regression tests using historical portfolios (verify 2023-01-15 optimization produces same results). Implement continuous integration pipeline, discuss how to test financial calculations (floating point precision, numerical stability), and establish quality gates (80% code coverage, <10s optimization time, zero critical bugs in production for 30 days).`,
            type: 'discussion' as const,
            sampleAnswer: `**Testing pyramid: 70% unit tests, 20% integration tests, 10% end-to-end tests. Unit tests: pytest fixtures with synthetic data, verify (1) unconstrained optimization finds analytical solution, (2) constraints bind correctly (tech sector stops at 30%), (3) numerical stability (condition number < 1e10). Integration tests: Docker Compose with all services, test full workflow with realistic data, verify results within 0.01% tolerance. Performance tests: Locust/JMeter simulating 100 concurrent users, target p95 <5s, p99 <10s, measure throughput (optimizations/second). Regression tests: golden dataset from 2023-01-15, re-run monthly, assert results match within 0.1%. CI/CD: GitHub Actions running pytest on every commit, deployment gate: all tests pass + code coverage ≥80%. Financial calculation testing: use decimal.Decimal for money, assert results match expected within epsilon (1e-6), cross-validate with Bloomberg/FactSet. Quality gates: (1) Zero Sev1 bugs in prod for 30 days, (2) p95 latency <5s, (3) 99.9% uptime monthly, (4) All optimizations converge (no numerical failures). Test data: anonymized real portfolios (with client permission), synthetic stress cases (ill-conditioned covariance matrices). Result: production platform with <0.1% defect rate, 99.95% uptime, trusted by institutional clients.**`,
            keyPoints: [
                'Testing pyramid: 70% unit (fast, isolated), 20% integration (realistic), 10% end-to-end (slow, brittle)',
                'Unit test challenges: verify optimization correctness (analytical solutions), constraint binding, numerical stability',
                'Integration tests: Docker Compose all services, test full workflow with realistic data, verify within tolerance',
                'Performance testing: Locust 100 concurrent users, measure p50/p95/p99 latency, target p95 <5s, p99 <10s',
                'Regression tests: golden dataset from production, re-run monthly, assert results match within 0.1% (tolerate minor drift)',
                'Financial calculation precision: use Decimal for money, epsilon tolerance 1e-6, cross-validate with Bloomberg',
                'CI/CD pipeline: GitHub Actions pytest on every commit, coverage ≥80%, deployment gate requires all tests pass',
                'Quality gates: zero Sev1 bugs 30 days, 99.9% uptime, <5s p95 latency, 100% optimization convergence'
            ]
        }
    ]
};

