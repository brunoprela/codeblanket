export default {
  id: 'fin-m15-s16-quiz',
  title: 'Risk Management Platform Project - Quiz',
  questions: [
    {
      id: 1,
      question:
        'A risk platform must calculate VaR with <100ms latency for pre-trade checks. Which architectural choice is most critical?',
      options: [
        'Using the most accurate VaR model (Monte Carlo)',
        'In-memory position cache (Redis) + fast parametric/incremental VaR',
        'Distributed database across 100 servers',
        'Blockchain for immutability',
      ],
      correctAnswer: 1,
      explanation:
        "Sub-100ms latency requires: (1) In-memory data store (Redis) for instant position lookup (<1ms), (2) Fast VaR approximation (parametric or incremental) that trades accuracy for speed (~10ms), (3) Pre-calculated sensitivities (cached). Total: <100ms. Option A is wrong—Monte Carlo takes seconds, unacceptable for pre-trade. Option C helps throughput but not individual latency. Option D adds latency, no benefit here. The tradeoff: Pre-trade checks use 95% accurate fast method. EOD uses 99.9% accurate slow method (Monte Carlo). Both are needed—can't use slow accurate for real-time (trading stops), can't use fast approximate for official reporting (regulatory rejection). Architecture: Layered approach with different methods for different latency requirements: Real-time (<100ms): parametric. Hourly (<10s): historical. EOD (can take minutes): Monte Carlo.",
    },
    {
      id: 2,
      question:
        'For a risk platform handling 100,000 trades per day, which data infrastructure component is most critical for reliability?',
      options: [
        'Kafka message queue for reliable async processing',
        'Excel spreadsheets for data storage',
        'Manual data entry for accuracy',
        'Single database server for simplicity',
      ],
      correctAnswer: 0,
      explanation:
        "Kafka (or similar message queue) provides: (1) Reliable delivery—guaranteed every trade processed even if systems temporarily down, (2) Async processing—trading system doesn't wait for risk system, (3) Replay capability—if bug, reprocess from message log, (4) Scalability—handle spikes (10× normal volume). Option B is toy approach. Option C doesn't scale. Option D is single point of failure. Real-world: Trade executed → Kafka topic → Multiple consumers (risk system, accounting, reporting) process in parallel. If risk system crashes, Kafka retains messages; when risk system restarts, processes backlog. Without message queue: Direct calls between systems create tight coupling, no retry mechanism, data loss on failures. Kafka (or alternatives: RabbitMQ, AWS Kinesis) is industry standard for reliable event streaming in financial systems. Cost: $0 (open source) to $100K/year (enterprise support). Benefit: No data loss, no missed trades, recoverable errors.",
    },
    {
      id: 3,
      question:
        'A pre-trade compliance system must check trades against limits before execution. If the check takes too long (>200ms), what happens?',
      options: [
        'Trade execution is delayed (unacceptable in fast markets)',
        'Trade executes without check (violates risk control)',
        'Trader gets fired',
        'System is working correctly—200ms is fast',
      ],
      correctAnswer: 0,
      explanation:
        '>200ms is too slow for real-time trading. In fast markets, prices move every 10ms—200ms delay means you miss the price or get adverse selection (only slow prices fill). Option A describes the problem: Delayed execution = poor execution quality = trader complains. Option B is even worse—executing without check defeats the purpose of pre-trade compliance. Option C is misdirected blame. Option D is wrong—200ms is slow (target: <100ms, ideally <50ms). Solutions: (1) Faster calculations (parametric VaR, cached sensitivities, in-memory data), (2) Parallel processing (check limits while assembling order), (3) Hierarchical checks (fast checks first, detailed checks async). Balance: Need speed (<100ms) AND reliability (no false rejects). Fast but unreliable = traders bypass system. Slow but reliable = traders complain about execution quality. Target: 50ms 99th percentile latency with <0.1% false rejection rate.',
    },
    {
      id: 4,
      question:
        'For audit trail requirements (7-year retention, immutable, queryable), which database choice is best?',
      options: [
        'Redis (in-memory cache)',
        'TimescaleDB (time-series database with compression) or similar append-only log',
        'Excel files on shared drive',
        'Blockchain',
      ],
      correctAnswer: 1,
      explanation:
        'TimescaleDB (or InfluxDB, Snowflake) optimized for: (1) Immutability—append-only, no updates/deletes, (2) Time-series—audit events ordered by time, (3) Compression—7 years of data compressed 10-20×, (4) Fast queries—compliance needs to query "all breaches last quarter" (sub-second). Option A is wrong—Redis is cache (volatile, no long-term storage). Option C is wrong—Excel doesn\'t scale to billions of audit records. Option D (blockchain) is overkill—unnecessary complexity and cost for internal audit log. Requirements: Every limit check (~100K/day) + Every breach (~10/day) + Every override (~2/day) × 7 years = ~250M records. TimescaleDB handles this easily with automatic compression and fast queries. Alternative: Data warehouse (Snowflake, BigQuery) for immutable storage + fast analytics. Key: Audit trail is regulatory requirement—must be bulletproof (no data loss), performant (fast queries for exams), and cost-effective (7-year retention is expensive if not compressed).',
    },
    {
      id: 5,
      question:
        'A risk platform stores positions in PostgreSQL (persistent) and Redis (cache). A trade arrives—which should be updated first?',
      options: [
        'PostgreSQL (persistent) then Redis (cache)',
        'Redis (cache) then PostgreSQL (persistent) asynchronously',
        'Both simultaneously',
        'Only PostgreSQL—Redis not needed',
      ],
      correctAnswer: 1,
      explanation:
        'Update Redis (cache) first for speed, then PostgreSQL async for durability. This gives best of both: (1) Redis update <1ms → real-time dashboards instantly reflect trade, (2) PostgreSQL update async (100ms) → durable storage without blocking. If PostgreSQL first: 100ms delay before dashboards update (users see stale data). If simultaneous: Complex coordination, slower. Option D is wrong—cache dramatically speeds reads (10,000× faster). Pattern: Write-through cache—update cache immediately (speed), persist to database asynchronously (durability), handle failures gracefully (if PostgreSQL write fails, retry from message queue). Risk: Brief window where Redis updated but PostgreSQL not yet (if system crashes, lose trade). Mitigation: Kafka message log provides backup—can replay. This architecture (cache-first, persist-async) enables sub-second updates for real-time systems while maintaining durability. Used by: Twitter, Facebook, trading systems. Tradeoff: Favor speed (cache first) over immediate durability (async persistence), justified by message queue backup.',
    },
  ],
} as const;
