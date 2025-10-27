export const marketDataStorageQuiz = [
  {
    id: 'market-data-storage-q-1',
    question:
      'Design a tick database system using TimescaleDB to store 1 year of market data for 500 symbols at 100 ticks/second per symbol. Requirements: (1) Calculate storage requirements with compression, (2) Optimize for time-range queries, (3) Implement automated data retention policy, (4) Provide backup strategy. Include SQL schema and Python implementation.',
    sampleAnswer:
      "TimescaleDB tick storage design:\n\n**Storage Calculation**: 500 symbols × 100 ticks/sec × 86,400 sec/day × 365 days = 1.58 trillion ticks/year. Per tick: timestamp(8) + symbol(10) + exchange(4) + price(8) + size(4) + conditions(10) = 44 bytes. Raw storage: 1.58T × 44 bytes = 69 TB. With TimescaleDB compression (20×): 69 TB / 20 = 3.5 TB compressed. Plus indexes (~20%): 4.2 TB total.\n\n**Schema Optimization**: Create hypertable partitioned by time (1-day chunks). Index on (symbol, time DESC) for fast symbol lookups. Use DECIMAL(12,4) for prices (exact precision). Enable compression on chunks older than 7 days (segment by symbol for better compression ratio).\n\n**Retention Policy**: Automated via TimescaleDB: add_retention_policy('ticks', INTERVAL '1 year'). Older data automatically dropped. For archival, export to S3 before deletion (compress to Parquet format, 50× compression).\n\n**Backup Strategy**: (1) Daily incremental backups to S3 (pg_dump compressed), (2) Continuous WAL archiving for point-in-time recovery, (3) Weekly full backups (offline copy), (4) Cross-region replication for disaster recovery. Test restore quarterly.\n\n**Query Performance**: Time-range query for AAPL (1 day): < 100ms on 8.6M ticks. Aggregation queries (OHLC): < 500ms. Use continuous aggregates for common queries (1min bars pre-calculated).",
    keyPoints: [
      'Storage: 1.58T ticks × 44 bytes = 69 TB raw, 3.5 TB with 20× compression, 4.2 TB with indexes',
      'Schema: Hypertable with 1-day chunks, index on (symbol, time DESC), DECIMAL prices for precision',
      'Compression: Enable on chunks > 7 days old, segment by symbol, achieves 20× reduction in production',
      'Retention: Automated 1-year policy, archive to S3 (Parquet) before deletion, 50× additional compression',
      'Performance: Time-range queries < 100ms, aggregations < 500ms, use continuous aggregates for frequent queries',
    ],
  },
  {
    id: 'market-data-storage-q-2',
    question:
      'Compare TimescaleDB vs QuestDB vs InfluxDB for storing market data. Analyze: (1) Write throughput, (2) Query performance, (3) Compression ratios, (4) Operational complexity, (5) Cost at scale (10TB data). Recommend which database to use for different scenarios.',
    sampleAnswer:
      'Time-series database comparison for market data:\n\n**TimescaleDB**:\n- Write throughput: 100K rows/sec (single node), 1M rows/sec (distributed)\n- Query: PostgreSQL compatibility, excellent for complex queries, JOINs\n- Compression: 10-20× (columnar compression on old chunks)\n- Complexity: Medium (PostgreSQL knowledge required)\n- Cost (10TB): ~$500/mo (AWS RDS, 3 replicas)\n- Pros: ACID, mature, PostgreSQL ecosystem\n- Cons: Not fastest for pure time-series\n\n**QuestDB**:\n- Write throughput: 4M rows/sec (fastest in class)\n- Query: Fast time-series queries, SQL support\n- Compression: 5-10× (columnar storage)\n- Complexity: Low (simple deployment)\n- Cost (10TB): ~$200/mo (self-hosted)\n- Pros: Fastest writes, low latency queries\n- Cons: Newer product, smaller ecosystem\n\n**InfluxDB**:\n- Write throughput: 500K rows/sec\n- Query: InfluxQL/Flux, optimized for metrics\n- Compression: 20-40× (TSM format)\n- Complexity: Medium (clustering complex)\n- Cost (10TB): ~$800/mo (InfluxDB Cloud)\n- Pros: Best compression, great for metrics\n- Cons: No SQL, eventual consistency\n\n**Recommendations**:\n\n*HFT Tick Storage (millions writes/sec)*: **QuestDB** - fastest writes (4M rows/sec), low query latency (< 1ms), simple deployment.\n\n*Institutional (complex analytics)*: **TimescaleDB** - PostgreSQL compatibility means existing tools work (pandas, BI tools), ACID transactions for critical data, excellent JOIN performance for multi-table analysis.\n\n*Metrics/Monitoring (high compression)*: **InfluxDB** - 40× compression saves storage costs, purpose-built for metrics, downsampling for long-term storage.\n\n*Budget-Conscious*: **QuestDB** - self-hosted costs 60% less than managed services, excellent performance, simple operations.\n\n**Decision Matrix**: Write speed > 1M rows/sec → QuestDB. Need PostgreSQL compatibility → TimescaleDB. Focus on compression/cost → InfluxDB.',
    keyPoints: [
      'TimescaleDB: 100K writes/sec, PostgreSQL compatibility, 10-20× compression, $500/mo for 10TB, best for complex queries',
      'QuestDB: 4M writes/sec (fastest), columnar storage, 5-10× compression, $200/mo self-hosted, best for HFT',
      'InfluxDB: 500K writes/sec, 20-40× compression (best), purpose-built for metrics, $800/mo cloud',
      'Recommendation: HFT → QuestDB (speed), Institutional → TimescaleDB (compatibility), Monitoring → InfluxDB (compression)',
      'Cost factor: QuestDB self-hosted = 60% cheaper than managed services at 10TB scale',
    ],
  },
  {
    id: 'market-data-storage-q-3',
    question:
      'You need to query historical tick data for backtesting: "Get all AAPL ticks from Jan 1-31, 2024" returns 2.1M rows and takes 45 seconds (too slow). Optimize: (1) Identify bottlenecks, (2) Add appropriate indexes, (3) Use compression, (4) Implement caching. Target: < 1 second query time.',
    sampleAnswer:
      "Query optimization strategy:\n\n**Bottleneck Analysis**:\n1. Sequential scan on 8.6M ticks/day × 31 days = 266M rows scanned\n2. No index on (symbol, time) → full table scan\n3. Decompression overhead (compressed chunks)\n4. Network transfer of 2.1M rows\n5. No result caching\n\n**Optimization 1: Index**\n```sql\nCREATE INDEX idx_symbol_time ON ticks (symbol, time DESC);\n```\nBenefit: Index seek directly to AAPL data, skip other 499 symbols. Reduces scan from 266M → 2.1M rows (126× reduction). Query time: 45s → 5s.\n\n**Optimization 2: Compression Strategy**\n```sql\n-- Decompress only queried chunks\nALTER TABLE ticks SET (\n    timescaledb.compress,\n    timescaledb.compress_segmentby = 'symbol'\n);\n```\nSegment by symbol means AAPL chunks decompress independently (not entire day). Query time: 5s → 2s.\n\n**Optimization 3: Continuous Aggregates**\n```sql\n-- Pre-calculate 1-min bars\nCREATE MATERIALIZED VIEW ticks_1min\nWITH (timescaledb.continuous) AS\nSELECT \n    time_bucket('1 minute', time) AS bucket,\n    symbol,\n    first(bid_price, time) AS open,\n    max(bid_price) AS high,\n    min(ask_price) AS low,\n    last(ask_price, time) AS close\nFROM ticks\nGROUP BY bucket, symbol;\n```\nFor many backtests, 1-min bars sufficient (not full ticks). 2.1M ticks → 44,640 bars (47× reduction). Query time: 2s → 0.1s.\n\n**Optimization 4: Redis Caching**\n```python\nimport redis\nimport pickle\n\nclass TickCache:\n    def __init__(self):\n        self.redis = redis.Redis()\n    \n    def get_ticks(self, symbol, start, end):\n        # Check cache\n        cache_key = f\"ticks:{symbol}:{start}:{end}\"\n        cached = self.redis.get(cache_key)\n        \n        if cached:\n            return pickle.loads(cached)  # 10ms from cache\n        \n        # Query database\n        ticks = query_database(symbol, start, end)  # 2s\n        \n        # Store in cache (1 hour TTL)\n        self.redis.setex(cache_key, 3600, pickle.dumps(ticks))\n        \n        return ticks\n```\nFirst query: 2s (cache miss). Subsequent queries: 10ms (cache hit). Cache hit rate: 80%+ for backtesting (same date ranges repeatedly tested).\n\n**Final Performance**:\n- Cold query (no cache): 2s (45s → 2s, 22.5× improvement)\n- Warm query (cached): 10ms (4500× improvement)\n- Storage: +20% for indexes, +5% for continuous aggregates\n- Achieved < 1 second target ✓\n\n**Additional Optimizations**:\n- Partition pruning: Only scan Jan 2024 chunks (exclude other months)\n- Parallel query: Use TimescaleDB parallel workers (4× speedup)\n- Columnar projection: Select only needed columns (price, time), not all fields",
    keyPoints: [
      'Index on (symbol, time DESC): Reduces scan from 266M → 2.1M rows (126×), 45s → 5s query time',
      'Compression segment by symbol: Decompress only AAPL chunks (not full day), 5s → 2s',
      'Continuous aggregates: Pre-calculate 1-min bars (2.1M ticks → 44K bars), 2s → 0.1s for bar queries',
      'Redis caching: First query 2s, subsequent 10ms (450× faster), 80%+ hit rate in backtesting',
      'Final: 45s → 2s cold query (22.5×), 10ms warm query (4500×), < 1s target achieved',
    ],
  },
];
