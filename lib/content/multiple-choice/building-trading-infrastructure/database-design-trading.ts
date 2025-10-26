export const databaseDesignTradingMC = [
    {
        id: 'database-design-trading-mc-1',
        question:
            'Which database is BEST for storing orders in a trading system?',
        options: [
            'PostgreSQL (ACID transactions)',
            'MongoDB (NoSQL)',
            'Redis (in-memory)',
            'Cassandra (distributed)',
        ],
        correctAnswer: 0,
        explanation:
            'Answer: PostgreSQL (ACID transactions).\n\n' +
            'Orders require ACID:\n' +
            '- **Atomicity**: Order insert must complete fully or not at all\n' +
            '- **Consistency**: Order state must be consistent\n' +
            '- **Isolation**: Concurrent orders don\'t interfere\n' +
            '- **Durability**: Order persists even after crash\n\n' +
            'Why others are unsuitable:\n' +
            '- **MongoDB**: Eventual consistency (orders could be lost)\n' +
            '- **Redis**: In-memory (orders lost on crash unless persistence enabled)\n' +
            '- **Cassandra**: Eventual consistency, complex to maintain consistency\n\n' +
            'Real-world: All brokers (Interactive Brokers, TD Ameritrade, etc.) use PostgreSQL or similar RDBMS for orders.',
    },
    {
        id: 'database-design-trading-mc-2',
        question:
            'Your market data table has 1 billion rows. A query to get the last hour of data takes 60 seconds. What is the BEST optimization?',
        options: [
            'Add more RAM',
            'Create an index on the timestamp column',
            'Use a faster SSD',
            'Partition the table by date',
        ],
        correctAnswer: 3,
        explanation:
            'Answer: Partition the table by date.\n\n' +
            'Problem: Scanning 1 billion rows to find last hour.\n\n' +
            'Solution - Partitioning:\n' +
            '```sql\n' +
            'CREATE TABLE market_data_2025_10_26\n' +
            'PARTITION OF market_data\n' +
            'FOR VALUES FROM (\'2025-10-26\') TO (\'2025-10-27\');\n' +
            '```\n\n' +
            'Query now scans only today\'s partition (~100M rows) instead of 1B rows:\n' +
            '- Before: 60 seconds (scan 1B rows)\n' +
            '- After: 6 seconds (scan 100M rows)\n' +
            '- With index: 0.1 seconds\n\n' +
            'Why index alone isn\'t enough: Index on 1B rows is huge (10GB+), still slow to traverse.',
    },
    {
        id: 'database-design-trading-mc-3',
        question:
            'What is a "materialized view" and why is it useful for trading dashboards?',
        options: [
            'A view that automatically refreshes on every query',
            'A pre-computed query result stored as a table',
            'A view that only shows material (important) data',
            'A view stored in memory',
        ],
        correctAnswer: 1,
        explanation:
            'Answer: A pre-computed query result stored as a table.\n\n' +
            'Regular view (slow):\n' +
            '```sql\n' +
            'CREATE VIEW daily_pnl AS\n' +
            'SELECT date, SUM(pnl) FROM trades GROUP BY date;\n' +
            '-- Re-computes every query (slow for 100M rows)\n' +
            '```\n\n' +
            'Materialized view (fast):\n' +
            '```sql\n' +
            'CREATE MATERIALIZED VIEW daily_pnl AS\n' +
            'SELECT date, SUM(pnl) FROM trades GROUP BY date;\n' +
            '-- Pre-computed, stored as table (fast to query)\n' +
            '\n' +
            '-- Refresh every 1 minute\n' +
            'REFRESH MATERIALIZED VIEW daily_pnl;\n' +
            '```\n\n' +
            'Dashboard benefit: Query returns instantly (no aggregation needed).',
    },
    {
        id: 'database-design-trading-mc-4',
        question:
            'Why use Redis for real-time positions instead of PostgreSQL?',
        options: [
            'Redis has better ACID guarantees',
            'Redis is 100x faster (in-memory vs disk)',
            'Redis has better query language',
            'Redis has better data durability',
        ],
        correctAnswer: 1,
        explanation:
            'Answer: Redis is 100x faster (in-memory vs disk).\n\n' +
            'Latency comparison:\n' +
            '- **Redis**: 0.1-1ms (all data in RAM)\n' +
            '- **PostgreSQL**: 10-100ms (disk-based, index lookups)\n\n' +
            'For real-time positions:\n' +
            '- Update on every fill (1000s of updates/sec)\n' +
            '- Query positions frequently (100s of queries/sec)\n' +
            '- Need <1ms latency\n\n' +
            'Production pattern:\n' +
            '- **Hot path**: Redis for real-time positions\n' +
            '- **Cold path**: Async write to PostgreSQL every 1 second for durability\n\n' +
            'Trade-off: Redis is less durable (lost on crash), but that\'s acceptable for real-time positions (can rebuild from PostgreSQL).',
    },
    {
        id: 'database-design-trading-mc-5',
        question:
            'Your database has 10TB of historical market data. What is the BEST strategy to reduce storage costs?',
        options: [
            'Delete old data',
            'Compress historical data and move to cheaper storage (S3 Glacier)',
            'Buy more disk space',
            'Use a smaller database',
        ],
        correctAnswer: 1,
        explanation:
            'Answer: Compress historical data and move to cheaper storage.\n\n' +
            'Data lifecycle strategy:\n\n' +
            '1. **Hot data** (last 7 days):\n' +
            '   - TimescaleDB (fast queries)\n' +
            '   - Cost: $0.10/GB/month (SSD)\n' +
            '   - Size: 7 days × 4GB/day = 28GB = $2.80/month\n\n' +
            '2. **Warm data** (7-90 days):\n' +
            '   - TimescaleDB with compression (10x)\n' +
            '   - Cost: $0.05/GB/month (HDD)\n' +
            '   - Size: 83 days × 0.4GB/day = 33GB = $1.65/month\n\n' +
            '3. **Cold data** (>90 days):\n' +
            '   - S3 Glacier (compressed)\n' +
            '   - Cost: $0.004/GB/month (archive)\n' +
            '   - Size: 9,939GB = $39.76/month\n\n' +
            'Total: $44/month (vs $1,000/month keeping all on SSD).',
    },
];

