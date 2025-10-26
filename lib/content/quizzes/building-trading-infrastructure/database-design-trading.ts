export const databaseDesignTradingQuiz = [
    {
        id: 'database-design-trading-q-1',
        question:
            'Your trading system stores 100 million market data ticks per day. Should you use PostgreSQL, MongoDB, or TimescaleDB? Explain your reasoning.',
        sampleAnswer:
            'Use TimescaleDB for Market Data:\n\n' +
            '**Why TimescaleDB:**\n' +
            '1. **Time-series optimized**: Built for time-stamped data\n' +
            '2. **Compression**: 10x compression for historical data\n' +
            '3. **Fast queries**: Optimized for time-range queries\n' +
            '4. **PostgreSQL compatible**: Standard SQL\n\n' +
            '**Architecture:**\n' +
            '- 100M ticks/day = 1,157 ticks/sec average\n' +
            '- Peak: 10K ticks/sec during market open\n' +
            '- Storage: 100M ticks × 40 bytes = 4GB/day uncompressed → 400MB compressed\n' +
            '- Retention: 365 days = 146GB compressed\n\n' +
            '**Why not PostgreSQL:** Too slow for time-series queries, no compression.\n' +
            '**Why not MongoDB:** Slower than TimescaleDB for time-range queries.',
        keyPoints: [
            'TimescaleDB best: Time-series optimized, 10x compression, fast time-range queries, PostgreSQL compatible',
            'Scale: 100M ticks/day = 1,157/sec average, 10K/sec peak, 4GB/day → 400MB compressed',
            'Storage: 365 days = 146GB compressed (vs 1.5TB uncompressed)',
            'PostgreSQL: Too slow for time-series, no compression, would require partitioning',
            'MongoDB: Slower than TimescaleDB for time-range queries, less SQL compatibility',
        ],
    },
    {
        id: 'database-design-trading-q-2',
        question:
            'Design a database schema for storing options positions. What tables/columns do you need? How would you calculate total portfolio Greeks?',
        sampleAnswer:
            'Options Position Schema:\n\n' +
            '```sql\n' +
            'CREATE TABLE options_positions (\n' +
            '    position_id BIGSERIAL PRIMARY KEY,\n' +
            '    underlying VARCHAR(10) NOT NULL,  -- AAPL\n' +
            '    strike DECIMAL(18,4) NOT NULL,    -- 150.00\n' +
            '    expiration DATE NOT NULL,         -- 2025-11-15\n' +
            '    option_type VARCHAR(4) NOT NULL,  -- CALL or PUT\n' +
            '    quantity INT NOT NULL,            -- 10 contracts\n' +
            '    avg_cost DECIMAL(18,4),           -- $5.00 premium\n' +
            '    account VARCHAR(50),\n' +
            '    strategy VARCHAR(50),\n' +
            '    \n' +
            '    -- Greeks (updated real-time)\n' +
            '    delta DECIMAL(8,6),               -- 0.600000\n' +
            '    gamma DECIMAL(8,6),\n' +
            '    vega DECIMAL(8,6),\n' +
            '    theta DECIMAL(8,6),\n' +
            '    \n' +
            '    updated_at TIMESTAMP DEFAULT NOW()\n' +
            ');\n' +
            '```\n\n' +
            'Portfolio Greeks:\n' +
            '```sql\n' +
            'SELECT\n' +
            '    SUM(quantity * 100 * delta) AS portfolio_delta,\n' +
            '    SUM(quantity * 100 * gamma) AS portfolio_gamma,\n' +
            '    SUM(quantity * 100 * vega) AS portfolio_vega,\n' +
            '    SUM(quantity * 100 * theta) AS portfolio_theta\n' +
            'FROM options_positions;\n' +
            '```',
        keyPoints: [
            'Schema: underlying, strike, expiration, option_type, quantity (contracts), avg_cost (premium)',
            'Greeks: Store delta, gamma, vega, theta per position, update real-time from options pricing model',
            'Portfolio Greeks: SUM(quantity × 100 × greek) to get total exposure (100 shares per contract)',
            'Indexes: (underlying, expiration) for fast queries, (account, strategy) for attribution',
            'Additional: Store implied_vol, current_price, intrinsic_value, time_value for analytics',
        ],
    },
    {
        id: 'database-design-trading-q-3',
        question:
            'Your orders table has 100 million rows and queries are slow (10+ seconds). How would you optimize it?',
        sampleAnswer:
            'Database Optimization:\n\n' +
            '1. **Partitioning** (by date):\n' +
            '```sql\n' +
            'CREATE TABLE orders_2025_10 PARTITION OF orders\n' +
            'FOR VALUES FROM (\'2025-10-01\') TO (\'2025-11-01\');\n' +
            '```\n' +
            '- Query only current month\'s partition (1M rows vs 100M)\n' +
            '- Result: 10s → 0.1s\n\n' +
            '2. **Indexes** (on query columns):\n' +
            '```sql\n' +
            'CREATE INDEX idx_symbol_status_created\n' +
            'ON orders (symbol, status, created_at DESC);\n' +
            '```\n' +
            '- Covers common query pattern: WHERE symbol AND status ORDER BY created_at\n\n' +
            '3. **Archival** (move old orders):\n' +
            '- Move orders >90 days to orders_archive table\n' +
            '- Keeps main table small (<10M rows)\n\n' +
            '4. **Denormalization** (for common queries):\n' +
            '- Materialized view for order_summary\n' +
            '- Refresh every 1 minute',
        keyPoints: [
            'Partitioning: Partition by month, queries scan 1M rows (current month) vs 100M (all time), 10s → 0.1s',
            'Indexes: Multi-column index on (symbol, status, created_at) for common query patterns',
            'Archival: Move orders >90 days to orders_archive, keeps main table <10M rows for fast queries',
            'Denormalization: Materialized views for common aggregates, refresh every 1 minute',
            'Vacuuming: VACUUM ANALYZE to update statistics, REINDEX to rebuild indexes',
        ],
    },
];

