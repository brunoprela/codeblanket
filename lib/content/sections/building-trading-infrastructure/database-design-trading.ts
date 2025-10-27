export const databaseDesignTrading = {
  title: 'Database Design for Trading Systems',
  id: 'database-design-trading',
  content: `
# Database Design for Trading Systems

## Introduction

**Database design** for trading requires balancing performance, reliability, and scalability with zero tolerance for data loss.

**Critical Requirements:**
- **ACID transactions**: Orders must never be lost or duplicated
- **Low latency**: Query positions in <100ms, orders in <10ms
- **High throughput**: Handle 1M+ ticks/second during market hours
- **Data integrity**: Prevent double-fills, incorrect P&L calculations
- **Audit trail**: Store every state change for regulatory compliance
- **Time-series efficiency**: Market data grows at 1TB+ per day

**Real-World Database Stack:**
- **Goldman Sachs**: Oracle + TimeTen (in-memory) + custom time-series DBs
- **Jane Street**: PostgreSQL + custom columnar stores
- **Interactive Brokers**: SQL Server + custom tick databases
- **Bloomberg**: Custom distributed database (B-Unit)

**Database types by use case:**

| Data Type | Database | Why |
|-----------|----------|-----|
| Orders, Fills | PostgreSQL | ACID transactions, referential integrity |
| Positions, Accounts | PostgreSQL | ACID, complex queries |
| Market Data (tick) | TimescaleDB | Time-series optimization, compression |
| Real-time Positions | Redis | In-memory, microsecond latency |
| Historical Analytics | ClickHouse | Columnar, fast aggregations |
| Audit Logs | PostgreSQL | Immutable, compliance |

This section covers production database design patterns for trading systems.

---

## PostgreSQL Schema for Orders

\`\`\`sql
-- =============================================================================
-- ORDERS AND FILLS SCHEMA
-- =============================================================================

-- Orders table (ACID transactions critical)
CREATE TABLE orders (
    -- Primary key
    order_id BIGSERIAL PRIMARY KEY,
    
    -- Client reference
    client_order_id VARCHAR(50) NOT NULL UNIQUE,
    parent_order_id BIGINT REFERENCES orders(order_id),  -- For child orders
    
    -- Security identification
    symbol VARCHAR(10) NOT NULL,
    security_id VARCHAR(50),  -- CUSIP, ISIN, etc.
    exchange VARCHAR(10),
    
    -- Order details
    side VARCHAR(4) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    order_type VARCHAR(20) NOT NULL CHECK (order_type IN ('MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT')),
    quantity DECIMAL(18,4) NOT NULL CHECK (quantity > 0),
    price DECIMAL(18,4),
    stop_price DECIMAL(18,4),
    
    -- Time in force
    time_in_force VARCHAR(10) NOT NULL CHECK (time_in_force IN ('DAY', 'GTC', 'IOC', 'FOK', 'GTD')),
    expire_time TIMESTAMP,
    
    -- Execution tracking
    status VARCHAR(20) NOT NULL CHECK (status IN (
        'NEW', 'PENDING_RISK', 'ACCEPTED', 'PARTIALLY_FILLED', 
        'FILLED', 'CANCELLED', 'REJECTED', 'EXPIRED'
    )),
    filled_quantity DECIMAL(18,4) NOT NULL DEFAULT 0,
    remaining_quantity DECIMAL(18,4) NOT NULL,
    avg_fill_price DECIMAL(18,4),
    
    -- Ownership
    account VARCHAR(50) NOT NULL,
    strategy VARCHAR(50),
    trader VARCHAR(50),
    
    -- Routing
    destination VARCHAR(50),  -- Broker, exchange, or venue
    route TEXT,  -- JSON array of venues tried
    
    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    submitted_at TIMESTAMP,
    accepted_at TIMESTAMP,
    completed_at TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Metadata
    metadata JSONB,  -- Flexible field for custom data
    
    -- Indexes for fast queries
    CONSTRAINT positive_quantities CHECK (
        quantity > 0 AND 
        filled_quantity >= 0 AND 
        remaining_quantity >= 0 AND
        filled_quantity <= quantity
    )
);

-- Indexes for performance
CREATE INDEX idx_orders_symbol_status ON orders(symbol, status) WHERE status NOT IN ('FILLED', 'CANCELLED', 'REJECTED');
CREATE INDEX idx_orders_account_created ON orders(account, created_at DESC);
CREATE INDEX idx_orders_strategy ON orders(strategy, created_at DESC) WHERE strategy IS NOT NULL;
CREATE INDEX idx_orders_status_created ON orders(status, created_at DESC);
CREATE INDEX idx_orders_client_order_id ON orders(client_order_id);

-- Partial index for open orders (most common query)
CREATE INDEX idx_orders_open ON orders(account, symbol, status) 
    WHERE status NOT IN ('FILLED', 'CANCELLED', 'REJECTED', 'EXPIRED');

-- Fills table (execution reports)
CREATE TABLE fills (
    -- Primary key
    fill_id BIGSERIAL PRIMARY KEY,
    
    -- Order reference
    order_id BIGINT NOT NULL REFERENCES orders(order_id),
    execution_id VARCHAR(50) NOT NULL UNIQUE,  -- Exchange execution ID
    
    -- Fill details
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(4) NOT NULL,
    quantity DECIMAL(18,4) NOT NULL CHECK (quantity > 0),
    price DECIMAL(18,4) NOT NULL CHECK (price > 0),
    
    -- Timing
    fill_time TIMESTAMP NOT NULL DEFAULT NOW(),
    exchange_timestamp TIMESTAMP,
    
    -- Venue
    exchange VARCHAR(10) NOT NULL,
    
    -- Fees
    commission DECIMAL(18,2) NOT NULL DEFAULT 0,
    sec_fee DECIMAL(18,2) NOT NULL DEFAULT 0,
    taf_fee DECIMAL(18,2) NOT NULL DEFAULT 0,
    other_fees DECIMAL(18,2) NOT NULL DEFAULT 0,
    commission_currency VARCHAR(3) DEFAULT 'USD',
    
    -- Liquidity indicator
    liquidity_flag VARCHAR(10),  -- MAKER, TAKER, etc.
    
    -- Calculated
    notional_value DECIMAL(18,2) GENERATED ALWAYS AS (quantity * price) STORED,
    
    -- Metadata
    metadata JSONB,
    
    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_fills_order_id ON fills(order_id, fill_time DESC);
CREATE INDEX idx_fills_symbol_time ON fills(symbol, fill_time DESC);
CREATE INDEX idx_fills_exchange ON fills(exchange, fill_time DESC);

-- Trigger to update order on fill
CREATE OR REPLACE FUNCTION update_order_on_fill()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE orders
    SET 
        filled_quantity = filled_quantity + NEW.quantity,
        remaining_quantity = quantity - (filled_quantity + NEW.quantity),
        avg_fill_price = CASE
            WHEN filled_quantity = 0 THEN NEW.price
            ELSE (filled_quantity * COALESCE(avg_fill_price, 0) + NEW.quantity * NEW.price) / (filled_quantity + NEW.quantity)
        END,
        status = CASE
            WHEN (filled_quantity + NEW.quantity) >= quantity THEN 'FILLED'
            ELSE 'PARTIALLY_FILLED'
        END,
        updated_at = NOW()
    WHERE order_id = NEW.order_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_order_on_fill
AFTER INSERT ON fills
FOR EACH ROW
EXECUTE FUNCTION update_order_on_fill();


-- =============================================================================
-- POSITIONS SCHEMA
-- =============================================================================

CREATE TABLE positions (
    -- Primary key
    position_id BIGSERIAL PRIMARY KEY,
    
    -- Position identification
    symbol VARCHAR(10) NOT NULL,
    account VARCHAR(50) NOT NULL,
    strategy VARCHAR(50),
    
    -- Position details
    quantity DECIMAL(18,4) NOT NULL,
    avg_cost DECIMAL(18,4) NOT NULL,
    
    -- P&L
    realized_pnl DECIMAL(18,2) DEFAULT 0,
    unrealized_pnl DECIMAL(18,2) DEFAULT 0,
    total_pnl DECIMAL(18,2) GENERATED ALWAYS AS (realized_pnl + unrealized_pnl) STORED,
    
    -- Market data
    current_price DECIMAL(18,4),
    price_updated_at TIMESTAMP,
    
    -- Timestamps
    opened_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    closed_at TIMESTAMP,
    
    -- Metadata
    metadata JSONB,
    
    -- Unique constraint
    UNIQUE(symbol, account, strategy),
    
    -- Check constraints
    CONSTRAINT valid_position CHECK (quantity != 0 OR closed_at IS NOT NULL)
);

-- Indexes
CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_positions_account ON positions(account);
CREATE INDEX idx_positions_strategy ON positions(strategy) WHERE strategy IS NOT NULL;
CREATE INDEX idx_positions_open ON positions(account, symbol) WHERE closed_at IS NULL;

-- View for current positions
CREATE VIEW current_positions AS
SELECT 
    p.*,
    (quantity * current_price) AS market_value,
    (quantity * (current_price - avg_cost)) AS unrealized_pnl_calculated
FROM positions
WHERE closed_at IS NULL
  AND quantity != 0;


-- =============================================================================
-- ACCOUNTS SCHEMA
-- =============================================================================

CREATE TABLE accounts (
    account_id VARCHAR(50) PRIMARY KEY,
    account_name VARCHAR(100) NOT NULL,
    account_type VARCHAR(20) NOT NULL CHECK (account_type IN ('CASH', 'MARGIN', 'PROP')),
    
    -- Balances
    cash_balance DECIMAL(18,2) NOT NULL DEFAULT 0,
    buying_power DECIMAL(18,2),
    margin_used DECIMAL(18,2) DEFAULT 0,
    
    -- Risk limits
    max_position_value DECIMAL(18,2),
    max_daily_loss DECIMAL(18,2),
    max_order_value DECIMAL(18,2),
    
    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'SUSPENDED', 'CLOSED')),
    
    -- Ownership
    owner VARCHAR(100),
    
    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Metadata
    metadata JSONB
);

-- Trigger to update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
   NEW.updated_at = NOW();
   RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_accounts_updated_at BEFORE UPDATE ON accounts
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
\`\`\`

---

## TimescaleDB for Market Data

\`\`\`sql
-- =============================================================================
-- MARKET DATA SCHEMA (TimescaleDB)
-- =============================================================================

-- Raw tick data
CREATE TABLE market_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    price DECIMAL(18,4) NOT NULL,
    quantity DECIMAL(18,4) NOT NULL,
    side VARCHAR(4),  -- BID, ASK, or NULL for trade
    exchange VARCHAR(10),
    sequence_number BIGINT,
    metadata JSONB
);

-- Convert to hypertable (TimescaleDB-specific)
SELECT create_hypertable('market_data', 'time', chunk_time_interval => INTERVAL '1 day');

-- Create indexes
CREATE INDEX idx_market_data_symbol_time ON market_data (symbol, time DESC);
CREATE INDEX idx_market_data_exchange ON market_data (exchange, time DESC);

-- Compression (saves 90% storage)
ALTER TABLE market_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol, exchange',
    timescaledb.compress_orderby = 'time DESC'
);

-- Compress chunks older than 1 day
SELECT add_compression_policy('market_data', INTERVAL '1 day');

-- Retention policy (drop data older than 1 year)
SELECT add_retention_policy('market_data', INTERVAL '1 year');


-- =============================================================================
-- CONTINUOUS AGGREGATES (Pre-computed OHLCV)
-- =============================================================================

-- 1-minute OHLCV
CREATE MATERIALIZED VIEW market_data_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    symbol,
    exchange,
    first(price, time) AS open,
    max(price) AS high,
    min(price) AS low,
    last(price, time) AS close,
    sum(quantity) AS volume,
    count(*) AS tick_count
FROM market_data
WHERE side IS NULL  -- Only trades, not quotes
GROUP BY bucket, symbol, exchange;

-- Refresh policy (refresh every 1 minute)
SELECT add_continuous_aggregate_policy('market_data_1min',
    start_offset => INTERVAL '2 minutes',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute');

-- 5-minute OHLCV
CREATE MATERIALIZED VIEW market_data_5min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('5 minutes', time) AS bucket,
    symbol,
    exchange,
    first(price, time) AS open,
    max(price) AS high,
    min(price) AS low,
    last(price, time) AS close,
    sum(quantity) AS volume,
    count(*) AS tick_count
FROM market_data
WHERE side IS NULL
GROUP BY bucket, symbol, exchange;

-- 1-hour OHLCV
CREATE MATERIALIZED VIEW market_data_1hour
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    symbol,
    exchange,
    first(price, time) AS open,
    max(price) AS high,
    min(price) AS low,
    last(price, time) AS close,
    sum(quantity) AS volume,
    count(*) AS tick_count
FROM market_data
WHERE side IS NULL
GROUP BY bucket, symbol, exchange;

-- =============================================================================
-- USEFUL QUERIES
-- =============================================================================

-- Get last 1 hour of AAPL ticks
SELECT * FROM market_data
WHERE symbol = 'AAPL'
  AND time >= NOW() - INTERVAL '1 hour'
ORDER BY time DESC;

-- Get 1-minute OHLCV for today
SELECT * FROM market_data_1min
WHERE symbol = 'AAPL'
  AND bucket >= DATE_TRUNC('day', NOW())
ORDER BY bucket DESC;

-- Calculate VWAP over last hour
SELECT 
    symbol,
    SUM(price * quantity) / SUM(quantity) AS vwap
FROM market_data
WHERE symbol = 'AAPL'
  AND time >= NOW() - INTERVAL '1 hour'
  AND side IS NULL
GROUP BY symbol;

-- Get order book snapshot (latest bid/ask)
WITH latest_quotes AS (
    SELECT DISTINCT ON (symbol, side)
        symbol,
        side,
        price,
        quantity,
        time
    FROM market_data
    WHERE symbol = 'AAPL'
      AND side IS NOT NULL
      AND time >= NOW() - INTERVAL '10 seconds'
    ORDER BY symbol, side, time DESC
)
SELECT 
    MAX(CASE WHEN side = 'BID' THEN price END) AS bid_price,
    MAX(CASE WHEN side = 'BID' THEN quantity END) AS bid_size,
    MAX(CASE WHEN side = 'ASK' THEN price END) AS ask_price,
    MAX(CASE WHEN side = 'ASK' THEN quantity END) AS ask_size
FROM latest_quotes;
\`\`\`

---

## Redis for Real-Time Data

\`\`\`python
"""
Redis for Real-Time Position Tracking

Benefits:
- In-memory: Microsecond latency
- Simple data structures: Hashes, sorted sets
- Pub/Sub: Real-time updates
- Persistence: Optional RDB/AOF

Use cases:
- Current positions (microsecond lookups)
- Order book snapshots
- Real-time P&L
- Active orders cache
"""

import redis
from redis import RedisCluster
import json
from typing import Dict, Optional, List
from decimal import Decimal
from datetime import datetime

class RedisPositionTracker:
    """
    Real-time position tracking in Redis
    
    Data structure:
    - Hash: pos:{account}:{symbol} → {quantity, avg_cost, unrealized_pnl, updated_at}
    - Sorted Set: pos:{account}:symbols → {symbol: timestamp}
    - Pub/Sub: pos:updates → {account, symbol, position}
    """
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        self.redis = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )
        
        # Test connection
        self.redis.ping()
        print("[Redis] Connected to Redis")
    
    def set_position(
        self,
        account: str,
        symbol: str,
        quantity: Decimal,
        avg_cost: Decimal,
        unrealized_pnl: Decimal = Decimal('0')
    ):
        """
        Set/update position
        
        Stored as hash for atomic updates
        """
        key = f"pos:{account}:{symbol}"
        
        position = {
            'symbol': symbol,
            'account': account,
            'quantity': str(quantity),
            'avg_cost': str(avg_cost),
            'unrealized_pnl': str(unrealized_pnl),
            'updated_at': datetime.utcnow().isoformat()
        }
        
        # Store as hash
        self.redis.hset(key, mapping=position)
        
        # Add to sorted set (for quick listing)
        self.redis.zadd(
            f"pos:{account}:symbols",
            {symbol: datetime.utcnow().timestamp()}
        )
        
        # Publish update
        self.redis.publish('pos:updates', json.dumps(position))
        
        print(f"[Redis] Set position: {account} {symbol} {quantity}")
    
    def get_position(
        self,
        account: str,
        symbol: str
    ) -> Optional[Dict]:
        """Get position (<1ms)"""
        key = f"pos:{account}:{symbol}"
        
        position = self.redis.hgetall(key)
        
        if position:
            return {
                'symbol': position['symbol'],
                'account': position['account'],
                'quantity': Decimal(position['quantity']),
                'avg_cost': Decimal(position['avg_cost']),
                'unrealized_pnl': Decimal(position['unrealized_pnl']),
                'updated_at': position['updated_at']
            }
        
        return None
    
    def get_all_positions(self, account: str) -> List[Dict]:
        """Get all positions for account"""
        # Get symbols from sorted set
        symbols = self.redis.zrange(f"pos:{account}:symbols", 0, -1)
        
        positions = []
        for symbol in symbols:
            position = self.get_position(account, symbol)
            if position:
                positions.append(position)
        
        return positions
    
    def update_unrealized_pnl(
        self,
        account: str,
        symbol: str,
        current_price: Decimal
    ):
        """
        Update unrealized P&L based on current price
        
        Uses atomic operation (HINCRBY not needed, HSET is atomic)
        """
        position = self.get_position(account, symbol)
        if not position:
            return
        
        quantity = position['quantity']
        avg_cost = position['avg_cost']
        
        unrealized_pnl = quantity * (current_price - avg_cost)
        
        # Update P&L
        self.redis.hset(
            f"pos:{account}:{symbol}",
            'unrealized_pnl',
            str(unrealized_pnl)
        )
    
    def delete_position(self, account: str, symbol: str):
        """Delete closed position"""
        key = f"pos:{account}:{symbol}"
        
        # Delete hash
        self.redis.delete(key)
        
        # Remove from sorted set
        self.redis.zrem(f"pos:{account}:symbols", symbol)
        
        print(f"[Redis] Deleted position: {account} {symbol}")
    
    def subscribe_to_updates(self, callback):
        """Subscribe to position updates (pub/sub)"""
        pubsub = self.redis.pubsub()
        pubsub.subscribe('pos:updates')
        
        print("[Redis] Subscribed to position updates")
        
        for message in pubsub.listen():
            if message['type'] == 'message':
                position = json.loads(message['data'])
                callback(position)


class RedisOrderBookCache:
    """
    Cache order book snapshots in Redis
    
    Data structure:
    - Sorted Set: ob:{symbol}:bids → {price: quantity}
    - Sorted Set: ob:{symbol}:asks → {price: quantity}
    - Expire: 60 seconds (refresh from source)
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    def set_order_book(
        self,
        symbol: str,
        bids: List[tuple[Decimal, int]],
        asks: List[tuple[Decimal, int]]
    ):
        """Set order book snapshot"""
        
        # Clear old data
        self.redis.delete(f"ob:{symbol}:bids", f"ob:{symbol}:asks")
        
        # Store bids (sorted by price descending)
        if bids:
            bid_dict = {str(price): quantity for price, quantity in bids}
            self.redis.zadd(f"ob:{symbol}:bids", bid_dict)
        
        # Store asks (sorted by price ascending)
        if asks:
            ask_dict = {str(price): quantity for price, quantity in asks}
            self.redis.zadd(f"ob:{symbol}:asks", ask_dict)
        
        # Set expiration (60 seconds)
        self.redis.expire(f"ob:{symbol}:bids", 60)
        self.redis.expire(f"ob:{symbol}:asks", 60)
    
    def get_best_bid_ask(self, symbol: str) -> tuple[Optional[Decimal], Optional[Decimal]]:
        """Get best bid and ask (NBBO)"""
        
        # Best bid (highest price)
        best_bid = self.redis.zrange(
            f"ob:{symbol}:bids",
            -1, -1,  # Last element (highest)
            withscores=False
        )
        
        # Best ask (lowest price)
        best_ask = self.redis.zrange(
            f"ob:{symbol}:asks",
            0, 0,  # First element (lowest)
            withscores=False
        )
        
        bid_price = Decimal(best_bid[0]) if best_bid else None
        ask_price = Decimal(best_ask[0]) if best_ask else None
        
        return (bid_price, ask_price)


# Example usage
def redis_example():
    """Demonstrate Redis real-time tracking"""
    
    tracker = RedisPositionTracker()
    
    print("=" * 80)
    print("REDIS REAL-TIME POSITION TRACKING")
    print("=" * 80)
    
    # Set positions
    print("\\nSetting positions...")
    tracker.set_position('ACC-001', 'AAPL', Decimal('100'), Decimal('150.00'))
    tracker.set_position('ACC-001', 'GOOGL', Decimal('50'), Decimal('140.00'))
    tracker.set_position('ACC-001', 'TSLA', Decimal('-200'), Decimal('250.00'))  # Short
    
    # Get position
    print("\\nGetting AAPL position...")
    position = tracker.get_position('ACC-001', 'AAPL')
    print(f"  Quantity: {position['quantity']}, Avg Cost: ${position['avg_cost']}")
    
    # Update P&L
    print("\\nUpdating unrealized P&L...")
    tracker.update_unrealized_pnl('ACC-001', 'AAPL', Decimal('152.00'))
    
    position = tracker.get_position('ACC-001', 'AAPL')
    print(f"  Unrealized P&L: ${position['unrealized_pnl']}")
    
    # Get all positions
    print("\\nAll positions for ACC-001:")
    positions = tracker.get_all_positions('ACC-001')
    for pos in positions:
        print(f"  {pos['symbol']}: {pos['quantity']} @ ${pos['avg_cost']}")

# redis_example()
\`\`\`

---

## Summary

**Database Selection by Use Case:**

| Data Type | Database | Latency | Throughput | Persistence | ACID |
|-----------|----------|---------|------------|-------------|------|
| Orders, Fills | PostgreSQL | ~10ms | 10K/s | Yes | Yes |
| Positions | PostgreSQL | ~10ms | 10K/s | Yes | Yes |
| Market Data (tick) | TimescaleDB | ~5ms write | 1M/s | Yes | No |
| Real-time Positions | Redis | <1ms | 1M/s | Optional | No |
| Order Book Cache | Redis | <1ms | 1M/s | No | No |
| Historical Analytics | ClickHouse | ~100ms | 10M rows/s | Yes | No |

**Production Best Practices:**
1. **PostgreSQL**: Use for transactional data (orders, positions, accounts)
2. **TimescaleDB**: Use for time-series data (market data, P&L history)
3. **Redis**: Use for real-time caching (positions, order books)
4. **Replication**: Master-slave for read scalability
5. **Partitioning**: Partition orders/fills by date (monthly)
6. **Archival**: Move old data to cold storage (S3, Glacier)
7. **Monitoring**: Track query latency, connection pool, slow queries
8. **Backup**: Continuous backup with point-in-time recovery

**Next Section**: Module 14.12 - System Monitoring and Alerting
`,
};
