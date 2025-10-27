export const equityMarketsDeepDiveQuiz = [
  {
    id: 'fm-1-1-q-1',
    question:
      "You're building a stock trading platform for retail investors. The product manager wants to show users 'the current stock price' prominently on each stock page. Explain why showing a single 'current price' can be misleading, and design a better UI that accurately represents how equity markets work. What data should you display, and how should you explain bid-ask spreads to non-technical users?",
    sampleAnswer: `**Why a Single "Current Price" is Misleading:**

There is no single "current price" in equity markets. At any moment, there are:
- **Best Bid**: Highest price buyers are willing to pay
- **Best Ask**: Lowest price sellers are willing to accept
- **Last Trade**: Price of the most recent transaction
- **Mid-Price**: (Bid + Ask) / 2

Showing only "last trade price" misleads users because:
- They can't buy at that price (must pay the ask)
- They can't sell at that price (must hit the bid)
- For illiquid stocks, last trade could be hours old
- Bid-ask spread represents real trading cost

**Better UI Design:**

Display format:
\`\`\`
AAPL - Apple Inc.
Bid: $180.20 x 500  →  Ask: $180.25 x 800  (Spread: $0.05 / 2.8 bps)
Last Trade: $180.22 at 2:45:32 PM ET

If you BUY now:  ~$180.25 (you pay the ask)
If you SELL now: ~$180.20 (you receive the bid)
\`\`\`

**For Non-Technical Users:**

"Think of it like buying concert tickets:
- **Bid** = highest price someone will buy your ticket for
- **Ask** = lowest price someone will sell for
- **Spread** = the difference (your cost to trade)

When you buy, you pay the Ask (seller's price).
When you sell, you get the Bid (buyer's price).

Tight spread (1 cent) = very liquid, easy to trade
Wide spread (50 cents) = less liquid, costs more"

**Implementation:**

\`\`\`python
# WebSocket real-time quotes
async def stream_quotes (ticker: str):
    quote = await get_latest_quote (ticker)
    return {
        'bid': quote.bid,
        'ask': quote.ask,
        'spread_bps': ((quote.ask - quote.bid) / quote.mid) * 10000,
        'last': quote.last,
        'warning': 'Wide spread!' if spread_bps > 50 else None
    }
\`\`\`

**Key Points:**
- Always show bid AND ask, not just last
- Explain spread as trading cost
- Warn users about wide spreads (illiquid)
- Use analogies (concert tickets) for clarity
- Show what THEY will pay/receive, not abstract "price"`,
    keyPoints: [
      'No single "current price" - bid, ask, last, and mid all differ',
      'Users pay ASK when buying, receive BID when selling',
      'Spread is real cost - tight = liquid, wide = illiquid',
      'Use simple analogies (concert tickets) to explain',
      'Always display what user will actually pay/receive',
    ],
  },
  {
    id: 'fm-1-1-q-2',
    question:
      'The Efficient Market Hypothesis (EMH) states that stock prices reflect all available information. However, quantitative hedge funds consistently profit from market inefficiencies. Analyze: (1) Which form of EMH (weak/semi-strong/strong) is most defensible? (2) What types of inefficiencies can quant strategies exploit? (3) As markets become more efficient (more quants), where do future alpha sources lie?',
    sampleAnswer: `**Which Form of EMH is Most Defensible:**

**Weak Form (prices reflect past prices):**
- MOSTLY true for liquid stocks
- Simple momentum strategies rarely work anymore
- BUT: Order flow momentum (microstructure) still exploitable
- Verdict: Defensible for major stocks, fails for small-cap/illiquid

**Semi-Strong Form (prices reflect all public info):**
- PARTIALLY true - major news gets priced in seconds/minutes
- BUT: Complex information (10-K filings, satellite data) takes time to process
- Earnings surprise still moves stocks (info processing takes time)
- Verdict: Mostly holds for simple info, fails for complex data

**Strong Form (prices reflect ALL info including private):**
- CLEARLY false - insider trading profits prove this
- Corporate insiders outperform market consistently
- M&A leaks show private info not priced in
- Verdict: Indefensible

**Most Defensible: Semi-strong for liquid stocks, weak for illiquid**

**Exploitable Inefficiencies for Quants:**

\`\`\`python
class MarketInefficiencies:
    """Types of alpha sources for quants"""
    
    inefficiencies = {
        'Microstructure': {
            'type': 'Order flow patterns, toxic flow detection',
            'timeframe': 'Milliseconds to seconds',
            'why_persists': 'Requires speed, infrastructure ($$$)',
            'example': 'Large orders create temporary imbalance'
        },
        'Behavioral': {
            'type': 'Overreaction, underreaction, momentum',
            'timeframe': 'Days to weeks',
            'why_persists': 'Human psychology unchanged',
            'example': 'Earnings surprise → drift for 60 days'
        },
        'Structural': {
            'type': 'Index rebalancing, forced flows',
            'timeframe': 'Days',
            'why_persists': 'Institutional mandates',
            'example': 'S&P inclusion → forced buying'
        },
        'Information Processing': {
            'type': 'Alternative data, NLP on filings',
            'timeframe': 'Hours to days',
            'why_persists': 'Requires tech + domain expertise',
            'example': 'Satellite images of parking lots → retail sales'
        },
        'Statistical Arbitrage': {
            'type': 'Mean reversion, pairs trading',
            'timeframe': 'Days to weeks',
            'why_persists': 'Capacity constrained',
            'example': 'Correlated stocks diverge → converge'
        }
    }
\`\`\`

**Why These Persist:**1. **Speed barriers**: HFT requires millions in infrastructure
2. **Capacity limits**: StatArb works until too much capital chases it
3. **Complexity**: Alternative data requires ML + domain knowledge
4. **Risk**: Arbitrage requires capital, leverage, risk appetite
5. **Behavioral**: Human psychology doesn't change

**Future Alpha Sources (As Markets Efficiency Increases):**

**Near Future (2025-2030):**
- Alternative data (satellite, credit card, social media)
- NLP on unstructured data (earnings calls, analyst reports)
- Cross-asset arbitrage (equity-credit, equity-options)
- Microstructure at millisecond level

**Long-term (2030+):**
- AI-generated fundamental insights (deep learning on 10-Ks)
- Behavioral finance 2.0 (neuro-economics)
- Crypto-equity arbitrage (DeFi inefficiencies)
- Quantum computing arbitrage (solve optimization faster)

**The Paradox:**
As markets become more efficient:
- Alpha gets harder to find (good)
- But new data sources emerge (good for tech-savvy)
- Winner-take-all: Best tech + data wins

**Conclusion:**
Semi-strong EMH mostly holds for simple public info on liquid stocks. But complex info processing, alternative data, and microstructure create persistent inefficiencies. Future alpha = technology + unique data + speed, not clever ideas alone.`,
    keyPoints: [
      'Semi-strong EMH mostly defensible for liquid stocks, simple information',
      'Inefficiencies persist due to speed barriers, complexity, behavioral biases',
      'Microstructure (milliseconds), alternative data, structural flows still exploitable',
      'Future alpha: AI on complex data, alternative data, technology advantage',
      'As efficiency increases, alpha shifts to tech-enabled strategies',
    ],
  },
  {
    id: 'fm-1-1-q-3',
    question:
      'Design a real-time stock screener that scans 3000+ US stocks for anomalies (unusual volume spikes, price breakouts, liquidity changes). Architecture requirements: (1) Handle 100K+ quotes/sec, (2) Sub-second latency for alerts, (3) Scale to institutional users. Include data ingestion, processing pipeline, storage, and alert delivery.',
    sampleAnswer: `**Architecture Design for Real-Time Stock Screener:**

**System Requirements:**
- Input: 100K+ quotes/second (3000 stocks × 30-50 quotes/sec each)
- Latency: <1 second from quote to alert
- Scale: 1000+ concurrent users
- Storage: Real-time + historical (3 months)
- Alerts: Email, SMS, WebSocket push

**Architecture:**

\`\`\`python
"""
Data Flow:
Exchange Feeds → Load Balancer → Ingestion Layer → Stream Processing → 
Storage Layer → Alert Engine → Delivery (WebSocket/SMS/Email)
"""

# 1. Data Ingestion Layer
class MarketDataIngestion:
    """
    Connect to exchanges via FIX/WebSocket
    Normalize data from multiple sources
    """
    def __init__(self):
        self.feeds = []  # NYSE, Nasdaq, IEX, etc.
        self.normalizer = DataNormalizer()
        self.publisher = KafkaPublisher()
    
    async def ingest_feed (self, exchange: str):
        """Ingest from single exchange"""
        async for quote in self.connect_exchange (exchange):
            # Normalize to common format
            normalized = self.normalizer.normalize (quote)
            
            # Publish to Kafka topic
            await self.publisher.publish(
                topic=f'quotes.{normalized.symbol}',
                message=normalized
            )

# 2. Stream Processing (Apache Flink or Kafka Streams)
class AnomalyDetector:
    """
    Real-time anomaly detection
    
    Uses sliding windows to calculate:
    - Volume spikes (current vs 20-day avg)
    - Price breakouts (current vs 52-week high/low)
    - Liquidity changes (spread widening)
    """
    
    def process_quote (self, quote, historical_stats):
        """
        Process single quote for anomalies
        """
        anomalies = []
        
        # Check 1: Volume spike
        avg_volume = historical_stats.avg_volume_20d
        if quote.volume > avg_volume * 3:  # 3x normal volume
            anomalies.append({
                'type': 'VOLUME_SPIKE',
                'severity': 'HIGH',
                'details': f'Volume {quote.volume / avg_volume:.1f}x normal'
            })
        
        # Check 2: Price breakout
        if quote.price > historical_stats.high_52w:
            anomalies.append({
                'type': 'BREAKOUT_HIGH',
                'severity': 'MEDIUM',
                'details': f'New 52-week high: \${quote.price:.2f}'
            })
        
        # Check 3: Spread widening (liquidity concern)
spread_bps = (quote.ask - quote.bid) / quote.mid * 10000
if spread_bps > historical_stats.avg_spread_bps * 2:
    anomalies.append({
        'type': 'LIQUIDITY_CONCERN',
        'severity': 'MEDIUM',
        'details': f'Spread {spread_bps:.0f} bps (2x normal)'
            })

return anomalies

# 3. Storage Layer(TimescaleDB + Redis)
class StorageLayer:
"""
    Hot data: Redis (last quote, last 1hr ticks)
    Warm data: TimescaleDB(last 3 months)
    Cold data: S3(historical archive)
"""
    def __init__(self):
self.redis = RedisClient()
self.timescale = TimescaleDB()
self.s3 = S3Client()
    
    async def store_quote (self, quote):
"""Multi-tier storage"""
        # Hot: Redis (instant access)
await self.redis.set(
    f'quote:{quote.symbol}:latest',
    quote,
    ttl = 3600  # 1 hour
)
        
        # Warm: TimescaleDB(queryable)
await self.timescale.insert(
    table = 'quotes',
    data = quote
)
        
        # Update rolling stats
await self.update_rolling_stats (quote)
    
    async def update_rolling_stats (self, quote):
"""
        Maintain rolling statistics in Redis
    - 20 - day avg volume
        - 52 - week high / low
            - Average spread
"""
await self.redis.update_rolling_avg(
    key = f'stats:{quote.symbol}:volume_20d',
    value = quote.volume,
    window = 20 * 6.5 * 3600  # 20 trading days
)

# 4. Alert Engine
class AlertEngine:
"""
    Match anomalies to user subscriptions
    Deliver via multiple channels
"""
    def __init__(self):
self.subscriptions = {}  # user_id -> filters
self.delivery = AlertDelivery()
    
    async def process_anomaly (self, symbol, anomalies):
"""
        Find matching subscriptions and deliver
"""
        # Find users subscribed to this symbol + anomaly type
matching_users = self.find_subscribers (symbol, anomalies)

for user in matching_users:
    alert = self.format_alert (symbol, anomalies, user.preferences)
            
            # Deliver based on user preference
await self.delivery.send(
    user_id = user.id,
    channels = user.alert_channels,  #['websocket', 'email']
                alert = alert
)

# 5. WebSocket Server (for real - time push to UI)
class WebSocketAlertServer:
"""
    Push alerts to connected clients
"""
    async def on_connect (self, websocket, user_id):
"""User connects to alert stream"""
self.active_connections[user_id] = websocket
        
        # Subscribe to user's alert topic
await self.kafka_consumer.subscribe(
    topic = f'alerts.{user_id}'
)
    
    async def stream_alerts (self, user_id):
"""Push alerts as they arrive"""
        async for alert in self.kafka_consumer.consume (f'alerts.{user_id}'):
    await self.active_connections[user_id].send_json (alert)

# 6. API Layer (for querying historical data)
    @app.get('/api/screener/results')
    async def get_screener_results(
        symbol: Optional[str],
        anomaly_type: Optional[str],
        start_time: datetime,
        end_time: datetime
    ):
"""
    Query historical anomalies
"""
results = await timescaledb.query(
    '''
        SELECT * FROM anomalies
        WHERE timestamp BETWEEN % s AND % s
        AND($1:: text IS NULL OR symbol = $1)
        AND($2:: text IS NULL OR anomaly_type = $2)
        ORDER BY timestamp DESC
        LIMIT 100
        ''',
        start_time, end_time, symbol, anomaly_type
)
return results
\`\`\`

**Technology Stack:**

**Ingestion:**
- Apache Kafka (message queue, 1M+ msgs/sec throughput)
- Kafka Connect (FIX/WebSocket adapters)

**Processing:**
- Apache Flink (stream processing, sub-second latency)
- Flink SQL (declarative anomaly rules)

**Storage:**
- Redis (hot data, <1ms reads)
- TimescaleDB (warm data, time-series optimized PostgreSQL)
- S3 (cold archival)

**Delivery:**
- WebSocket (Socket.io for browser push)
- Twilio (SMS alerts)
- SendGrid (email)

**Infrastructure:**
- Kubernetes (container orchestration)
- AWS EKS (managed Kubernetes)
- Prometheus + Grafana (monitoring)

**Performance Optimizations:**1. **Partitioning**: Kafka topics by symbol (parallel processing)
2. **Caching**: Redis for hot data (avoid DB queries)
3. **Compression**: Protobuf for wire format (smaller than JSON)
4. **Batch Processing**: Group alerts to avoid spam
5. **Load Balancing**: Multiple Flink workers (horizontal scale)

**Scaling:**

- **Horizontal**: Add more Kafka partitions + Flink workers
- **Vertical**: Upgrade to r6i instances (128GB RAM) for in-memory processing
- **Expected**: 100K quotes/sec = 10 Kafka brokers + 20 Flink workers

**Cost Estimate:**
- AWS infrastructure: $5K-10K/month (for 3000 stocks, 1000 users)
- Market data: $10K-50K/month (direct exchange feeds)
- Total: $15K-60K/month

**Key Decisions:**
- Kafka (not RabbitMQ): Better for high-throughput streams
- Flink (not Spark): Lower latency for real-time processing
- TimescaleDB (not Cassandra): Better for time-series queries
- Redis (not Memcached): Richer data structures (sorted sets for rolling stats)

This architecture handles 100K+ quotes/sec with <1s latency and scales horizontally.`,
    keyPoints: [
      'Kafka for ingestion (1M+ msgs/sec), Flink for stream processing (<1s latency)',
      'Multi-tier storage: Redis (hot), TimescaleDB (warm), S3 (cold)',
      'Real-time anomaly detection: volume spikes, breakouts, spread widening',
      'WebSocket push alerts, SMS/email for critical events',
      'Horizontal scaling: partition by symbol, multiple workers',
    ],
  },
];
