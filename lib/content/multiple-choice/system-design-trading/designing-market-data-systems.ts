import { MultipleChoiceQuestion } from '@/lib/types';

export const designingMarketDataSystemsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'dmds-mc-1',
      question:
        'For storing 3 years of tick data (1M msgs/sec), which storage strategy minimizes cost while maintaining query performance?',
      options: [
        'Store everything in TimescaleDB on SSD',
        'Store everything in DynamoDB with on-demand pricing',
        'Tiered storage: Hot (TimescaleDB), Warm (ClickHouse), Cold (S3/Parquet)',
        'Store everything in Redis for fastest queries',
      ],
      correctAnswer: 2,
      explanation:
        'Tiered storage minimizes cost: Hot tier (7 days) in TimescaleDB for <10ms queries (\$1K/mo), Warm tier (90 days) in ClickHouse for analytics ($500/mo), Cold tier (3 years) in S3/Parquet for archival ($50/mo). Total ~$1550/mo. Single-tier solutions: TimescaleDB for 3 years = $50K+/mo (too expensive), Redis = $100K+/mo (RAM cost), DynamoDB = $10K+/mo (high write costs). Parquet compression (50:1) makes S3 very cheap.',
    },
    {
      id: 'dmds-mc-2',
      question:
        'What is the primary reason to normalize market data from different exchanges into a common format?',
      options: [
        'To reduce storage requirements through compression',
        'To enable uniform processing logic across all data sources',
        'To improve query performance in the database',
        'To comply with regulatory requirements',
      ],
      correctAnswer: 1,
      explanation:
        "Normalization enables uniform processing: strategies, risk systems, and analytics work with single Tick format regardless of whether data came from Binance (WebSocket/JSON), NYSE (FIX), or Alpha Vantage (REST). Without normalization, you'd need custom parsing logic everywhere. Normalization includes: standardizing timestamps (all to microseconds), symbol names (AAPL vs AAPL.NASDAQ), field names (price vs last_trade_price). Reduces bugs and complexity.",
    },
    {
      id: 'dmds-mc-3',
      question:
        'Why is Kafka commonly used as a message queue between feed handlers and storage in market data systems?',
      options: [
        'Kafka provides the lowest latency (<1ms)',
        'Kafka offers persistence and replay capability for exactly-once processing',
        'Kafka is cheaper than alternatives like RabbitMQ',
        'Kafka automatically normalizes data formats',
      ],
      correctAnswer: 1,
      explanation:
        "Kafka provides durability + replay: messages persisted to disk (configurable retention), consumers can replay from any offset. Critical for market data: if storage fails, can replay from Kafka without losing ticks. Exactly-once semantics via transactions. Alternative (Redis Streams) has lower latency (<1ms vs Kafka ~3-10ms) but less durable. RabbitMQ doesn't support replay well. Kafka doesn't normalize data (feed handlers do) or provide lowest latency (in-memory queues faster).",
    },
    {
      id: 'dmds-mc-4',
      question:
        'When aggregating tick data into OHLCV bars, how should late-arriving ticks be handled?',
      options: [
        'Ignore late ticks completely to maintain consistency',
        'Always update historical bars when late ticks arrive',
        'Use a buffer window (5-10s) and publish revised bars if needed',
        'Automatically adjust timestamps to current bar',
      ],
      correctAnswer: 2,
      explanation:
        'Buffer window + revisions: Keep bars open for 5-10s after bar close time to accept late ticks (common due to network delays). If very late tick arrives (>10s), publish revised bar with is_revision=true flag. Consumers decide whether to use revisions. Ignoring late ticks causes data loss. Always updating historical bars is expensive (requires database updates). Adjusting timestamps is incorrect (falsifies data). Buffer balances accuracy vs latency.',
    },
    {
      id: 'dmds-mc-5',
      question:
        "What is the purpose of storing bar timestamps in the exchange's local timezone rather than UTC?",
      options: [
        'To save storage space by using shorter timestamps',
        'To comply with exchange regulations',
        'To align bars with market hours that are meaningful in local time',
        'To avoid daylight saving time complications',
      ],
      correctAnswer: 2,
      explanation:
        'Market hours meaningful in local time: NYSE opens at 09:30 Eastern Time. Storing bars in ET means 09:30 bar always represents market open, regardless of UTC offset. In UTC, market open is 14:30 (winter) or 13:30 (summer, DST). Aligning to local time makes bar interpretation simpler: strategies can check "if time == market_open" without timezone conversions. DST still needs handling, but less confusing than shifting UTC times. Storage space same (timestamps include timezone offset).',
    },
  ];
