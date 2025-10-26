import { MultipleChoiceQuestion } from '@/lib/types';

export const marketDataStorageMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'market-data-storage-mc-1',
    question: 'TimescaleDB hypertable stores 1 billion ticks. Each chunk is 1 day. How many chunks for 1 year of data?',
    options: ['365 chunks', '12 chunks', '52 chunks', '1 chunk'],
    correctAnswer: 0,
    explanation: 'Hypertables partition data by time intervals called chunks. With chunk_time_interval = 1 day, TimescaleDB creates one chunk per day. 1 year = 365 days = 365 chunks. Each chunk stores all ticks for that day (all symbols). Benefit: Fast queries (only read relevant chunks), efficient compression (compress old chunks), easy deletion (drop old chunks for retention policy). Chunk size affects query performance: too small = overhead, too large = scan waste. 1-day chunks optimal for market data.',
  },
  {
    id: 'market-data-storage-mc-2',
    question: 'QuestDB ingests 4 million rows/second. How many 8-byte price fields can it write per second?',
    options: ['32 MB/s (4M rows × 8 bytes)', '4 GB/s', '500 MB/s', '8 MB/s'],
    correctAnswer: 0,
    explanation: '4M rows/sec × 8 bytes/row = 32 MB/sec write throughput. Complete tick (timestamp 8B + symbol 8B + price 8B + size 4B = 28 bytes) = 4M × 28 = 112 MB/sec. This is why QuestDB is fastest time-series DB - optimized columnar storage, direct memory mapping, SIMD instructions. Comparison: TimescaleDB ~100K rows/sec = 2.8 MB/sec (35× slower). InfluxDB ~500K rows/sec = 14 MB/sec (7× slower). For HFT (millions ticks/sec), QuestDB required.',
  },
  {
    id: 'market-data-storage-mc-3',
    question: 'Your tick database is 50 TB raw. With 20× compression, what is compressed size?',
    options: ['2.5 TB', '1 TB', '10 TB', '5 TB'],
    correctAnswer: 0,
    explanation: '50 TB / 20 = 2.5 TB compressed. TimescaleDB achieves 10-20× compression using columnar compression on old chunks. Technique: (1) Dictionary encoding for symbols (AAPL → 1, GOOGL → 2), (2) Delta encoding for timestamps (store differences, not absolute), (3) Run-length encoding for repeated values. Fresh data (< 7 days): uncompressed for fast writes. Old data (> 7 days): compressed automatically via compression policy. Storage cost: $25/TB/month × 2.5 TB = $62.50/month vs $1,250/month uncompressed (20× savings).',
  },
  {
    id: 'market-data-storage-mc-4',
    question: 'Query: SELECT * FROM ticks WHERE symbol=AAPL AND time >= X takes 30 seconds. Why?',
    options: ['Missing index on (symbol, time)', 'Too much data returned', 'Network bottleneck', 'CPU overload'],
    correctAnswer: 0,
    explanation: 'Missing index forces sequential scan of entire table. With index on (symbol, time DESC), database can: (1) Use index to find AAPL rows instantly (B-tree lookup), (2) Scan only relevant time range. Query time: 30s → 100ms (300× faster). Index cost: ~20% storage overhead. For 1 billion rows, scanning without index reads all 1B rows. With index, reads only matching rows (e.g., 1M AAPL ticks). Always index: (symbol, time) for market data, (user_id, timestamp) for user events, (symbol, exchange, time) if multi-exchange.',
  },
  {
    id: 'market-data-storage-mc-5',
    question: 'TimescaleDB continuous aggregate pre-calculates 1-min bars. How much data reduction for ticks?',
    options: ['~60× reduction (60 ticks → 1 bar)', '10× reduction', '100× reduction', 'No reduction'],
    correctAnswer: 0,
    explanation: 'Continuous aggregate materializes common queries. For 1 tick/second: 60 ticks/minute become 1 bar (OHLC). Reduction: 60 ticks × 40 bytes = 2400 bytes vs 1 bar × 40 bytes = 40 bytes (60× less). Query speedup: Querying 1 day of 1-min bars (1440 bars) vs 86,400 ticks (60× fewer rows). Refresh: Runs automatically in background. Use cases: Dashboard displaying 1-min bars, backtesting on bar data, technical indicators (MA, RSI on bars not ticks). Cost: Extra storage (~2%) for materialized view, worth it for 60× query speedup.',
  },
];
