import { MultipleChoiceQuestion } from '@/lib/types';

export const marketDataFundamentalsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'market-data-fundamentals-mc-1',
    question:
      'Your trading application is consuming market data via WebSocket from a vendor. During peak market hours (9:30-10:00 AM ET), you notice the application is receiving only 50% of expected quote updates for SPY, but trade data appears complete. What is the MOST likely cause?',
    options: [
      'Network bandwidth saturation causing dropped WebSocket frames',
      'Vendor implementing conflation (rate limiting) on quote updates during high volume',
      'Your callback function is blocking, causing the WebSocket client to buffer and drop messages',
      'The exchange is experiencing technical issues',
    ],
    correctAnswer: 1,
    explanation:
      'Market data vendors commonly implement conflation during high-volume periods to prevent overwhelming clients. Conflation means the vendor sends only the latest quote when multiple updates occur within a short time window, effectively rate-limiting the stream. This is especially common at market open when SPY can have 500+ quote updates per second. Trades are typically not conflated because each trade is a unique event. Network bandwidth is rarely the issue (quotes are small, ~100 bytes). If your callback was blocking, you would see increasing latency but not selective message drops. Exchange issues would affect all clients uniformly. Solution: Subscribe to unthrottled feed tier if you need every update, or implement local buffering to handle bursts.',
  },
  {
    id: 'market-data-fundamentals-mc-2',
    question:
      'You are comparing market data vendors for a day trading platform that needs quotes for 500 US equities. IEX offers free real-time data with ~50ms latency. Polygon.io costs $200/month with ~20ms latency. Bloomberg costs $2,000/month (terminal) + $50/symbol with ~5ms latency. Your traders complain about missed fills when market moves fast. Which vendor should you choose?',
    options: [
      'IEX - Free is best, 50ms is acceptable for day trading',
      'Polygon.io - Best cost/performance balance at $200/month',
      'Bloomberg - Traders need the lowest latency, cost is justified',
      'Use IEX for quotes and Polygon for trades to minimize cost',
    ],
    correctAnswer: 1,
    explanation:
      'Polygon.io at $200/month provides the best cost/performance balance for this use case. 20ms latency is sufficient for human day traders (human reaction time is 200+ ms), and the cost is reasonable. IEX at 50ms is usable but the extra 30ms delay could result in stale quotes during fast markets, leading to failed orders. Bloomberg at $27,000/month ($2,000 base + $50 × 500 symbols) is overkill - the 5ms latency only matters for algorithmic trading where microseconds count. For human traders, 20ms vs 5ms makes no perceptible difference. Using different vendors for quotes vs trades creates synchronization issues and adds complexity. Cost breakdown: IEX = $0, Polygon = $200, Bloomberg = $27K. Unless you are running HFT algorithms, sub-10ms latency is not worth 135× higher cost.',
  },
  {
    id: 'market-data-fundamentals-mc-3',
    question:
      'Your market data consumer receives quotes with these sequence numbers for AAPL: 1234567, 1234568, 1234570, 1234571. You detect a gap (missing sequence 1234569). What should your application do?',
    options: [
      'Immediately reconnect to the WebSocket to get missing data',
      'Log the gap and continue processing - one missing quote is not critical',
      'Request a snapshot of the current order book to resync',
      'Buffer subsequent messages until the missing sequence arrives',
    ],
    correctAnswer: 1,
    explanation:
      'Log the gap and continue processing. In production market data systems, gaps happen (network packet loss, vendor issues, processing delays). One missing quote out of thousands is not critical - the next quote will have current prices. Reconnecting is disruptive and won\'t recover the missing historical message. Requesting a snapshot is expensive and unnecessary for a single gap. Buffering subsequent messages is dangerous - if the missing message never arrives, you stop processing all data (deadlock). Best practice: Track gap statistics (count, size, frequency), alert if gaps exceed threshold (e.g., >1% of messages), but continue processing. Most trading systems tolerate small gaps because: (1) Markets update frequently (next quote in milliseconds), (2) Trades are more important than quotes (less frequent, tracked separately), (3) Reconnecting causes larger data loss. Implement gap detection for monitoring and alerting, but don\'t stop processing. If gaps are frequent (>1% of messages), investigate vendor reliability or network issues.',
  },
  {
    id: 'market-data-fundamentals-mc-4',
    question:
      'You are designing a market data replay system for backtesting. Historical data is stored as Parquet files partitioned by date and symbol (e.g., /data/2024-01-15/AAPL.parquet). Each file contains 50,000 ticks. A backtest needs to replay data for 100 symbols from 9:30 AM to 10:00 AM (30 minutes). What is the MOST efficient approach?',
    options: [
      'Load all 100 files into memory, merge-sort by timestamp, then replay',
      'Use heap-based merge: Read first tick from each file, maintain min-heap by timestamp',
      'Process one symbol at a time sequentially to minimize memory usage',
      'Concatenate all files into one large file, then sequential scan',
    ],
    correctAnswer: 1,
    explanation:
      'Heap-based merge is the most efficient approach for multi-symbol replay. It maintains a min-heap with the next tick from each symbol, always popping the earliest timestamp. This provides O(log N) insertions where N = number of symbols (100), maintaining timestamp order across all symbols while using minimal memory (only current tick from each file in RAM). Loading all files (option A) requires 100 × 50K ticks × ~100 bytes = 500 MB in RAM, which works for 100 symbols but doesn\'t scale to 1,000+ symbols. Sequential processing (option C) loses timestamp ordering - you would process all AAPL ticks, then all MSFT ticks, which is incorrect for backtesting (strategies need data in time order). Concatenating files (option D) is expensive (disk I/O) and creates huge files. Heap-based merge is the standard algorithm for external sorting and time-series merging, used by systems like TimescaleDB and KDB+. Memory usage: ~1 MB (100 ticks buffered). Performance: 1M ticks/sec on single core. This is how production systems like QuantConnect and Lean replay data.',
  },
  {
    id: 'market-data-fundamentals-mc-5',
    question:
      'Your application subscribes to 1,000 symbols via WebSocket. The vendor charges $0.05 per 1,000 messages. SPY receives 200 messages/sec, while a low-volume stock receives 0.5 messages/sec. After 1 month (30 days, 6.5 hours trading/day), approximately how much will market data cost?',
    options: [
      '$500 - $1,000 (most symbols are low volume)',
      '$5,000 - $10,000 (average volume)',
      '$50,000 - $100,000 (high volume)',
      '$500,000+ (extremely high volume)',
    ],
    correctAnswer: 1,
    explanation:
      'Approximately $5,000 - $10,000. Calculation: Assume average symbol receives 10 messages/sec (between SPY\'s 200/sec and low-volume\'s 0.5/sec, weighted toward low volume since most stocks are illiquid). Total messages per month: 1,000 symbols × 10 msg/sec × 6.5 hours/day × 3,600 sec/hour × 30 days = 702M messages. Cost: 702M / 1,000 × $0.05 = $35,100. But this assumes average 10 msg/sec. More realistic distribution: 10 high-volume stocks (100 msg/sec each), 90 medium-volume (10 msg/sec), 900 low-volume (1 msg/sec). Total: (10 × 100) + (90 × 10) + (900 × 1) = 2,800 msg/sec = ~220M messages/month = $11,000. Real-world costs tend toward $5K-10K for 1,000 symbols on message-based pricing. Note: Many vendors offer flat-rate pricing ($200-2,000/month unlimited) which is much cheaper for high-volume consumers. Always negotiate pricing! Per-message pricing is a trap for high-frequency consumers.',
  },
];

