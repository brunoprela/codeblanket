export const dataNormalizationQuiz = [
  {
    id: 'data-normalization-q-1',
    question: 'Design a multi-exchange normalization system for AAPL trading on NASDAQ, NYSE, and ARCA. Handle: (1) Different price formats (NASDAQ uses 1/10000ths, NYSE uses decimals), (2) Size conventions (NYSE quotes in round lots), (3) Calculate NBBO in real-time, (4) Handle timestamp skew between exchanges. Provide architecture and Python implementation.',
    sampleAnswer: 'Multi-exchange normalization: (1) Define unified NormalizedQuote format with Decimal prices and int sizes. (2) Create exchange-specific normalizers (NASDAQ: divide by 10000, NYSE: multiply lots by 100). (3) NBBOCalculator maintains latest quote from each exchange, finds max bid and min ask. (4) Timestamp handling: Use exchange_timestamp for ordering, not receive_timestamp. Handle clock skew by allowing 100ms tolerance window. Architecture: Raw feeds → Exchange normalizers → Unified quotes → NBBO calculator → Strategies. Performance: Process 100K quotes/sec with < 5μs normalization latency per quote.',
    keyPoints: [
      'Unified format: NormalizedQuote with Decimal prices, int shares, datetime timestamps',
      'Exchange-specific: NASDAQ÷10000 for prices, NYSE×100 for sizes (round lots→shares)',
      'NBBO: Track latest quote per exchange, best bid=max(all bids), best ask=min(all asks)',
      'Timestamp skew: Use exchange timestamp, allow 100ms tolerance for out-of-order quotes',
      'Performance: 100K quotes/sec, < 5μs normalization, strict ordering per symbol',
    ],
  },
  {
    id: 'data-normalization-q-2',
    question: 'AAPL had a 4:1 stock split on August 31, 2020. Pre-split price was $400. Explain: (1) How to adjust historical prices, (2) Impact on volume, (3) Handling options contracts, (4) Backtesting implications. Calculate adjusted prices and volumes.',
    sampleAnswer: 'Stock split adjustment: (1) Divide historical prices by split ratio: $400 / 4 = $100 adjusted. All pre-split prices divided by 4. (2) Multiply historical volume by ratio: 1M shares pre-split × 4 = 4M adjusted. (3) Options: Adjust strike (400 call → 100 call) and multiply contracts by ratio (1 contract = 400 shares post-split). (4) Backtesting: MUST adjust historical data or strategies will see artificial 75% drop on split date. Always use adjusted prices for indicators (MA, RSI). Unadjusted prices only for visual charts to show actual traded prices.',
    keyPoints: [
      'Price adjustment: Divide by ratio ($400 / 4 = $100), apply to all pre-split prices',
      'Volume adjustment: Multiply by ratio (1M shares × 4 = 4M), maintain dollar volume',
      'Options: Adjust strike ($400 → $100) and contract size (1 → 4 contracts)',
      'Backtesting: Use adjusted prices for indicators, or strategy sees false 75% crash',
      'Dollar volume preserved: $400 × 1M = $100 × 4M = $400M (no change)',
    ],
  },
  {
    id: 'data-normalization-q-3',
    question: 'You receive quotes from 3 exchanges with timestamps: NASDAQ 10:30:00.123456, NYSE 10:30:00.125789 (+2.3ms), ARCA 10:30:00.121000 (-2.4ms). Clocks are not synchronized. How to: (1) Detect clock skew, (2) Order quotes correctly, (3) Calculate accurate NBBO, (4) Handle in production?',
    sampleAnswer: 'Clock skew handling: (1) Detection: Compare exchange timestamps to atomic reference (GPS, PTP). If NYSE consistently 2ms ahead, that\'s clock skew. Monitor drift over time. (2) Ordering: Don\'t trust absolute timestamps. Use sequence numbers per exchange (monotonic ordering). For cross-exchange ordering, apply known offsets (NYSE-2ms). (3) NBBO: Calculate per-exchange BBO first, then combine. Don\'t mix timestamps across exchanges. (4) Production: Sync system clock with NTP/PTP. Log timestamp discrepancies. Alert if skew > 10ms. Use sequence numbers as primary ordering mechanism.',
    keyPoints: [
      'Clock skew: Exchanges have unsynchronized clocks (±10ms typical), detect by comparison to GPS time',
      'Ordering: Use sequence numbers per exchange (monotonic), not absolute timestamps across exchanges',
      'NBBO: Calculate per-exchange first, then combine (don\'t directly compare cross-exchange timestamps)',
      'Production: Sync with NTP (<1ms accuracy) or PTP (<1μs), log discrepancies, alert if >10ms',
      'Sequence numbers: Primary ordering mechanism, timestamps secondary for approximate timing',
    ],
  },
];
