import { MultipleChoiceQuestion } from '@/lib/types';

export const dataQualityValidationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'data-quality-validation-mc-1',
    question:
      'Quote shows bid=$150.26, ask=$150.24 (bid > ask). What is this called?',
    options: [
      'Crossed market (error condition)',
      'Locked market (bid = ask)',
      'Wide spread',
      'Normal market',
    ],
    correctAnswer: 0,
    explanation:
      'Crossed market: bid > ask, physically impossible (cannot buy for more than you can sell). Causes: Data error, out-of-order updates, exchange malfunction. Action: Reject quote immediately, alert operators, do NOT trade on crossed market (illegal). Locked market (bid = ask): Allowed, occurs across different exchanges. Wide spread: Large difference between bid/ask, legitimate. Regulation NMS prohibits brokers from routing orders to crossed markets. Validators must catch this before strategies see the data.',
  },
  {
    id: 'data-quality-validation-mc-2',
    question:
      'AAPL price jumps from $150 to $1500 in 1ms. Price spike or legitimate?',
    options: [
      'Price spike (10× jump in 1ms impossible)',
      'Legitimate (stock split)',
      'Legitimate (volatility)',
      'Cannot determine',
    ],
    correctAnswer: 0,
    explanation:
      'Price spike detection: 10× move in 1ms = clearly erroneous. Legitimate scenarios: Stock splits take hours (not instant), earnings moves max ~20% (not 1000%), flash crashes ~10% (not 1000%). Detection: Z-score > 5 sigma from rolling mean. Validation: Check other vendors - if all show $150, spike is error. If all show $1500, might be real (but investigate). Response: Reject quote, alert operators, log incident. Never auto-trade on 10× spikes without manual confirmation. Fat finger errors cost millions (Knight Capital $440M loss).',
  },
  {
    id: 'data-quality-validation-mc-3',
    question:
      'Data quality score: Accuracy 99%, Completeness 95%, Timeliness 90%. Overall score (weights: 40%, 30%, 30%)?',
    options: [
      '95.5 (weighted average)',
      '94.7 (simple average)',
      '99 (best component)',
      '90 (worst component)',
    ],
    correctAnswer: 0,
    explanation:
      'Weighted average calculation: (99 × 0.4) + (95 × 0.3) + (90 × 0.3) = 39.6 + 28.5 + 27 = 95.1 ≈ 95.5. Weights reflect importance: Accuracy most critical (40%) - wrong data = wrong decisions. Completeness (30%) - gaps cause missed opportunities. Timeliness (30%) - stale data less valuable. Score interpretation: 95.5 = good quality, continue trading. < 95 = reduce positions. < 90 = pause trading. < 85 = halt all trading. Monitor daily, investigate degradation trends.',
  },
  {
    id: 'data-quality-validation-mc-4',
    question:
      'You process 100K ticks/second with 10μs validation per tick. What is CPU time per second?',
    options: [
      '1 second (100K × 10μs = 1s CPU)',
      '0.1 seconds',
      '10 seconds',
      '100 milliseconds',
    ],
    correctAnswer: 0,
    explanation:
      "CPU calculation: 100,000 ticks/sec × 10μs/tick = 1,000,000μs = 1 second CPU time. At 100% single-core capacity. Solution: (1) Optimize validation (< 5μs target), (2) Parallel processing (use 2 cores = 50% each), (3) GPU acceleration for statistical checks. With 10μs validation, system can't scale beyond 100K ticks/ sec on single core.At 5μs: 200K ticks / sec possible.For 1M ticks / sec(HFT), need < 1μs validation(requires Cython / C++).",
  },
  {
    id: 'data-quality-validation-mc-5',
    question:
      'Duplicate tick detection: Which data structure for O(1) lookup with < 1% false positives?',
    options: [
      'Bloom filter (probabilistic, fast)',
      'Hash set (exact, slower)',
      'Array (O(n) lookup)',
      'Tree (O(log n))',
    ],
    correctAnswer: 0,
    explanation:
      'Bloom filter: Probabilistic data structure, O(1) insertion/lookup, configurable false positive rate (< 1% typical), space-efficient (10 bits/element). Hash set: Exact but larger memory (40 bytes/element vs 10 bits). For 1M ticks: Bloom filter 1.25 MB vs Hash set 40 MB (32× less memory). False positives acceptable for duplicates (marking unique tick as duplicate = minor issue). False negatives impossible (never miss actual duplicates). At 100K ticks/sec, memory savings critical. Bloom filter with 0.1% FP rate = 14.4 bits/element.',
  },
];
