import { MultipleChoiceQuestion } from '@/lib/types';

export const tickDataProcessingMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'tick-data-processing-mc-1',
        question:
            'Your tick processor stores 1 million ticks in memory using Python objects (500 bytes each = 500 MB). You switch to NumPy arrays (40 bytes each). How much memory do you save?',
        options: [
            '460 MB (92% reduction)',
            '250 MB (50% reduction)',
            '400 MB (80% reduction)',
            '100 MB (20% reduction)',
        ],
        correctAnswer: 0,
        explanation:
            'Memory savings: 500 bytes - 40 bytes = 460 bytes per tick. For 1M ticks: 460 MB saved (92% reduction). Python objects have massive overhead: each object needs type info, reference count, and dict for attributes (totaling 400+ bytes). NumPy stores raw data in contiguous arrays with shared metadata. 1M ticks: Python objects = 500 MB, NumPy = 40 MB. This 12.5× memory reduction is why production tick processors use NumPy. At 1B ticks: Python = 500 GB (won\'t fit in RAM), NumPy = 40 GB (fits on single machine). Additional benefits: NumPy operations are vectorized (10-100× faster) and CPU cache-friendly. For extremely large datasets, consider further compression: Delta encoding (store price differences) reduces to 10 bytes/tick, Parquet with Snappy compression achieves 5-10 bytes/tick.',
    },
    {
        id: 'tick-data-processing-mc-2',
        question:
            'You receive ticks with these sequence numbers: 1, 2, 4, 5, 3, 6. Using a sequence tracker with buffer size 1000, which ticks are emitted immediately vs buffered?',
        options: [
            'Emitted: [1, 2], Buffered: [4, 5], then emit [3, 4, 5, 6] when 3 arrives',
            'Emitted: [1, 2, 3, 4, 5, 6] (all immediately, ignore ordering)',
            'Emitted: [1, 2], drop [4, 5], emit [3], emit [6]',
            'Buffer all until end, then emit sorted: [1, 2, 3, 4, 5, 6]',
        ],
        correctAnswer: 0,
        explanation:
            'Sequence trackers emit in-order ticks immediately and buffer out-of-order ticks. Process: Receive 1 → emit [1]. Receive 2 → emit [2]. Receive 4 → buffer (expecting 3). Receive 5 → buffer (still expecting 3). Receive 3 → emit [3], check buffer, find [4, 5], emit them too = emit [3, 4, 5]. Receive 6 → emit [6]. This maintains order while tolerating network reordering. Option B (emit all immediately) loses ordering. Option C (dropping) loses data. Option D (buffer until end) adds unbounded latency. Buffer size limits memory usage: If sequence 3 never arrives and buffer fills (1000 ticks), force flush oldest buffered tick. Real-world: Out-of-order rate is typically 1-5% of ticks. UDP-based feeds (ITCH) have higher reordering (5-10%) than TCP feeds (< 1%). Production systems monitor gap statistics and alert if > 10% out-of-order (indicates network issues).',
    },
    {
        id: 'tick-data-processing-mc-3',
        question:
            'A tick processor handles 100,000 ticks/sec. Each tick requires 10 microseconds of processing. What CPU utilization is needed?',
        options: [
            '100% of 1 core (100K × 10μs = 1 second per second)',
            '10% of 1 core (100K × 10μs = 0.1 seconds per second)',
            '1000% (requires 10 cores)',
            '50% of 1 core (headroom for bursts)',
        ],
        correctAnswer: 0,
        explanation:
            'Calculation: 100,000 ticks/sec × 10 microseconds/tick = 1,000,000 microseconds = 1 second of CPU time per second = 100% of one core. This is theoretical minimum - production systems need headroom for bursts and variance. Solution: (1) Optimize to < 5 μs/tick (50% CPU), (2) Use multiple cores (partition ticks by symbol), (3) Reduce processing (skip unnecessary validations). Real-world example: If processing spikes to 200K ticks/sec during market open, you need 200% CPU = 2 cores. Best practice: Target 60-70% average utilization to handle 2× bursts without dropping ticks. Performance optimization: Use Cython or Numba for hot path (10× faster), batch processing (process 1000 ticks at once with NumPy = 1 μs per tick). At 1M ticks/sec: Need 10 cores OR optimize to 1 μs/tick (1 core). HFT firms achieve < 0.1 μs/tick using C++ and FPGA.',
    },
    {
        id: 'tick-data-processing-mc-4',
        question:
            'Your spike detector flags a tick as anomalous when price deviates > 5% from 100-tick moving average. AAPL normally trades at $150. A tick arrives at $160 (+6.7% from mean). What should the system do?',
        options: [
            'Reject the tick (likely bad data)',
            'Flag as warning but process normally (could be legitimate)',
            'Automatically correct to $150 (revert to mean)',
            'Halt processing and alert operators',
        ],
        correctAnswer: 1,
        explanation:
            'Flag as warning but process normally. A 6.7% move is unusual but possible - AAPL can legitimately jump on news (earnings, acquisitions, market events). Example: In January 2024, AAPL dropped 3.6% in minutes on weak earnings. In March 2020, circuit breakers triggered on 7% drops. Rejecting the tick risks missing real market movements. Better approach: (1) Flag for review, (2) Check if other exchanges show similar price, (3) Check if preceded by news, (4) Process but mark as "unconfirmed" until validated. Correction strategies: Never auto-correct prices (destroys real data). Auto-correction is only acceptable for obvious errors (negative prices, inverted spreads). For suspected spikes: Process but tag with quality score (e.g., confidence = 70%), allow downstream systems to decide. Production systems use multi-stage validation: (1) Check against other exchanges (if NYSE shows $150 but NASDAQ shows $160, investigate), (2) Check trade volume (price with no volume = suspect), (3) Check historical volatility (5-sigma moves are rare but happen). Flash crashes: On May 6, 2010, market dropped 9% in minutes then recovered - these ticks were real and should not be rejected.',
    },
    {
        id: 'tick-data-processing-mc-5',
        question:
            'You are consolidating ticks from 3 exchanges (NASDAQ bid $150.25, NYSE bid $150.24, BATS bid $150.26). What is the National Best Bid (NBB)?',
        options: [
            '$150.26 (BATS - highest bid across all exchanges)',
            '$150.25 (NASDAQ - weighted average)',
            '$150.24 (NYSE - first exchange alphabetically)',
            '$150.25 (median of three bids)',
        ],
        correctAnswer: 0,
        explanation:
            'National Best Bid (NBB) is the highest bid price across all exchanges = $150.26 from BATS. National Best Offer (NBO) is the lowest ask across all exchanges. Together they form NBBO (National Best Bid and Offer). Regulation NMS (2005) requires brokers to route orders to exchange offering best price. Example: If investor wants to sell, they get filled at NBB ($150.26 on BATS). If buying, they pay NBO (lowest ask). NBBO protects retail investors from getting inferior prices. Calculating NBBO: (1) Collect best bid and ask from each exchange, (2) NBB = max(all bids), (3) NBO = min(all asks), (4) NBBO = (NBB, NBO). Locked market: NBB = NBO (e.g., bid $150.25, ask $150.25) - rare but legal. Crossed market: NBB > NBO (e.g., bid $150.26, ask $150.25) - illegal, indicates stale data or system error. Production systems calculate NBBO continuously (updates 1000+ times per second for liquid stocks), broadcast to clients, used for best execution compliance. Market makers must respect NBBO when displaying quotes.',
    },
];

