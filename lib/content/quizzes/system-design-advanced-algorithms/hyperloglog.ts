/**
 * Quiz questions for HyperLogLog section
 */

export const hyperloglogQuiz = [
  {
    id: 'q1',
    question:
      'Explain the fundamental insight behind HyperLogLog: how does counting leading zeros in hashes allow you to estimate cardinality? Use the coin flip analogy.',
    sampleAnswer:
      'COIN FLIP ANALOGY: If you flip coins and see 10 heads in a row, probability is (1/2)^10 = 1/1024. You can infer you probably flipped ~1024 times to see this rare event. HYPERLOGLOG INSIGHT: Hash functions produce uniformly random bits. Hash "user123" → 0x00ABC123 (binary: 0000000010101011...). Leading zeros: 7. This is like flipping 7 heads in a row (probability: 1/2^7 = 1/128). To see 7 leading zeros, you likely hashed ~2^7 = 128 distinct elements. More leading zeros = more unique elements. MATHEMATICAL BASIS: If you have N unique elements and hash them uniformly, the probability of seeing at least one hash with k leading zeros is approximately 1 - (1 - 1/2^k)^N. When this probability ≈ 0.5, then k ≈ log₂(N). Thus, max leading zeros ≈ log₂(cardinality). PROBLEM: High variance. One lucky hash with many zeros skews estimate. SOLUTION: Use multiple buckets (16K) and harmonic mean to average out variance. Each bucket tracks max leading zeros for its subset. This reduces standard error to ~1.04/√buckets. With 16,384 buckets, error is 0.81%. This is the genius of HyperLogLog: convert counting problem to statistics problem.',
    keyPoints: [
      'Leading zeros in hash ≈ coin flips (probability 1/2^k)',
      'Max leading zeros seen ≈ log₂(cardinality)',
      'Multiple buckets reduce variance (harmonic mean)',
      'Error ≈ 1.04/√buckets (16K buckets → 0.81% error)',
      'Convert counting to statistics: constant memory regardless of N',
    ],
  },
  {
    id: 'q2',
    question:
      'How does HyperLogLog achieve constant memory usage (16 KB) regardless of whether you have 1 million or 1 billion unique elements? Explain why exact counting requires O(N) memory.',
    sampleAnswer:
      "EXACT COUNTING: Must store every unique element to verify uniqueness. Set/hash table grows with N unique elements. 1 million UUIDs: 1M × 16 bytes = 16 MB. 1 billion UUIDs: 1B × 16 bytes = 16 GB. Memory = O(N) where N = cardinality. HYPERLOGLOG CONSTANT MEMORY: Doesn't store elements, only stores STATISTICS. Uses fixed number of buckets (e.g., 16,384). Each bucket stores one integer (1 byte): max leading zeros seen for elements hashing to that bucket. Total: 16,384 buckets × 1 byte = 16 KB. REGARDLESS OF N: Whether you hash 1M or 1B elements, you still have 16,384 buckets storing 1 byte each. The values in buckets change (larger max leading zeros for larger N), but the NUMBER of buckets stays constant. TRADE-OFF: Lost ability to retrieve elements or verify membership. Gained constant memory and ability to count arbitrarily large sets. ANALOGY: Like taking a statistical sample instead of a census. Sample size (buckets) stays constant, estimates entire population. This is why HyperLogLog can count billions of unique visitors using only 16 KB while exact counting would need gigabytes.",
    keyPoints: [
      'Exact counting: O(N) memory (must store every unique element)',
      'HyperLogLog: O(1) memory (fixed number of buckets)',
      'Each bucket stores single statistic (max leading zeros)',
      'Bucket values change with N, but bucket COUNT is constant',
      'Trade-off: Cannot retrieve elements for constant memory',
    ],
  },
  {
    id: 'q3',
    question:
      'Facebook needs to count Daily Active Users (DAU) across 10,000 servers. Explain how they would use HyperLogLog with merging to get a global unique count efficiently.',
    sampleAnswer:
      'DISTRIBUTED HYPERLOGLOG ARCHITECTURE: (1) LOCAL COUNTING: Each of 10,000 servers maintains local HyperLogLog (precision=14, 16 KB). As users interact, add user_id to local HLL. Server 1: hll1.add (user_123), hll1.add (user_456). Server 2: hll2.add (user_456), hll2.add (user_789). (2) HOURLY AGGREGATION: Every hour, each server sends its HLL to central aggregator. Data transfer: 10,000 servers × 16 KB = 160 MB per hour. (3) MERGING: Aggregator merges all HLLs by taking MAX of each register. merged.registers[i] = max (hll1.registers[i], hll2.registers[i], ..., hll10000.registers[i]). Why MAX works: If user_456 was seen on Server 1 and Server 2, both computed same hash (deterministic). Hash falls into bucket B. Both servers updated register[B]. Taking MAX ensures we count user_456 once (not twice). (4) GLOBAL COUNT: merged.count() returns global unique DAU with ~1% accuracy. WHY THIS BEATS ALTERNATIVES: Exact counting would require: (1) Distributed set across servers (complex, expensive). (2) OR each server sends all user IDs to aggregator (10K servers × millions of IDs = terabytes transfer). HyperLogLog: 160 MB transfer, simple merge, 1% error acceptable for DAU metric. PRODUCTION SCALE: Facebook actually uses this. 500M DAU counted using HyperLogLog with minimal overhead.',
    keyPoints: [
      'Each server maintains local HLL (16 KB)',
      'Periodic merge: send HLLs to aggregator (160 MB for 10K servers)',
      'Merge by taking MAX of each register across HLLs',
      'Deterministic hashing ensures same user not double-counted',
      'Alternative (exact) would need terabytes transfer or distributed set',
    ],
  },
];
