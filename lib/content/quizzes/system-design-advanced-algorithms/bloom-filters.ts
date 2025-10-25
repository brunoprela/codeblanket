/**
 * Quiz questions for Bloom Filters section
 */

export const bloomfiltersQuiz = [
  {
    id: 'q1',
    question:
      'Explain why Bloom filters never produce false negatives but can produce false positives. Walk through a concrete example.',
    sampleAnswer:
      'Bloom filters never produce false negatives because of how the lookup operation works: To check if element X exists, we compute k hash functions and check if ALL corresponding bits are set to 1. If even ONE bit is 0, we know with certainty that element X was never inserted (because insertion would have set that bit to 1). This is a definitive "NO". False positives occur when we check element Y that was never inserted, but all k bits happen to be 1 (set by OTHER elements). Example: Insert "alice" (sets bits 3,7,11). Insert "bob" (sets bits 2,7,14). Now check "charlie" which hashes to bits 2,3,7. All three bits are 1 (set by alice and bob), so Bloom filter says "probably present" even though "charlie" was never inserted. This is a false positive. The key insight: bits can only transition 0→1 (never 1→0), so a 0 bit means "definitely not inserted", but all 1s means "maybe inserted".',
    keyPoints: [
      'Lookup checks if ALL k bits are set to 1',
      'If any bit is 0, element was definitely not inserted (no false negative)',
      'If all bits are 1, they might have been set by other elements (false positive)',
      'Bits only transition 0→1, never 1→0 in standard Bloom filter',
      'As more elements added, more bits become 1, increasing collision probability',
    ],
  },
  {
    id: 'q2',
    question:
      'You need to build a Bloom filter for a web crawler that will track 10 billion URLs. Your target is a 0.1% false positive rate. Calculate the required memory and explain the trade-offs.',
    sampleAnswer:
      'For 10 billion URLs with 0.1% FPR: Formula: m = -n * ln (p) / (ln(2))² where n=10B, p=0.001. m ≈ 14.4 bits per element. Total: 10B * 14.4 bits = 144 billion bits = 18 GB. Optimal k = (m/n) * ln(2) ≈ 10 hash functions. TRADE-OFFS: (1) Memory: 18 GB vs 10B * 100 bytes = 1 TB for hash table (55x savings). (2) Speed: O(10) hash operations vs O(1) hash table lookup (slightly slower but still fast). (3) False positives: 0.1% of NEW URLs will be incorrectly marked as "already seen" and skipped. For 10B URLs, that\'s 10M URLs missed (acceptable for web crawler). (4) Cannot remove URLs if needed. (5) As filter fills up, FPR increases beyond 0.1%, requiring periodic rebuilds. Decision: Use Bloom filter because 18 GB fits in memory of single server, whereas 1 TB hash table requires distributed system. Missing 0.1% of URLs is acceptable for web crawling. This is the classic space-accuracy trade-off Bloom filters excel at.',
    keyPoints: [
      'Formula: m = -n*ln (p)/(ln(2))² gives bits needed',
      '18 GB Bloom filter vs 1 TB hash table (55x smaller)',
      '0.1% FPR means 10M URLs missed out of 10B (acceptable)',
      'Space savings enable single-server solution',
      'Periodic rebuilds needed as filter fills up',
    ],
  },
  {
    id: 'q3',
    question:
      'Google BigTable uses Bloom filters in front of SSTables. Explain how this works and why it dramatically improves read performance.',
    sampleAnswer:
      'BigTable stores data in immutable SSTables on disk. Each SSTable has an in-memory Bloom filter. READ PATH: (1) Query: "Does row key X exist in SSTable Y?" (2) Check Bloom filter first (microseconds). (3) If Bloom filter says "NO" → Skip disk read entirely (saves 10-100ms). (4) If Bloom filter says "MAYBE" → Read from disk to confirm. WHY IT WORKS: Most reads are for non-existent keys (cache misses, deleted data). Bloom filter eliminates 70-90% of unnecessary disk reads by definitively saying "this key is NOT in this SSTable". FALSE POSITIVES: 1% FPR means 1% of queries do unnecessary disk read (acceptable cost). PERFORMANCE IMPACT: Without Bloom filter: 100ms disk read for every negative lookup. With Bloom filter: Skip 90% of disk reads, 10% false positives still read disk. Result: 90% reduction in disk I/O for negative lookups. CONFIGURATION: Cassandra uses similar approach with configurable FPR (default 0.01%). More bits per key = lower FPR = fewer wasted disk reads but more memory. Typical: 10-20 bits per key (1-2% FPR). This is WHY distributed databases use Bloom filters: they\'re the perfect pre-filter for expensive disk operations.',
    keyPoints: [
      'Bloom filter sits in memory in front of on-disk SSTables',
      'Eliminates 70-90% of unnecessary disk reads for non-existent keys',
      '1% false positives do unnecessary disk read (acceptable)',
      'Each disk read saved is 10-100ms improvement',
      'Trade-off: Small memory cost for large disk I/O savings',
    ],
  },
];
