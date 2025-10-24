/**
 * Multiple choice questions for Anti-Entropy section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const antientropyMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the primary advantage of using Merkle trees for anti-entropy compared to comparing every key-value pair?',
    options: [
      'Merkle trees use less disk space',
      'Merkle trees enable O(log N) comparison to identify differing data ranges, avoiding full dataset transfers',
      'Merkle trees automatically repair data without network communication',
      'Merkle trees work only for small datasets',
    ],
    correctAnswer: 1,
    explanation:
      'Merkle trees provide O(log N) comparison efficiency by hierarchically organizing hash values, allowing identification of differing data ranges without scanning the entire dataset. Without Merkle trees (naive approach): Must compare all 1 million keys between replicas. Transfer all 1 million key-value pairs (gigabytes of data). Time: hours to days. With Merkle trees: Build tree with 15 levels (32K leaf nodes, each covering ~30 keys). Compare roots: If match, done (datasets identical). If differ, compare children of root (2 hashes). Descend into differing branches only. At leaves, identify specific differing ranges. Only transfer differing ranges (~100K keys if 10% differ). Efficiency example: 1M keys, 10% differ: Naive: Transfer 1M keys = 10× overhead. Merkle: Compare ~1000 hashes (traverse tree), transfer ~100K keys = close to actual differences. Time savings: Hours → Minutes. Network savings: GBs → MBs of comparison overhead + actual diff transfer. This is why production systems (Cassandra, Dynamo) use Merkle trees—anti-entropy becomes practical at scale. Option 1 (disk space) is incorrect—minimal impact. Option 3 (no network) is incorrect—still need to transfer diffs. Option 4 (small datasets) is opposite—Merkle trees especially valuable for large datasets.',
  },
  {
    id: 'mc2',
    question:
      'In Cassandra, why is it recommended to run anti-entropy repair at least once per gc_grace_seconds (default 10 days)?',
    options: [
      'To improve read performance',
      'To prevent deleted data (tombstones) from resurrecting on replicas that missed the deletion',
      'To reduce disk usage',
      'To elect a new leader',
    ],
    correctAnswer: 1,
    explanation:
      "Running repair within gc_grace_seconds prevents tombstone resurrection—deleted data reappearing on replicas that missed the original deletion. The problem (tombstone resurrection): T=0: DELETE user:123 executed. Tombstone written to Replicas A and B. Replica C is down (missed deletion). T=1-10 days: Tombstone exists on A and B (prevents reads from returning deleted data). C still has user:123 (stale). T=10 days: gc_grace_seconds expires. A and B delete tombstone (garbage collection). Now: A and B have no record of user:123 (correct, deleted). C still has user:123 (missed deletion). T=11 days: Repair runs. C has user:123, A and B don't. Repair sees C has data that A and B don't. Incorrectly syncs user:123 from C to A and B. Result: Deleted user:123 resurrected! Solution: Run repair before gc_grace_seconds (before tombstones deleted): Repair runs at T=9 days. C still has user:123. A and B have tombstone. Repair sees tombstone on A and B, sends to C. C deletes user:123 (tombstone received). T=10 days: All replicas delete tombstone. No resurrection. Recommendation: repair interval < gc_grace_seconds. Default gc_grace_seconds=10 days → repair every 7-9 days. Critical for data correctness. Options 1 and 3 are secondary benefits. Option 4 is unrelated—anti-entropy doesn't elect leaders.",
  },
  {
    id: 'mc3',
    question:
      'Why does anti-entropy typically take much longer to complete than read repair for the same amount of inconsistent data?',
    options: [
      'Anti-entropy uses slower network protocols',
      'Anti-entropy must scan the entire dataset to build Merkle trees and find inconsistencies, not just accessed keys',
      'Anti-entropy repairs only one key at a time',
      'Anti-entropy requires manual intervention',
    ],
    correctAnswer: 1,
    explanation:
      "Anti-entropy takes longer because it must comprehensively scan the entire dataset to build Merkle trees and identify all inconsistencies, regardless of whether data is accessed. The timing comparison: Read repair: Triggered by reads (piggybacks on client operations). Only checks keys that are read (hot data). Immediate for those keys (fixed on first read after inconsistency). Time for 100K inconsistent keys: If all read within hours/days, all fixed quickly. Cold keys never fixed (blind spot). Anti-entropy: Background scan of entire dataset (billions of keys). Must read all data to compute Merkle tree hashes. Find inconsistencies (100K keys). Transfer and apply differences. Time for 100GB dataset: 12-24 hours to scan and build trees. Additional hours to transfer differences. Example - 10TB database: Read repair: Fixes hot 1% in hours (frequently accessed). Instant gratification for accessed keys. Anti-entropy: Scans all 10TB in days (comprehensive). Catches remaining 99% (cold data). Resource intensity: Read repair: Minimal overhead (part of normal reads). Anti-entropy: High CPU (hash computation), disk I/O (full scan), network (tree comparison). Why anti-entropy is still necessary: Guarantees eventual consistency for ALL data (cold keys). Finds and fixes inconsistencies read repair never sees. Safety net—ensures no data left behind. Production approach: Both mechanisms complement each other (speed for hot data + completeness for all data). Options 1, 3, and 4 are incorrect—anti-entropy's slowness is due to comprehensive scope, not implementation details.",
  },
  {
    id: 'mc4',
    question:
      'What is the trade-off in choosing Merkle tree depth (e.g., 10 levels vs 20 levels) for anti-entropy?',
    options: [
      'Deeper trees are always better',
      'Deeper trees have more comparison overhead but transfer less unnecessary data; shallower trees have less comparison overhead but transfer more',
      'Shallower trees are more secure',
      "Tree depth doesn't matter for performance",
    ],
    correctAnswer: 1,
    explanation:
      'Tree depth trades comparison efficiency (fewer levels = faster tree traversal) against transfer efficiency (more levels = finer granularity, less unnecessary data transfer). Shallow tree (10 levels, 1024 leaves): Comparison overhead: 10 levels to traverse = fast (~20 hash comparisons). Leaf size: 100GB / 1024 = ~100MB per leaf. Transfer overhead: If 10% differs, must transfer entire 100MB leaves even if only small portion within leaf differs. Potentially 10GB transferred for 1GB actual differences (10× waste). Deep tree (20 levels, 1,048,576 leaves): Comparison overhead: 20 levels to traverse = slower (~40 hash comparisons). Leaf size: 100GB / 1M = ~100KB per leaf. Transfer overhead: If 10% differs, transfer 100KB leaves. Much closer to actual differences (~1GB for 1GB differences). Optimal balance (typical: 15 levels): Moderate comparison overhead (15 levels). Moderate leaf granularity (~3MB per leaf). Good balance for most workloads. Example decision: Small dataset (1GB): Shallow tree okay (10 levels, comparison dominates). Large dataset (10TB) with sparse diffs: Deep tree better (20 levels, avoid transferring GBs of unchanged data within large leaves). Production (Cassandra): Configurable depth (default ~15), balances both. Monitor: If transfer size >> actual diffs, increase depth. If comparison time excessive, decrease depth. Options 1 and 3 are incorrect. Option 4 is false—depth significantly impacts performance.',
  },
  {
    id: 'mc5',
    question:
      'In a distributed system, why might you run anti-entropy more frequently than weekly despite the high resource cost?',
    options: [
      'To improve read performance',
      'To meet compliance requirements, handle high write volumes with frequent failures, or prevent tombstone resurrection',
      'To reduce the number of nodes needed',
      'To eliminate the need for replication',
    ],
    correctAnswer: 1,
    explanation:
      "More frequent anti-entropy is justified when consistency requirements, failure patterns, or data integrity guarantees demand it, despite resource costs. Scenarios requiring more frequent anti-entropy: 1. Compliance/regulatory: Financial system must prove all replicas consistent within 24 hours (regulation). Solution: Daily anti-entropy (meet compliance). Cost: High CPU/network daily, but necessary for regulatory compliance. 2. High write volume + frequent failures: 100K writes/sec, frequent node failures. Hinted handoff TTL: 3 hours (after that, relies on anti-entropy). Node down >3 hours accumulates significant missing data. Weekly anti-entropy: Up to 7-day inconsistency window (unacceptable). Solution: Daily anti-entropy (max 1-day inconsistency). 3. Tombstone resurrection prevention: gc_grace_seconds = 10 days. Anti-entropy must run within 10 days to prevent deleted data resurrection. Weekly: Risky if one cycle missed (9-16 day gap possible). Solution: Every 7-9 days (safety margin). 4. Silent corruption detection: Large dataset (TBs), risk of bit flips. Monthly full anti-entropy with checksum validation catches corruption early. Cost mitigation: Off-peak scheduling (2-6 AM). Throttling (extend duration, reduce impact). Incremental (partition dataset, different partitions different days). Dedicated nodes (don't impact serving nodes). The decision: Cost of frequent anti-entropy < cost of extended inconsistency or compliance violation. Option 1 (read performance) is secondary. Options 3 and 4 are incorrect—anti-entropy doesn't reduce nodes or eliminate replication.",
  },
];
