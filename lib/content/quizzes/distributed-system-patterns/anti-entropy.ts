/**
 * Quiz questions for Anti-Entropy section
 */

export const antientropyQuiz = [
  {
    id: 'q1',
    question:
      'Explain how Merkle trees enable efficient anti-entropy by reducing the amount of data that needs to be compared. Walk through a comparison between two replicas where only 10% of the data differs.',
    sampleAnswer:
      'Merkle trees enable logarithmic comparison by allowing nodes to quickly identify which portions of data differ without transferring everything. Merkle tree structure: Dataset with 1 million keys divided into ranges with a tree depth of 4 levels. Level 3 has 16 leaf ranges of 62,500 keys each, where each leaf hash equals the hash of all keys in that range. Level 2 has 8 internal nodes, Level 1 has 4 internal nodes, and Level 0 has 1 root node. Both replicas build identical structures. In a scenario where 10% differs (100,000 keys out of 1 million spread across ranges 3, 7, 11, and 14), Replica A has Root_A and Replica B has Root_B which differ because Range3_B differs from Range3_A. The comparison process: Step 1 - Compare roots, detect difference, descend into tree (avoiding the need to transfer all 1M keys). Step 2 - Compare level 1 children, identify which branches differ (AB, CD, GH) and skip matching branches (EF). Step 3 - Descend into differing branches like AB, compare A_A vs A_B. Step 4 - At leaves, identify specific differing ranges like Range3. Step 5 - Transfer actual data only for differing ranges (4 ranges × 62,500 keys = 250,000 keys transferred vs 1M naively). Efficiency calculation shows ~20 hash comparisons at 32 bytes each for 640 bytes overhead, transferring 250K keys vs 1M without Merkle trees (4-10x improvement). Real-world Cassandra uses configurable depth (~15 levels) with 1-10GB per leaf range, balancing comparison overhead vs transfer overhead.',
    keyPoints: [
      'Merkle tree: Hierarchical hashes, compare roots first, descend only into differing branches',
      'Efficiency: O(log N) comparisons vs O(N) naive approach',
      'Example: 1M keys, 10% differ, compare ~20 hashes, transfer 100-250K keys (4-10× improvement)',
      'Granularity trade-off: Smaller leaves = more comparisons but less transfer',
      'Cassandra: Configurable depth (~15 levels), 1-10GB per leaf range',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare the resource costs (CPU, memory, network, disk) of anti-entropy versus read repair. In what scenarios would you increase anti-entropy frequency despite higher costs?',
    sampleAnswer:
      'Anti-entropy and read repair have vastly different resource profiles. Read repair has low CPU (piggybacks on reads, only checks read keys), minimal memory (KB-MB per read), low network (proportional to read traffic), low disk I/O (sequential reads plus occasional repairs), and instant fixing for hot keys but never fixes cold keys. Anti-entropy has high CPU (full dataset scan, Merkle tree rebuild can spike 50% CPU for hours), moderate-high memory (100MB-1GB for tree structure on 1TB dataset), high network (exchange Merkle trees plus GB-TB of differing ranges), high disk I/O (full sequential scan), and is slow (12-24 hours for 1TB) but guarantees eventual completion. Scenarios requiring increased anti-entropy frequency: (1) High write volume with failures - 100K writes/sec with frequent node failures where nodes down >3 hours (hinted handoff TTL) accumulate significant missing data, requiring daily anti-entropy to achieve max 1-day inconsistency window vs unacceptable 7-day window with weekly anti-entropy. (2) Compliance requirements - Financial systems must prove all replicas consistent within 24 hours, requiring daily anti-entropy since read repair does not cover cold data. (3) Tombstone resurrection prevention - Cassandra deletes use tombstones with gc_grace_seconds of 10 days, requiring anti-entropy every 7-9 days to prevent tombstones from resurrecting on replicas that missed deletions. (4) Silent corruption detection - Monthly full anti-entropy with checksum validation on large TB datasets to detect bit flips in cold data before data loss from multiple replicas. (5) Post-incident recovery - After major incidents like datacenter failures, run immediate continuous anti-entropy to restore consistency ASAP. Cost mitigation strategies include off-peak scheduling (2-6 AM), throttling disk I/O and network bandwidth, incremental partitioning of dataset, and dedicated nodes for anti-entropy in large clusters.',
    keyPoints: [
      'Resource costs: Anti-entropy high (CPU, disk, network scan), read repair low (incremental)',
      'Increase frequency: High writes + failures, compliance (24h SLA), tombstone cleanup (< gc_grace)',
      'Silent corruption: Monthly full scan with checksums detects bit flips in cold data',
      'Post-incident: Immediate continuous anti-entropy to restore consistency after major failure',
      'Mitigation: Off-peak scheduling, throttling, incremental partitioning, dedicated nodes',
    ],
  },
  {
    id: 'q3',
    question:
      'Design a Merkle tree-based anti-entropy system for a database with 10TB of data across 100 nodes. What tree depth, leaf size, and incremental update strategy would you use?',
    sampleAnswer:
      'For 10TB across 100 nodes (100GB per node), design considerations include: Tree depth selection using formula depth = log₂(dataset_size / leaf_size). Comparing options - Depth 10 gives 1024 leaves with 100MB per leaf (fast comparison, wasteful transfer), Depth 20 gives 1M leaves with 100KB per leaf (more comparison overhead, efficient transfer). Recommended depth 15 provides 32K leaves with 3MB per leaf (30K keys at 100 bytes each), balancing moderate comparison overhead (15 levels) with efficient transfer granularity. Total tree nodes would be 65K nodes using 2MB storage (negligible overhead). Incremental update strategy options: (1) Write-triggered updates propagate 15 hash updates per write up the tree (cheap, always current but per-write overhead), (2) Batch updates buffer dirty keys and recompute leaf hashes every 60 seconds (amortized cost, 60-second staleness), (3) Snapshot-based rebuilds scan entire dataset weekly before anti-entropy (1-2 hour rebuild, no per-write overhead). Recommended hybrid approach uses batch updates every 60 seconds during low traffic plus weekly full rebuild for accurate anti-entropy. Performance estimates: Weekly anti-entropy between node pairs takes ~15-20 minutes (1 second for 32K hash comparisons, 96 seconds to transfer 9.6GB for 10% differences, 5-10 minutes to apply updates). System-wide with 100 nodes and RF=3 requires 100 syncs parallelized 10 at a time for 3.3 hours total. Monitoring should track tree_depth, leaf_count, tree_rebuild_duration with alerts if rebuild exceeds 4 hours. Optimizations include compression (2-5x reduction), incremental transfer of only differing keys, parallel sync of multiple replicas, and adaptive depth where hot ranges use deeper trees.',
    keyPoints: [
      'Tree depth: 15 levels (32K leaves) balances comparison speed vs transfer efficiency',
      'Leaf size: 3MB (~30K keys) provides reasonable granularity for 100GB per node',
      'Update strategy: Hybrid - batch incremental (60s) + weekly snapshot rebuild',
      'Performance: 15-20 min per node pair, 3.3 hours for 100-node cluster (parallelized)',
      'Monitoring: Track rebuild duration, transfer size, alert on anomalies',
    ],
  },
];
