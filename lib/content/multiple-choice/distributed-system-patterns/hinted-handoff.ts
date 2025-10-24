/**
 * Multiple choice questions for Hinted Handoff section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const hintedhandoffMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'In Cassandra, why do hints NOT count toward the read quorum?',
    options: [
      'Hints are encrypted and cannot be read',
      "Hints are temporary stand-ins for unavailable replicas and don't contain the actual data for reads",
      'Hints are only stored in memory',
      'Hints are automatically deleted after being created',
    ],
    correctAnswer: 1,
    explanation:
      'Hints do not count toward read quorum because they are meta-data about writes that need to be delivered, not actual replicas containing queryable data. When a write fails to a replica (say, Node C is down), a hint is stored on another node (Node D): "When C comes back, write key=X with value=Y." The hint is a reminder/instruction, not a copy of the data. When reads occur: Client requests R=2 replicas for key=X. Coordinator reads from Nodes A and B (actual replicas). Node D has a hint, but cannot return the value for key=X (it\'s just a delivery instruction, not data storage). R=2 must come from actual replicas (A, B), not hints. Why this design: Hints may be stale (data updated since hint created). Hints may have expired (TTL passed). Hints are for write availability and catch-up, not serving reads. If hints counted toward R, consistency would be violated (reading incomplete/stale data). Example: RF=3, W=2, R=2. C down, write to A, B, hint on D. Read: Must get R=2 from {A, B}, not D\'s hint. This maintains consistency (R+W>RF). Option 1 is incorrect—encryption is irrelevant. Option 3 is incorrect—hints can be persistent. Option 4 is incorrect—hints persist until replayed or TTL.',
  },
  {
    id: 'mc2',
    question:
      'What is the default TTL (time-to-live) for hints in Apache Cassandra, and why is it limited?',
    options: [
      '30 minutes to reduce network overhead',
      '3 hours to prevent unlimited hint accumulation',
      '24 hours to maximize data retention',
      'Unlimited to ensure all data is eventually delivered',
    ],
    correctAnswer: 1,
    explanation:
      "Cassandra's default hint TTL is 3 hours to balance recovery speed with resource constraints, preventing unbounded hint accumulation. The problem with unlimited hints: Node C down for days. Write rate: 10K/sec targeting C. Hint accumulation: 10K × 3600 sec/hour × 24 hours = 864 million hints per day! Storage: Hints consume disk space (potentially TBs). Memory: Hint metadata in memory. Recovery: Replaying 864M hints takes hours/days. The 3-hour TTL represents: Typical hardware replacement/recovery time (most failures resolved within 3 hours). Reasonable storage overhead (10K × 3600 × 3 = 108M hints = manageable). Trade-off: Nodes down <3 hours: Hints provide fast catch-up (seconds to minutes to replay). Nodes down >3 hours: Hints discarded, rely on anti-entropy/repair (slower but comprehensive). Configurable (max_hint_window_in_ms) based on environment: Cloud (fast instance replacement): 1-2 hours might suffice. On-prem (slower hardware replacement): 3-6 hours more appropriate. The limit prevents a single failed node from exhausting cluster storage. Option 1 (30min) too short for typical recovery. Option 3 (24h) risks excessive accumulation. Option 4 (unlimited) would cause resource exhaustion.",
  },
  {
    id: 'mc3',
    question:
      'When replaying hints to a recovered node, what problem can occur if hints are replayed too quickly?',
    options: [
      'The hints become corrupted',
      'The recovered node becomes overwhelmed and cannot keep up',
      'Network security is compromised',
      'Other nodes start failing',
    ],
    correctAnswer: 1,
    explanation:
      'Replaying hints too quickly can overwhelm the recovered node, which is already catching up and may not have full capacity, causing it to fall behind again or crash. The scenario: Node C down for 2 hours. Hints accumulated: 72 million (10K writes/sec × 7200 sec). Node C recovers. Multiple nodes start replaying hints to C simultaneously at full speed. Node C receives: Normal traffic (new writes). Replay traffic (72M hints being streamed). C cannot keep up: Write queue backs up. Memory exhausted. CPU saturated. C either falls out of ISR again (too slow) or crashes (OOM). Solutions: Throttle hint replay: Limit rate (e.g., 1 MB/sec per node). Cassandra: hinted_handoff_throttle_in_kb configures this. Stagger replay: Nodes replay sequentially, not all at once. Backpressure: C signals "slow down" if it\'s struggling. Example: 72M hints at 1 MB/sec throttle = ~20 hours to replay (but C remains healthy). Without throttling at 100 MB/sec: C overwhelmed in minutes, falls out of ISR, hints never complete. The recovering node needs time to stabilize—aggressive hint replay defeats the purpose. Option 1 (corruption) is unlikely. Option 3 (security) is unrelated. Option 4 (cascade failure) is possible but not the direct problem.',
  },
  {
    id: 'mc4',
    question:
      'How can stale hints (hints containing old data) be prevented from overwriting newer data on a recovered node?',
    options: [
      'By deleting all hints before replay',
      "By using timestamps or version vectors to reject hints older than the node's current data",
      'By requiring manual approval for each hint',
      'By only storing hints in memory',
    ],
    correctAnswer: 1,
    explanation:
      "Stale hints are prevented by including timestamps or version vectors with hints and rejecting replays that are older than the node's current data. The race condition: T=0: Node C down, write X=1 to A and B, hint stored on D. T=1: C recovers. T=2: Write X=2 directly to A, B, C (all healthy). C now has X=2. T=3: D begins replaying hints. T=4: D sends hint \"write X=1 to C\" (stale!). T=5: Without protection, C overwrites X=2 with X=1 (regression). Protection mechanisms: Timestamp check: Hint includes timestamp from original write (T=0). C compares: hint_timestamp (T=0) < local_timestamp (T=2). Reject hint, keep X=2. Version vector check: Hint includes version vector from T=0. C's current version vector dominates hint's vector. Reject hint (causally older). Last-write-wins: Even if hint applied, subsequent comparisons would prefer newer timestamp. Example in Cassandra: Every write includes timestamp (writetime). Hints include original write timestamp. On replay, C uses normal write path with timestamp. If C's current data has higher timestamp, hint rejected. This ensures consistency—newest data always wins. Option 1 (delete all) loses valid hints. Option 3 (manual) impractical. Option 4 (memory-only) doesn't solve staleness.",
  },
  {
    id: 'mc5',
    question:
      'What happens if the node storing hints crashes before replaying them to the recovered node?',
    options: [
      'The hints are automatically transferred to another node',
      'The cluster immediately fails',
      'The hints are lost, and anti-entropy must eventually catch the inconsistency',
      'The recovered node permanently loses that data',
    ],
    correctAnswer: 2,
    explanation:
      'If the node storing hints crashes, those hints are lost (unless replicated, which is rare), and the system must rely on anti-entropy to eventually synchronize the missing data. The scenario: Node C down. Hints stored on Node D for writes targeting C. Node D crashes before C recovers. Hints on D are lost (typically not replicated). C recovers but never receives those hints. Result: C is missing some writes. Hinted handoff failed. Recovery mechanisms: Anti-entropy/repair: Periodic process (e.g., weekly in Cassandra via nodetool repair). Compares replicas using Merkle trees. Detects C is missing data. Synchronizes from A and B. Eventually C catches up (days/weeks, depending on repair schedule). Read repair: If those missing keys are read, read repair detects inconsistency. Fixes on-the-fly. Why this is acceptable: Hints are best-effort, not guaranteed delivery. Anti-entropy provides guaranteed eventual consistency (safety net). Most hints succeed (node crashes while holding hints are rare). Design philosophy: Optimize for common case (hints work, fast catch-up). Handle rare case (hint loss) with comprehensive but slower mechanism (anti-entropy). Option 1 (transfer) rarely implemented (adds complexity). Option 2 (cluster fails) too severe. Option 4 (permanent loss) incorrect—anti-entropy eventually recovers.',
  },
];
