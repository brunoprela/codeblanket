/**
 * Multiple choice questions for Split-Brain Resolution section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const splitbrainresolutionMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'In a 7-node cluster, what is the minimum number of nodes (quorum) required to elect a leader and prevent split-brain?',
    options: ['3 nodes', '4 nodes', '5 nodes', '7 nodes'],
    correctAnswer: 1,
    explanation:
      'Quorum is calculated as ⌈N/2⌉ (ceiling of N divided by 2), which for 7 nodes is ⌈7/2⌉ = ⌈3.5⌉ = 4 nodes. This ensures that only one partition can have a majority. Example network partition scenarios: Partition {4 nodes} vs {3 nodes}: 4-node partition has quorum (4 ≥ 4), can elect leader. 3-node partition lacks quorum (3 < 4), cannot elect leader. Partition {5 nodes} vs {2 nodes}: 5-node partition has quorum, can operate. 2-node partition cannot. Only one partition can have ≥4 nodes, preventing two leaders. Why 4, not 3? If quorum were 3: Partition {4 nodes} has quorum (4 ≥ 3). Partition {3 nodes} also has quorum (3 ≥ 3). Both could elect leaders → Split-brain! By requiring 4 (more than half), at most one partition can have quorum. The formula ⌈N/2⌉ ensures this mathematically for any cluster size. For odd N (5, 7, 9): Clear majority needed. For even N (4, 6, 8): Need strictly more than half, preventing 50/50 split from electing.',
  },
  {
    id: 'mc2',
    question:
      'What is the primary purpose of fencing tokens in preventing split-brain?',
    options: [
      'To encrypt messages between nodes',
      'To provide defense-in-depth by rejecting operations from stale leaders even if quorum mechanisms fail',
      'To improve network performance',
      'To balance load across nodes',
    ],
    correctAnswer: 1,
    explanation:
      "Fencing tokens provide defense-in-depth: even if quorum-based election fails (due to bugs, misconfiguration), tokens prevent stale leaders from corrupting data. Each elected leader receives a monotonically increasing token (generation number). Resources track the highest token seen and reject lower tokens. Example: Leader A with token=5, network partition, new Leader B elected with token=6. Both try to write to shared storage: A sends write (data, token=5). B sends write (data, token=6). Storage accepts B's write (token=6 > current), rejects A's (token=5 < 6). Even though A thinks it's leader, it cannot corrupt data. This is \"defense-in-depth\": Layer 1 (quorum) prevents split-brain election. Layer 2 (fencing) prevents stale leader damage if Layer 1 fails. Layer 3 (monitoring) detects anomalies. Real-world systems (Google Chubby, HDFS, Patroni) use fencing tokens because bugs happen—relying on a single mechanism is risky for critical systems. Option 1 (encryption) is unrelated. Options 3 and 4 don't relate to split-brain prevention.",
  },
  {
    id: 'mc3',
    question:
      'In a split-brain scenario with Last-Write-Wins conflict resolution, what is a major risk?',
    options: [
      'The system becomes permanently unavailable',
      'Clock skew can cause incorrect resolution, accepting stale data as "latest"',
      'Network usage increases exponentially',
      'All data is permanently lost',
    ],
    correctAnswer: 1,
    explanation:
      'Last-Write-Wins (LWW) relies on timestamps to determine which write is "latest," making it vulnerable to clock skew causing incorrect resolution. Example of the problem: Partition A: Write X=5 at timestamp 100 (clock accurate). Partition B: Write X=3 at timestamp 150 (clock 1 hour fast, actual time=90). After partition heals: LWW compares timestamps: 150 > 100. Keeps X=3 (from B). But X=5 was actually the later write (real time 100 > 90)! X=3 is stale, incorrectly chosen due to clock skew. Result: Lost the correct update, kept outdated data, potential data corruption. This is why production systems using LWW require: Strict NTP synchronization (clocks within 100ms). Monitoring of clock drift, alerts on skew. Sometimes hybrid approaches (LWW with vector clocks as backup). Financial systems avoid LWW entirely—prefer manual resolution or vector clocks. Option 1 (unavailable) is a different failure mode. Option 3 (network) is unrelated to LWW. Option 4 (total loss) is exaggerated—LWW loses one write version, not all data.',
  },
  {
    id: 'mc4',
    question:
      'Why is using an odd number of nodes (e.g., 3, 5, 7) recommended for quorum-based systems?',
    options: [
      'Odd numbers perform better computationally',
      'Odd numbers provide better fault tolerance per node added than even numbers',
      'Odd numbers are easier to configure',
      'Odd numbers use less memory',
    ],
    correctAnswer: 1,
    explanation:
      "Odd numbers provide better fault tolerance per node because even numbers don't improve the maximum tolerable failures compared to the next lower odd number. Fault tolerance analysis (for quorum systems): 3 nodes: Quorum=2, tolerates 1 failure (2-1=1). 4 nodes: Quorum=3, tolerates 1 failure (3-1=1). Same as 3 nodes! 5 nodes: Quorum=3, tolerates 2 failures (3-1=2). 6 nodes: Quorum=4, tolerates 2 failures (4-1=2). Same as 5 nodes! 7 nodes: Quorum=4, tolerates 3 failures. 8 nodes: Quorum=5, tolerates 3 failures. Same as 7 nodes! Pattern: Adding one node to an odd cluster (making it even) increases cost (one more node to maintain) but doesn't improve fault tolerance. Going from 3→4 nodes: Same fault tolerance (1), but 33% more hardware/cost. From 5→6 nodes: Same fault tolerance (2), but 20% more cost. Therefore, production systems use 3, 5, or 7 nodes: 3 for moderate availability, 5 for high availability, 7 for critical systems. Even numbers (4, 6, 8) are wasteful—pay for extra node without benefit. Options 1, 3, and 4 are incorrect—the benefit is specifically about fault tolerance efficiency, not performance, configuration simplicity, or memory.",
  },
  {
    id: 'mc5',
    question:
      'What is the CAP theorem trade-off made when using quorum-based split-brain prevention?',
    options: [
      'Consistency over Availability during network partitions',
      'Availability over Consistency',
      'Partition Tolerance over Consistency',
      'Performance over Security',
    ],
    correctAnswer: 0,
    explanation:
      'Quorum-based split-brain prevention chooses Consistency over Availability during network partitions, which is the fundamental CAP theorem trade-off. CAP theorem: Can have at most 2 of: Consistency, Availability, Partition tolerance. Partition tolerance is required (networks do partition), so must choose between C and A. Quorum approach (CP system): During partition: Minority partition cannot form quorum. Minority becomes unavailable (cannot elect leader, accept writes). Majority partition remains available. Result: Consistency maintained (no split-brain, no conflicting writes). Availability sacrificed (minority partition down). Example: 5-node cluster partitions {3} vs {2}. Minority {2} unavailable, majority {3} operates. Users in minority experience downtime. Alternative (AP system): Both partitions continue operating (availability). Accept conflicting writes (inconsistency). Resolve conflicts later (merge, LWW, etc.). Example: Multi-master databases, DynamoDB with eventual consistency. Quorum systems (etcd, Consul, ZooKeeper, strongly consistent databases) explicitly choose C over A because: Data correctness is critical. Brief unavailability (minority partition) acceptable. Split-brain consequences worse than downtime. Option 2 (A over C) describes AP systems. Option 3 misunderstands CAP (P is assumed). Option 4 is unrelated to CAP.',
  },
];
