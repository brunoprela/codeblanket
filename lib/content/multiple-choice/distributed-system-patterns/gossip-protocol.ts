/**
 * Multiple choice questions for Gossip Protocol section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const gossipprotocolMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'In a gossip protocol with fanout=3, approximately how many rounds does it take for information to reach all 1000 nodes in a cluster?',
    options: [
      'About 7 rounds (log₃ 1000)',
      'About 10 rounds (log₁₀ 1000)',
      'About 333 rounds (1000/3)',
      'About 1000 rounds (one per node)',
    ],
    correctAnswer: 0,
    explanation:
      'With fanout=3, each informed node tells 3 others per round, causing approximately exponential growth: 3^k nodes informed after k rounds. To reach N nodes, we need 3^k ≈ N, so k ≈ log₃(N). For 1000 nodes: log₃(1000) = log(1000)/log(3) ≈ 6.3 rounds. In practice, ~7 rounds reach all 1000 nodes. Round 0: 1 node. Round 1: 4 nodes (1 + 3). Round 2: 13 nodes (each of 4 tells 3, with some overlap). Rounds 3-7: Exponential growth continues. At 1 second per round, full propagation in ~7 seconds. This O(log N) propagation time is why gossip scales well—adding nodes increases propagation time only logarithmically. Option 2 is for fanout=10. Option 3 assumes linear propagation (incorrect). Option 4 assumes sequential one-by-one spread (ignores exponential nature of gossip).',
  },
  {
    id: 'mc2',
    question:
      'What is the primary difference between push gossip and pull gossip?',
    options: [
      'Push is faster but pull is more reliable',
      'Push gossip has nodes send updates to others; pull gossip has nodes request updates from others',
      'Push uses TCP; pull uses UDP',
      'Push is used for small clusters; pull is used for large clusters',
    ],
    correctAnswer: 1,
    explanation:
      "The fundamental difference is the direction of information flow. Push gossip (rumor mongering): Nodes with information actively push it to random peers. Node A has update → sends to B, C. Fast initial spread (exponential), but may not reach 100% (some nodes never randomly selected). Pull gossip: Nodes without information actively pull from random peers. Node D doesn't know → asks E \"what's your state?\". Slower initial spread, but ensures eventual convergence (uninformed nodes keep asking). Push-pull (hybrid): Nodes both push their state and pull peer's state in one exchange. Combines benefits—fast initial spread (push) plus guaranteed convergence (pull). Most production systems (Cassandra, Consul) use push-pull. Option 1 is partially true but doesn't explain the mechanism. Option 3 is incorrect—protocol choice is independent of transport. Option 4 is incorrect—not determined by cluster size.",
  },
  {
    id: 'mc3',
    question:
      'In gossip-based failure detection, why might a node be marked as "suspected" before being marked as "dead"?',
    options: [
      'To give operators time to manually intervene',
      'To distinguish between temporary network issues and actual node failure',
      'To reduce the number of gossip messages',
      'To comply with security policies',
    ],
    correctAnswer: 1,
    explanation:
      '"Suspected" state provides a grace period to distinguish temporary issues from permanent failures, reducing false positives. In Cassandra\'s gossip with phi accrual: Node\'s phi value exceeds threshold (e.g., 8) → Mark "suspected". Node remains suspected while phi stays high. If node recovers (phi drops), mark "alive" again (no unnecessary removal). If suspected for extended period (e.g., 30s) and phi remains high → Mark "down". This two-phase approach prevents thrashing: a node experiencing temporary network congestion might briefly be suspected but not removed, avoiding expensive operations (rebalancing data, updating routes). If immediately marked dead on first suspicion, a network blip causes unnecessary cluster changes. When node recovers seconds later, it must rejoin (expensive). Example: Node C has transient 10-second network delay. Without "suspected": Marked dead, cluster rebalances, C rejoins 10s later (wasteful). With "suspected": Marked suspected, network recovers, back to alive (no rebalancing). Option 1 is incorrect—automated systems don\'t wait for humans. Options 3 and 4 are unrelated to the suspected state\'s purpose.',
  },
  {
    id: 'mc4',
    question:
      'What optimization can reduce gossip network overhead when most nodes already have the same information?',
    options: [
      'Increase the gossip interval',
      'Use digest-based gossip (send hashes first, full state only if different)',
      'Reduce the cluster size',
      'Stop gossip entirely once converged',
    ],
    correctAnswer: 1,
    explanation:
      'Digest-based gossip (or anti-entropy) sends compact hashes/digests first, exchanging full state only when differences are detected. Process: Node A sends digest (hash of its state): digest_A = hash("state version X"). Node B compares with its digest: digest_B. If digest_A == digest_B: States match, no exchange needed (optimization!). If digest_A ≠ digest_B: Actual state differs, exchange full state. Benefit: When most nodes have the same state (common after convergence), most gossip exchanges are just small hash comparisons (32 bytes) instead of full state transfers (potentially KB-MB). This dramatically reduces network traffic. Example: 1000-node cluster, steady state (few updates). Without digests: Every gossip exchange sends full 10KB state = 10KB × 1000 × fanout 3 = 30MB/sec. With digests: Most exchanges send 32-byte digest (match) = 32B × 1000 × 3 = 96KB/sec (99% reduction). Only occasional state differences require full exchange. Cassandra and many gossip systems use this optimization. Option 1 slows propagation. Option 3 is impractical. Option 4 breaks continuous membership updates.',
  },
  {
    id: 'mc5',
    question:
      'In a gossip protocol, what problem does exponential decay help solve?',
    options: [
      'It prevents old/stale information from being gossiped indefinitely',
      'It reduces the cluster size over time',
      'It encrypts messages for security',
      'It elects a leader node',
    ],
    correctAnswer: 0,
    explanation:
      'Exponential decay reduces the probability of gossiping old information over time, preventing nodes from continuously spreading information that everyone already knows. Without decay: Node A learns X at T=0, gossips it every round forever. Even after all nodes know X (T=10), A continues gossiping X (wasteful). With exponential decay: Round 1-3: Gossip X every round (100% probability, rapid spread). Round 4-6: Gossip X probabilistically (50% probability, still spreading but less). Round 7+: Gossip X rarely (10% probability, almost stopped). After N rounds, effectively stop gossiping X. Benefit: Network bandwidth conserved—only new/recent information is actively spread, old information naturally fades. Example: Configuration update propagated to all 1000 nodes in 10 rounds. Without decay: Wasted bandwidth for next 1000 rounds gossiping same config. With decay: After 20 rounds, config gossip effectively stops (everyone knows). Cassandra uses a variant where updates have a "generation" number, and old generations are gossiped less frequently. Option 2 is incorrect—decay applies to information propagation, not cluster membership. Options 3 and 4 are unrelated to decay.',
  },
];
