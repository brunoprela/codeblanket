import { MultipleChoiceQuestion } from '@/lib/types';

export const distributedConsensusFundamentalsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'dcf-mc-1',
      question:
        'According to the CAP theorem, which two properties does Bitcoin choose to guarantee?',
      options: [
        'Consistency and Availability',
        'Consistency and Partition Tolerance',
        'Availability and Partition Tolerance',
        'Bitcoin violates CAP theorem entirely',
      ],
      correctAnswer: 1,
      explanation:
        'Bitcoin chooses Consistency and Partition Tolerance (CP). It prioritizes having a single consistent ledger (all nodes eventually agree on same blockchain) and tolerates network partitions (works even if network splits). Availability is sacrificed: during network partition, nodes on minority fork will eventually abandon their chain once they see the majority fork has more work. Real example: 2013 Bitcoin fork (v0.7 vs v0.8) created two chains for 6 hours. Eventually nodes converged on single chain, but some transactions were reversed. This is acceptable in blockchain because consistency (no permanent double-spends) matters more than availability (some nodes temporarily unreachable). Traditional databases often choose CA (assuming no partitions in datacenter), but Bitcoin operates on unreliable internet requiring P. Available + Partition Tolerant (AP) would allow conflicting states, unacceptable for money.',
    },
    {
      id: 'dcf-mc-2',
      question:
        'The FLP impossibility theorem proves that deterministic consensus is impossible in asynchronous systems. How does Bitcoin achieve consensus despite this?',
      options: [
        'Bitcoin uses synchronous communication with guaranteed message delivery times',
        'Bitcoin uses probabilistic (not deterministic) consensus with synchrony assumptions',
        'Bitcoin disproves FLP theorem through proof-of-work',
        'Bitcoin uses a centralized coordinator to ensure consensus',
      ],
      correctAnswer: 1,
      explanation:
        'Bitcoin circumvents FLP impossibility through probabilistic consensus with synchrony assumptions. FLP assumes: (1) Asynchronous system (unbounded delays), (2) Deterministic algorithm, (3) Even one faulty process. Bitcoin relaxes these: Uses synchrony assumptions (10-minute blocks expect bounded propagation), probabilistic finality (never 100% certain, confidence grows exponentially with confirmations), economic incentives making faults expensive. After 6 confirmations (~1 hour), reversal probability < 0.1%, good enough for practice even without theoretical guarantee. This is "eventual consistency" not "strong consistency." Answer A is wrong - Bitcoin doesn\'t guarantee message timing. Answer C is wrong - PoW doesn\'t violate impossibility theorem, just works around it. Answer D is wrong - Bitcoin is decentralized.',
    },
    {
      id: 'dcf-mc-3',
      question:
        'In Byzantine Fault Tolerance, what is the minimum number of total nodes (n) required to tolerate f Byzantine (malicious) nodes?',
      options: ['n ≥ 2f + 1', 'n ≥ 3f + 1', 'n ≥ 4f', 'n ≥ f²'],
      correctAnswer: 1,
      explanation:
        "BFT requires n ≥ 3f + 1 nodes to tolerate f Byzantine faults. Why? Need 2f+1 votes for agreement. With f malicious nodes voting incorrectly, remaining honest votes (n-f) must be ≥ 2f+1. Solving: n-f ≥ 2f+1 gives n ≥ 3f+1. Example: 10 nodes (n=10) tolerates 3 Byzantine (f=3): 3×3+1=10. With 3 malicious, 7 honest remain. Need 7 votes (2f+1=7) for agreement, which honest nodes can provide. With f=4, would need n≥13. PBFT, Tendermint, and most BFT protocols use this 3f+1 requirement. Bitcoin's 51% attack threshold is different: needs >50% *hash power* (not node count) for attack, which is harder to achieve than 33% nodes in permission systems. Answer A (2f+1) only handles crash faults, not Byzantine faults.",
    },
    {
      id: 'dcf-mc-4',
      question:
        'What is the key difference between "safety" and "liveness" in consensus protocols?',
      options: [
        'Safety means transactions complete quickly, liveness means they eventually complete',
        'Safety means nothing bad ever happens, liveness means something good eventually happens',
        'Safety is about security, liveness is about availability',
        'Safety and liveness are the same concept with different names',
      ],
      correctAnswer: 1,
      explanation:
        "Safety = nothing bad ever happens (no conflicting blocks permanently accepted). Liveness = something good eventually happens (new valid blocks eventually added). Blockchain examples - Safety violation: Two conflicting blocks both permanently finalized (double-spend). Liveness violation: Network stops producing blocks entirely. Bitcoin prioritizes liveness: during partition, keeps producing blocks (liveness maintained) but may create conflicting chains temporarily (safety violated until partition heals). Ethereum PoS prioritizes safety: during partition, may stop finalizing blocks (liveness violated) but never finalizes conflicting blocks (safety maintained). Trade-off: Can't guarantee both in asynchronous system (FLP). Most blockchains choose eventual safety with temporary liveness issues over risking permanent safety violations. Real scenario: 51% attack violates safety (can finalize conflicting blocks), network partition violates liveness (can't make progress).",
    },
    {
      id: 'dcf-mc-5',
      question:
        'Why does PBFT (Practical Byzantine Fault Tolerance) have O(n²) message complexity, and why does this limit its scalability?',
      options: [
        'Each node must send messages to every other node during consensus rounds',
        'The algorithm requires n² computational operations per consensus',
        'Storage requirements grow quadratically with number of nodes',
        'Network bandwidth decreases quadratically as nodes increase',
      ],
      correctAnswer: 0,
      explanation:
        "PBFT requires all-to-all communication: each node broadcasts to all others, and waits for 2f+1 responses. With n nodes: each sends n-1 messages, total = n(n-1) ≈ n² messages per consensus round. Example: 10 nodes = 90 messages, 100 nodes = 9,900 messages, 1,000 nodes = ~1M messages per block. This limits scalability: at 1,000 nodes with 1-second blocks, each node processes 1M messages/sec. At 10Kb per message = 10GB/sec bandwidth per node. Compare to Bitcoin/Ethereum: nodes only communicate with neighbors (O(n) messages), though consensus takes longer. This is why PBFT variants (Tendermint, Hotstuff) limit validators to <200 typically, while Bitcoin has 10,000+ full nodes. Trade-off: PBFT gets fast finality (1-3 seconds) but can't scale to thousands of nodes. Bitcoin gets massive decentralization but slow finality (10 minutes).",
    },
  ];
