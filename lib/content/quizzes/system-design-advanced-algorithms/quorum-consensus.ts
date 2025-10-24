/**
 * Quiz questions for Quorum Consensus section
 */

export const quorumconsensusQuiz = [
  {
    id: 'q1',
    question:
      'Explain the mathematical foundation of why W + R > N guarantees strong consistency in quorum systems. Walk through a concrete example with N=5, W=3, R=3.',
    sampleAnswer:
      'MATHEMATICAL PROOF: W + R > N ensures read and write quorums must overlap by at least one server. Why? Total servers = N. Write touches W servers. Read touches R servers. If W + R > N, then W + R ≥ N + 1, meaning the union of write and read quorums contains at least N+1 server instances, but only N servers exist. Therefore, at least ONE server must be in BOTH quorums (pigeonhole principle). CONCRETE EXAMPLE: N=5 servers {A,B,C,D,E}. W=3, R=3. Write "user:123 = v2" succeeds on servers {A,B,C}. Later, read "user:123" from servers {C,D,E}. Server C is in both quorums! C has v2, so read returns v2 (latest). Even if we read from different 3 servers {B,D,E}, server B overlaps. The key: With W=3 and R=3, we touch 6 server instances total, but only 5 servers exist, so minimum 1 overlap guaranteed. If W=2, R=2 (W+R=4≤5), could read from {D,E} and write to {A,B} with no overlap → stale data. The formula W+R>N is not arbitrary, it is mathematically necessary and sufficient for strong consistency.',
    keyPoints: [
      'W + R > N ensures at least one server in both quorums (pigeonhole principle)',
      'Overlapping server has latest write',
      'Example: W=3, R=3, N=5 → 6 instances, 5 servers → 1+ overlap',
      'Without overlap (W+R≤N), could read stale data',
      'Mathematical necessity: not just best practice, but provably required',
    ],
  },
  {
    id: 'q2',
    question:
      'Compare strict quorum vs sloppy quorum. When would you choose sloppy quorum despite the consistency trade-off? Give a real-world example.',
    sampleAnswer:
      'STRICT QUORUM: Write must succeed on designated replicas. Example: N=3 designated servers {A,B,C}, W=2. If A and B are down, write FAILS (availability sacrifice for consistency). SLOPPY QUORUM (Amazon Dynamo): Write can succeed on ANY N healthy servers, not just designated replicas. If A and B down, write to C and temporary server D. Later, when A/B recover, "hinted handoff" transfers data from D back to A/B. TRADE-OFFS: Strict = strong consistency, lower availability. Sloppy = higher availability, eventual consistency. WHEN TO CHOOSE SLOPPY: (1) Availability more critical than consistency. (2) Temporary failures common (network hiccups). (3) Acceptable to read slightly stale data. (4) Shopping cart, session store, click tracking. REAL-WORLD EXAMPLE: Amazon DynamoDB shopping cart. If 2 of 3 designated replicas are down, strict quorum would reject add-to-cart operations (terrible UX!). Sloppy quorum accepts write on any 2 healthy servers. Cart might be slightly inconsistent for seconds, but user can checkout (revenue preserved). For e-commerce, availability > perfect consistency. Hinted handoff ensures eventual consistency. This is why Amazon invented sloppy quorum for DynamoDB—optimizing for availability in production.',
    keyPoints: [
      'Strict quorum: requires designated replicas (strong consistency, lower availability)',
      'Sloppy quorum: accepts any N healthy servers (higher availability, eventual consistency)',
      'Hinted handoff: transfers data when designated replicas recover',
      'Use sloppy when availability more critical (e-commerce, carts, sessions)',
      'DynamoDB invented this for shopping cart availability',
    ],
  },
  {
    id: 'q3',
    question:
      'You are designing a read-heavy distributed database with 5 replicas. How would you configure N, W, R to optimize for read performance while maintaining strong consistency? Justify your choice.',
    sampleAnswer:
      'REQUIREMENT: Read-heavy workload (95% reads, 5% writes), need strong consistency. CONFIGURATION: N=5, W=4, R=2. VERIFICATION: W + R = 4 + 2 = 6 > 5 ✓ (strong consistency guaranteed). REASONING: (1) LOW R=2: Read only needs 2 responses (fast! wait for 2 servers, not 4). Read latency = P50 of 2 fastest servers. Much faster than R=3. (2) HIGH W=4: Write needs 4 acknowledgments (slower). But only 5% of requests, acceptable trade-off. (3) QUORUM OVERLAP: Read from 2 servers, write touched 4 servers → always overlap by at least 1 server (4+2=6>5). (4) FAULT TOLERANCE: Reads tolerate 3 failures (only need 2 servers). Writes tolerate 1 failure (need 4 servers). ALTERNATIVE (worse): N=5, W=3, R=3 is balanced but both operations slower. R=3 means wait for 3 servers (50% slower read). COMPARISON: W=4,R=2: Reads use P50 of 2 servers (~10ms). W=3,R=3: Reads use P50 of 3 servers (~15ms). For read-heavy workload, 50% read speedup is massive (95% of requests). The 5% writes being slower is acceptable. This is the classic tuning: optimize R for read-heavy, optimize W for write-heavy, maintaining W+R>N.',
    keyPoints: [
      'Read-heavy optimization: low R, high W (still W+R>N)',
      'N=5, W=4, R=2 → only need 2 responses for reads (fast)',
      'Writes slower (W=4) but acceptable for 5% of traffic',
      'Maintains strong consistency (W+R=6>5)',
      'Opposite for write-heavy: W=2, R=4',
    ],
  },
];
