/**
 * Quiz questions for Vector Clocks & Version Vectors section
 */

export const vectorclocksQuiz = [
  {
    id: 'q1',
    question:
      'Explain why wall-clock timestamps are unreliable for ordering events in distributed systems. Provide a concrete scenario where timestamps give wrong ordering.',
    sampleAnswer:
      'PROBLEM: Wall clocks cannot be perfectly synchronized across servers due to clock skew (initial difference), clock drift (different advancement rates), and NTP limitations (~100ms accuracy). CONCRETE SCENARIO: Server A (fast clock): 10:00:00.500. Server B (slow clock): 10:00:00.200. Event 1: Client writes to Server A at actual time T1. A records timestamp 10:00:00.500. Event 2: 100ms later (actual time T1+100ms), client writes to Server B. B records timestamp 10:00:00.300. Result: timestamp(Event2)=10:00:00.300 < timestamp(Event1)=10:00:00.500, even though Event2 happened AFTER Event1! If we use timestamps to determine causality, we would incorrectly conclude Event1 happened after Event2. This breaks causal consistency. REAL-WORLD IMPACT: Database replication could apply operations in wrong order (delete before insert → data loss). Vector clocks solve this by using logical counters independent of physical time, tracking actual happens-before relationships through message passing and event ordering.',
    keyPoints: [
      'Clock skew and drift make clocks unreliable across servers',
      'NTP has ~100ms accuracy, not precise enough',
      'Later events can have earlier timestamps',
      'Breaks causal consistency in distributed systems',
      'Vector clocks use logical time to track true causality',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk through how two vector clocks [2,1,0] and [1,2,0] are compared. What relationship exists between them and what does this mean for conflict resolution?',
    sampleAnswer:
      'COMPARISON ALGORITHM: Check if VC1 ≤ VC2 (all elements) AND VC1 ≠ VC2. VC1=[2,1,0], VC2=[1,2,0]. Element 0: 2 > 1 (VC1 not ≤ VC2). Element 1: 1 < 2 (VC2 not ≤ VC1). Element 2: 0 = 0 (equal). RESULT: Neither VC1 happens-before VC2, nor VC2 happens-before VC1 → They are CONCURRENT. MEANING: Events with these vector clocks happened independently without causal relationship (no message passing between them). EXAMPLE: Server A wrote user:123="Alice" with VC:[2,1,0]. Server B wrote user:123="Alicia" with VC:[1,2,0]. These are concurrent writes! CONFLICT RESOLUTION REQUIRED: (1) Cannot automatically determine which is "newer". (2) System must preserve both versions or use resolution strategy (LWW, merge, CRDT, client resolution). (3) In shopping cart: merge both items. (4) In bank account: flag for manual review. This is the fundamental value of vector clocks: they DETECT when events are concurrent and conflicts exist, unlike timestamps which would pick arbitrary "winner".',
    keyPoints: [
      'Compare element-wise: check if all elements ≤',
      '[2,1,0] vs [1,2,0]: neither dominates (concurrent)',
      'Concurrent = events happened independently',
      'Requires conflict resolution (merge, LWW, CRDT)',
      'Vector clocks detect conflicts timestamps would miss',
    ],
  },
  {
    id: 'q3',
    question:
      'Amazon Dynamo uses vector clocks for shopping carts. Explain the complete flow: how conflicts are detected and how shopping cart conflicts should be resolved.',
    sampleAnswer:
      'DYNAMO SHOPPING CART FLOW: (1) READ: Client fetches cart, server returns cart + vector clock. Example: cart=[item1], VC:[1,0,0]. (2) MODIFY: Client adds item2 locally. (3) WRITE: Client writes cart=[item1,item2] with original VC:[1,0,0]. Server increments its counter → new VC:[2,0,0]. (4) CONCURRENT WRITE: Another client simultaneously added item3 via different server. Their write: cart=[item1,item3], VC:[1,1,0]. (5) CONFLICT DETECTION: Next read returns both versions. VC:[2,0,0] vs VC:[1,1,0]. Compare: 2>1 (first element favors first). 0<1 (second element favors second). CONCURRENT! (6) RESOLUTION: Shopping cart conflicts resolve via MERGE (union of items). Merged cart: [item1, item2, item3]. New VC: [2,1,0] (takes max of each element, then increments on write). WHY THIS WORKS: Users expect additive behavior for shopping carts. Losing items would be terrible UX. Merge strategy ensures no item loss. This is domain-specific: shopping carts use merge, but bank accounts would flag for review. PRODUCTION LESSON: Dynamo returns ALL conflicting versions to client, client merges and writes back resolved version with combined VC. This is client-side conflict resolution.',
    keyPoints: [
      'Read returns data + vector clock',
      'Write includes previous VC (causality tracking)',
      'Concurrent writes detected via VC comparison',
      'Shopping carts: merge strategy (union of items)',
      'Client-side resolution: merge and write back',
    ],
  },
];
