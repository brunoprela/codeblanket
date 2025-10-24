/**
 * Multiple choice questions for Vector Clocks & Version Vectors section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const vectorclocksMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'Why are wall-clock timestamps unreliable for ordering distributed events?',
    options: [
      'They are too slow to compute',
      'They use too much memory',
      'Clocks on different servers can be skewed and drift',
      'They only work in a single datacenter',
    ],
    correctAnswer: 2,
    explanation:
      'Wall clocks on different servers cannot be perfectly synchronized due to clock skew (initial difference), clock drift (different rates), and NTP limitations (~100ms accuracy). This means a later event might have an earlier timestamp, breaking causal ordering. Vector clocks solve this with logical time.',
  },
  {
    id: 'mc2',
    question:
      'Given vector clocks VC1=[3,2,1] and VC2=[4,3,2], what is their relationship?',
    options: [
      'VC1 and VC2 are concurrent',
      'VC1 happens before VC2',
      'VC2 happens before VC1',
      'They are equal',
    ],
    correctAnswer: 1,
    explanation:
      'VC1=[3,2,1] happens before VC2=[4,3,2] because ALL elements of VC1 are ≤ corresponding elements in VC2 (3≤4, 2≤3, 1≤2) AND at least one is strictly less. This means all events in VC1 causally precede VC2. VC2 is a later state that has seen all events from VC1.',
  },
  {
    id: 'mc3',
    question:
      'Given vector clocks VC1=[2,1,0] and VC2=[1,2,0], what is their relationship?',
    options: [
      'VC1 happens before VC2',
      'VC2 happens before VC1',
      'They are concurrent (conflict)',
      'VC1 is newer than VC2',
    ],
    correctAnswer: 2,
    explanation:
      'VC1=[2,1,0] and VC2=[1,2,0] are CONCURRENT because neither dominates: 2>1 (VC1 ahead in position 0) but 1<2 (VC2 ahead in position 1). This indicates the events happened independently without causal relationship, representing a conflict that requires resolution.',
  },
  {
    id: 'mc4',
    question:
      'What is the main disadvantage of vector clocks in very large systems?',
    options: [
      'They are too slow',
      'They cannot detect conflicts',
      'Their size grows with the number of servers',
      'They require synchronized physical clocks',
    ],
    correctAnswer: 2,
    explanation:
      'Vector clocks require one counter per server, so size is O(N) where N is number of servers. For 1000 servers, each object needs 1000 integers (4KB). This becomes prohibitive at scale. Solutions: dotted version vectors (prune old entries) or hybrid logical clocks (constant size).',
  },
  {
    id: 'mc5',
    question: 'Which production systems use vector clocks or version vectors?',
    options: [
      'Only academic research systems',
      'Amazon DynamoDB and Riak',
      'Only single-server databases',
      'Only systems that use physical timestamps',
    ],
    correctAnswer: 1,
    explanation:
      'Vector clocks/version vectors are production-proven: Amazon DynamoDB (shopping carts), Riak (multi-master replication), Voldemort (LinkedIn), CouchDB (document conflicts), Git (commit causality). Any multi-master system with concurrent writes needs causality tracking. This is industry standard, not theory.',
  },
];
