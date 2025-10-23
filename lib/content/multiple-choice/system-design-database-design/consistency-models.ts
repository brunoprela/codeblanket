/**
 * Multiple choice questions for Consistency Models section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const consistencymodelsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'consistency-models-q1',
    question:
      'What is the key difference between Linearizability and Sequential Consistency?',
    options: [
      'Linearizability is faster than Sequential Consistency',
      'Linearizability preserves real-time ordering of operations, Sequential Consistency does not',
      'Sequential Consistency is stronger than Linearizability',
      'Linearizability only works with single-node systems',
    ],
    correctAnswer: 1,
    explanation:
      "The key difference is that Linearizability preserves real-time ordering - if operation A completes before operation B starts (in real-time), all processes must see A before B. Sequential Consistency only requires a total order that respects each process's program order, but doesn't guarantee real-time ordering between different processes. For example, with Sequential Consistency, even if Process A writes X=1 and completes before Process B reads X in real-time, B might still read the old value, as long as there exists some valid ordering. Linearizability would not allow this.",
  },
  {
    id: 'consistency-models-q2',
    question:
      "Your social media app allows users to post updates and reply to posts. Users report seeing replies before the original posts. Which consistency model guarantees this won't happen?",
    options: [
      'Eventual Consistency',
      'Read-Your-Writes Consistency',
      'Causal Consistency',
      'Monotonic Read Consistency',
    ],
    correctAnswer: 2,
    explanation:
      "Causal Consistency is the right answer because it guarantees that causally-related operations are seen in order by all processes. A reply is causally dependent on the original post (you can't reply without seeing the post first), so Causal Consistency ensures the post is always seen before the reply. Eventual Consistency provides no ordering guarantees. Read-Your-Writes only ensures you see your own writes, not others'. Monotonic Reads prevents going backward in time but doesn't guarantee causal ordering between different users' operations.",
  },
  {
    id: 'consistency-models-q3',
    question:
      'A user updates their profile on your website. They refresh the page and expect to see their changes. Which consistency model guarantee is most important here?',
    options: [
      'Linearizability',
      'Read-Your-Writes Consistency',
      'Sequential Consistency',
      'Monotonic Write Consistency',
    ],
    correctAnswer: 1,
    explanation:
      "Read-Your-Writes Consistency is the most important guarantee here. It ensures that a process (user) always sees its own writes. This is critical for good user experience - if a user updates their profile and refreshes, they expect to see their changes immediately. This can be achieved without full Linearizability (which would be overkill) by techniques like session affinity (routing user to same replica) or client-side caching with version tracking. Linearizability is stronger than needed. Sequential/Monotonic Write don't specifically guarantee seeing your own writes.",
  },
  {
    id: 'consistency-models-q4',
    question:
      'Which of the following systems REQUIRES linearizability (strong consistency) and would break with eventual consistency?',
    options: [
      'Social media like/comment counter',
      'Product catalog for e-commerce site',
      'Distributed lock for leader election',
      'DNS record propagation',
    ],
    correctAnswer: 2,
    explanation:
      "Distributed lock for leader election requires linearizability. Leader election must ensure that all nodes agree on exactly one leader at any time - this requires strong consistency. With eventual consistency, you could have split-brain scenarios where two nodes each think they're the leader during network partitions or replication lag. Social media counters can tolerate being off by a few (eventual consistency fine). Product catalogs rarely change and can tolerate staleness. DNS explicitly uses eventual consistency with TTL-based propagation.",
  },
  {
    id: 'consistency-models-q5',
    question:
      'Amazon DynamoDB offers both eventually consistent reads (default) and strongly consistent reads (opt-in). Why might you choose eventually consistent reads despite the risk of reading stale data?',
    options: [
      'Eventually consistent reads are always more accurate',
      'Eventually consistent reads are cheaper and have lower latency',
      'Strongly consistent reads are not actually consistent',
      'There is no difference in practice',
    ],
    correctAnswer: 1,
    explanation:
      'Eventually consistent reads are cheaper and have lower latency. They can be served from any replica (typically the nearest one), resulting in ~1-5ms latency and lower cost. Strongly consistent reads must coordinate across replicas to ensure you get the latest data, resulting in ~10-20ms latency and higher cost (charged at 2x the rate). For many use cases (product catalogs, user profiles), the risk of reading slightly stale data (typically only 1-2 seconds old) is acceptable in exchange for faster, cheaper reads. The choice depends on your specific requirements.',
  },
];
