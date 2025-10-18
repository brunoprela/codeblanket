/**
 * Multiple choice questions for CAP Theorem Deep Dive section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const captheoremMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'cap-theorem-q1',
    question:
      'During a network partition between two datacenters, your Cassandra cluster (with consistency level ONE) continues accepting reads and writes in both datacenters. Some users see slightly stale data for a few seconds until the partition heals. Which CAP properties is the system providing?',
    options: [
      'Consistency and Availability (CA)',
      'Consistency and Partition Tolerance (CP)',
      'Availability and Partition Tolerance (AP)',
      'All three (CAP) because the system eventually becomes consistent',
    ],
    correctAnswer: 2,
    explanation:
      'The system is AP (Availability + Partition Tolerance). During the partition, the system remains available in both datacenters, accepting reads and writes, which means it prioritizes Availability. The fact that users see stale data means it sacrificed strong Consistency. Partition Tolerance is demonstrated because the system continues operating despite the network partition. The eventual consistency doesn\'t mean the system is "CAP" - CAP theorem states you can only have 2 of 3 during a partition, and this system chose AP. Option D is incorrect because "eventually consistent" is not the same as "consistent" in CAP terms (which means strongly consistent).',
    difficulty: 'medium',
  },
  {
    id: 'cap-theorem-q2',
    question:
      'You are designing a banking system for account transfers. During a network partition, a user tries to transfer $1000 from their account. The primary database node cannot communicate with the replica nodes to confirm the replication. What should the system do to maintain correctness?',
    options: [
      'Accept the transfer on the primary node and replicate later when partition heals (AP)',
      'Reject the transfer and return an error until partition heals (CP)',
      'Accept the transfer on primary and replica independently, resolve conflicts later (AP)',
      'Split the $1000 transfer across available nodes (partition-aware)',
    ],
    correctAnswer: 1,
    explanation:
      'The system should reject the transfer (CP approach) because banking requires strong consistency - account balances must always be accurate. Accepting the transfer without confirmed replication (Option A) risks data loss if the primary fails. Option C could result in inconsistent balances or double-transfers. Option D makes no sense for financial transactions. In banking, correctness (consistency) is more important than availability. Better to show "service temporarily unavailable" than to risk incorrect account balances. This is why banks typically use CP systems (PostgreSQL with synchronous replication) rather than AP systems.',
    difficulty: 'hard',
  },
  {
    id: 'cap-theorem-q3',
    question: 'Which of the following statements about CAP theorem is CORRECT?',
    options: [
      'NoSQL databases are always AP, while SQL databases are always CP',
      'You must choose 2 properties at design time and cannot change them',
      'Partition tolerance is optional if you have a reliable network',
      'Some databases (like Cassandra) allow tunable consistency per query, choosing between CP and AP behavior',
    ],
    correctAnswer: 3,
    explanation:
      'Option D is correct: Modern databases like Cassandra and DynamoDB support tunable consistency, allowing you to choose CP or AP behavior per query. For example, Cassandra with consistency level QUORUM behaves like CP (requires majority, unavailable without quorum), while consistency level ONE behaves like AP (high availability, eventual consistency). Option A is wrong because MySQL async replication is AP, while MongoDB with majority writes is CP. Option B is wrong because consistency can be tuned per operation. Option C is wrong because network partitions always happen in distributed systems - partition tolerance is mandatory, not optional.',
    difficulty: 'hard',
  },
  {
    id: 'cap-theorem-q4',
    question:
      'A distributed coordination service like ZooKeeper is used for leader election in a cluster. During a network partition, one group of nodes cannot communicate with another group. The minority partition stops accepting writes. What CAP classification is this and why?',
    options: [
      'AP - Because it uses eventual consistency to maintain availability',
      'CP - Because it requires majority quorum and becomes unavailable in minority partition',
      'CA - Because it provides both consistency and availability when possible',
      'AP - Because the majority partition remains available to some nodes',
    ],
    correctAnswer: 1,
    explanation:
      'ZooKeeper is CP (Consistency + Partition Tolerance). It requires a majority quorum for writes, meaning if a network partition splits nodes into majority and minority groups, the minority partition becomes unavailable (stops accepting writes) to maintain consistency. This prevents split-brain scenarios where two leaders could be elected simultaneously. Coordination services must prioritize consistency over availability because having two conflicting leaders would break the system. While the majority partition remains available (Option D mentions this), the system as a whole is classified as CP because it sacrifices availability in the minority partition to maintain consistency.',
    difficulty: 'medium',
  },
  {
    id: 'cap-theorem-q5',
    question:
      "Your Instagram-like social media feed system uses Cassandra. During a network partition, users can still post photos and view feeds, but some users don't see the latest posts for 2-3 seconds until replication completes. Why is this AP design appropriate for this use case?",
    options: [
      "It's actually incorrect - Instagram should use CP to ensure users always see the latest posts",
      'AP is appropriate because users prefer seeing slightly stale feeds over getting "service unavailable" errors',
      'AP is required because Instagram has too much data for a CP system',
      'AP is used only to save costs, not for technical reasons',
    ],
    correctAnswer: 1,
    explanation:
      'Option B is correct: For social media feeds, availability is more important than immediate consistency. Users expect the app to always work (availability) and won\'t notice or care if a feed is 2-3 seconds stale. Showing a slightly outdated feed is far better UX than showing "service unavailable" error. Option A is wrong because strong consistency is not critical for social feeds - the business impact of seeing a post 3 seconds late is negligible. Option C is wrong because scale doesn\'t force AP choice (you can shard CP systems too). Option D is wrong because AP is a conscious design choice for user experience, not cost savings. This demonstrates understanding of matching CAP choices to business requirements.',
    difficulty: 'medium',
  },
];
