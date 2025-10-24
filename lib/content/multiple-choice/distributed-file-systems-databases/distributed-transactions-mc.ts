/**
 * Multiple choice questions for Distributed Transactions section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const distributedTransactionsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What are the two phases in Two-Phase Commit (2PC)?',
    options: [
      'Read and Write',
      'Prepare (voting) and Commit (decision)',
      'Lock and Unlock',
      'Backup and Restore',
    ],
    correctAnswer: 1,
    explanation:
      'Two-Phase Commit has two phases: (1) Prepare phase - coordinator asks all participants "can you commit?", participants vote YES/NO. (2) Commit phase - coordinator decides (COMMIT if all YES, ABORT if any NO) and tells all participants. All participants must follow coordinator\'s decision.',
  },
  {
    id: 'mc2',
    question: 'What is the main problem with Two-Phase Commit (2PC)?',
    options: [
      'Too fast',
      'Too expensive',
      'Blocking - coordinator failure leaves participants waiting indefinitely',
      'Cannot handle more than 2 nodes',
    ],
    correctAnswer: 2,
    explanation:
      "2PC is a blocking protocol. If coordinator fails after participants vote but before sending decision, participants are stuck holding locks and cannot proceed. They don't know if coordinator decided COMMIT or ABORT. Manual intervention required. This makes 2PC unsuitable for high-availability systems.",
  },
  {
    id: 'mc3',
    question: 'What is the Saga pattern used for?',
    options: [
      'Data compression',
      'Long-running transactions using local transactions + compensating transactions',
      'Query optimization',
      'Data replication',
    ],
    correctAnswer: 1,
    explanation:
      'Saga pattern breaks long-running transactions into sequence of local transactions, each with compensating transaction for rollback. On failure, execute compensating transactions in reverse order. Provides eventual consistency without distributed locks. Better availability than 2PC but weaker consistency guarantees.',
  },
  {
    id: 'mc4',
    question: 'What consensus algorithm does Raft use for leader election?',
    options: [
      'Two-Phase Commit',
      'Majority voting with terms and log replication',
      'Paxos Multi-Decree',
      'Gossip protocol',
    ],
    correctAnswer: 1,
    explanation:
      'Raft uses majority voting for leader election. Followers timeout and become candidates, request votes from other nodes. Candidate receiving majority votes becomes leader. Leader sends heartbeats to maintain authority. Raft is simpler than Paxos while providing same guarantees. Used by etcd, Consul, TiKV.',
  },
  {
    id: 'mc5',
    question:
      'What does Google Spanner use to achieve global strong consistency?',
    options: [
      'Two-Phase Commit only',
      'TrueTime API with GPS and atomic clocks',
      'Eventual consistency',
      'Gossip protocol',
    ],
    correctAnswer: 1,
    explanation:
      'Spanner uses TrueTime API which provides time intervals with bounded uncertainty (<10ms) using GPS and atomic clocks in every datacenter. Spanner waits out uncertainty before committing, guaranteeing global ordering. This enables external consistency - strongest consistency guarantee for distributed database.',
  },
];
