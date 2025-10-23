/**
 * Multiple choice questions for PACELC Theorem section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const pacelctheoremMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'pacelc-q1',
    question:
      'What does the "E" and "L" in PACELC theorem represent, and why is this important?',
    options: [
      'E=Error handling, L=Load balancing; it addresses how systems handle failures',
      'E=Else (no partition), L=Latency; it addresses trade-offs during normal operation',
      'E=Encryption, L=Logging; it addresses security and observability',
      'E=Eventual consistency, L=Linearizability; it addresses consistency models',
    ],
    correctAnswer: 1,
    explanation:
      'PACELC\'s "E" stands for "Else" (when there is NO partition) and "L" stands for Latency. This is important because CAP theorem only addresses behavior during partitions (which are rare), but PACELC recognizes that systems must make trade-offs even during normal operation (99.9% of the time). The ELC part states: during normal operation, you must choose between Low Latency (reading from any replica, potential staleness) and Consistency (reading from authoritative source, higher latency). This explains why Cassandra is faster than HBase even when both are healthy - Cassandra chooses Latency (EL) while HBase chooses Consistency (EC).',
  },
  {
    id: 'pacelc-q2',
    question:
      'Cassandra is classified as PA/EL. What does this mean in practice when the system has NO network partition and is operating normally?',
    options: [
      'Cassandra reads from all replicas and returns the most recent value, ensuring consistency but higher latency',
      'Cassandra reads from the nearest replica for fast response, but might return slightly stale data (eventual consistency)',
      'Cassandra rejects read requests to maintain strong consistency',
      'Cassandra waits for quorum acknowledgment before responding, ensuring consistency',
    ],
    correctAnswer: 1,
    explanation:
      "PA/EL means during normal operation (EL), Cassandra chooses Latency over Consistency. It reads from the nearest/fastest replica (often just ONE replica) to minimize latency, which means it might return slightly stale data if that replica hasn't received the latest write yet. This is eventual consistency - the system will become consistent over time, but reads prioritize speed over guaranteed freshness. Option A describes PC/EC behavior (HBase). Option C describes unavailability. Option D describes QUORUM reads (which would be PA/EC configuration).",
  },
  {
    id: 'pacelc-q3',
    question:
      'Why does HBase (PC/EC) have higher read latency than Cassandra (PA/EL) even when both systems are healthy with no network partitions?',
    options: [
      'HBase is written in Java while Cassandra is optimized in C++',
      'HBase stores more data per node than Cassandra',
      'HBase prioritizes consistency over latency (EC), requiring reads from authoritative source, while Cassandra prioritizes latency (EL), reading from any replica',
      'HBase uses synchronous replication while Cassandra uses no replication',
    ],
    correctAnswer: 2,
    explanation:
      'The latency difference comes from the PACELC trade-off during normal operation. HBase (PC/EC) chooses Consistency over Latency even without partitions - reads must go to the authoritative RegionServer which might not be the closest one, adding network latency. Writes wait for WAL sync and replication acknowledgment. Cassandra (PA/EL) chooses Latency over Consistency - reads can go to any replica (usually the nearest), providing sub-millisecond response but potentially stale data. This architectural difference (EC vs EL) is why HBase has higher latency even when healthy. Options A, B, D are not the fundamental reasons.',
  },
  {
    id: 'pacelc-q4',
    question:
      'Your system needs <5ms read latency for a globally distributed user base and can tolerate seeing slightly stale data (1-2 seconds old). Which PACELC classification best fits your requirements?',
    options: [
      'PC/EC - Strong consistency ensures data accuracy which is most important',
      'PA/EL - High availability and low latency with eventual consistency fits the requirements',
      'PA/EC - Availability during partitions but consistency normally',
      "PC/EL - This doesn't exist as a common pattern",
    ],
    correctAnswer: 1,
    explanation:
      'PA/EL (like Cassandra or DynamoDB) best fits these requirements. The "<5ms latency" requirement strongly suggests needing to read from local replicas without waiting for quorum or authoritative sources, which is the EL (Latency over Consistency) choice. The ability to "tolerate slightly stale data" means eventual consistency is acceptable, which aligns with both PA (availability during partitions) and EL (low latency normally). PC/EC systems like HBase typically have 10-50ms latency because they prioritize consistency. PA/EC might work but still has higher latency than PA/EL during normal operation.',
  },
  {
    id: 'pacelc-q5',
    question:
      'Why is PACELC theorem considered more practical than CAP theorem for day-to-day system design decisions?',
    options: [
      'PACELC is newer and replaces CAP theorem entirely',
      'CAP only addresses partition behavior (rare), while PACELC addresses normal operation behavior (99.9% of time)',
      'PACELC is simpler to understand than CAP',
      'PACELC allows you to have all three properties (Consistency, Availability, Partition Tolerance)',
    ],
    correctAnswer: 1,
    explanation:
      "PACELC is more practical because it addresses the trade-offs that affect your system during normal operation (ELC), which is 99.9% of the time. CAP only describes behavior during network partitions, which are rare events (minutes per year). The daily performance of your system is determined by the ELC trade-off: do you want low latency (read from any replica, eventual consistency) or strong consistency (read from authoritative source, higher latency)? This explains real-world performance differences between Cassandra and HBase even when both are healthy. PACELC extends CAP, it doesn't replace it (Option A wrong). It's not simpler (Option C wrong). It still has the same fundamental trade-offs (Option D wrong).",
  },
];
