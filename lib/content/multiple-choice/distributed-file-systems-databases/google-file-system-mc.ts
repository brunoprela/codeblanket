/**
 * Multiple choice questions for Google File System section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const gfsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the default chunk size in Google File System?',
    options: ['4 MB', '16 MB', '64 MB', '128 MB'],
    correctAnswer: 2,
    explanation:
      'GFS uses 64 MB chunks (unusually large compared to traditional file systems). This reduces metadata at the master, reduces client-master communication, and allows long-lived TCP connections. The trade-off is wasted space for small files and potential hot spots for popular small files.',
  },
  {
    id: 'mc2',
    question: 'How does GFS handle the read path to avoid master bottlenecks?',
    options: [
      'All reads go through the master',
      'Master routes all data through itself',
      'Client gets metadata from master, then reads directly from chunkservers',
      'Chunkservers cache all metadata locally',
    ],
    correctAnswer: 2,
    explanation:
      'GFS separates control flow from data flow. Clients contact the master only for metadata (which chunkservers hold which chunks), then read data directly from chunkservers. The master is NOT involved in data transfer, preventing it from becoming a bottleneck.',
  },
  {
    id: 'mc3',
    question: 'What consistency guarantee does GFS provide for record appends?',
    options: [
      'Strong consistency (exactly-once)',
      'Linearizability',
      'At-least-once (may have duplicates and gaps)',
      'No consistency guarantees',
    ],
    correctAnswer: 2,
    explanation:
      'GFS provides "at-least-once" semantics for record appends. Records are guaranteed to be appended but may be duplicated (if retries) or have padding/gaps. Applications must handle deduplication using record checksums and unique IDs. This weak consistency enables high availability and performance.',
  },
  {
    id: 'mc4',
    question: 'What happens when the GFS master fails?',
    options: [
      'All data is permanently lost',
      'System is unavailable until manual intervention',
      'Shadow master is promoted; operation log + checkpoint used for recovery',
      'Chunkservers automatically elect a new master',
    ],
    correctAnswer: 2,
    explanation:
      'GFS master state is persistent (operation log + checkpoints replicated to multiple machines). Shadow masters (read-only replicas) can take over. New master loads checkpoint, replays operation log, polls chunkservers for chunk locations, and resumes operations. Brief unavailability during failover.',
  },
  {
    id: 'mc5',
    question: 'Why does GFS use a lease mechanism for writes?',
    options: [
      'To encrypt data',
      'To grant one replica authority to order mutations without master coordination',
      'To backup data to tape',
      'To compress chunks',
    ],
    correctAnswer: 1,
    explanation:
      'Leases (60-second timeout) grant one replica (primary) authority to order mutations. This allows the primary to serialize writes without contacting the master for every operation, reducing master bottleneck and improving write performance. Master grants lease, primary orders writes, lease prevents split-brain.',
  },
];
