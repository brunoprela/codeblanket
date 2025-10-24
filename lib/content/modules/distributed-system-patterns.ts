/**
 * Distributed System Patterns Module
 * Aggregates sections, quizzes, and multiple choice questions
 */

import { Module } from '../../types';

// Import sections
import { leaderelectionSection } from '../sections/distributed-system-patterns/leader-election';
import { writeaheadlogSection } from '../sections/distributed-system-patterns/write-ahead-log';
import { segmentedlogSection } from '../sections/distributed-system-patterns/segmented-log';
import { highwatermarkSection } from '../sections/distributed-system-patterns/high-water-mark';
import { leaseSection } from '../sections/distributed-system-patterns/lease';
import { heartbeatSection } from '../sections/distributed-system-patterns/heartbeat';
import { gossipprotocolSection } from '../sections/distributed-system-patterns/gossip-protocol';
import { phiaccrualfailuredetectorSection } from '../sections/distributed-system-patterns/phi-accrual-failure-detector';
import { splitbrainresolutionSection } from '../sections/distributed-system-patterns/split-brain-resolution';
import { hintedhandoffSection } from '../sections/distributed-system-patterns/hinted-handoff';
import { readrepairSection } from '../sections/distributed-system-patterns/read-repair';
import { antientropySection } from '../sections/distributed-system-patterns/anti-entropy';

// Import quizzes
import { leaderelectionQuiz } from '../quizzes/distributed-system-patterns/leader-election';
import { writeaheadlogQuiz } from '../quizzes/distributed-system-patterns/write-ahead-log';
import { segmentedlogQuiz } from '../quizzes/distributed-system-patterns/segmented-log';
import { highwatermarkQuiz } from '../quizzes/distributed-system-patterns/high-water-mark';
import { leaseQuiz } from '../quizzes/distributed-system-patterns/lease';
import { heartbeatQuiz } from '../quizzes/distributed-system-patterns/heartbeat';
import { gossipprotocolQuiz } from '../quizzes/distributed-system-patterns/gossip-protocol';
import { phiaccrualfailuredetectorQuiz } from '../quizzes/distributed-system-patterns/phi-accrual-failure-detector';
import { splitbrainresolutionQuiz } from '../quizzes/distributed-system-patterns/split-brain-resolution';
import { hintedhandoffQuiz } from '../quizzes/distributed-system-patterns/hinted-handoff';
import { readrepairQuiz } from '../quizzes/distributed-system-patterns/read-repair';
import { antientropyQuiz } from '../quizzes/distributed-system-patterns/anti-entropy';

// Import multiple choice
import { leaderelectionMultipleChoice } from '../multiple-choice/distributed-system-patterns/leader-election';
import { writeaheadlogMultipleChoice } from '../multiple-choice/distributed-system-patterns/write-ahead-log';
import { segmentedlogMultipleChoice } from '../multiple-choice/distributed-system-patterns/segmented-log';
import { highwatermarkMultipleChoice } from '../multiple-choice/distributed-system-patterns/high-water-mark';
import { leaseMultipleChoice } from '../multiple-choice/distributed-system-patterns/lease';
import { heartbeatMultipleChoice } from '../multiple-choice/distributed-system-patterns/heartbeat';
import { gossipprotocolMultipleChoice } from '../multiple-choice/distributed-system-patterns/gossip-protocol';
import { phiaccrualfailuredetectorMultipleChoice } from '../multiple-choice/distributed-system-patterns/phi-accrual-failure-detector';
import { splitbrainresolutionMultipleChoice } from '../multiple-choice/distributed-system-patterns/split-brain-resolution';
import { hintedhandoffMultipleChoice } from '../multiple-choice/distributed-system-patterns/hinted-handoff';
import { readrepairMultipleChoice } from '../multiple-choice/distributed-system-patterns/read-repair';
import { antientropyMultipleChoice } from '../multiple-choice/distributed-system-patterns/anti-entropy';

export const distributedSystemPatternsModule: Module = {
  id: 'distributed-system-patterns',
  title: 'Distributed System Patterns',
  description:
    'Essential patterns for building robust distributed systems including leader election, replication, failure detection, and consistency mechanisms',
  category: 'undefined',
  difficulty: 'undefined',
  estimatedTime: 'undefined',
  prerequisites: [],
  icon: '🔄',
  keyTakeaways: [
    'Leader election ensures single coordination point while preventing split-brain through quorum and fencing tokens',
    'Write-Ahead Log (WAL) provides durability through sequential writes, enabling fast commits and crash recovery',
    'Segmented logs enable efficient compaction, retention policies, and parallel processing without blocking writes',
    'High-Water Mark coordinates replication by distinguishing committed (safe to read) from uncommitted data',
    'Leases provide time-bound resource ownership with automatic expiration, preventing deadlocks from node failures',
    'Heartbeats enable failure detection, with intervals and timeouts tuned based on network characteristics',
    'Gossip protocol achieves O(log N) information propagation with no single point of failure or coordination bottleneck',
    'Phi Accrual Failure Detector adapts to network conditions by learning heartbeat patterns, providing continuous suspicion levels',
    'Split-brain resolution uses quorum (only majority operates) and fencing tokens (reject stale leaders) for consistency',
    "Hinted handoff improves write availability during temporary failures but doesn't count toward read quorum",
    'Read repair fixes hot data quickly by piggybacking on reads, while anti-entropy ensures completeness for all data',
    'Merkle trees enable O(log N) anti-entropy comparison, identifying differing ranges without full dataset transfer',
  ],
  learningObjectives: [
    'Understand when and how to implement leader election using Raft, ZooKeeper, or etcd',
    'Design WAL-based systems with appropriate checkpointing strategies for durability and performance',
    'Implement segmented logs with compaction strategies for efficient storage and retention',
    'Use high-water marks to coordinate replication and provide consistency guarantees',
    'Apply leases for distributed locking and leader election with proper renewal strategies',
    'Configure heartbeat intervals and timeouts to balance detection speed vs false positives',
    'Implement gossip protocols for scalable cluster membership and failure detection',
    'Apply phi accrual failure detection for adaptive, network-aware failure detection',
    'Prevent split-brain scenarios using quorum-based decisions and fencing tokens',
    'Use hinted handoff to improve write availability while maintaining consistency',
    'Combine read repair and anti-entropy for both fast convergence and guaranteed completeness',
    'Design Merkle tree-based anti-entropy with appropriate depth and leaf granularity',
  ],
  sections: [
    {
      ...leaderelectionSection,
      quiz: leaderelectionQuiz,
      multipleChoice: leaderelectionMultipleChoice,
    },
    {
      ...writeaheadlogSection,
      quiz: writeaheadlogQuiz,
      multipleChoice: writeaheadlogMultipleChoice,
    },
    {
      ...segmentedlogSection,
      quiz: segmentedlogQuiz,
      multipleChoice: segmentedlogMultipleChoice,
    },
    {
      ...highwatermarkSection,
      quiz: highwatermarkQuiz,
      multipleChoice: highwatermarkMultipleChoice,
    },
    {
      ...leaseSection,
      quiz: leaseQuiz,
      multipleChoice: leaseMultipleChoice,
    },
    {
      ...heartbeatSection,
      quiz: heartbeatQuiz,
      multipleChoice: heartbeatMultipleChoice,
    },
    {
      ...gossipprotocolSection,
      quiz: gossipprotocolQuiz,
      multipleChoice: gossipprotocolMultipleChoice,
    },
    {
      ...phiaccrualfailuredetectorSection,
      quiz: phiaccrualfailuredetectorQuiz,
      multipleChoice: phiaccrualfailuredetectorMultipleChoice,
    },
    {
      ...splitbrainresolutionSection,
      quiz: splitbrainresolutionQuiz,
      multipleChoice: splitbrainresolutionMultipleChoice,
    },
    {
      ...hintedhandoffSection,
      quiz: hintedhandoffQuiz,
      multipleChoice: hintedhandoffMultipleChoice,
    },
    {
      ...readrepairSection,
      quiz: readrepairQuiz,
      multipleChoice: readrepairMultipleChoice,
    },
    {
      ...antientropySection,
      quiz: antientropyQuiz,
      multipleChoice: antientropyMultipleChoice,
    },
  ],
};
