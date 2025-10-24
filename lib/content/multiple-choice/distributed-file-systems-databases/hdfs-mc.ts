/**
 * Multiple choice questions for HDFS section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const hdfsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the default block size in HDFS?',
    options: ['64 MB', '128 MB', '256 MB', '512 MB'],
    correctAnswer: 1,
    explanation:
      'HDFS uses 128 MB blocks by default (configurable). This is larger than GFS (64 MB) to further reduce metadata overhead at NameNode. Larger blocks mean fewer blocks per file, less metadata, and more efficient sequential I/O. Trade-off is wasted space for small files.',
  },
  {
    id: 'mc2',
    question: 'What is the role of the NameNode in HDFS?',
    options: [
      'Store actual data blocks',
      'Manage metadata and coordinate DataNodes',
      'Execute MapReduce jobs',
      'Compress data',
    ],
    correctAnswer: 1,
    explanation:
      'NameNode is the master that manages file system namespace (directory structure), block-to-DataNode mapping, and coordinates DataNode operations. It does NOT store actual data - DataNodes store blocks. NameNode keeps all metadata in memory for fast access.',
  },
  {
    id: 'mc3',
    question: 'How does HDFS achieve write reliability?',
    options: [
      'Write to single DataNode only',
      'Write to RAID array',
      'Pipeline replication: write to 3 DataNodes in sequence',
      'Broadcast to all DataNodes simultaneously',
    ],
    correctAnswer: 2,
    explanation:
      'HDFS uses pipeline replication: Client → DN1 → DN2 → DN3. Each DataNode forwards data to next in pipeline. This uses 1/3 the client bandwidth compared to star topology (client writing to all replicas). ACKs flow back through pipeline. All replicas must ACK before write is successful.',
  },
  {
    id: 'mc4',
    question: 'What handles HDFS NameNode failover in HA configuration?',
    options: [
      'DataNodes vote for new NameNode',
      'ZooKeeper coordinates automatic failover',
      'Client applications manage failover',
      'No automatic failover; manual intervention required',
    ],
    correctAnswer: 1,
    explanation:
      'ZooKeeper coordinates automatic failover via ZKFC (ZooKeeper Failover Controller). Active NameNode maintains session with ZK. On failure, ZK detects and triggers failover to Standby NameNode. ZK ensures only one Active NameNode (prevents split-brain). Failover takes ~30 seconds.',
  },
  {
    id: 'mc5',
    question: 'Why is HDFS Federation useful?',
    options: [
      'Increases data replication factor',
      'Enables multiple NameNodes managing different namespaces, scaling metadata',
      'Compresses data more efficiently',
      'Provides automatic backups',
    ],
    correctAnswer: 1,
    explanation:
      'HDFS Federation allows multiple independent NameNodes, each managing a portion of the namespace. This horizontally scales metadata capacity and throughput (single NameNode memory and CPU are bottlenecks). Different NameNodes manage different directories. All share the same DataNode pool.',
  },
];
