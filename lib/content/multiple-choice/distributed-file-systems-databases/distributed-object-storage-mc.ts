/**
 * Multiple choice questions for Distributed Object Storage section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const distributedObjectStorageMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question: 'What is the CRUSH algorithm in Ceph used for?',
      options: [
        'Data compression',
        'Deterministic object placement without central metadata lookup',
        'User authentication',
        'Data encryption',
      ],
      correctAnswer: 1,
      explanation:
        'CRUSH (Controlled Replication Under Scalable Hashing) deterministically calculates where objects should be stored based on object ID and cluster map. No central metadata server needed! Any node can calculate location, eliminating metadata bottleneck and enabling massive scalability.',
    },
    {
      id: 'mc2',
      question: 'What are the three types of storage Ceph provides?',
      options: [
        'Object, Block, and File',
        'RAM, SSD, and HDD',
        'Hot, Warm, and Cold',
        'Public, Private, and Hybrid',
      ],
      correctAnswer: 0,
      explanation:
        'Ceph provides unified storage: (1) Object storage via RADOS Gateway (RGW) with S3/Swift API, (2) Block storage via RADOS Block Device (RBD) for VMs and containers, (3) File storage via CephFS (POSIX filesystem). All built on RADOS core.',
    },
    {
      id: 'mc3',
      question: 'What is the main advantage of MinIO over Ceph?',
      options: [
        'Larger scale capacity',
        'Block storage support',
        'Simplicity and easier operations',
        'Older and more mature',
      ],
      correctAnswer: 2,
      explanation:
        'MinIO is significantly simpler than Ceph - faster deployment, easier operations, Kubernetes-native, and straightforward S3-compatible interface. While Ceph offers more features (block, file) and scales larger, MinIO wins on simplicity and operational ease. Ideal for cloud-native applications.',
    },
    {
      id: 'mc4',
      question: 'How does erasure coding (8+4) differ from 3x replication?',
      options: [
        'Erasure coding uses more storage space',
        'Erasure coding uses 1.5x storage overhead vs 3x for replication',
        'Erasure coding is faster for all operations',
        'Erasure coding provides less durability',
      ],
      correctAnswer: 1,
      explanation:
        'Erasure coding (8+4) splits data into 8 data chunks + 4 parity chunks, providing 1.5x storage overhead while tolerating loss of any 4 chunks. 3x replication uses 3x overhead and tolerates loss of 2 copies. Erasure coding saves 50% storage but requires more CPU and has slightly slower reads.',
    },
    {
      id: 'mc5',
      question: 'What is the role of Monitors (MON) in Ceph?',
      options: [
        'Store actual data',
        'Maintain cluster map and provide consensus via Paxos',
        'Handle client authentication only',
        'Compress data',
      ],
      correctAnswer: 1,
      explanation:
        'Ceph Monitors maintain the cluster map (OSD status, placement groups, CRUSH rules) and provide consensus via Paxos. They do NOT store actual data. Minimum 3 monitors recommended for high availability. Monitors coordinate cluster state but are not in the data path.',
    },
  ];
