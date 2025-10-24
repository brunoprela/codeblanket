/**
 * Multiple choice questions for S3 Architecture section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const s3ArchitectureMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What consistency model does S3 provide as of December 2020?',
    options: [
      'Eventual consistency',
      'Strong read-after-write consistency',
      'No consistency guarantees',
      'Causal consistency',
    ],
    correctAnswer: 1,
    explanation:
      'As of December 2020, S3 provides strong read-after-write consistency. After a successful PUT, GET immediately returns the latest version. Overwrites are immediately visible. Deletes immediately return 404. List operations immediately reflect changes. This simplifies application development significantly.',
  },
  {
    id: 'mc2',
    question: 'What is the maximum size of a single S3 object?',
    options: ['1 GB', '5 GB', '5 TB', 'Unlimited'],
    correctAnswer: 2,
    explanation:
      'S3 objects can be up to 5 TB. For uploads larger than 100 MB, multipart upload is recommended. Multipart upload allows uploading in parts (5 MB to 5 GB each) with up to 10,000 parts. This enables parallel uploads, resumable uploads, and faster large file uploads.',
  },
  {
    id: 'mc3',
    question: 'What is S3 Intelligent-Tiering?',
    options: [
      'Manual tier selection by user',
      'Automatic tier transitions based on access patterns with no retrieval fees',
      'Immediate deletion of old data',
      'Data compression service',
    ],
    correctAnswer: 1,
    explanation:
      'S3 Intelligent-Tiering automatically moves objects between access tiers based on access patterns: Frequent Access (< 30 days), Infrequent Access (30-90 days), Archive Instant Access (> 90 days), with optional Archive and Deep Archive tiers. No retrieval fees and no overhead to retrieve data. Small monthly monitoring fee.',
  },
  {
    id: 'mc4',
    question: 'How many 9s of durability does S3 Standard provide?',
    options: [
      '99.9% (3 nines)',
      '99.99% (4 nines)',
      '99.999999999% (11 nines)',
      '100%',
    ],
    correctAnswer: 2,
    explanation:
      'S3 Standard provides 99.999999999% (11 nines) durability. This means if you store 10 million objects, you can expect to lose 1 object every 10,000 years. Achieved through replication across multiple Availability Zones, erasure coding, continuous verification, and automatic healing.',
  },
  {
    id: 'mc5',
    question: 'What is the purpose of S3 Transfer Acceleration?',
    options: [
      'Compress data before upload',
      'Use CloudFront edge locations to speed up uploads to distant S3 regions',
      'Increase bandwidth limits',
      'Automatically shard large files',
    ],
    correctAnswer: 1,
    explanation:
      'S3 Transfer Acceleration uses CloudFront edge locations to upload data over AWS optimized network paths instead of public internet. Client uploads to nearest edge location, then data transfers to S3 region over AWS backbone. Can improve upload speeds by 50-500% for distant regions. Additional cost: $0.04-$0.08/GB.',
  },
];
