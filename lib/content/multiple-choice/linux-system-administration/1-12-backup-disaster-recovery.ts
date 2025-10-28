/**
 * Multiple choice questions for Backup & Disaster Recovery
 */

import { MultipleChoiceQuestion } from '../../../types';

export const backupDisasterRecoveryMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'backup-mc-1',
    question: 'What is RPO (Recovery Point Objective)?',
    options: [
      'Maximum acceptable downtime',
      'Maximum acceptable data loss',
      'Backup frequency',
      'Restore speed',
    ],
    correctAnswer: 1,
    explanation:
      'RPO (Recovery Point Objective) is the maximum acceptable amount of data loss measured in time. If RPO is 1 hour, you can tolerate losing up to 1 hour of data. This determines backup frequency. RTO is recovery time objective (downtime).',
    difficulty: 'easy',
    topic: 'DR Concepts',
  },
  {
    id: 'backup-mc-2',
    question: 'Which backup type requires the least storage space?',
    options: [
      'Full backup',
      'Differential backup',
      'Incremental backup',
      'Mirror backup',
    ],
    correctAnswer: 2,
    explanation:
      'Incremental backups store only changes since the last backup (full or incremental), requiring the least storage. Differential stores changes since last full backup (more than incremental). Full backup stores everything (most storage). Trade-off: incremental requires all backups to restore.',
    difficulty: 'medium',
    topic: 'Backup Types',
  },
  {
    id: 'backup-mc-3',
    question: 'What is the 3-2-1 backup rule?',
    options: [
      '3 backups daily, 2 weekly, 1 monthly',
      '3 copies, 2 different media, 1 offsite',
      '3 servers, 2 regions, 1 backup',
      '3 full, 2 differential, 1 incremental',
    ],
    correctAnswer: 1,
    explanation:
      'The 3-2-1 backup rule: maintain 3 copies of data (original + 2 backups), on 2 different types of media (e.g., disk + tape, or local + cloud), with 1 copy offsite (different location). This protects against hardware failure, site disasters, and human error.',
    difficulty: 'easy',
    topic: 'Backup Best Practices',
  },
  {
    id: 'backup-mc-4',
    question:
      'Which AWS service provides automated, policy-based backup across AWS services?',
    options: [
      'EBS Snapshots',
      'S3 Versioning',
      'AWS Backup',
      'AWS Storage Gateway',
    ],
    correctAnswer: 2,
    explanation:
      'AWS Backup is a fully managed service that centralizes and automates backups across AWS services (EBS, RDS, DynamoDB, EFS, etc.) with policy-based retention, cross-region copy, and compliance reporting. EBS Snapshots are manual/automated for EBS only.',
    difficulty: 'medium',
    topic: 'AWS Backup Services',
  },
  {
    id: 'backup-mc-5',
    question: 'For RPO of 5 minutes, what backup strategy is needed?',
    options: [
      'Daily full backups',
      'Hourly incremental backups',
      'Continuous replication or 5-minute snapshots',
      'Weekly backups',
    ],
    correctAnswer: 2,
    explanation:
      'RPO of 5 minutes means you can only lose 5 minutes of data. This requires continuous replication (database streaming replication, S3 CRR) or snapshots/backups every 5 minutes. Hourly or daily backups would violate the RPO requirement. This is typically for mission-critical systems.',
    difficulty: 'advanced',
    topic: 'RPO/RTO Planning',
  },
];
