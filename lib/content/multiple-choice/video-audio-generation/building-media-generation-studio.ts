/**
 * Multiple choice questions for Building a Media Generation Studio section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const buildingmediagenerationstudioMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'vag-studio-mc-1',
      question:
        'Why is a queue system essential for a media generation platform?',
      options: [
        'It makes the application faster',
        'It handles concurrent requests, manages GPU resources, and provides progress tracking',
        'It reduces storage costs',
        'It improves video quality',
      ],
      correctAnswer: 1,
      explanation:
        'Queue systems (Celery, Bull) serialize GPU-intensive tasks, prevent resource exhaustion, enable priority handling, provide progress tracking, and allow graceful scaling.',
    },
    {
      id: 'vag-studio-mc-2',
      question:
        'What is the recommended approach for storing large video files (>100MB) in a media generation platform?',
      options: [
        'Store directly in PostgreSQL database',
        'Store on local filesystem',
        'Use S3/cloud storage and store only URLs in database',
        'Compress videos to <10MB',
      ],
      correctAnswer: 2,
      explanation:
        'S3 (or equivalent cloud storage) is designed for large media files with CDN delivery, versioning, and lifecycle policies. The database stores only metadata and S3 URLs.',
    },
    {
      id: 'vag-studio-mc-3',
      question:
        'What is the primary benefit of implementing GPU pooling in a media generation studio?',
      options: [
        'It reduces GPU costs',
        'It allows multiple workers to share GPU resources efficiently',
        'It improves video quality',
        'It eliminates the need for a queue',
      ],
      correctAnswer: 1,
      explanation:
        'GPU pooling allows multiple worker processes to use available GPUs efficiently. Idle workers can grab GPUs from the pool, maximizing utilization and throughput.',
    },
    {
      id: 'vag-studio-mc-4',
      question:
        'Which metric is most important for tracking media generation platform costs?',
      options: [
        'Number of users',
        'Cost per generation (by type: text-to-video, image-to-video, etc.)',
        'Total storage used',
        'API response time',
      ],
      correctAnswer: 1,
      explanation:
        'Cost per generation by type allows accurate pricing, margin analysis, and optimization focus. Track GPU time, API calls, and storage per generation type separately.',
    },
    {
      id: 'vag-studio-mc-5',
      question:
        'What is the recommended strategy for handling failed media generation jobs?',
      options: [
        'Immediately retry the job',
        'Delete the job and notify the user',
        'Implement exponential backoff retry with maximum attempts',
        'Never retry; require user to resubmit',
      ],
      correctAnswer: 2,
      explanation:
        'Exponential backoff (retry after 1s, 2s, 4s, 8s...) with max attempts (e.g., 3) handles transient failures without overwhelming APIs or resources. Log failures for analysis.',
    },
  ];
