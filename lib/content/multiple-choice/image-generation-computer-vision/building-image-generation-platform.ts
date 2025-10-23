import { MultipleChoiceQuestion } from '../../../types';

export const buildingimmagegenerationplatformMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'igcv-platform-mc-1',
      question:
        'What is the purpose of a queue system in an image generation platform?',
      options: [
        'Faster generation',
        'Async processing and load management',
        'Better quality',
        'Cheaper costs',
      ],
      correctAnswer: 1,
      explanation:
        'Queues enable async processing, allowing API to respond immediately while workers process in background.',
    },
    {
      id: 'igcv-platform-mc-2',
      question: 'Why use WebSockets in an image generation platform?',
      options: [
        'Faster generation',
        'Real-time status updates to users',
        'Cheaper hosting',
        'Better security',
      ],
      correctAnswer: 1,
      explanation:
        'WebSockets provide real-time updates on generation progress without polling.',
    },
    {
      id: 'igcv-platform-mc-3',
      question: 'Where should generated images be stored for production?',
      options: ['Local disk', 'Database', 'S3 + CDN', 'In memory'],
      correctAnswer: 2,
      explanation:
        'S3 with CDN provides scalable, fast image storage and delivery.',
    },
    {
      id: 'igcv-platform-mc-4',
      question:
        'What is the estimated monthly cost for a basic production platform?',
      options: ['$50', '$150', '$500', '$1,350+'],
      correctAnswer: 3,
      explanation:
        'A basic platform with 2 GPU workers, database, and storage costs ~$1,350/month base.',
    },
    {
      id: 'igcv-platform-mc-5',
      question: 'What should you do when a generation job fails?',
      options: [
        'Delete it',
        'Retry automatically with exponential backoff',
        'Ignore it',
        'Always manual retry',
      ],
      correctAnswer: 1,
      explanation:
        'Implement automatic retry with exponential backoff for transient failures.',
    },
  ];
