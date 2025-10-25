/**
 * Multiple choice questions for Push vs Pull Models section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const pushvspullmodelsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question: 'What is the main advantage of the push model?',
    options: [
      'Easier to scale horizontally',
      'Uses less server resources',
      'Near real-time updates with low latency',
      'Simpler to implement',
    ],
    correctAnswer: 2,
    explanation:
      'The main advantage of push is near real-time updates with low latency (< 100ms). Server pushes updates immediately when data changes, so clients see changes instantly. This is critical for chat apps, live sports scores, and stock trading. However, push requires persistent connections which use more server resources and are harder to scale.',
  },
  {
    id: 'mc2',
    question: 'Which scenario is best suited for the pull model?',
    options: [
      'Real-time chat messages',
      'Live stock price updates',
      'Static website images served by CDN',
      'Push notifications',
    ],
    correctAnswer: 2,
    explanation:
      "Static website images served by CDN are perfect for pull model (origin pull). Client requests image when needed, CDN pulls from origin on cache miss and caches for future requests. This is efficient, scalable, and doesn't require real-time updates. Chat, stocks, and push notifications all require real-time push.",
  },
  {
    id: 'mc3',
    question: 'Why does Twitter use a hybrid push/pull approach for timelines?',
    options: [
      'To make the system more complex',
      'Because push works for regular users but is too expensive for celebrities with millions of followers',
      'To reduce storage costs only',
      'Because pull is always better than push',
    ],
    correctAnswer: 1,
    explanation:
      'Twitter uses hybrid because push (fan-out on write) works well for regular users with few followers, but becomes impossibly expensive for celebrities. If Taylor Swift (200M followers) tweets, push would require 200M timeline writes! Instead, Twitter uses push for regular users (<10K followers) and pulls celebrity tweets on-demand when followers load their timeline.',
  },
  {
    id: 'mc4',
    question:
      'What is the main disadvantage of polling (pull model) for real-time updates?',
    options: [
      "It doesn't work at all",
      'Higher latency and wasted bandwidth from polling even when no new data exists',
      'It requires WebSockets',
      "It\'s more complex than push",
    ],
    correctAnswer: 1,
    explanation:
      "Polling has higher latency (average delay = polling interval / 2) and wastes bandwidth by making requests even when there's no new data. If polling every 30 seconds, average latency is 15 seconds. Also, if there's no new data, the poll still consumes server resources and network bandwidth unnecessarily. WebSockets (push) solve this by keeping a connection open and only sending data when it changes.",
  },
  {
    id: 'mc5',
    question: 'When should you use CDN origin push instead of origin pull?',
    options: [
      'For all content by default',
      'For user-generated content',
      'For critical assets during product launches to ensure no cache misses',
      'Never, origin pull is always better',
    ],
    correctAnswer: 2,
    explanation:
      'Use origin push for critical assets during product launches or high-traffic events to ensure files are pre-cached on all edge servers (no cache misses). For example, Apple pushes iPhone product images before announcement. For normal content, origin pull is better (automatic caching, efficient). Push requires manual uploads and uses more storage.',
  },
];
