/**
 * Multiple choice questions for Batch Processing vs Stream Processing section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const batchvsstreamprocessingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the primary advantage of batch processing over stream processing?',
    options: [
      'Lower latency',
      'Real-time results',
      'Higher throughput and cost-effectiveness for scheduled workloads',
      'More complex capabilities',
    ],
    correctAnswer: 2,
    explanation:
      'Batch processing excels at high throughput and cost-effectiveness for scheduled workloads. It can process billions of records efficiently in a single job, and only runs when needed (e.g., nightly). Stream processing has lower latency but is more expensive (always running) and has lower per-record throughput due to processing overhead.',
  },
  {
    id: 'mc2',
    question: 'Which scenario is best suited for stream processing?',
    options: [
      'Monthly financial reports',
      'Machine learning model training on historical data',
      'Real-time fraud detection on credit card transactions',
      'Daily ETL jobs for data warehousing',
    ],
    correctAnswer: 2,
    explanation:
      'Real-time fraud detection requires stream processing because it must detect and block fraudulent transactions within seconds, before they complete. Batch processing would detect fraud too late (hours/days later). Monthly reports, ML training, and daily ETL are better suited for batch processing (high throughput, latency not critical).',
  },
  {
    id: 'mc3',
    question: 'What is a "tumbling window" in stream processing?',
    options: [
      'A window that overlaps with other windows',
      'A fixed-size, non-overlapping time window',
      'A window based on user session inactivity',
      'A window that processes data in reverse chronological order',
    ],
    correctAnswer: 1,
    explanation:
      'A tumbling window is a fixed-size, non-overlapping time window. For example, with a 5-minute tumbling window, events are grouped into [0-5min], [5-10min], [10-15min], etc. Each event belongs to exactly one window. This is different from sliding windows (overlapping) or session windows (based on inactivity).',
  },
  {
    id: 'mc4',
    question: 'What is the Lambda Architecture?',
    options: [
      'A cloud computing serverless pattern',
      'A hybrid approach combining batch (accuracy) and stream (latency) processing',
      'A type of machine learning algorithm',
      'A database sharding strategy',
    ],
    correctAnswer: 1,
    explanation:
      'Lambda Architecture combines batch processing (for comprehensive accuracy on historical data) and stream processing (for low-latency updates on recent data). A serving layer merges both to provide queries with complete, up-to-date results. This provides both the high throughput of batch and the low latency of stream, at the cost of higher complexity.',
  },
  {
    id: 'mc5',
    question:
      'Why is batch processing often more cost-effective than stream processing?',
    options: [
      'It always uses cheaper hardware',
      'It only runs when needed (e.g., nightly) vs. stream running 24/7',
      'It uses less memory',
      'It requires fewer engineers',
    ],
    correctAnswer: 1,
    explanation:
      'Batch processing is more cost-effective because it only runs when needed (e.g., 1 hour per day for a nightly job) vs. stream processing which runs continuously 24/7. A batch job might cost $10/month (1 hour/day) while the equivalent stream processing would cost $2,000/month (24/7), a 200x difference. This is only acceptable when real-time results are actually required.',
  },
];
