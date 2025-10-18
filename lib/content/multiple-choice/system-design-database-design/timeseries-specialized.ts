/**
 * Multiple choice questions for Time-Series and Specialized Databases section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const timeseriesspecializedMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'ts-1',
    question:
      'Why are traditional RDBMS (PostgreSQL, MySQL) poorly suited for high-volume time-series data?',
    options: [
      "They don't support timestamp data types",
      "They can't handle millions of rows",
      'B-tree index updates are expensive, compression is poor, and no built-in downsampling',
      "They don't support time-based queries or aggregations",
    ],
    correctAnswer: 2,
    explanation:
      'Option C is correct. Traditional RDBMS struggle with time-series data because: (1) B-tree index updates become very expensive at billions of rows, (2) They achieve only 1-2x compression vs 10-20x in TSDBs, (3) No built-in downsampling or retention policies, (4) Scanning billions of rows for time-range queries is slow. Option A is false (they support timestamps). Option B is false (they can handle many rows, just not efficiently for this use case). Option D is false (they support these queries, just not efficiently).',
    difficulty: 'medium' as const,
  },
  {
    id: 'ts-2',
    question: 'What is downsampling in time-series databases?',
    options: [
      'Reducing the sample rate at which data is collected',
      'Deleting old data to save storage space',
      'Aggregating high-resolution data into lower-resolution summaries over time',
      'Compressing data using specialized algorithms',
    ],
    correctAnswer: 2,
    explanation:
      'Option C is correct. Downsampling is the process of aggregating high-resolution data (e.g., 1-second intervals) into lower-resolution summaries (e.g., 1-minute or 1-hour averages) as data ages. For example: keep raw data for 7 days, then downsample to 1-minute aggregates for 90 days, then 1-hour aggregates long-term. This saves storage while retaining important patterns. Option A describes changing collection rate (different concept). Option B is deletion, not downsampling. Option D describes compression (related but different).',
    difficulty: 'medium' as const,
  },
  {
    id: 'ts-3',
    question: 'What is the main difference between Prometheus and InfluxDB?',
    options: [
      'Prometheus is for logs, InfluxDB is for metrics',
      'Prometheus uses a pull model and is monitoring-focused; InfluxDB uses a push model and is general-purpose',
      'Prometheus stores data in SQL, InfluxDB uses NoSQL',
      'Prometheus is open-source, InfluxDB is commercial only',
    ],
    correctAnswer: 1,
    explanation:
      'Option B is correct. Prometheus uses a pull model (scrapes metrics from targets), is specifically designed for monitoring with built-in alerting, and has local-only storage by default. InfluxDB uses a push model (applications send data), is a general-purpose time-series database for various use cases (IoT, analytics, monitoring), and has more flexible data models. Option A is false (both are for metrics). Option C is false (neither uses SQL as storage). Option D is false (both have open-source versions).',
    difficulty: 'easy' as const,
  },
  {
    id: 'ts-4',
    question:
      'How do time-series databases achieve 10-20x better compression than traditional RDBMS?',
    options: [
      'They use standard gzip compression on all data',
      'They store only the most recent data and delete old data',
      'They use delta encoding, delta-of-delta encoding, and run-length encoding for patterns',
      'They reduce precision by rounding all numeric values',
    ],
    correctAnswer: 2,
    explanation:
      'Option C is correct. Time-series databases use specialized compression algorithms: (1) Delta encoding - store differences between consecutive values (1000, 1001, 1002 → 1000, +1, +1), (2) Delta-of-delta encoding - detect patterns in deltas, (3) Run-length encoding - compress repeated values (5, 5, 5, 5 → 5 count=4), (4) Specialized timestamp compression. These exploit the predictable nature of time-series data. Option A is too generic. Option B is deletion, not compression. Option D would lose data precision.',
    difficulty: 'hard' as const,
  },
  {
    id: 'ts-5',
    question: 'When should you NOT use a time-series database?',
    options: [
      'When you have millions of sensor readings per second',
      'When you need complex JOINs across many relational tables with ACID transactions',
      'When you need to store application metrics and perform aggregations',
      'When you need to automatically downsample and delete old data',
    ],
    correctAnswer: 1,
    explanation:
      'Option B is correct. Time-series databases are NOT suitable when you need: (1) Complex relationships with many JOINs, (2) Strong ACID transaction guarantees across multiple entities, (3) Frequent updates to existing data, (4) Low write volume (<1000/sec), (5) Unpredictable ad-hoc queries. Use a traditional RDBMS (PostgreSQL, MySQL) for these scenarios. Options A, C, and D are all ideal use cases FOR time-series databases, not reasons to avoid them.',
    difficulty: 'medium' as const,
  },
];
