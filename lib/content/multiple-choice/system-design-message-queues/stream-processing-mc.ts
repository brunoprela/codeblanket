/**
 * Multiple Choice Questions for Stream Processing
 */

import { MultipleChoiceQuestion } from '../../../types';

export const streamProcessingMC: MultipleChoiceQuestion[] = [
  {
    id: 'stream-processing-mc-1',
    question:
      'What is the difference between stream processing and batch processing?',
    options: [
      { id: 'a', text: 'Stream processing is slower than batch processing' },
      {
        id: 'b',
        text: 'Stream processing operates on data as it arrives (continuous), batch processing operates on bounded datasets (periodic)',
      },
      {
        id: 'c',
        text: 'Stream processing uses more memory than batch processing',
      },
      { id: 'd', text: 'They are the same' },
    ],
    correctAnswer: 'b',
    explanation:
      'Stream processing operates continuously on unbounded data streams as events arrive (real-time or near-real-time), providing low-latency results. Batch processing processes bounded datasets periodically (e.g., every hour or day). For example, stream processing calculates real-time click rates (updated every second), while batch processing generates daily reports (run overnight). Stream processing enables real-time analytics, fraud detection, and monitoring. Batch processing is suitable for historical analysis and periodic aggregations.',
  },
  {
    id: 'stream-processing-mc-2',
    question: 'What is a tumbling window in stream processing?',
    options: [
      { id: 'a', text: 'A window that overlaps with adjacent windows' },
      { id: 'b', text: 'A fixed-size, non-overlapping time-based window' },
      { id: 'c', text: 'A window based on inactivity gaps' },
      { id: 'd', text: 'A window that grows dynamically' },
    ],
    correctAnswer: 'b',
    explanation:
      'A tumbling window is a fixed-size, non-overlapping time window. For example, 5-minute tumbling windows: [00:00-00:05), [00:05-00:10), [00:10-00:15). Each event belongs to exactly one window. Use for periodic aggregations like "page views per 5 minutes" or "sales per hour". This differs from hopping windows (overlapping), sliding windows (continuous), or session windows (based on inactivity gaps). Tumbling windows are simplest and most commonly used for time-based aggregations.',
  },
  {
    id: 'stream-processing-mc-3',
    question: 'What is watermarking in stream processing?',
    options: [
      { id: 'a', text: 'Adding timestamps to messages' },
      {
        id: 'b',
        text: 'A mechanism to track event-time progress and handle late-arriving data',
      },
      { id: 'c', text: 'Compressing messages to save bandwidth' },
      { id: 'd', text: 'Encrypting messages for security' },
    ],
    correctAnswer: 'b',
    explanation:
      'Watermarks track event-time progress in a stream, indicating "all events with timestamp < watermark have been received". This helps determine when to close time windows and how to handle late data. For example, watermark = max(event_time) - 30 seconds. A window [10:00-10:05) closes when watermark >= 10:05:00 (i.e., max event_time >= 10:05:30). Late data arriving before watermark is included; after watermark, it\'s dropped or triggers late-data handling. Essential for out-of-order event processing.',
  },
  {
    id: 'stream-processing-mc-4',
    question: 'What is the purpose of state in stateful stream processing?',
    options: [
      { id: 'a', text: 'To cache messages for faster processing' },
      {
        id: 'b',
        text: 'To maintain aggregations, counts, or other computations across multiple events',
      },
      { id: 'c', text: 'To compress data' },
      { id: 'd', text: 'To replicate data across nodes' },
    ],
    correctAnswer: 'b',
    explanation:
      'State in stream processing maintains aggregations, counts, or other computations across multiple events. For example, counting clicks per user (state: Map<userId, count>) or detecting fraud patterns (state: recent transactions per card). State can be local (RocksDB, in-memory) or distributed. Frameworks like Kafka Streams and Flink provide managed state with fault tolerance (changelog topics, checkpoints). Stateless processing handles each event independently; stateful processing requires state for aggregations, joins, pattern detection.',
  },
  {
    id: 'stream-processing-mc-5',
    question:
      'What does exactly-once processing guarantee in stream processing frameworks like Kafka Streams?',
    options: [
      { id: 'a', text: 'Messages are never lost' },
      {
        id: 'b',
        text: 'Each input event affects the output exactly once, even if failures occur',
      },
      { id: 'c', text: 'Processing is faster' },
      { id: 'd', text: 'Messages are processed in order' },
    ],
    correctAnswer: 'b',
    explanation:
      'Exactly-once processing guarantees that each input event affects the final output exactly once, even if processing failures and retries occur. This means no duplicate outputs and no lost messages. Achieved through idempotent operations, transactional writes, and checkpointing. For example, if a stream processor crashes after writing results but before committing input offsets, upon restart it reprocesses the same input, but the duplicate output is prevented. Set processing.guarantee=exactly_once_v2 in Kafka Streams to enable.',
  },
];
