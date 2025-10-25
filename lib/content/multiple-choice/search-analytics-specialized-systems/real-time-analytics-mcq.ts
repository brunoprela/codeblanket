import { MultipleChoiceQuestion } from '@/lib/types';

export const realTimeAnalyticsMCQ: MultipleChoiceQuestion[] = [
  {
    id: 'rta-mcq-1',
    question:
      'You need to count unique visitors to your website in real-time across billions of page views. An exact count requires storing all visitor IDs (GBs of memory). Which algorithm provides approximate unique counts with <1% error using only 12KB of memory?',
    options: [
      'Bloom Filter',
      'HyperLogLog',
      'Count-Min Sketch',
      'HashMap with LRU eviction',
    ],
    correctAnswer: 1,
    explanation:
      'HyperLogLog is designed specifically for cardinality estimation (unique counts) with minimal memory. It uses only ~12KB to estimate billions of unique values with 1-2% error. Bloom Filters test membership (is X in set?) not counts. Count-Min Sketch estimates frequency (how many times did X appear?) not unique counts. HashMap with LRU would require far more memory and still lose data when evicting entries. HyperLogLog achieves this efficiency through probabilistic counting based on hash function properties—it examines the distribution of leading zeros in hashed values to estimate cardinality. This is why Redis, Druid, and BigQuery all use HyperLogLog for DISTINCT queries on massive datasets.',
  },
  {
    id: 'rta-mcq-2',
    question:
      'Your real-time analytics pipeline uses tumbling windows of 1 minute. Events arrive with timestamps: 10:00:05, 10:00:55, 10:01:03, 10:00:58. The watermark is set to "max event time - 10 seconds". Which events are included in the 10:00-10:01 window?',
    options: [
      'Only 10:00:05 and 10:00:55 (events within window bounds)',
      '10:00:05, 10:00:55, and 10:00:58 (watermark extends window)',
      'All events (windows always wait for late data)',
      '10:00:05, 10:00:55, 10:01:03 (processing time determines window)',
    ],
    correctAnswer: 1,
    explanation:
      "Watermarks determine when a window closes. The 10:00-10:01 window closes when watermark reaches 10:01:00. Event sequence: (1) 10:00:05 arrives → watermark=9:59:55 (10:00:05-10s), window open. (2) 10:00:55 arrives → watermark=10:00:45, window still open. (3) 10:01:03 arrives → watermark=10:00:53, window still open (hasn't reached 10:01:00). (4) 10:00:58 arrives late but within watermark tolerance (10:00:58 < 10:00:53+10s), so it's accepted. The window closes only when an event arrives with timestamp > 10:01:10 (watermark would exceed 10:01:00). So 10:00:05, 10:00:55, and 10:00:58 are all included. The 10:01:03 event goes to the next window (10:01-10:02). This demonstrates why watermarks are critical for handling out-of-order events in real-time systems.",
  },
  {
    id: 'rta-mcq-3',
    question:
      'Apache Flink processes 50,000 events per second with stateful operations (tracking per-user spend). Where does Flink store this state to achieve both high throughput and fault tolerance?',
    options: [
      'In-memory only (fastest but lost on failure)',
      'In-memory with periodic snapshots to persistent storage (RocksDB backend)',
      'Directly to database (PostgreSQL) for each event',
      'In Kafka topics as a changelog',
    ],
    correctAnswer: 1,
    explanation:
      "Flink uses in-memory state with periodic snapshots (checkpoints) to persistent storage for optimal performance and fault tolerance. State is stored locally in memory (or RocksDB for large state) for fast access during processing. Every few seconds (e.g., every 60 seconds), Flink takes a consistent snapshot of all state across all operators and writes it to durable storage (HDFS, S3). If a node fails, Flink restores from the last checkpoint. This provides both high throughput (in-memory access) and fault tolerance (durable checkpoints). Writing to PostgreSQL for every event (option C) would create massive bottleneck—50k events/sec would overwhelm the database. In-memory only (option A) loses data on failure. Kafka changelog (option D) is used by Kafka Streams but not Flink's primary mechanism. This architecture is why Flink achieves millions of events/sec with exactly-once guarantees.",
  },
  {
    id: 'rta-mcq-4',
    question:
      'Your dashboard needs to show "page views in the last 5 minutes" updated every second. Which window type is most appropriate?',
    options: [
      'Tumbling window of 5 minutes',
      'Sliding window of 5 minutes with 1-second slide',
      'Session window with 5-minute gap',
      'Global window',
    ],
    correctAnswer: 1,
    explanation:
      'Sliding window of 5 minutes with 1-second slide provides the continuously updating 5-minute window. At 10:00:00, it shows events from 9:55:00-10:00:00. At 10:00:01, it shows 9:55:01-10:00:01. The window "slides" forward every second, always maintaining a 5-minute span. Tumbling windows (option A) create non-overlapping 5-minute buckets but update only every 5 minutes, not every second. Session windows (option C) are for grouping events by inactivity gaps, not fixed time ranges. Global window (option D) never closes. The trade-off: sliding windows create more windows (every second) and more computation, but provide the smooth, continuously updating experience users expect in real-time dashboards. This is why live monitoring systems use sliding windows despite the computational cost.',
  },
  {
    id: 'rta-mcq-5',
    question:
      "You're building a real-time analytics system and debating between Spark Structured Streaming (micro-batches) and Apache Flink (true streaming). Your use case requires <100ms latency. Which should you choose and why?",
    options: [
      'Spark Structured Streaming because it handles streaming data',
      'Apache Flink because micro-batches have minimum ~200ms latency',
      'Either works fine for <100ms latency requirements',
      'Spark Structured Streaming because micro-batches are more reliable',
    ],
    correctAnswer: 1,
    explanation:
      'Apache Flink is required for <100ms latency. Spark Structured Streaming uses micro-batches (process events in small batches every 200ms-1s). Even with 200ms micro-batches, you have minimum 200ms latency, which violates the <100ms requirement. Flink processes events one-at-a-time as they arrive (true streaming), achieving single-digit millisecond latencies. Trade-off: Flink is more complex to operate than Spark. But for latency-critical applications (fraud detection, real-time bidding, trading), Flink is the only viable option. Use Spark Structured Streaming when: (1) latency requirements are >1 second, (2) you already have Spark expertise, (3) you need tight integration with Spark ML. Use Flink when: (1) latency must be <100ms, (2) complex event processing with state, (3) exactly-once guarantees critical. The micro-batch vs true streaming debate is fundamental to streaming architecture choices.',
  },
];
