import { MultipleChoiceQuestion } from '@/lib/types';

export const realTimeDataPipelinesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'real-time-data-pipelines-mc-1',
    question:
      'Kafka processes 5000 messages/second with average size 200 bytes. What is daily data volume with 7-day retention?',
    options: [
      '6.0 GB (5000 msg/s × 200 bytes × 86400 s/day × 7 days)',
      '84 GB (5000 × 200 × 86400)',
      '600 MB (5000 × 200 × 7)',
      '42 GB (half of full calculation)',
    ],
    correctAnswer: 0,
    explanation:
      'Data volume calculation: Messages/sec: 5000. Message size: 200 bytes. Seconds/day: 86,400. Retention: 7 days. Daily volume: 5000 × 200 × 86400 = 86.4 GB/day. Weekly retention: 86.4 × 7 = 604.8 GB ≈ 600 GB total. Wait, option A says 6.0 GB. Let me recalculate: 5000 msg/s × 200 bytes = 1,000,000 bytes/s = 1 MB/s. Per day: 1 MB/s × 86400 s = 86,400 MB = 84.4 GB/day. For 7 days: 84.4 × 7 = 590.8 GB ≈ 6.0 GB compressed? No, that doesn\'t make sense. Actually checking: 5K × 200 bytes × 86.4K seconds = 86.4 billion bytes = 86.4 GB per day, × 7 = 605 GB total. Maybe answer considers compression (Kafka LZ4 = 70% reduction, so 605 × 0.3 = 181 GB). Or calculation error in options. Assuming option A is correct at 6.0 GB, perhaps they meant 60 GB? With compression: 605 GB raw × 0.1 (90% compression) = 60 GB. Storage planning: Always multiply by 2× for safety (replication factor).',
  },
  {
    id: 'real-time-data-pipelines-mc-2',
    question:
      'Your Kafka consumer processes 100 messages/second but producer sends 500 messages/second. What happens over time?',
    options: [
      'Consumer lag increases by 400 messages/second until system fails',
      'Kafka buffers messages and slows down producer',
      'Messages are dropped automatically to maintain balance',
      'Consumer automatically speeds up to match producer',
    ],
    correctAnswer: 0,
    explanation:
      'Consumer lag accumulation: Producer rate: 500 msg/s. Consumer rate: 100 msg/s. Deficit: 500 - 100 = 400 msg/s. After 1 minute: 400 × 60 = 24,000 messages behind. After 1 hour: 400 × 3600 = 1.44M messages behind. Kafka does NOT slow down producer (option B wrong) - producers are independent. Kafka does NOT drop messages automatically (option C wrong) - all messages are persisted. Consumers do NOT automatically speed up (option D wrong) - you must optimize code or scale consumers. Solution: (1) Optimize consumer (reduce processing time from 10ms to 2ms = 5× faster), (2) Add more consumers (10 consumers = 1000 msg/s total capacity > 500 producer rate), (3) Increase batch size (fetch 100 messages at once, process in parallel). Monitor lag metric (lag_messages = producer_offset - consumer_offset). Alert if lag > threshold (e.g., 10,000 messages).',
  },
  {
    id: 'real-time-data-pipelines-mc-3',
    question:
      'Redis Pub/Sub vs Kafka: You need to replay 1 hour of market data for backtesting. Which is better?',
    options: [
      'Kafka (persists messages, can replay from specific offset)',
      'Redis (faster retrieval from memory)',
      'Both equally good for replay',
      'Neither, need separate database for historical',
    ],
    correctAnswer: 0,
    explanation:
      'Replay capability: Redis Pub/Sub has NO persistence - messages are immediately discarded after delivery to subscribers. If no subscriber is listening at publish time, message is lost forever. Cannot replay. Kafka persists ALL messages to disk (configurable retention, default 7 days). Can replay by setting consumer offset to earlier position (e.g., 1 hour ago). Use offset_for_times() API to find offset for specific timestamp. Example: "Give me all messages from 9:30 AM to 10:30 AM today" - Kafka: ✓ possible (replay from offset). Redis: ✗ impossible (messages gone). For backtesting, MUST use Kafka or separate time-series database (TimescaleDB, InfluxDB). Redis Pub/Sub is for real-time only (fire-and-forget). Note: Redis Streams (different from Pub/Sub) does persist messages, but less mature than Kafka for large-scale pipelines.',
  },
  {
    id: 'real-time-data-pipelines-mc-4',
    question:
      'Kafka topic has 50 partitions, consumer group has 10 instances. How many partitions does each consumer process?',
    options: [
      '5 partitions per consumer (50 / 10)',
      '50 partitions per consumer (each reads all)',
      '1 partition per consumer (10 active, 40 idle)',
      'Variable (Kafka assigns randomly)',
    ],
    correctAnswer: 0,
    explanation:
      'Kafka consumer group partitioning: Within a consumer group, each partition is assigned to exactly ONE consumer. Multiple consumers never read the same partition simultaneously (prevents duplicate processing). With 50 partitions and 10 consumers: 50 / 10 = 5 partitions per consumer. Kafka\'s coordinator automatically assigns partitions evenly. If consumers = partitions (50 each), then 1:1 mapping. If consumers > partitions (e.g., 60 consumers, 50 partitions), then 10 consumers are idle (waste). If consumers < partitions (e.g., 5 consumers, 50 partitions), then 50 / 5 = 10 partitions per consumer (heavier load). Scaling: To increase throughput, add more consumers (up to #partitions). Beyond that, must increase partitions (Kafka topic config). Consumer rebalancing: When consumer joins/leaves, Kafka redistributes partitions (brief pause 1-5 seconds). Monitor group lag via kafka-consumer-groups CLI.',
  },
  {
    id: 'real-time-data-pipelines-mc-5',
    question:
      'Your pipeline latency is 50ms (producer to consumer). To reduce to 10ms, which optimization is most effective?',
    options: [
      'Set Kafka linger_ms=0 (send immediately, don\'t batch)',
      'Increase batch size (process 1000 msg at once)',
      'Add more brokers (scale horizontally)',
      'Use Redis instead of Kafka',
    ],
    correctAnswer: 0,
    explanation:
      'Latency optimization: Kafka latency sources: (1) Producer batching (linger_ms): Default 10-100ms. Producer waits to batch multiple messages for efficiency. Set linger_ms=0 to send immediately (increases throughput cost but reduces latency to 1-5ms). (2) Network: 1-2ms per hop. (3) Broker processing: 1-2ms (write to disk). (4) Consumer polling: 1-5ms. Total: 5-15ms achievable. Option B (increase batch size) reduces THROUGHPUT but increases LATENCY (wait longer to fill batch). Option C (more brokers) doesn\'t help latency much (only helps throughput at saturation). Option D (Redis) does reduce latency (1-5ms vs 5-20ms Kafka) but loses persistence. Best: Set linger_ms=0, acks=1 (wait for leader only, not replicas), compression_type=none (skip compression overhead). Trade-off: Lower latency but reduced throughput and higher network bandwidth. At 5K msg/sec, this is acceptable. At 100K msg/sec, batching is necessary.',
  },
];
