import { MultipleChoiceQuestion } from '@/lib/types';

export const realTimePlatformProjectMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'real-time-platform-project-mc-1',
    question: 'Platform processes 50K ticks/second with 10ms latency. You add 5 more app servers. New throughput?',
    options: ['300K ticks/sec (6× servers = 6× throughput)', '50K ticks/sec (no change)', '100K ticks/sec (2× only)', '500K ticks/sec'],
    correctAnswer: 0,
    explanation: 'Horizontal scaling: Linear throughput increase if stateless. Before: 1 server × 50K = 50K ticks/sec. After: 6 servers × 50K = 300K ticks/sec. Latency unchanged (still 10ms per server). Requirements: (1) Stateless servers (no shared state), (2) Load balancer distributes traffic, (3) Kafka partitions increase (50 → 300 for 6 servers). Bottleneck shifts to Kafka (300K ticks/sec = 30 MB/sec, need 3+ brokers). Cost: 6× servers = 6× cost. Efficiency: Perfect scaling for stateless data pipelines.',
  },
  {
    id: 'real-time-platform-project-mc-2',
    question: 'Kafka topic has 50 partitions, consumer group has 100 instances. How many consumers are active?',
    options: ['50 active, 50 idle (max 1 consumer per partition)', '100 active', '25 active', '1 active'],
    correctAnswer: 0,
    explanation: 'Kafka partitioning: Each partition assigned to exactly ONE consumer in a group. With 50 partitions and 100 consumers: 50 consumers active (1 per partition), 50 consumers idle (no partitions assigned). Idle consumers waste resources. Optimal: Consumers = Partitions (50 each). If Consumers < Partitions (e.g., 25): Each consumer handles 2 partitions (higher load). If Consumers > Partitions (e.g., 100): Half are idle. Solution: Scale consumers to match partitions. For more throughput: Increase partitions (50 → 100), then add consumers.',
  },
  {
    id: 'real-time-platform-project-mc-3',
    question: 'SLA requires 99.9% uptime. How much downtime allowed per year?',
    options: ['8.76 hours (365 days × 24 hrs × 0.1%)', '3.65 days', '87.6 hours', '36.5 hours'],
    correctAnswer: 0,
    explanation: 'SLA calculation: 99.9% uptime = 0.1% downtime allowed. Per year: 365 days × 24 hours = 8760 hours total. Downtime: 8760 × 0.001 = 8.76 hours/year = 525.6 minutes. Per month: 8.76 / 12 = 43.8 minutes. Common SLAs: 99% = 3.65 days/year. 99.9% = 8.76 hours/year (three nines). 99.99% = 52.6 minutes/year (four nines). 99.999% = 5.26 minutes/year (five nines, extremely expensive). For trading: 99.9% minimum (market hours = 6.5 hrs/day, cannot afford daily outages).',
  },
  {
    id: 'real-time-platform-project-mc-4',
    question: 'Load test shows P99 latency = 15ms (target < 10ms). Where to optimize first?',
    options: ['Identify slowest component via distributed tracing', 'Add more servers', 'Upgrade CPU', 'Reduce data volume'],
    correctAnswer: 0,
    explanation: 'Optimization approach: (1) Measure: Use distributed tracing (Jaeger/Zipkin) to find bottleneck. Latency breakdown: Ingestion 2ms + Normalization 5ms + Kafka 7ms + Consume 1ms = 15ms. Bottleneck: Kafka (7ms). (2) Optimize: Reduce Kafka latency (linger_ms=0, compression=none, local broker). Result: 7ms → 2ms, total 10ms ✓. Adding servers will not help (not a throughput issue). CPU upgrade minimal impact (not CPU-bound). Reducing volume defeats purpose. Always profile before optimizing - do not guess!',
  },
  {
    id: 'real-time-platform-project-mc-5',
    question: 'Platform stores 50 billion ticks (2.5 TB compressed). Backup to S3 costs $0.023/GB. Monthly cost?',
    options: ['$58/month (2500 GB × $0.023)', '$575/month', '$23/month', '$230/month'],
    correctAnswer: 0,
    explanation: 'S3 storage cost: 2.5 TB = 2500 GB. Cost: 2500 × $0.023 = $57.50/month. Additional costs: PUT requests ($0.005/1000) = negligible. Data transfer OUT: $0.09/GB if querying from S3 (expensive). Glacier for archival: $0.004/GB = $10/month (4× cheaper but slower retrieval). Recommendation: Hot data (7 days) on TimescaleDB, warm data (30 days) on S3 Standard, cold data (> 30 days) on S3 Glacier. Total: $1,700 (TimescaleDB) + $58 (S3) + $10 (Glacier) = $1,768/month for complete solution.',
  },
];
