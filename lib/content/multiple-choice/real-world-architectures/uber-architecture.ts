/**
 * Multiple choice questions for Uber Architecture section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const uberarchitectureMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What geospatial indexing system does Uber use to efficiently find nearby drivers?',
    options: [
      'Quadtree spatial indexing',
      'Geohash with 12-character precision',
      'H3 hexagonal hierarchical geospatial indexing',
      'R-tree for bounding box queries',
    ],
    correctAnswer: 2,
    explanation:
      'Uber uses H3 (Hexagonal Hierarchical Geospatial Index) developed by Uber Engineering. H3 divides Earth into hexagonal cells at multiple resolutions. To find nearby drivers, Uber converts a location to an H3 hex ID and queries neighboring hexes (k-ring). This reduces the search space from millions of drivers to just hundreds in nearby hexes, enabling sub-100ms queries. Hexagons are preferred over squares because they have equal distance from center to all edges.',
  },
  {
    id: 'mc2',
    question: "What is DISCO in Uber\'s architecture?",
    options: [
      "Uber's distributed caching system",
      'Dispatch Optimization algorithm for matching drivers to orders',
      'Dynamic Inventory Sorting and Classification service',
      'Distributed Service Coordination framework',
    ],
    correctAnswer: 1,
    explanation:
      'DISCO (Dispatch Optimization) is Uber\'s algorithm for optimally matching drivers to delivery orders. Unlike a simple "nearest driver" greedy approach, DISCO performs batch matching every 30 seconds, considering factors like estimated pickup time, driver acceptance probability, driver quality, future demand prediction, and stacked delivery opportunities. It formulates this as an optimization problem to minimize total delivery time, resulting in 15% improvement over naive assignment.',
  },
  {
    id: 'mc3',
    question:
      "How frequently does Uber\'s driver app send GPS location updates?",
    options: [
      'Every second for real-time accuracy',
      'Every 4 seconds via WebSocket',
      'Every 10 seconds to conserve battery',
      'Only when significant movement is detected',
    ],
    correctAnswer: 1,
    explanation:
      "Uber\'s driver app (Dasher app) sends GPS coordinates every 4 seconds via WebSocket connection. This frequency balances real-time tracking accuracy with battery consumption and network bandwidth. Location updates are sent to the Location Gateway, published to Kafka for processing, and stored in Redis (current location with 60s TTL) and Cassandra (historical tracking). This enables sub-100ms queries for nearby drivers.",
  },
  {
    id: 'mc4',
    question:
      'Which stream processing technology does Uber use for real-time location tracking?',
    options: [
      'Apache Storm for stream processing',
      'Apache Flink for stateful stream processing',
      'AWS Kinesis for managed streaming',
      'RabbitMQ with custom processors',
    ],
    correctAnswer: 1,
    explanation:
      "Uber uses Apache Flink for real-time stream processing of location updates. Flink processes the Kafka stream of location events, performs stateful operations (aggregations, windowing), updates Redis for current locations, converts to H3 hex IDs for indexing, and writes historical data to Cassandra. Flink\'s exactly-once processing semantics and low-latency capabilities make it ideal for Uber's real-time location tracking at scale.",
  },
  {
    id: 'mc5',
    question:
      'What storage technology does Uber use for historical location data?',
    options: [
      'MySQL with time-series partitioning',
      'Cassandra partitioned by dasher_id and clustered by timestamp',
      'InfluxDB time-series database',
      'DynamoDB with TTL enabled',
    ],
    correctAnswer: 1,
    explanation:
      'Uber uses Cassandra for storing historical location data, partitioned by dasher_id and clustered by timestamp. This schema enables efficient queries like "get dasher location history for last hour" by querying a single partition ordered by time. Cassandra\'s high write throughput (billions of location updates daily) and scalability make it well-suited for this use case. Redis stores only current locations for low-latency queries.',
  },
];
