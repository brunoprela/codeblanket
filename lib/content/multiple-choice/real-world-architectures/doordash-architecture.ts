/**
 * Multiple choice questions for DoorDash Architecture section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const doordasharchitectureMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'mc1',
        question: 'How often does DoorDash\'s dispatch optimization (DISCO) perform batch matching?',
        options: [
            'Every 5 seconds for real-time assignment',
            'Every 30 seconds for batch optimization',
            'Every 2 minutes for load balancing',
            'Continuously using streaming algorithms',
        ],
        correctAnswer: 1,
        explanation: 'DISCO performs batch matching every 30 seconds. It collects pending orders and available Dashers, formulates an optimization problem considering pickup time, delivery time, acceptance probability, stacking potential, and future demand, then solves using Mixed Integer Linear Programming or greedy heuristics. The 30-second batch window enables global optimization (15% better than greedy nearest-Dasher) while maintaining acceptable assignment latency.',
    },
    {
        id: 'mc2',
        question: 'What percentage of DoorDash\'s ETA predictions are accurate within 5 minutes?',
        options: [
            'Approximately 50%',
            'Approximately 65%',
            'Approximately 80%',
            'Approximately 95%',
        ],
        correctAnswer: 2,
        explanation: 'DoorDash achieves 80% accuracy within 5 minutes for ETA predictions. The ML models (gradient boosted trees) predict: restaurant prep time (based on order complexity, time of day, queue depth), Dasher pickup time (routing with traffic), and delivery time. Features include restaurant historical data, order details, traffic patterns, and time features. Models are continuously retrained on millions of historical orders to improve accuracy.',
    },
    {
        id: 'mc3',
        question: 'Which geospatial indexing system does DoorDash use for finding nearby Dashers?',
        options: [
            'Geohash with 9-character precision',
            'H3 hexagonal hierarchical indexing',
            'S2 geometry from Google',
            'QuadTree spatial indexing',
        ],
        correctAnswer: 1,
        explanation: 'DoorDash uses H3 (Hexagonal Hierarchical geospatial indexing), the same system Uber developed. H3 divides Earth into hexagonal cells at multiple resolutions. To find nearby Dashers, DoorDash converts restaurant location to H3 hex ID and queries neighboring hexes (k-ring). This reduces search space from potentially millions of Dashers to hundreds in nearby hexes, enabling sub-100ms queries. Hexagons provide better distance approximation than squares.',
    },
    {
        id: 'mc4',
        question: 'How does DoorDash handle experiment safety in its A/B testing platform?',
        options: [
            'Manual review of all experiments before launch',
            'Guardrail metrics with auto-stop + gradual rollout from 1% to 50%',
            'Post-hoc analysis after full rollout',
            'Shadow testing in production without user impact',
        ],
        correctAnswer: 1,
        explanation: 'DoorDash uses multiple safety mechanisms: (1) Guardrail metrics monitoring error rates and crash rates with auto-stop if violated. (2) Gradual rollout starting at 1%, increasing to 5%, 10%, 50% if metrics are positive. (3) Senior engineer review for high-risk experiments. (4) Statistical significance testing with confidence intervals. This prevents bad experiments from impacting many users while enabling thousands of concurrent A/B tests safely.',
    },
    {
        id: 'mc5',
        question: 'Which stream processing technology does DoorDash use for real-time location tracking?',
        options: [
            'Kafka Streams',
            'Apache Flink',
            'Spark Streaming',
            'AWS Kinesis',
        ],
        correctAnswer: 1,
        explanation: 'DoorDash uses Apache Flink for real-time stream processing of location updates. Flink consumes the Kafka stream of GPS coordinates (sent every 4 seconds from Dasher app), updates Redis for current locations (TTL 60s), converts to H3 hex IDs for spatial indexing, and writes historical data to Cassandra. Flink\'s exactly-once semantics, stateful processing, and low latency make it ideal for real-time location tracking at scale.',
    },
];

