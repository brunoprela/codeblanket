/**
 * Multiple choice questions for Pinterest Architecture section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const pinterestarchitectureMultipleChoice: MultipleChoiceQuestion[] = [
    {
        id: 'mc1',
        question: 'What sharding strategy does Pinterest use for its graph data?',
        options: [
            'User-based sharding (all user data on one shard)',
            'Hash-based sharding with consistent hashing',
            'Object-based sharding (pins, boards, users sharded separately)',
            'Geographic sharding (users by region)',
        ],
        correctAnswer: 2,
        explanation: 'Pinterest uses object-based sharding where pins, boards, and users are sharded separately by their IDs. For example, all pins are sharded by pin_id across 4096 shards. This prevents hot shard problems—a celebrity with 10M pins won\'t overload one shard because their pins are distributed. Shard ID = (object_id mod 4096). Trade-off: Some queries require scatter-gather, but caching and denormalization mitigate this.',
    },
    {
        id: 'mc2',
        question: 'Which technologies does Pinterest use for visual search (Pinterest Lens)?',
        options: [
            'TensorFlow CNNs for embeddings + Redis for caching',
            'TensorFlow CNNs for embeddings + FAISS for nearest neighbor search',
            'OpenCV for feature extraction + Elasticsearch for search',
            'YOLO object detection + Cassandra for storage',
        ],
        correctAnswer: 1,
        explanation: 'Pinterest uses TensorFlow-trained CNNs to extract feature embeddings (128-512 dimensions) from images, and FAISS (Facebook AI Similarity Search) for approximate nearest neighbor search. When a user uploads an image, objects are detected, embeddings extracted, and FAISS finds similar pins via cosine similarity in sub-100ms. Results are ranked by visual similarity, engagement, and personalization. This powers 600M+ visual searches per month.',
    },
    {
        id: 'mc3',
        question: 'What percentage of Pinterest engagement comes from the personalized home feed?',
        options: [
            'Approximately 30%',
            'Approximately 50%',
            'Approximately 70%',
            'Approximately 90%',
        ],
        correctAnswer: 2,
        explanation: 'Approximately 70% of engagement comes from Pinterest\'s personalized home feed. The feed uses a two-stage ranking system: candidate generation (followed boards, topics, related pins, trending, collaborative filtering → 1000 candidates) and deep ML ranking (user/pin embeddings, predict save/click probability → ordered feed). Real-time updates via Flink streaming adjust user embeddings when users save pins, ensuring fresh personalization.',
    },
    {
        id: 'mc4',
        question: 'How many shards does Pinterest typically use for object-based sharding?',
        options: [
            '256 shards',
            '1024 shards',
            '4096 shards',
            '16384 shards',
        ],
        correctAnswer: 2,
        explanation: 'Pinterest typically uses 4096 shards for object-based sharding. Shard ID is calculated as (object_id mod 4096). This provides fine-grained distribution to prevent hot shards while remaining manageable operationally. With 4096 shards, even users with millions of pins have their data distributed across many shards. Consistent hashing is used for adding/removing shards without full resharding.',
    },
    {
        id: 'mc5',
        question: 'Which stream processing technology does Pinterest use for real-time user embedding updates?',
        options: [
            'Apache Kafka Streams',
            'Apache Flink',
            'Apache Storm',
            'Spark Streaming',
        ],
        correctAnswer: 1,
        explanation: 'Pinterest uses Apache Flink for real-time stream processing, including updating user embeddings when users save pins. When a user saves a pin, the event flows through Kafka to Flink, which updates the user\'s embedding vector in real-time. This updated embedding is immediately used for feed ranking, ensuring that the home feed reflects users\' latest interests. Flink\'s stateful processing and low latency make it ideal for this use case.',
    },
];

