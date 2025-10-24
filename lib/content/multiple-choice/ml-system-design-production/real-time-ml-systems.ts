import { MultipleChoiceQuestion } from '@/types/curriculum';

export const realTimeMlSystemsQuestions: MultipleChoiceQuestion[] = [
  {
    id: 'rtms-mc-1',
    question:
      "You're building an online learning system that updates a fraud detection model in real-time as new transactions are labeled. The model currently uses batch gradient descent. What modification is most critical for enabling effective online learning?",
    options: [
      'Switch to stochastic gradient descent (SGD) or mini-batch SGD to process individual/small batches of examples',
      'Increase the learning rate to adapt quickly to new patterns',
      'Use a simpler model (linear regression) that updates faster',
      'Implement distributed training across multiple GPUs',
    ],
    correctAnswer: 0,
    explanation:
      "Online learning requires processing examples as they arrive rather than waiting for a full batch. SGD or mini-batch SGD enables this by updating model weights after each example or small batch. This is fundamental for online learning. Increasing learning rate (option B) without SGD doesn't enable online updates and can cause instability. Simpler models (option C) may help but don't address the batch vs. online distinction. Distributed training (option D) is for throughput, not online learning specifically.",
    difficulty: 'intermediate',
    topic: 'Real-Time ML Systems',
  },
  {
    id: 'rtms-mc-2',
    question:
      'Your real-time recommendation system uses a streaming pipeline (Kafka → Flink → Model) to compute features and make predictions. You notice that during traffic spikes, prediction latency increases from 50ms to 500ms. What is the most likely bottleneck and solution?',
    options: [
      'Model inference is slow; optimize the model or add more serving instances',
      'Streaming pipeline backpressure; scale up Flink parallelism and add buffering',
      'Kafka message lag; increase Kafka partitions and consumer groups',
      'Feature computation is slow; cache computed features',
    ],
    correctAnswer: 1,
    explanation:
      "Latency increases during traffic spikes suggest streaming pipeline backpressure: Flink can't keep up with message rate, causing queuing delays. Solutions include increasing Flink parallelism (more workers), optimizing streaming operations, and adding buffers. This addresses the root cause. Option A (model inference) would show consistent slow latency, not spike-dependent. Option C (Kafka lag) is a symptom, not the bottleneck—Flink can't consume fast enough. Option D (feature caching) helps but doesn't address throughput limits during spikes.",
    difficulty: 'advanced',
    topic: 'Real-Time ML Systems',
  },
  {
    id: 'rtms-mc-3',
    question:
      "You're implementing real-time feature computation for a user action prediction model. Some features require aggregating the user's last 100 actions, which are stored in a database. Computing this at prediction time takes 80ms, exceeding your 50ms latency budget. What is the most effective solution?",
    options: [
      'Use a faster database with better query performance',
      'Implement a real-time feature store with streaming aggregations (e.g., using Flink or Spark Streaming)',
      'Reduce to last 50 actions to speed up computation',
      'Cache the aggregated features with 5-minute TTL',
    ],
    correctAnswer: 1,
    explanation:
      'Real-time feature store with streaming aggregations pre-computes features as events arrive, maintaining up-to-date aggregates in low-latency storage (Redis, DynamoDB). At prediction time, features are retrieved in <5ms. This is the standard approach for real-time ML with complex aggregations. Option A (faster DB) helps but query 100 records will still be slow. Option C (fewer actions) reduces feature quality. Option D (caching with TTL) creates staleness issues—features are up to 5 minutes old.',
    difficulty: 'advanced',
    topic: 'Real-Time ML Systems',
  },
  {
    id: 'rtms-mc-4',
    question:
      'Your real-time ML system detects concept drift: model performance has degraded over the past 2 hours due to changing user behavior. What is the most appropriate automated response?',
    options: [
      'Immediately retrain the model on recent data and deploy',
      'Switch to a backup model trained on different data',
      'Trigger an alert for human investigation while gradually increasing the weight on recent training examples',
      'Roll back to the previous model version',
    ],
    correctAnswer: 2,
    explanation:
      "Gradual adaptation with human oversight is safest: trigger alerts for investigation while using online learning or sliding window retraining to adapt to recent patterns. This balances responsiveness with caution. Option A (immediate retrain/deploy) is risky—the drift might be temporary (e.g., flash sale, event), and rushing deployment risks bugs. Option B (backup model) may not address the drift cause. Option D (rollback) doesn't help if drift is in the data, not the model.",
    difficulty: 'advanced',
    topic: 'Real-Time ML Systems',
  },
  {
    id: 'rtms-mc-5',
    question:
      "You're building a real-time bidding system that must make predictions within 50ms. The model needs features from three sources: user service (20ms), ad service (15ms), and context service (10ms). Serial fetching takes 45ms, leaving only 5ms for model inference. How can you optimize this?",
    options: [
      'Fetch features in parallel and implement timeout fallbacks for missing features',
      'Cache features with short TTL to avoid real-time fetching',
      'Combine all feature sources into a single service',
      'Use a simpler model that requires fewer features',
    ],
    correctAnswer: 0,
    explanation:
      "Parallel feature fetching with timeouts is optimal: fetch from all services simultaneously (max 20ms, not 45ms), use default values for features that don't return in time, and ensure model can handle missing features gracefully. This meets latency requirements while maximizing feature availability. Option B (caching) introduces staleness. Option C (single service) requires extensive infrastructure changes and creates a single point of failure. Option D (fewer features) sacrifices model quality unnecessarily.",
    difficulty: 'advanced',
    topic: 'Real-Time ML Systems',
  },
];
