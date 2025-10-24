import { MultipleChoiceQuestion } from '@/lib/types';

export const scalabilityPerformanceQuestions: MultipleChoiceQuestion[] = [
  {
    id: 'sp-mc-1',
    question:
      'Your deployed model serves 50ms p50 latency but 800ms p99 latency, causing timeout issues for a small percentage of requests. The model itself has consistent inference time (~40ms). What is the most likely cause and solution?',
    options: [
      'Model inference variability; optimize model architecture for consistent latency',
      'Cold start issues; implement connection pooling and keep-alive mechanisms',
      'Resource contention and queueing; implement request prioritization and load shedding',
      'Network latency spikes; deploy model servers closer to clients',
    ],
    correctAnswer: 2,
    explanation:
      "The large gap between p50 (50ms) and p99 (800ms) with consistent model inference time (~40ms) indicates tail latency from queueing and resource contention. During traffic spikes, requests queue up waiting for resources (CPU/GPU/memory). Solutions include: request prioritization (important requests first), load shedding (reject excess requests with fast 503 responses rather than timing out), and auto-scaling. Option A is wrong—model time is consistent. Option B (cold start) would show in all percentiles. Option D (network) doesn't explain why only p99 is affected.",
    difficulty: 'advanced',
    topic: 'Scalability & Performance',
  },
  {
    id: 'sp-mc-2',
    question:
      "You're optimizing a transformer model for production inference. The model has 350M parameters and serves 1000 QPS with 100ms latency requirement. Which optimization technique would provide the best latency improvement without significant accuracy loss?",
    options: [
      'Dynamic quantization to INT8 for linear layers',
      'Knowledge distillation to a smaller student model',
      'Operator fusion and graph optimization using TorchScript or ONNX Runtime',
      'Reduce sequence length and batch size',
    ],
    correctAnswer: 0,
    explanation:
      'Dynamic quantization to INT8 (converting weights to 8-bit integers) provides 2-4x speedup for transformer models with minimal accuracy loss (<1% typically). It reduces memory bandwidth requirements and enables faster computation on CPU/GPU. This is the fastest optimization with the least engineering effort. Knowledge distillation (option B) can work but requires retraining and validation, taking weeks. Operator fusion (option C) helps but typically provides smaller gains (10-30%) than quantization. Reducing sequence length (option D) changes functionality and may not be acceptable.',
    difficulty: 'advanced',
    topic: 'Scalability & Performance',
  },
  {
    id: 'sp-mc-3',
    question:
      'Your model serving system uses Redis for feature caching, but cache hit rate is only 40% despite features being stable. Profiling shows most requests have unique feature combinations. How can you improve cache effectiveness?',
    options: [
      'Increase Redis memory and cache TTL to store more combinations',
      'Implement feature-level caching instead of full feature vector caching',
      'Switch to a distributed cache with more nodes',
      'Pre-compute and cache predictions for all possible feature combinations',
    ],
    correctAnswer: 1,
    explanation:
      "Feature-level caching stores individual features rather than entire feature vectors. If each feature changes independently, caching them separately increases hit rate dramatically. For example, if you have 10 features each with 100 possible values, full vector caching requires 10^10 entries, but feature-level caching needs only 1,000 entries. Option A doesn't solve the combinatorial explosion problem. Option C (more nodes) doesn't address the fundamental caching strategy issue. Option D is infeasible with high cardinality feature combinations.",
    difficulty: 'advanced',
    topic: 'Scalability & Performance',
  },
  {
    id: 'sp-mc-4',
    question:
      "You're deploying a model that performs dynamic batching to improve throughput. However, latency is unpredictable: sometimes 20ms, sometimes 100ms for the same request. What batching parameter adjustment would stabilize latency while maintaining good throughput?",
    options: [
      'Increase maximum batch size to improve throughput',
      'Implement adaptive batching with a timeout threshold (e.g., max 50ms wait time)',
      'Disable batching entirely to ensure consistent latency',
      'Use a fixed batch size and wait for the batch to fill',
    ],
    correctAnswer: 1,
    explanation:
      "Adaptive batching with a timeout combines the benefits of batching (high throughput) with latency guarantees. Requests batch together up to a maximum size, but if the batch doesn't fill within the timeout (e.g., 50ms), process whatever's available. This prevents requests from waiting indefinitely during low traffic. Option A increases max batch size, which can worsen latency during low traffic. Option C (disable batching) sacrifices throughput. Option D (fixed batch size) causes high latency during low traffic when batches don't fill quickly.",
    difficulty: 'advanced',
    topic: 'Scalability & Performance',
  },
  {
    id: 'sp-mc-5',
    question:
      'Your GPU-based model serving has low GPU utilization (30%) despite high request load. CPU usage is 80%. What is the most likely bottleneck and solution?',
    options: [
      'GPU compute capacity is insufficient; use multiple GPUs',
      'CPU-bound preprocessing is bottleneck; move preprocessing to GPU or use multi-threading',
      "Model architecture doesn't utilize GPU efficiently; redesign the model",
      'Memory transfer between CPU and GPU is slow; reduce batch size',
    ],
    correctAnswer: 1,
    explanation:
      "High CPU usage (80%) with low GPU usage (30%) clearly indicates CPU-bound preprocessing is the bottleneck. The GPU is waiting for the CPU to prepare data. Solutions include: move preprocessing to GPU using CUDA/RAPIDS, use multi-threaded preprocessing, or optimize preprocessing code. Option A (more GPUs) doesn't help—the existing GPU is underutilized. Option C (model redesign) is unnecessary when the model isn't the bottleneck. Option D (reduce batch size) would reduce GPU utilization further and hurt throughput.",
    difficulty: 'intermediate',
    topic: 'Scalability & Performance',
  },
];
