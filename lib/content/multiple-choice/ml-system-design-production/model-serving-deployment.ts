import { MultipleChoiceQuestion } from '@/types/curriculum';

export const modelServingDeploymentQuestions: MultipleChoiceQuestion[] = [
  {
    id: 'msd-mc-1',
    question:
      "You're deploying a recommendation model that serves 10,000 requests per second with a p99 latency requirement of 50ms. The model inference takes 30ms. Which deployment strategy would best meet this requirement?",
    options: [
      'Deploy a single high-performance GPU instance with request queuing',
      'Use horizontal scaling with load balancing across multiple CPU instances with model caching',
      'Implement a multi-tier architecture with edge caching and model serving layer with batching',
      'Deploy on serverless functions with automatic scaling',
    ],
    correctAnswer: 2,
    explanation:
      'A multi-tier architecture is optimal: edge caching handles popular items (reducing backend load), and the model serving layer uses dynamic batching to improve throughput while maintaining latency. This approach handles the high QPS while meeting p99 latency. Option A (single instance) cannot handle 10K QPS even with queuing. Option B (CPU with caching) helps but batching is more effective for throughput. Option D (serverless) typically has cold start issues and unpredictable latency, making p99 guarantees difficult.',
    difficulty: 'advanced',
    topic: 'Model Serving & Deployment',
  },
  {
    id: 'msd-mc-2',
    question:
      'Your deployed model needs to handle varying traffic patterns: 1000 QPS during business hours and 100 QPS at night. The model requires 2 seconds to load and initialize. Which scaling strategy is most cost-effective while maintaining performance?',
    options: [
      'Keep enough instances running for peak load at all times',
      'Use autoscaling with metric-based triggers and a minimum instance count of 2',
      'Deploy on spot instances with aggressive scale-down policies',
      'Use predictive scaling based on historical patterns with pre-warming',
    ],
    correctAnswer: 3,
    explanation:
      'Predictive scaling based on historical patterns allows pre-warming instances before traffic increases, avoiding the 2-second cold start during scale-up. This balances cost (scaling down during low traffic) and performance (no latency spikes). Option A wastes resources during low traffic. Option B (reactive autoscaling) causes latency spikes during sudden traffic increases due to the 2-second initialization time. Option C (spot instances with aggressive scale-down) risks availability and causes frequent cold starts.',
    difficulty: 'advanced',
    topic: 'Model Serving & Deployment',
  },
  {
    id: 'msd-mc-3',
    question:
      "You're comparing model serialization formats for a large transformer model (1.5GB). The model will be loaded frequently on serving instances. Which format provides the best tradeoff between load time and deployment flexibility?",
    options: [
      'Pickle format (.pkl) with compression',
      'ONNX format with runtime optimization',
      'PyTorch TorchScript (.pt) format',
      'SavedModel format with XLA compilation',
    ],
    correctAnswer: 1,
    explanation:
      "ONNX provides excellent load times, framework independence (can switch between PyTorch, TensorFlow, etc.), and runtime optimization for inference (fused operations, quantization support). It's specifically designed for production deployment. Pickle (option A) is Python-specific, has security issues, and is slower. TorchScript (option C) is PyTorch-only and has limited optimization compared to ONNX. SavedModel with XLA (option D) is TensorFlow-specific and XLA compilation adds overhead during loading.",
    difficulty: 'intermediate',
    topic: 'Model Serving & Deployment',
  },
  {
    id: 'msd-mc-4',
    question:
      'Your deployed model is experiencing memory leaks, with memory usage growing by 50MB per hour until the instance crashes. The model uses custom preprocessing and caching. What is the most systematic approach to diagnose and fix this issue?',
    options: [
      'Increase instance memory and implement periodic restarts',
      'Profile memory usage over time, identify growing objects, implement proper cleanup and resource limits',
      'Switch to a stateless serving architecture without any caching',
      'Implement aggressive garbage collection with reduced GC thresholds',
    ],
    correctAnswer: 1,
    explanation:
      "Systematic memory profiling (using tools like memory_profiler, tracemalloc, or py-spy) identifies which objects are growing. Common culprits include uncapped caches, unreleased file handles, or circular references. Once identified, implement proper cleanup (LRU cache limits, context managers, weak references). Option A masks the problem without fixing it. Option C (no caching) may hurt performance unnecessarily. Option D (aggressive GC) adds CPU overhead and doesn't fix the underlying leak—objects are still being retained.",
    difficulty: 'advanced',
    topic: 'Model Serving & Deployment',
  },
  {
    id: 'msd-mc-5',
    question:
      'You need to deploy a model update with significant architectural changes. The new model has a different input schema and slower inference time. What deployment strategy minimizes risk and allows for quick rollback?',
    options: [
      'Blue-green deployment: deploy new version alongside old, switch traffic instantly, keep old version for rollback',
      'Rolling update: gradually replace instances with new version',
      'Canary deployment: route 5% traffic to new version, monitor metrics, gradually increase if successful',
      'Shadow deployment: run new model in parallel, compare outputs, then switch traffic after validation',
    ],
    correctAnswer: 3,
    explanation:
      "Shadow deployment is safest for significant changes: the new model processes real requests but doesn't serve responses (shadow mode). You can compare outputs, monitor latency, and validate behavior with production data before switching traffic. This catches issues without impacting users. Canary (option C) is good but exposes 5% of users to potential issues. Blue-green (option A) is risky for architectural changes—issues affect 100% of traffic immediately upon switch. Rolling updates (option B) expose users during rollout and can be harder to rollback mid-update.",
    difficulty: 'advanced',
    topic: 'Model Serving & Deployment',
  },
];
