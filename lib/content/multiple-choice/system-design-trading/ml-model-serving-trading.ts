import { MultipleChoiceQuestion } from '@/lib/types';

export const mlModelServingTradingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mmst-mc-1',
    question:
      'What is the typical inference latency for a LightGBM model with 100 trees and depth 6?',
    options: [
      '1-5 microseconds',
      '20-50 microseconds',
      '1-5 milliseconds',
      '50-100 milliseconds',
    ],
    correctAnswer: 1,
    explanation:
      'LightGBM with 100 trees depth 6: 20-50μs inference latency (fast!). This makes it suitable for HFT where <100μs required. Linear models faster (5-10μs), deep neural networks much slower (1-10ms). LightGBM sweet spot: good accuracy with acceptable latency. More trees or depth → slower but more accurate. Trade-off depends on strategy requirements.',
  },
  {
    id: 'mmst-mc-2',
    question:
      'What is the primary advantage of using ONNX Runtime for model serving in trading?',
    options: [
      'It supports only TensorFlow models',
      'It provides 2-5x speedup vs native framework with framework portability',
      'It requires GPU for inference',
      'It automatically retrains models',
    ],
    correctAnswer: 1,
    explanation:
      "ONNX Runtime provides 2-5x speedup through graph optimization, operator fusion, and optimized C++ implementation. Works with models from PyTorch, TensorFlow, scikit-learn (portable). Example: RandomForest 200μs native → 50μs ONNX. Not TensorFlow-only (framework-agnostic), works on CPU or GPU (not GPU-required), doesn't retrain (just serves). Best choice for production trading where latency critical.",
  },
  {
    id: 'mmst-mc-3',
    question: 'What is training-serving skew and why is it problematic?',
    options: [
      'When training data is older than serving data',
      'When features computed differently in training vs serving, causing model to fail',
      'When model is retrained too frequently',
      'When serving latency is higher than training time',
    ],
    correctAnswer: 1,
    explanation:
      "Training-serving skew: features computed differently in training vs serving → model fails in production. Example: training uses df.rolling().mean() (may include future data), serving computes manually (correct). Results don't match → model predicts poorly. Solution: same code for both (shared feature functions), version features, comprehensive tests. Not about data age (different issue), not training frequency, not latency comparison.",
  },
  {
    id: 'mmst-mc-4',
    question: 'What is the purpose of blue-green deployment for ML models?',
    options: [
      'To train two models simultaneously',
      'To switch between models instantly with zero downtime',
      'To reduce model file size',
      'To improve prediction accuracy',
    ],
    correctAnswer: 1,
    explanation:
      'Blue-green deployment: run two model versions (blue=current, green=new). Load new model to green while blue serves traffic. Test green, then switch instantly (change pointer). Zero downtime, instant rollback if issues. Not for training (deployment only), not for compression, not for accuracy (deployment strategy). Enables safe model updates in production without service interruption.',
  },
  {
    id: 'mmst-mc-5',
    question:
      'Why would you store pre-computed features in Redis for model serving?',
    options: [
      'Redis automatically retrains the model',
      'To avoid expensive feature computation on every prediction request',
      'Redis provides better prediction accuracy',
      'To comply with regulatory requirements',
    ],
    correctAnswer: 1,
    explanation:
      "Redis feature store: pre-compute expensive features (SMA, RSI) in batch pipeline, store in Redis, serve to model in <1ms. Avoids recomputing on every request (saves latency). Example: Computing 20-day SMA requires 20 historical prices—expensive. Pre-compute once, serve 1000x. Redis doesn't train models, doesn't affect accuracy (just caching), not for compliance. Enables sub-millisecond serving by separating expensive computation from inference.",
  },
];
