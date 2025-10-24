import { MultipleChoiceQuestion } from '@/types/curriculum';

export const mlSystemCaseStudiesQuestions: MultipleChoiceQuestion[] = [
  {
    id: 'mscs-mc-1',
    question:
      "You're designing a recommendation system for an e-commerce platform with 10M products and 100M users. The system must generate personalized recommendations in real-time (<100ms) when users visit the homepage. Which architecture would best meet these requirements?",
    options: [
      'Two-stage funnel: candidate generation (recall ~1000 items) + ranking model (deep neural network)',
      'Single end-to-end neural network that directly ranks all 10M products',
      'Collaborative filtering matrix factorization computed in real-time',
      'Content-based filtering using pre-computed product embeddings',
    ],
    correctAnswer: 0,
    explanation:
      'Two-stage funnel is the industry-standard approach: (1) Candidate generation quickly retrieves ~1000 potentially relevant items using efficient methods (collaborative filtering, content-based, popularity), (2) Ranking model applies expensive deep learning to rank these candidates. This meets latency requirements while maintaining quality. Option B (rank all 10M) is computationally infeasible in 100ms. Option C (real-time matrix factorization) is too slow. Option D (content-based only) misses collaborative signals and personalization.',
    difficulty: 'advanced',
    topic: 'ML System Case Studies',
  },
  {
    id: 'mscs-mc-2',
    question:
      "You're building a real-time fraud detection system for credit card transactions. The system must process 10,000 transactions per second with decisions in <50ms to avoid blocking purchases. A transaction flagged as fraud is sent to human review. How should you design this system?",
    options: [
      'Complex ensemble model (XGBoost + Neural Network) that evaluates all features for maximum accuracy',
      'Tiered system: fast rule-based filter → feature store → ML model for borderline cases',
      'Single lightweight model (logistic regression) for all transactions',
      'Batch processing: queue transactions and process in batches every minute',
    ],
    correctAnswer: 1,
    explanation:
      'Tiered architecture optimizes latency and accuracy: (1) Fast rule-based system catches obvious fraud/legitimate cases (~80% of traffic) in <5ms, (2) Feature store provides pre-computed features quickly, (3) ML model focuses compute on borderline cases (~20% of traffic). This meets latency requirements while maintaining detection quality. Option A (complex ensemble) is too slow for all transactions. Option C (simple model for all) sacrifices accuracy. Option D (batch processing) violates real-time requirement and allows fraud to complete.',
    difficulty: 'advanced',
    topic: 'ML System Case Studies',
  },
  {
    id: 'mscs-mc-3',
    question:
      "You're designing a search ranking system for a content platform with 1B documents. Users expect results in <200ms. The ranking model is a BERT-based neural network that takes 50ms per document. How can you make this feasible?",
    options: [
      'Use smaller BERT model (DistilBERT) to reduce inference time to <10ms per document',
      'Multi-stage ranking: retrieval (BM25) → pre-ranking (lightweight model for top 1000) → BERT re-ranking (top 20)',
      'Cache BERT embeddings for all documents and compute similarity at query time',
      'Use approximate nearest neighbor search with BERT query embeddings',
    ],
    correctAnswer: 1,
    explanation:
      'Multi-stage ranking is the standard approach for large-scale search: (1) Retrieval (BM25, etc.) quickly narrows to ~1000 candidates, (2) Lightweight neural pre-ranker (simple BERT or simpler) ranks these to top 100, (3) Expensive BERT re-ranks only top 20-50. This keeps latency <200ms (50ms × 20 documents = 1000ms budget for final stage). Option A (DistilBERT for all) is still too slow for 1000+ documents. Option C (cache embeddings) works for similarity search but ranking requires query-document interaction. Option D misses the ranking nuance needed for search quality.',
    difficulty: 'advanced',
    topic: 'ML System Case Studies',
  },
  {
    id: 'mscs-mc-4',
    question:
      "You're building a real-time bidding (RTB) system for ad auctions. The system must predict click-through rate (CTR) for an ad-user pair and submit a bid within 100ms of receiving the auction request. Which system design is most appropriate?",
    options: [
      'Deep learning model with real-time feature computation from user history',
      'Lightweight model (logistic regression/GBDT) with pre-computed user and ad features from feature store',
      'Hybrid: pre-compute user embeddings, compute ad features in real-time, use neural network for CTR',
      'Simple heuristic-based bidding without ML',
    ],
    correctAnswer: 2,
    explanation:
      'Hybrid approach balances latency and accuracy: pre-computed user embeddings (updated periodically) are fast to retrieve, ad features can be computed quickly (ad metadata is small), and a neural network captures complex interactions. This fits within 100ms. Option A (real-time history processing) is too slow—aggregating user history takes longer than 100ms. Option B (simple GBDT) may not capture complex patterns well enough for competitive bidding. Option D (no ML) leaves money on the table in the competitive ad auction environment.',
    difficulty: 'advanced',
    topic: 'ML System Case Studies',
  },
  {
    id: 'mscs-mc-5',
    question:
      "You're designing an algorithmic trading system that uses ML to predict short-term price movements and execute trades. The model must process market data and make decisions in <10ms. Which architecture would best meet this ultra-low latency requirement?",
    options: [
      'Deep neural network running on GPU for maximum prediction accuracy',
      'Ensemble of models (neural network + GBDT + linear) with voting',
      'Optimized lightweight model (linear/GBDT) running on CPU with hardware-accelerated feature computation',
      'Reinforcement learning agent that learns optimal trading strategy',
    ],
    correctAnswer: 2,
    explanation:
      "Ultra-low latency (<10ms) requires optimized lightweight models on CPU with minimal overhead. GPU has transfer latency (~1-5ms) that's too high. Linear models or small GBDT with hardware-optimized feature computation (SIMD instructions, careful memory access) can achieve <1ms inference. Trading profitability comes from speed + reasonable accuracy, not perfect accuracy. Option A (GPU) has too much latency. Option B (ensemble) is too slow and complex. Option D (RL) doesn't address latency requirements and online learning in production trading is risky.",
    difficulty: 'advanced',
    topic: 'ML System Case Studies',
  },
];
