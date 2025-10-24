import { MultipleChoiceQuestion } from '@/lib/types';

export const mlSystemDesignPrinciplesMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mlsdp-mc-1',
      question: 'What is the primary purpose of a feature store in ML systems?',
      options: [
        'Store model weights',
        'Centralize feature computation for training and serving consistency',
        'Store training data',
        'Cache predictions',
      ],
      correctAnswer: 1,
      explanation:
        'Feature store centralizes feature computation. Define features once, use in both training (offline store: S3/BigQuery) and serving (online store: Redis <5ms). Benefits: (1) Consistency (no training-serving skew), (2) Reusability (teams share features), (3) Point-in-time correctness (no data leakage). Example: user_30d_transactions computed once, used by all models. Without: Each team computes separately, inconsistencies, wasted compute.',
    },
    {
      id: 'mlsdp-mc-2',
      question:
        'In a canary deployment for ML models, what is the typical traffic split progression?',
      options: [
        '50% → 100%',
        '10% → 50% → 100%',
        '1% → 10% → 50% → 100%',
        '100% immediately',
      ],
      correctAnswer: 1,
      explanation:
        'Canary deployment: Gradual rollout. Standard progression: 10% traffic → monitor 1 hour (check accuracy, latency, error rate) → if OK, 50% → monitor 4 hours → if OK, 100%. Allows early detection of issues with minimal user impact. If metrics degrade at any stage → automatic rollback. Alternative: Blue-green (instant 100% switch). Canary safer but slower.',
    },
    {
      id: 'mlsdp-mc-3',
      question:
        'What is the main advantage of batch prediction over real-time prediction?',
      options: [
        'Lower latency',
        'Higher throughput and lower cost',
        'Better accuracy',
        'Easier debugging',
      ],
      correctAnswer: 1,
      explanation:
        'Batch prediction: Process millions of records offline (nightly Spark job). Advantages: (1) High throughput (parallel processing), (2) Low cost (spot instances, run once), (3) Simple infrastructure (no always-on API). Disadvantages: High latency (hours from data to prediction). Use when: Decision not urgent (email recommendations, credit score updates). Real-time: Low latency (<100ms) but higher cost (always-on servers), lower throughput per instance. Use when: Immediate decision needed (fraud detection at checkout).',
    },
    {
      id: 'mlsdp-mc-4',
      question:
        'Which metric is most important for detecting data drift in production ML models?',
      options: [
        'Model accuracy',
        'Inference latency',
        'Population Stability Index (PSI)',
        'API error rate',
      ],
      correctAnswer: 2,
      explanation:
        'PSI (Population Stability Index) detects distribution shifts. Formula: PSI = Σ (prod_pct - train_pct) × ln(prod_pct / train_pct). Interpretation: PSI <0.1 (no shift), 0.1-0.25 (moderate shift, monitor), >0.25 (major shift, retrain immediately). Example: Training data: mean age = 35. Production: mean age = 45. PSI = 0.32 → drift detected. Alternatives: KS test (per-feature), Evidently library (automated drift detection). Accuracy: Lagging indicator (drift causes accuracy drop later). PSI: Leading indicator (detect drift before accuracy degrades).',
    },
    {
      id: 'mlsdp-mc-5',
      question:
        'What is the primary benefit of using MLflow for experiment tracking?',
      options: [
        'Faster model training',
        'Version control for code only',
        'Track hyperparameters, metrics, artifacts, and model lineage',
        'Automated hyperparameter tuning',
      ],
      correctAnswer: 2,
      explanation:
        "MLflow tracks full experiment lineage: (1) Hyperparameters (learning_rate, batch_size), (2) Metrics (accuracy, loss curves), (3) Artifacts (model, plots, config files), (4) Code version (git commit hash), (5) Dataset version, (6) Environment. Benefits: Reproducibility (reproduce any model from run_id), comparison (compare 100 runs side-by-side), model registry (version models, promote Staging → Production). Example: Model v2 lineage: run_id=abc123, git commit=def456, data_v1.5, accuracy=0.92. Can reproduce exact model anytime. Not for: Training speed (doesn't speed up training), automated tuning (use Optuna for that).",
    },
  ];
