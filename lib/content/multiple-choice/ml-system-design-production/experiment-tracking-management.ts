import { MultipleChoiceQuestion } from '@/lib/types';

export const experimentTrackingManagementMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'etm-mc-1',
      question:
        'What information must be logged for full experiment reproducibility?',
      options: [
        'Only hyperparameters',
        'Only model weights',
        'Code version (git commit), data version (DVC hash), environment (Docker/conda), hyperparameters, random seeds',
        'Only training metrics',
      ],
      correctAnswer: 2,
      explanation:
        "Full reproducibility requires: (1) Code: git commit hash (exact code version), (2) Data: DVC hash or S3 path with timestamp (exact dataset), (3) Environment: requirements.txt (pinned versions) or Docker image tag (exact packages), (4) Hyperparameters: learning_rate, batch_size, etc., (5) Random seeds: Python, NumPy, PyTorch, CUDA seeds, (6) Config: Training config (YAML). MLflow tracks all. To reproduce: git checkout <commit>, dvc checkout <data>, docker run <image> python train.py. Verify: Metrics match within floating-point error. Without any: Can't reproduce. Example: Model trained 6 months ago needs debugging → retrieve all from MLflow → reproduce exactly.",
    },
    {
      id: 'etm-mc-2',
      question:
        'Which hyperparameter optimization method is most sample-efficient?',
      options: [
        'Grid search',
        'Random search',
        'Bayesian optimization (Optuna)',
        'Manual tuning',
      ],
      correctAnswer: 2,
      explanation:
        'Bayesian optimization (Optuna): Most sample-efficient. Builds probabilistic model (Gaussian Process) of performance function. Acquisition function suggests next hyperparameters (balance exploration vs exploitation). Finds optimal in 30-50 trials (vs 100-200 for random). Example: 50 trials × 5 min = 250 min. Random: 2-5× more efficient than grid but less than Bayesian. Grid: Exponential cost (3 params × 10 values each = 1000 trials). Manual: Inefficient, biased. Use Bayesian when: Expensive evaluations (long training), limited budget (50 trials), continuous search space. Use random when: Quick evaluations, many trials possible. Optuna implements Bayesian (TPE sampler) + early stopping (Hyperband).',
    },
    {
      id: 'etm-mc-3',
      question: 'What is the purpose of MLflow Model Registry?',
      options: [
        'Train models faster',
        'Version models and manage staging → production promotion',
        'Store training data',
        'Tune hyperparameters',
      ],
      correctAnswer: 1,
      explanation:
        'MLflow Model Registry: Version control for models. Features: (1) Versioning: Models auto-versioned (v1, v2, v3), (2) Stages: None → Staging → Production → Archived, (3) Lineage: Each version links to training run (code, data, hyperparameters, metrics), (4) Approval workflow: Promote requires approval (ML lead + PM), (5) Rollback: Keep last 3-5 versions, can rollback if new version fails. Example: fraud_detection model: v1 in Production (serving), v2 in Staging (testing), v3 in None (just trained). Promote v2 to Production (after A/B test success), archive v1. Rollback: If v2 fails, revert to v1 (1 command). Not for: Training speed, data storage, hyperparameter tuning (use Optuna).',
    },
    {
      id: 'etm-mc-4',
      question: 'Why is nested cross-validation important for AutoML systems?',
      options: [
        'Faster training',
        'Prevents overfitting to validation set when selecting models',
        'Reduces memory usage',
        'Improves accuracy',
      ],
      correctAnswer: 1,
      explanation:
        'Nested CV prevents validation set overfitting. Problem: AutoML tests 100 models on validation set, picks best. Best model overfit to validation (optimistic accuracy). Solution: Nested CV. Outer loop (5 folds): For each fold, use 4 for AutoML (including model selection), test on 5th (unseen). Inner loop (within AutoML): 4 folds for training, 1 for validation (model selection). Average test accuracy across outer folds → unbiased estimate. Example: Inner CV: Try 100 models, pick best (90% validation accuracy). Outer CV: Test on held-out fold (87% test accuracy, realistic). Single CV: Reports 90% (overfit). Nested CV: Reports 87% (honest). Cost: 5× slower (5 outer folds). Use when: AutoML, hyperparameter tuning, need unbiased accuracy estimate.',
    },
    {
      id: 'etm-mc-5',
      question:
        'What is the main advantage of using DVC (Data Version Control)?',
      options: [
        'Faster model training',
        'Version large datasets efficiently with git-like workflow',
        'Automated feature engineering',
        'Model deployment',
      ],
      correctAnswer: 1,
      explanation:
        "DVC versions large datasets efficiently. How: dvc add data/train.csv creates data/train.csv.dvc (hash + metadata, <1KB) stored in git. Actual data (100GB) stored in S3. Benefits: (1) Git-like workflow (dvc checkout, dvc push), (2) Efficient (doesn't duplicate data, stores hash), (3) Reproducible (retrieve exact dataset version), (4) Lightweight git (only metadata in git, not data). Example: data_v1.0 (train.csv, 10GB), data_v1.1 (train.csv, 12GB). Git stores 2KB of metadata, S3 stores 22GB (both versions). Reproduce: dvc checkout data@v1.0 pulls 10GB from S3. Without DVC: Store data in git (bloats repo) or manually track (error-prone). DVC standard for ML data versioning.",
    },
  ];
