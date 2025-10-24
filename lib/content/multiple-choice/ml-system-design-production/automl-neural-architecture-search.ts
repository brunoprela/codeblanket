import { MultipleChoiceQuestion } from '@/lib/types';

export const automlNeuralArchitectureSearchQuestions: MultipleChoiceQuestion[] =
  [
    {
      id: 'anas-mc-1',
      question:
        "You're using AutoML to find the best model for a tabular dataset with 50 features and 100K samples. After 6 hours, AutoML has evaluated 200 models but performance has plateaued. What is the most efficient next step?",
      options: [
        'Continue running AutoML for longer to find better models',
        'Analyze the top performing models to identify patterns, then manually tune in that direction',
        'Restart AutoML with a different random seed to explore different areas',
        'Increase the search space to include more complex models',
      ],
      correctAnswer: 1,
      explanation:
        "When AutoML plateaus, analyze the top models to identify patterns: which algorithms work best? Which hyperparameters are consistently good? This provides insights to manually refine the search space or perform targeted tuning, which is more efficient than blind searching. Option A (continue running) wastes compute if you've plateaued. Option C (different seed) explores the same space redundantly. Option D (expand search) adds complexity without addressing why simpler models have plateaued—it may just find more complex models with similar performance.",
      difficulty: 'advanced',
      topic: 'AutoML & Neural Architecture Search',
    },
    {
      id: 'anas-mc-2',
      question:
        "You're implementing Neural Architecture Search (NAS) for a computer vision task. Full model training takes 12 hours per architecture. Which NAS approach would most efficiently search the space?",
      options: [
        'Random search: evaluate 50 random architectures with full training',
        'Reinforcement learning-based NAS with early stopping predictors',
        'One-shot NAS with weight sharing (e.g., DARTS)',
        'Evolutionary algorithms with full training and population size 20',
      ],
      correctAnswer: 2,
      explanation:
        'One-shot NAS with weight sharing (like DARTS) trains a supernet once, where all candidate architectures share weights. Architecture search happens by optimizing continuous architecture parameters, requiring only one training run (12 hours) instead of training each architecture separately. Option A (random search 50 architectures) takes 600 hours. Option B (RL-based with early stopping) is better but still requires many partial training runs. Option D (evolutionary) needs multiple generations of full training, taking days or weeks.',
      difficulty: 'advanced',
      topic: 'AutoML & Neural Architecture Search',
    },
    {
      id: 'anas-mc-3',
      question:
        "You're using H2O AutoML for a classification problem. AutoML selected a stacked ensemble as the best model, combining 5 base models. However, this model is too complex for your production environment. What is the best approach?",
      options: [
        'Manually select the best single model from the leaderboard',
        'Use the ensemble but deploy only the top 2 models to reduce complexity',
        'Re-run AutoML with constraints on model complexity and ensemble settings',
        'Use knowledge distillation to compress the ensemble into a single model',
      ],
      correctAnswer: 2,
      explanation:
        'Re-running AutoML with appropriate constraints (disable ensembles, limit model types, set max_models, specify interpretability requirements) ensures the best model within your production constraints. Most AutoML tools support these settings. Option A (manual selection) works but may miss opportunities—AutoML with constraints might find a better single model. Option B (subset ensemble) is arbitrary—why 2 models? Option D (distillation) adds engineering complexity and may not be necessary if a good single model exists.',
      difficulty: 'intermediate',
      topic: 'AutoML & Neural Architecture Search',
    },
    {
      id: 'anas-mc-4',
      question:
        "You're comparing AutoML tools for a production ML pipeline. Your dataset has 1M rows, 100 features, and requires model updates every week. Which AutoML consideration is MOST important for this use case?",
      options: [
        'Model performance: highest accuracy on validation set',
        'Search efficiency: time to find good models',
        'Reproducibility: ability to reproduce the same model given the same data and seed',
        'Deployment flexibility: easy model export and serving',
      ],
      correctAnswer: 2,
      explanation:
        "Reproducibility is critical for weekly retraining pipelines. You need to reliably reproduce models for debugging, compliance, and stable deployments. Non-reproducible AutoML is problematic in production. Model performance (option A) is important but most AutoML tools achieve similar performance. Search efficiency (option B) matters but is one-time cost per training. Deployment flexibility (option D) is important but can be handled with standard export formats—reproducibility is harder to fix if the tool doesn't support it.",
      difficulty: 'advanced',
      topic: 'AutoML & Neural Architecture Search',
    },
    {
      id: 'anas-mc-5',
      question:
        "You're using automated feature engineering (e.g., Featuretools) that generates 500 features from your original 20 features. However, model training has become very slow and models are overfitting. What is the most appropriate strategy?",
      options: [
        'Use all 500 features but apply strong regularization',
        'Manually select features based on domain knowledge',
        'Apply automated feature selection on top of feature engineering to retain top N features',
        'Reduce the depth of automated feature engineering to generate fewer features',
      ],
      correctAnswer: 2,
      explanation:
        'Automated feature selection (using methods like mutual information, recursive feature elimination, or L1 regularization) on top of automated feature engineering provides the best of both worlds: discover useful feature interactions automatically, then systematically select the most predictive ones. This is more principled than manual selection and more thorough than limiting feature generation. Option A (regularization) may help but 500 features still slow training. Option B (manual selection) loses the benefits of automation. Option D (reduce depth) might discard valuable high-order features.',
      difficulty: 'advanced',
      topic: 'AutoML & Neural Architecture Search',
    },
  ];
