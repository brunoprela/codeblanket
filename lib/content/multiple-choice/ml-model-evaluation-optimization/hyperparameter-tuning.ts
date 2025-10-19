import { MultipleChoiceQuestion } from '../../../types';

export const hyperparameterTuningMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'hyperparameter-tuning-mc-1',
    question:
      'You have 5 hyperparameters, each with 5 possible values. How many configurations would grid search evaluate?',
    options: ['25', '125', '625', '3,125'],
    correctAnswer: 3,
    explanation:
      'Grid search tries all combinations: 5 × 5 × 5 × 5 × 5 = 5^5 = 3,125 configurations. This exponential growth is why grid search becomes impractical for high-dimensional hyperparameter spaces.',
  },
  {
    id: 'hyperparameter-tuning-mc-2',
    question:
      'Why does random search often perform as well as grid search with fewer iterations?',
    options: [
      'Random search uses a better optimization algorithm',
      'Not all hyperparameters are equally important, so random search explores important ones more thoroughly',
      'Random search automatically identifies the best hyperparameters',
      'Grid search wastes time on duplicate configurations',
    ],
    correctAnswer: 1,
    explanation:
      'Random search works well because typically only a few hyperparameters significantly impact performance. With the same budget, random search tries more unique values for each parameter than grid search, giving better coverage of the important dimensions.',
  },
  {
    id: 'hyperparameter-tuning-mc-3',
    question:
      'When tuning a learning rate hyperparameter, which search space is most appropriate?',
    options: [
      'Linear: [0.001, 0.002, 0.003, ..., 0.999]',
      'Logarithmic: [0.0001, 0.001, 0.01, 0.1, 1.0]',
      'Random uniform: uniform(0, 1)',
      'Categorical: [small, medium, large]',
    ],
    correctAnswer: 1,
    explanation:
      'Learning rates should be searched on a logarithmic scale because the effect of changing from 0.001 to 0.01 is similar to changing from 0.01 to 0.1 (10x multiplier). Linear spacing would waste evaluations in regions that behave similarly. Use loguniform or powers of 10.',
  },
  {
    id: 'hyperparameter-tuning-mc-4',
    question:
      'What is the purpose of nested cross-validation in hyperparameter tuning?',
    options: [
      'To make hyperparameter tuning faster',
      'To get an unbiased estimate of model performance with tuning',
      'To tune more hyperparameters simultaneously',
      'To avoid overfitting on the training set',
    ],
    correctAnswer: 1,
    explanation:
      'Nested CV provides an unbiased performance estimate by using an outer CV loop for evaluation and an inner CV loop for hyperparameter selection. This prevents overfitting to the validation set that occurs when you optimize hyperparameters using the same CV folds you report.',
  },
  {
    id: 'hyperparameter-tuning-mc-5',
    question:
      'You performed hyperparameter tuning and your best model has CV score of 0.85 but test score of 0.75. What likely happened?',
    options: [
      'The model is working perfectly',
      'You overfitted to the validation set during hyperparameter tuning',
      'You need more training data',
      'The hyperparameters are optimal',
    ],
    correctAnswer: 1,
    explanation:
      'The 0.10 gap between CV (0.85) and test (0.75) scores suggests you overfitted to the validation set during hyperparameter tuning. You optimized hyperparameters to maximize CV score, which makes it an optimistically biased estimate. This is why nested CV or a separate validation set is important.',
  },
];
