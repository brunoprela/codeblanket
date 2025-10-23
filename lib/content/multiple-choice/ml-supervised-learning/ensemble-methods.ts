/**
 * Multiple Choice Questions for Ensemble Methods
 */

import { MultipleChoiceQuestion } from '../../../types';

export const ensemblemethodsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'ensemble-methods-mc-1',
    question: 'What is the key requirement for an effective ensemble?',
    options: [
      'All models must have high accuracy',
      'Models must be diverse and make different errors',
      'All models must be the same type',
      'Models must be trained on the same data subset',
    ],
    correctAnswer: 1,
    explanation:
      "Diversity is crucial for ensembles. If all models make the same errors, combining them doesn't help. Diverse models (different algorithms, hyperparameters, or data subsets) make uncorrelated errors that cancel out when averaged. Even if individual models are mediocre, diversity enables the ensemble to perform better.",
  },
  {
    id: 'ensemble-methods-mc-2',
    question:
      'In a VotingClassifier, what is the difference between hard and soft voting?',
    options: [
      'Hard voting uses majority vote, soft voting averages predicted probabilities',
      'Hard voting is more accurate than soft voting',
      'Soft voting requires more computational power',
      'There is no difference, they are synonyms',
    ],
    correctAnswer: 0,
    explanation:
      'Hard voting uses majority vote (most common class prediction wins). Soft voting averages the predicted probabilities from each model and picks the class with highest average probability. Soft voting is usually more accurate because it considers confidence levels, not just final predictions.',
  },
  {
    id: 'ensemble-methods-mc-3',
    question: 'What is the main advantage of stacking over simple voting?',
    options: [
      'Stacking is faster to train',
      'Stacking learns optimal weights to combine base models',
      'Stacking requires fewer base models',
      'Stacking never overfits',
    ],
    correctAnswer: 1,
    explanation:
      'Stacking trains a meta-model that learns how to optimally combine base model predictions. This meta-model can learn complex, non-linear combination rules, potentially achieving better performance than simple averaging. However, it requires more training time and proper cross-validation to avoid overfitting.',
  },
  {
    id: 'ensemble-methods-mc-4',
    question: 'Why are out-of-fold predictions used in stacking?',
    options: [
      'To speed up training',
      'To prevent data leakage and overfitting in the meta-model',
      'To reduce memory usage',
      'To make predictions more accurate',
    ],
    correctAnswer: 1,
    explanation:
      'Out-of-fold (OOF) predictions prevent data leakage. If we used training predictions directly, base models would have seen that data during training, leading to overly optimistic predictions and meta-model overfitting. OOF predictions are generated on data each base model never saw during training, providing unbiased inputs for the meta-model.',
  },
  {
    id: 'ensemble-methods-mc-5',
    question:
      "You have a VotingClassifier with 5 base models: three achieve 80% accuracy, two achieve 75%. The ensemble achieves 78%. What's the most likely problem?",
    options: [
      'Not enough base models',
      'Base models are too similar (highly correlated predictions)',
      'Ensemble always performs worse than individual models',
      'Soft voting should be used instead of hard voting',
    ],
    correctAnswer: 1,
    explanation:
      "When an ensemble performs worse than its best components, the models are likely highly correlated - they make the same errors. Without diversity, averaging doesn't help and can actually hurt. Solution: use more diverse model types (e.g., mix tree-based, linear, and kernel methods) or different hyperparameters to decorrelate predictions.",
  },
];
