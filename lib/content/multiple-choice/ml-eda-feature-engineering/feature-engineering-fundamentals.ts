/**
 * Multiple choice questions for Feature Engineering Fundamentals section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const featureengineeringfundamentalsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'mc1',
      question:
        'Which of the following is an example of data leakage in feature engineering?',
      options: [
        'Creating a "customer_age" feature from birthdate',
        'Creating a "days_until_purchase" feature when predicting whether customer will purchase',
        'Creating a "log_income" feature from "income"',
        'Creating an interaction feature between "price" and "quantity"',
      ],
      correctAnswer: 1,
      explanation:
        '"days_until_purchase" uses future information (when the purchase happens) that wouldn\'t be known at prediction time. This is classic temporal leakage. The other features use only information available before prediction.',
    },
    {
      id: 'mc2',
      question: 'When should you fit a StandardScaler for feature scaling?',
      options: [
        'On the entire dataset (train + test combined)',
        'On the training set only, then apply to both train and test',
        'Separately on train and test sets',
        'Scaling is not necessary for feature engineering',
      ],
      correctAnswer: 1,
      explanation:
        'Always fit scalers on training data only to avoid data leakage, then apply the same transformation to both train and test sets. This maintains train/test consistency and prevents test statistics from influencing training.',
    },
    {
      id: 'mc3',
      question:
        'What is the primary purpose of creating interaction features (e.g., x1 * x2)?',
      options: [
        'To reduce the number of features',
        'To capture non-additive relationships where features work together',
        'To handle missing values',
        'To normalize feature distributions',
      ],
      correctAnswer: 1,
      explanation:
        'Interaction features capture non-additive relationships where the combined effect of two features differs from their sum. For example, "expensive product × high quantity" might have a special effect not captured by either feature alone.',
    },
    {
      id: 'mc4',
      question:
        'Which type of model benefits MOST from extensive feature engineering?',
      options: [
        'Deep neural networks',
        'Linear models (linear/logistic regression)',
        'Random Forest',
        'K-Nearest Neighbors',
      ],
      correctAnswer: 1,
      explanation:
        'Linear models benefit most from feature engineering because they can only model linear relationships. Good feature engineering (polynomial features, interactions, transformations) makes complex relationships linear. Tree-based models can discover some of these patterns automatically.',
    },
    {
      id: 'mc5',
      question: 'Why is documentation important in feature engineering?',
      options: [
        "It\'s not important; only the model performance matters",
        'To enable reproducibility, debugging, and understanding of what features represent',
        'To make the code look professional',
        'Documentation is only needed for complex features',
      ],
      correctAnswer: 1,
      explanation:
        'Documentation is critical for reproducibility (others can rebuild features), debugging (understand why a feature matters), production deployment (engineers need to implement features), and regulatory compliance (explain features to auditors). Undocumented features become technical debt.',
    },
  ];
