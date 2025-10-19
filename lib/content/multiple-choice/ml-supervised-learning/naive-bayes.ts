/**
 * Multiple Choice Questions for Naive Bayes
 */

import { MultipleChoiceQuestion } from '../../../types';

export const naivebayesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'naive-bayes-mc-1',
    question: 'What does the "naive" assumption in Naive Bayes mean?',
    options: [
      'Features are normally distributed',
      'Features are conditionally independent given the class',
      'The model is simple and easy to implement',
      'All classes are equally likely',
    ],
    correctAnswer: 1,
    explanation:
      'The naive assumption is that features are conditionally independent given the class label. This means P(x₁,x₂,...|y) = P(x₁|y)×P(x₂|y)×.... This assumption is rarely true in practice but allows efficient computation and often works well for classification despite being violated.',
    difficulty: 'medium',
  },
  {
    id: 'naive-bayes-mc-2',
    question:
      'In text classification, you encounter a word in the test set that never appeared in the training set. What problem does this cause without Laplace smoothing?',
    options: [
      'The model runs slowly',
      'The posterior probability becomes exactly 0 for all classes',
      'The model cannot make a prediction',
      'The posterior probability becomes exactly 0 for the class where the word was unseen',
    ],
    correctAnswer: 3,
    explanation:
      'Without Laplace smoothing, the probability P(word|class) = 0 for any word never seen with that class during training. Since Naive Bayes multiplies probabilities (P(x₁|y)×P(x₂|y)×...), a single zero makes the entire posterior probability 0 for that class, regardless of other features. Laplace smoothing adds a small count to prevent this.',
    difficulty: 'hard',
  },
  {
    id: 'naive-bayes-mc-3',
    question:
      'Which variant of Naive Bayes should you use for text classification with word counts?',
    options: [
      'Gaussian Naive Bayes',
      'Multinomial Naive Bayes',
      'Bernoulli Naive Bayes',
      'Linear Naive Bayes',
    ],
    correctAnswer: 1,
    explanation:
      'Multinomial Naive Bayes is designed for count/frequency data, making it ideal for text classification with word counts or TF-IDF features. Gaussian NB assumes continuous normally-distributed features. Bernoulli NB is for binary features (word present/absent). There is no "Linear Naive Bayes."',
    difficulty: 'easy',
  },
  {
    id: 'naive-bayes-mc-4',
    question:
      'Why does Naive Bayes often perform well despite the independence assumption being violated?',
    options: [
      'The assumption is actually true most of the time',
      'Classification only requires correct ranking of probabilities, not accurate probability values',
      "Feature dependence doesn't affect machine learning",
      'Laplace smoothing corrects for violations',
    ],
    correctAnswer: 1,
    explanation:
      'Naive Bayes works despite violated assumptions because classification only needs to rank classes correctly (argmax P(y|x)), not estimate exact probabilities. Even if absolute probability values are wrong due to independence violations, as long as the relative ordering is correct, classification succeeds. The model is robust to this because errors can cancel across features.',
    difficulty: 'hard',
  },
  {
    id: 'naive-bayes-mc-5',
    question: 'What is the main computational advantage of Naive Bayes?',
    options: [
      'It requires no training',
      'It has very fast training and prediction',
      'It uses less memory than other algorithms',
      'It handles missing data automatically',
    ],
    correctAnswer: 1,
    explanation:
      'Naive Bayes is extremely fast for both training and prediction. Training is O(nd) where it just computes mean/variance or counts for each feature-class combination. Prediction is also very fast as it multiplies pre-computed probabilities. This makes it ideal for real-time applications and large-scale text classification.',
    difficulty: 'easy',
  },
];
