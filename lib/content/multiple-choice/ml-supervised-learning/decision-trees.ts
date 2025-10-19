/**
 * Multiple Choice Questions for Decision Trees
 */

import { MultipleChoiceQuestion } from '../../../types';

export const decisiontreesMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'decision-trees-mc-1',
    question: 'What does a Gini impurity of 0 indicate for a node?',
    options: [
      'The node is completely impure (50-50 class split)',
      'The node is pure (all samples belong to one class)',
      'The node should not be split further',
      'The tree is overfitting',
    ],
    correctAnswer: 1,
    explanation:
      'Gini impurity of 0 means the node is completely pure - all samples belong to the same class. This is the stopping criterion for tree growth at that branch. Gini = 0.5 (for binary) indicates maximum impurity.',
    difficulty: 'easy',
  },
  {
    id: 'decision-trees-mc-2',
    question:
      'Which parameter directly controls the maximum depth of a decision tree?',
    options: [
      'min_samples_split',
      'min_samples_leaf',
      'max_depth',
      'criterion',
    ],
    correctAnswer: 2,
    explanation:
      'max_depth directly limits how deep the tree can grow. Setting max_depth=3 means the tree will have at most 3 levels of decisions. This is the most straightforward way to control tree complexity and prevent overfitting.',
    difficulty: 'easy',
  },
  {
    id: 'decision-trees-mc-3',
    question:
      'Why might a decision tree perform poorly on a dataset with strong linear relationships?',
    options: [
      'Trees cannot handle numerical features',
      'Trees create axis-aligned splits, requiring many splits to approximate diagonal lines',
      'Trees always overfit linear data',
      'Trees require feature scaling for linear relationships',
    ],
    correctAnswer: 1,
    explanation:
      'Decision trees make axis-aligned (parallel to axes) splits. To approximate a diagonal decision boundary (like y=x), they need many small rectangular splits, creating a staircase pattern. This is inefficient compared to linear models which directly learn diagonal boundaries. Trees excel at non-linear, interaction-rich relationships.',
    difficulty: 'hard',
  },
  {
    id: 'decision-trees-mc-4',
    question:
      'What is the key advantage of decision trees over logistic regression for interpretability?',
    options: [
      'Trees have fewer parameters',
      'Trees can be visualized as flowcharts showing exact decision logic',
      'Trees are always more accurate',
      'Trees require less training data',
    ],
    correctAnswer: 1,
    explanation:
      'Decision trees can be visualized as flowcharts showing the exact sequence of if-then-else decisions, making them extremely interpretable. You can trace any prediction through the tree. Logistic regression coefficients are interpretable but less intuitive than a visual flowchart.',
    difficulty: 'medium',
  },
  {
    id: 'decision-trees-mc-5',
    question: 'What happens to a decision tree as max_depth increases?',
    options: [
      'Training error increases, test error decreases',
      'Both training and test error decrease',
      'Training error decreases, test error eventually increases (overfitting)',
      'Nothing changes after a certain depth',
    ],
    correctAnswer: 2,
    explanation:
      'As max_depth increases, training error decreases (tree fits training data better). However, test error decreases initially then increases as the tree becomes too complex and overfits. This is the classic bias-variance tradeoff - deeper trees have lower bias but higher variance.',
    difficulty: 'medium',
  },
];
