/**
 * Discussion and Multiple Choice Questions for Decision Trees
 */

import { QuizQuestion } from '../../../types';

export const decisiontreesQuiz: QuizQuestion[] = [
  {
    id: 'decision-trees-q1',
    question:
      'Explain why decision trees are prone to overfitting and describe the techniques used to prevent it. Compare pre-pruning and post-pruning approaches.',
    hint: 'Think about what happens when trees grow too deep and how pruning controls complexity.',
    sampleAnswer:
      "Decision trees overfit because they can grow arbitrarily deep, creating a leaf for every training point if unconstrained. This memorizes training data including noise, failing to generalize. Pre-pruning (early stopping) limits growth during construction using max_depth, min_samples_split, or min_samples_leaf. Advantage: computationally efficient, prevents building useless branches. Disadvantage: may stop too early, missing useful splits. Post-pruning builds full tree then removes branches that don't improve validation performance, typically using cost-complexity pruning. Advantage: more accurate as it sees full picture. Disadvantage: computationally expensive (build full tree first). In practice, pre-pruning is more common in scikit-learn. Best approach: use cross-validation to tune pruning parameters.",
    keyPoints: [
      'Overfitting: deep trees memorize training data',
      'Pre-pruning: limit growth (max_depth, min_samples_split)',
      'Post-pruning: build full tree, remove branches',
      'Pre-pruning efficient but may stop early',
      'Tune pruning via cross-validation',
    ],
  },
  {
    id: 'decision-trees-q2',
    question:
      'Compare Gini impurity and entropy as splitting criteria. Do they produce significantly different trees? When might one be preferred over the other?',
    hint: 'Consider the mathematical differences and practical implications.',
    sampleAnswer:
      "Gini impurity and entropy both measure node homogeneity but differ mathematically. Gini: 1-Σpᵢ² (quadratic), entropy: -Σpᵢlog (pᵢ) (logarithmic). Entropy penalizes impurity more severely due to logarithm, while Gini is computationally faster (no log). In practice, they usually produce similar trees - differences are minor for most datasets. Entropy may create slightly more balanced trees. Gini is preferred in scikit-learn default due to speed (no log computation). Choose entropy when you need information-theoretic interpretation or slightly better balance. Choose Gini for faster training. Honestly, the choice matters far less than proper pruning and ensemble methods (Random Forests). Don't overthink this - use default Gini unless you have specific reason for entropy.",
    keyPoints: [
      'Gini: 1-Σpᵢ², entropy: -Σpᵢlog (pᵢ)',
      'Both measure homogeneity, usually similar results',
      'Gini faster (no logarithm)',
      'Entropy slightly more balanced',
      'Choice matters less than pruning',
    ],
  },
  {
    id: 'decision-trees-q3',
    question:
      "Why do decision trees handle mixed data types and missing values better than many other algorithms? What about feature scaling - why isn't it needed?",
    hint: 'Think about how trees make decisions and what information they use.',
    sampleAnswer:
      'Decision trees naturally handle mixed types because they make threshold-based splits, not distance calculations. For categorical features, they ask "is category in set S?" For numerical, "is value > threshold?" This doesn\'t require features to be on same scale or even numeric. Missing values can be handled by treating "missing" as its own category or using surrogate splits (backup splits using other features). Feature scaling isn\'t needed because trees only care about relative ordering within each feature, not absolute magnitudes. A split "age > 30" works identically whether age is in years, months, or standardized. This contrasts with distance-based methods (kNN, SVM) where scales directly affect calculations. However, trees still bias toward features with more unique values (more split opportunities), so feature engineering matters. The scale-invariance is why trees are popular for heterogeneous real-world data like customer databases mixing categorical IDs, numerical values, and missing data.',
    keyPoints: [
      'Threshold-based splits, not distance calculations',
      'Categorical: membership tests, Numerical: threshold comparisons',
      'Missing values: treat as category or surrogate splits',
      'Scale-invariant: only relative ordering matters',
      'Bias toward features with more unique values',
    ],
  },
];

export const decisiontreesMultipleChoice = [
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
