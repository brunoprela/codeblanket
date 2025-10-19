/**
 * Discussion Questions for Ensemble Methods
 */

import { QuizQuestion } from '../../../types';

export const ensemblemethodsQuiz: QuizQuestion[] = [
  {
    id: 'ensemble-methods-q1',
    question:
      'Explain the fundamental principle behind why ensembles work. Why do diverse models perform better when combined than the best single model alone?',
    hint: 'Think about error correlation and the "wisdom of crowds" concept.',
    sampleAnswer:
      "Ensembles work because diverse models make different errors. If models were identical, combining wouldn't help. Key insight: uncorrelated errors cancel out when averaged. Mathematically, if we average M independent models each with variance σ², ensemble variance is σ²/M. In practice, models aren't perfectly independent, so gains are less but still significant. Why diversity matters: a tree might overfit certain regions while logistic regression understates probabilities; an SVM might excel with complex boundaries while Naive Bayes captures global patterns. Their errors occur in different places. Combining via averaging (soft voting) or learning optimal weights (stacking) leverages each model's strengths. This is \"wisdom of crowds\" - many imperfect estimators collectively approach truth. Critical requirement: models must disagree meaningfully. Using 5 identical random forests gains nothing; using tree, linear, and SVM gains significantly. Best ensembles: different algorithm families, different hyperparameters, different feature subsets.",
    keyPoints: [
      'Diverse models make different, uncorrelated errors',
      'Averaging reduces variance: σ²/M for M models',
      'Each model has different strengths/weaknesses',
      'Errors cancel out when combined',
      'Diversity requirement: use different algorithms',
    ],
  },
  {
    id: 'ensemble-methods-q2',
    question:
      'Compare voting and stacking ensembles. When would you use each? What are the tradeoffs?',
    hint: 'Consider simplicity, performance, and computational cost.',
    sampleAnswer:
      "Voting averages predictions from base models (hard: majority vote, soft: average probabilities). Stacking trains meta-model to learn optimal combination weights. Voting: Simple, no additional training, works well with good base models, soft voting typically better than hard. Weights can be set manually if you know relative model strengths. Stacking: More sophisticated, meta-model learns complex combination rules (can be non-linear), usually achieves better performance, but requires CV to avoid overfitting (use out-of-fold predictions for meta-training). Stacking needs more data and computation. Tradeoffs: Use voting for production (simple, robust, fast), use stacking for competitions/maximum accuracy. Stacking shines when base models have very different characteristics (tree+linear+SVM). Risk with stacking: overfitting if not properly CV'd. Practical tip: Start with soft voting; if you need extra 0.5-1% accuracy and have enough data, try stacking. The gain from stacking over voting is often modest (1-3%) but can be decisive in competitions.",
    keyPoints: [
      'Voting: simple averaging or majority vote',
      'Stacking: meta-model learns optimal combination',
      'Voting: simpler, faster, robust',
      'Stacking: better performance, needs CV',
      'Stacking gain: 1-3% over voting',
    ],
  },
  {
    id: 'ensemble-methods-q3',
    question:
      'How do you ensure proper cross-validation in stacking to avoid overfitting? What is the purpose of using out-of-fold predictions?',
    hint: "Think about data leakage and why you can't use training predictions directly.",
    sampleAnswer:
      "Direct approach (wrong): Train base models on full training set, get their predictions, train meta-model on these predictions. Problem: data leakage! Base models saw this data during training, so predictions are overly optimistic. Meta-model learns on inflated scores, overfits terribly. Correct approach (out-of-fold predictions): Use k-fold CV. For each fold: (1) Train base models on other k-1 folds; (2) Predict on held-out fold; (3) Collect these OOF predictions. Result: meta-training data where base models never saw the prediction data during training - unbiased! Then train base models on full training data for test predictions. This ensures meta-model sees realistic base model performance, not inflated training performance. Implementation: scikit-learn's StackingClassifier handles this automatically with cv parameter. Manual: use cross_val_predict. Why critical: without OOF, stacking overfits dramatically - train accuracy 99%, test accuracy 70%. With OOF, stacking generalizes properly and genuinely improves over base models.",
    keyPoints: [
      'Direct training predictions cause overfitting',
      'Out-of-fold predictions prevent data leakage',
      'K-fold CV: train on k-1 folds, predict on held-out',
      'Meta-model sees unbiased base predictions',
      'StackingClassifier handles OOF automatically',
    ],
  },
];
