/**
 * Discussion Questions for Naive Bayes
 */

import { QuizQuestion } from '../../../types';

export const naivebayesQuiz: QuizQuestion[] = [
  {
    id: 'naive-bayes-q1',
    question:
      'Explain the "naive" independence assumption in Naive Bayes. Why is it called naive, why does it often work despite being violated, and when does it fail catastrophically?',
    hint: 'Consider what the assumption means mathematically, how classification differs from probability estimation, and specific scenarios where dependencies matter.',
    sampleAnswer:
      'The naive assumption states that features are conditionally independent given the class: P(x₁,x₂,...,xₙ|y) = ∏P(xᵢ|y). This means knowing one feature provides no information about others once you know the class. It\'s "naive" because this is almost never true in real data - features are typically correlated. Despite this strong violation, Naive Bayes often works because: (1) Classification only requires correct ranking of class probabilities, not accurate absolute probabilities. Even if P(y|x) estimates are wrong, as long as argmax_y P(y|x) is correct, classification succeeds. (2) Errors from independence assumption can cancel across features. (3) Simple models generalize better with limited data. However, it fails catastrophically when: (1) Features are highly redundant (word "free" appears 5 times counted separately amplifies its effect 5x), (2) Feature combinations define classes (XOR-like patterns where x₁ AND x₂ together determine class), (3) Adversarial scenarios where feature dependencies are exploited.',
    keyPoints: [
      'Independence assumption: P(x₁,...,xₙ|y) = ∏P(xᵢ|y)',
      'Violated in practice but works for classification (ranking, not probabilities)',
      'Fails with: redundant features, feature interactions, adversarial cases',
    ],
  },
  {
    id: 'naive-bayes-q2',
    question:
      'Why is Laplace smoothing necessary in Naive Bayes? Explain the zero-frequency problem with a concrete example and discuss how different smoothing values affect model behavior.',
    hint: 'Think about what happens when a feature value never appears with a class in training data.',
    sampleAnswer:
      'Laplace smoothing addresses the zero-frequency problem where a feature value unseen for a class during training causes P(xᵢ|y)=0, making the entire posterior P(y|x)=0 since Naive Bayes multiplies probabilities. Concrete example: Training spam detector with words {"free":spam 10x, "meeting":not-spam 8x}. Test email contains "conference" (never seen). Without smoothing, P("conference"|not-spam)=0, so P(not-spam|email with "conference")=0 regardless of other words! The email is forced into spam class. Laplace smoothing adds α to numerator and αV to denominator: P(xᵢ|y)=(count(xᵢ,y)+α)/(count(y)+αV). With α=1, "conference" gets small non-zero probability. Smoothing values: α=0 (no smoothing, zero-frequency problem), α=1 (standard Laplace), α<1 (less smoothing, trusts data more), α>1 (more smoothing, more uniform). Larger α makes model more conservative, smaller α more aggressive. Tune via cross-validation.',
    keyPoints: [
      'Zero-frequency: unseen feature→P=0→entire posterior=0',
      'Laplace: add α to counts, prevents zeros',
      'α=1 standard, tune via CV',
      'Larger α more conservative, smaller more aggressive',
    ],
  },
  {
    id: 'naive-bayes-q3',
    question:
      'Compare Naive Bayes with Logistic Regression for text classification. When would you prefer each, and what are the computational and performance tradeoffs?',
    hint: 'Consider training/prediction speed, data requirements, assumptions, and feature handling.',
    sampleAnswer:
      'For text classification, both are popular but differ fundamentally. Naive Bayes: Generative model (models P(x|y)), extremely fast training O(n), fast prediction O(features), works with small data, handles high dimensions well, assumes feature independence, provides probability estimates (though often poorly calibrated). Logistic Regression: Discriminative model (models P(y|x) directly), slower training (iterative optimization), still fast prediction, needs more data, can learn feature interactions with polynomial features, better probability calibration. Prefer Naive Bayes when: (1) Need extreme speed (millions of documents), (2) Limited training data (<1000 samples), (3) Baseline/prototype quickly, (4) Features genuinely independent-ish. Prefer Logistic when: (1) Need accurate probabilities, (2) Sufficient training data, (3) Feature interactions matter, (4) Want feature selection (L1 regularization). Computational: NB trains 10-100x faster, both predict fast. Performance: On large balanced text corpora, similar accuracy. NB better with small data, LR better with large data. Hybrid: Use NB for initial filtering (fast), LR for refined classification.',
    keyPoints: [
      'NB: generative, fast, small data, independence assumption',
      'LR: discriminative, slower, more data, learns interactions',
      'NB faster training, similar prediction speed',
      'Choose based on: data size, speed needs, probability accuracy requirements',
    ],
  },
];
