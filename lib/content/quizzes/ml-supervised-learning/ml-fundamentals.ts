/**
 * Discussion Questions for Machine Learning Fundamentals
 */

import { QuizQuestion } from '../../../types';

export const mlfundamentalsQuiz: QuizQuestion[] = [
  {
    id: 'ml-fundamentals-q1',
    question:
      'Compare and contrast supervised learning and unsupervised learning. Provide specific examples of when you would use each approach in a real-world business scenario. What are the main challenges associated with each type?',
    hint: 'Think about the role of labeled data, the types of insights each provides, and practical constraints like cost and time.',
    sampleAnswer:
      "Supervised learning and unsupervised learning differ fundamentally in their use of labeled data. Supervised learning requires labeled training data where we know the correct outputs, making it ideal for prediction tasks like credit card fraud detection (labeled as fraud/not fraud) or customer churn prediction (labeled as churned/retained). The algorithm learns patterns that map inputs to known outputs and can then predict outcomes for new data. However, supervised learning faces challenges including the cost and time of obtaining labeled data, potential label noise or errors, and the need for labels that accurately represent the prediction target.\n\nUnsupervised learning, by contrast, works with unlabeled data to discover hidden patterns and structures. It's particularly valuable when labeling is expensive or impractical, such as customer segmentation for marketing campaigns where we want to discover natural groupings without predefined categories. Other examples include anomaly detection in network traffic or exploratory data analysis to understand data structure. The main challenges include difficulty in evaluating results (no ground truth), interpreting discovered patterns, and ensuring the patterns are actionable for business needs.\n\nIn practice, many real-world systems use both approaches. For example, an e-commerce company might use unsupervised learning to segment customers into groups, then use supervised learning to predict which products each segment will purchase. The choice depends on data availability, business objectives, and the specific insights needed.",
    keyPoints: [
      'Supervised learning requires labeled data and is used for prediction tasks with known outcomes',
      'Unsupervised learning discovers patterns in unlabeled data and is used for exploration and segmentation',
      'Supervised learning challenges: labeling cost, label quality, representative samples',
      'Unsupervised learning challenges: evaluation difficulty, interpretation, actionability',
      'Real-world applications often combine both approaches for comprehensive solutions',
    ],
  },
  {
    id: 'ml-fundamentals-q2',
    question:
      'Explain the bias-variance tradeoff in machine learning. How does this tradeoff relate to overfitting and underfitting? Provide a practical example of how you would detect and address high bias vs. high variance in a model.',
    hint: 'Consider how model complexity affects both training and test performance, and what practical steps can diagnose and fix each issue.',
    sampleAnswer:
      "The bias-variance tradeoff is a fundamental concept that explains the relationship between model complexity and prediction error. Bias refers to error from overly simplistic assumptions in the learning algorithm - high bias models fail to capture important patterns (underfitting). Variance refers to error from excessive sensitivity to fluctuations in the training data - high variance models learn noise rather than signal (overfitting). Total error = Bias² + Variance + Irreducible Error, and there's an inherent tradeoff: reducing bias often increases variance and vice versa.\n\nHigh bias (underfitting) manifests as poor performance on both training and test sets. For example, using linear regression to model a clearly non-linear relationship between features and target. To detect: training error is high, and training/test errors are similar. Solutions include using a more complex model architecture, adding relevant features, reducing regularization strength, or training for more epochs.\n\nHigh variance (overfitting) appears as excellent training performance but poor test performance. For instance, a deep neural network with millions of parameters trained on only 100 samples might memorize the training data. Detection: low training error but high test error, with a large gap between them. Solutions include collecting more training data, using regularization techniques (L1/L2, dropout), reducing model complexity, using cross-validation for hyperparameter tuning, implementing early stopping, or applying data augmentation.\n\nA practical approach to diagnosis: train models with increasing complexity while monitoring both training and validation errors. Plot learning curves showing error vs. model complexity. The sweet spot occurs where validation error is minimized - this balances bias and variance optimally for your specific problem and dataset.",
    keyPoints: [
      'Bias: error from wrong assumptions, leads to underfitting, affects all datasets',
      'Variance: error from sensitivity to training data, leads to overfitting, specific to training set',
      'High bias diagnosis: poor train and test performance, errors are similar',
      'High variance diagnosis: excellent train but poor test performance, large error gap',
      'Solutions differ: high bias needs more complexity, high variance needs regularization or more data',
      'Learning curves help visualize and diagnose the tradeoff',
    ],
  },
  {
    id: 'ml-fundamentals-q3',
    question:
      'Why is proper data splitting (train/validation/test) crucial in machine learning? Explain what can go wrong if data splitting is done incorrectly. How would you handle data splitting for time series data, and why is it different from random splitting?',
    hint: 'Think about data leakage, evaluation bias, and the temporal nature of time series data.',
    sampleAnswer:
      "Proper data splitting is fundamental to building reliable machine learning models because it ensures honest evaluation of model performance on unseen data. The training set teaches the model patterns, the validation set helps tune hyperparameters and select model architectures, and the test set provides a final, unbiased estimate of real-world performance. This separation is crucial because models naturally perform better on data they've seen during training, and we need to know how they'll perform in production on new data.\n\nIncorrect data splitting can lead to several serious problems. Data leakage occurs when information from the test set influences the training process - for example, scaling features using statistics from the entire dataset before splitting means the training set \"knows\" about test set distributions. This inflates performance metrics and leads to models that fail in production. Another issue is using the test set multiple times for model selection, which effectively turns it into a validation set and loses the unbiased performance estimate. Temporal leakage in time series is particularly problematic - if future information leaks into training data, the model appears to work well but fails completely on real future predictions.\n\nTime series data requires special handling because observations have temporal dependencies and autocorrelation. Random splitting violates the temporal structure - a model shouldn't be trained on 2023 data and tested on 2022 data, as this represents an impossible scenario. Instead, use time-based splitting: train on earlier periods, validate on a middle period, and test on the most recent period. This mimics production use where we predict the future based on the past. Walk-forward validation is even better: train on months 1-10, validate on month 11, test on month 12; then train on months 1-11, validate on month 12, test on month 13, etc. This provides multiple evaluation points while respecting temporal ordering.\n\nFor financial trading models, this is especially critical. A model that \"predicts\" yesterday's stock prices using tomorrow's news would show perfect accuracy but be completely useless. Only time-aware splitting reveals true predictive power.",
    keyPoints: [
      'Training set: learn patterns; Validation set: tune hyperparameters; Test set: final evaluation',
      'Data leakage from improper splitting inflates performance metrics and causes production failures',
      'Common leakage: preprocessing before splitting, feature selection on full dataset, using test set for model selection',
      'Time series requires temporal splitting: train on past, test on future, never random splitting',
      'Walk-forward validation provides robust evaluation for sequential data',
      'Financial applications especially vulnerable to temporal leakage (lookahead bias)',
    ],
  },
];
