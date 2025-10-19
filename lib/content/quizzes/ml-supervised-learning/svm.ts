/**
 * Discussion Questions for Support Vector Machines
 */

import { QuizQuestion } from '../../../types';

export const svmQuiz: QuizQuestion[] = [
  {
    id: 'svm-q1',
    question:
      'Explain the concept of maximum margin in SVM. Why does maximizing the margin lead to better generalization? What are support vectors and why are they the only points that matter?',
    hint: 'Consider what the margin represents geometrically and how it relates to model robustness.',
    sampleAnswer:
      'The maximum margin is the distance between the decision boundary (hyperplane) and the nearest data points from each class. SVM finds the hyperplane that maximizes this margin. Geometrically, a larger margin means the decision boundary is farther from training points, creating a "buffer zone" that makes the classifier more robust to small perturbations and noise. Support vectors are the data points closest to the decision boundary - they lie exactly on the margin boundaries. These points are critical because: (1) They alone determine the optimal hyperplane position; (2) Removing non-support vectors doesn\'t change the decision boundary; (3) The model stores only support vectors, making it memory efficient. This is why SVM is robust - the decision depends only on the most difficult-to-classify points near the boundary, not on points far from it.',
    keyPoints: [
      'Maximum margin: distance from hyperplane to nearest points',
      'Larger margin → better generalization and robustness',
      'Support vectors: points on margin boundaries',
      'Only support vectors determine hyperplane',
      'Memory efficient: stores only support vectors',
    ],
  },
  {
    id: 'svm-q2',
    question:
      'Explain the kernel trick in SVM. How does it allow non-linear classification without explicitly computing high-dimensional transformations? Compare linear, polynomial, and RBF kernels.',
    hint: 'Think about the computational advantage of kernels and when each type is appropriate.',
    sampleAnswer:
      "The kernel trick allows SVM to learn non-linear decision boundaries efficiently. Instead of explicitly mapping data to high-dimensional space (φ(x)) which is computationally expensive, kernels compute inner products directly: K(x,x')=φ(x)·φ(x'). This is powerful because SVM only needs inner products, never explicit coordinates. Linear kernel K(x,x')=x·x' creates linear boundaries (like standard SVM). Polynomial kernel K(x,x')=(γx·x'+r)^d creates polynomial boundaries of degree d; good for moderately non-linear problems, but can overfit with high degree. RBF (Gaussian) kernel K(x,x')=exp(-γ||x-x'||²) maps to infinite-dimensional space, creating very flexible boundaries; γ controls locality (small γ=smooth, large γ=wiggly). RBF is most popular for general non-linear problems. Choose based on: linear when data linearly separable, polynomial for moderate non-linearity with known degree, RBF as default for complex patterns.",
    keyPoints: [
      'Kernel trick: compute inner products without explicit transformation',
      "Linear: x·x' for linear boundaries",
      "Polynomial: (γx·x'+r)^d for polynomial boundaries",
      "RBF: exp(-γ||x-x'||²) for flexible boundaries",
      'RBF most versatile but needs tuning',
    ],
  },
  {
    id: 'svm-q3',
    question:
      'Discuss the role of the C parameter in SVM. How does it affect the bias-variance tradeoff? Provide guidance on tuning C for different scenarios.',
    hint: 'Consider what happens with very small vs. very large C values.',
    sampleAnswer:
      'The C parameter controls the tradeoff between maximizing margin and minimizing training errors. Small C (e.g., 0.1) prioritizes large margin, allowing more misclassifications (soft margin). This creates simpler, smoother boundaries but may underfit. Large C (e.g., 1000) heavily penalizes misclassifications, forcing the model to classify training points correctly even if it means smaller margin. This can lead to complex, wiggly boundaries and overfitting. C directly impacts bias-variance: small C=high bias/low variance (underfitting), large C=low bias/high variance (overfitting). Tuning guidance: Start with C=1.0 as baseline. If training error is high, increase C. If large gap between train/test accuracy (overfitting), decrease C. Use cross-validation with C=[0.01, 0.1, 1, 10, 100]. For noisy data, prefer smaller C for robustness. For clean, well-separated data, larger C works well. Always tune C jointly with kernel parameters (gamma for RBF).',
    keyPoints: [
      'C controls margin vs. error tradeoff',
      'Small C: large margin, soft, may underfit',
      'Large C: small margin, hard, may overfit',
      'Start C=1.0, tune via cross-validation',
      'Noisy data→small C, clean data→large C',
    ],
  },
];
