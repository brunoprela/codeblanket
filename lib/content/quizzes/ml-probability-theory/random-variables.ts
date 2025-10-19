/**
 * Quiz questions for Random Variables section
 */

export const randomvariablesQuiz = [
  {
    id: 'q1',
    question:
      'Explain the fundamental difference between a random variable and a regular variable. Why is this distinction critical in machine learning?',
    hint: 'Think about determinism vs randomness.',
    sampleAnswer:
      'A regular variable has a fixed, deterministic value (x = 5). A random variable is a function that maps outcomes to numbers, and its value is determined by a random process (X = die roll result). The distinction is critical in ML because: (1) Data is random - we never know the exact values we\'ll see during training or testing. (2) Model outputs are random variables - predictions have uncertainty. (3) Training involves randomness - mini-batch sampling, initialization, dropout. (4) Losses are random variables - each batch gives different loss values. (5) Gradients are random variables in SGD. Without understanding random variables, we can\'t reason about uncertainty, generalization, or why we need validation sets. Every "value" in ML should be thought of as a realization of a random variable, not a fixed number. This probabilistic view is fundamental to understanding why ML works and its limitations.',
    keyPoints: [
      'Random variable: function mapping random outcomes to numbers',
      'Regular variable: fixed, deterministic value',
      'ML data, predictions, losses, gradients are all RVs',
      'Random variables have distributions, not single values',
      'Essential for reasoning about uncertainty',
    ],
  },
  {
    id: 'q2',
    question:
      'For continuous random variables, why is P(X = exact value) = 0? How do we calculate probabilities if every exact value has probability zero?',
    sampleAnswer:
      "For continuous RVs with uncountably many possible values, the probability of any exact value is zero. This is because probability must be distributed over infinitely many points - dividing any finite probability by infinity gives zero. Intuitively: if we measure height to infinite precision, the probability of exactly 170.000... cm (infinitely many decimal places) is zero. We calculate probabilities using intervals instead: P(a ≤ X ≤ b) = ∫ᵃᵇ f(x)dx, which is the area under the PDF curve. The PDF f(x) itself is NOT a probability - it's a density. It can even be > 1! The probability is the integral (area), not the function value (height). This is why we use ≤ vs < interchangeably for continuous RVs: P(X ≤ a) = P(X < a) because P(X = a) = 0. In ML, this means: don't use exact equality tests on continuous values (floating point), use intervals or epsilon-tolerance.",
    keyPoints: [
      'Continuous RV: uncountably many values',
      'P(X = exact value) = 0 for continuous RVs',
      'Calculate probabilities using intervals: P(a ≤ X ≤ b)',
      'PDF is density, not probability (can be > 1)',
      'Probability = area under PDF curve',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain the relationship between PDF, CDF, and their derivatives/integrals. Why is the CDF useful even though the PDF seems more intuitive?',
    sampleAnswer:
      'PDF and CDF are related through calculus: CDF F(x) is the integral of PDF: F(x) = ∫₋∞ˣ f(t)dt. Conversely, PDF is the derivative of CDF: f(x) = dF(x)/dx. The CDF F(x) = P(X ≤ x) is the cumulative probability up to x. The PDF f(x) is the rate of change of cumulative probability - how "dense" probability is at x. CDF advantages: (1) Works for both discrete and continuous RVs (PDF only for continuous). (2) Directly gives probabilities: P(X ≤ x) = F(x). (3) Easy to compute intervals: P(a ≤ X ≤ b) = F(b) - F(a). (4) CDF always exists, even when PDF doesn\'t (e.g., discrete RVs). (5) Quantile functions are inverse CDFs: x = F⁻¹(p) gives value with p cumulative probability. In ML: CDFs used for calibration curves, ROC curves (essentially CDFs), and percentile calculations. While PDFs are more intuitive for visualization, CDFs are mathematically cleaner and more general.',
    keyPoints: [
      'CDF F(x) = ∫ f(t)dt (integral of PDF)',
      'PDF f(x) = dF(x)/dx (derivative of CDF)',
      'CDF works for discrete and continuous RVs',
      'CDF directly gives P(X ≤ x)',
      'Use CDF for probability calculations',
    ],
  },
];
