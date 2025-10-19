/**
 * Quiz questions for Expectation & Variance section
 */

export const expectationvarianceQuiz = [
  {
    id: 'q1',
    question:
      'Explain why E[X+Y] = E[X] + E[Y] holds even when X and Y are dependent, but Var(X+Y) = Var(X) + Var(Y) only holds when X and Y are independent.',
    hint: 'Think about how covariance affects variance but not expectation.',
    sampleAnswer:
      "Expectation is linear: E[X+Y] = E[X] + E[Y] ALWAYS holds, even for dependent variables. This is because expectation of a sum equals sum of expectations by definition. No interaction terms appear. However, variance is NOT linear. Var(X+Y) = Var(X) + Var(Y) + 2Cov(X,Y). The covariance term captures dependency: if X and Y tend to deviate together (positive covariance), sum variance increases. If they deviate oppositely (negative covariance), sum variance decreases. Only when independent does Cov(X,Y)=0, giving Var(X+Y) = Var(X) + Var(Y). Intuition: Expectation cares about average behavior - dependencies average out. Variance cares about spread - dependencies affect how deviations combine. If X is high, does Y tend to be high too (amplifying spread) or low (canceling spread)? This matters for variance but not mean. In ML: Ensemble methods exploit this - combining correlated models doesn't help much (high covariance), but combining independent/negatively correlated models reduces variance effectively.",
    keyPoints: [
      'E[X+Y] = E[X] + E[Y] always (linearity of expectation)',
      'Var(X+Y) = Var(X) + Var(Y) + 2Cov(X,Y) (includes covariance)',
      'Variance adds only when independent (Cov=0)',
      'Dependencies affect spread but not center',
      'Important for ensemble methods in ML',
    ],
  },
  {
    id: 'q2',
    question:
      'When you scale a random variable by constant a: Y = aX, how do the mean and variance change? Why does variance scale by a² instead of a?',
    sampleAnswer:
      'Mean scales linearly: E[aX] = aE[X]. Variance scales quadratically: Var(aX) = a²Var(X), thus SD scales linearly: SD(aX) = |a|SD(X). Why a² for variance? Variance is E[(X-μ)²], measuring squared deviations. When we scale by a: Var(aX) = E[(aX-aμ)²] = E[a²(X-μ)²] = a²E[(X-μ)²] = a²Var(X). The squaring in variance definition causes a to be squared. Intuition: If you double all values (a=2), deviations double, but squared deviations quadruple. Practical implications: (1) When standardizing features (dividing by σ), variance becomes 1 (since 1/σ² × σ² = 1). (2) When changing units (meters to centimeters, ×100), variance multiplies by 10,000 but SD by 100. (3) In neural networks, weight scaling affects activation variance quadratically - crucial for initialization. This is why He initialization uses √(2/n) not 2/n - we want Var = 2/n, requiring std = √(2/n).',
    keyPoints: [
      'E[aX] = aE[X]: mean scales linearly',
      'Var(aX) = a²Var(X): variance scales quadratically',
      'Due to squaring in variance definition',
      'Standard deviation scales linearly: |a|σ',
      'Critical for feature scaling and weight initialization',
    ],
  },
  {
    id: 'q3',
    question:
      'In machine learning, loss functions are expectations over the data distribution. Explain what this means and why we use empirical risk minimization.',
    sampleAnswer:
      "A loss function L(y,ŷ) measures error for a single prediction. The true risk (expected loss) is R = E_(x,y)~P[L(y,ŷ)], the expectation over the true data distribution P(x,y). This is what we actually want to minimize. However, we don't know P(x,y) - we only have training samples. Empirical Risk Minimization (ERM) replaces true expectation with sample average: R̂ = (1/n)Σᵢ L(yᵢ,ŷᵢ). By Law of Large Numbers, as n→∞, R̂→R. We minimize R̂ hoping to minimize R. Challenges: (1) Finite data - R̂ ≠ R, causing generalization gap. (2) Overfitting - model minimizes R̂ but not R. (3) Distribution shift - training distribution ≠ test distribution. Solutions: (1) Regularization - prefer simpler models. (2) Cross-validation - estimate true R from held-out data. (3) More data - improve R̂ estimate. This expectation view explains why: validation loss matters more than training loss (better estimates true risk), we need i.i.d. data (to estimate expectation correctly), and large batch sizes help (reduce gradient noise = better risk estimate).",
    keyPoints: [
      'True risk: R = E[L(y,ŷ)] over data distribution',
      "Don't know true distribution, only have samples",
      'Empirical risk: R̂ = (1/n)Σ L(yᵢ,ŷᵢ) from training data',
      'Law of Large Numbers: R̂ → R as n → ∞',
      'Generalization gap when R̂ ≠ R',
    ],
  },
];
