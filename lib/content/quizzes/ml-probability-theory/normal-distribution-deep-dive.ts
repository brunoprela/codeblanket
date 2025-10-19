/**
 * Quiz questions for Normal Distribution Deep Dive section
 */

export const normaldistributiondeepdiveQuiz = [
  {
    id: 'q1',
    question:
      'Explain why the normal distribution appears so frequently in machine learning and nature. What is the Central Limit Theorem and how does it relate to this ubiquity?',
    hint: 'Think about sums of random variables.',
    sampleAnswer:
      "The normal distribution appears everywhere because of the Central Limit Theorem (CLT): when you sum many independent random variables (regardless of their individual distributions), the sum approaches a normal distribution as the number of variables increases. CLT states: if X₁,...,Xₙ are i.i.d. with mean μ and finite variance σ², then (X₁+...+Xₙ - nμ)/(σ√n) → N(0,1) as n→∞. This explains ubiquity: (1) Measurement errors are sums of many small independent factors → normal. (2) Heights are determined by many genes → normal. (3) ML gradients in SGD are averages of mini-batch gradients → approximately normal. (4) Activations in deep networks are sums of weighted inputs → tend toward normal. (5) Any process influenced by many independent factors tends normal. Additionally: (1) Maximum entropy distribution given mean and variance. (2) Mathematically tractable - easy to work with analytically. (3) Fully specified by two parameters. The CLT is why we can assume normality in many contexts even when individual components aren't normal.",
    keyPoints: [
      'Central Limit Theorem: sums of RVs → normal',
      'Applies regardless of individual distributions',
      'Explains appearance in measurement errors, biology, physics',
      'ML gradients, activations often approximately normal',
      'Maximum entropy distribution for given mean/variance',
    ],
  },
  {
    id: 'q2',
    question:
      'When adding two independent normal random variables X₁ ~ N(μ₁, σ₁²) and X₂ ~ N(μ₂, σ₂²), explain why variances add but standard deviations do NOT simply add. Provide intuition and a concrete example.',
    sampleAnswer:
      "Variances add: Var(X₁ + X₂) = σ₁² + σ₂², but σ₁ + σ₂ ≠ √(σ₁² + σ₂²). Intuition: Variance measures average squared deviation. When summing independent variables, squared deviations add linearly. But standard deviation is the square root, and √(a²+b²) ≠ a+b due to non-linearity. Example: X₁ ~ N(0, 9) so σ₁=3. X₂ ~ N(0, 16) so σ₂=4. If std devs added: σ₁+σ₂ = 3+4 = 7. But actually: Var(X₁+X₂) = 9+16 = 25, so σ = √25 = 5, not 7! Why? Independence means deviations don't always align. Sometimes X₁ is high when X₂ is low (canceling). Average squared deviation grows slower than if deviations always aligned. This is Pythagorean theorem: √(3²+4²) = √(9+16) = 5. Practical implication: Adding noise sources increases variance less than you might think. In ML: Total uncertainty from independent sources combines via variance addition, making ensemble methods effective - errors partially cancel.",
    keyPoints: [
      'Variances add for independent normals: σ₁² + σ₂²',
      'Standard deviations do NOT add: σ ≠ σ₁ + σ₂',
      'Due to square root non-linearity',
      'Analogous to Pythagorean theorem',
      'Deviations partially cancel when independent',
    ],
  },
  {
    id: 'q3',
    question:
      "The 68-95-99.7 rule is often used for outlier detection. Explain the rule, how it's used to detect outliers, and discuss potential pitfalls when the data isn't actually normally distributed.",
    sampleAnswer:
      'The 68-95-99.7 rule (empirical rule): For N(μ,σ²), approximately 68% of data within μ±σ, 95% within μ±2σ, 99.7% within μ±3σ. Outlier detection: Compute Z-score = (x-μ)/σ. If |Z| > 3, only ~0.3% of normal data would be this extreme, so flag as outlier. If |Z| > 2, ~5% would be this extreme (less strong evidence). Pitfalls when data isn\'t normal: (1) Heavy-tailed distributions (t, Cauchy) have more extremes than normal - you\'d flag too many "outliers" that are actually typical. (2) Skewed distributions - one tail has more extremes than the other, asymmetric outlier detection needed. (3) Multimodal distributions - "outliers" might be from a second mode, not errors. (4) Small samples - estimated μ, σ are imprecise, many false positives. Better approaches: (1) Use robust statistics (median, MAD instead of mean, std). (2) Visual inspection (Q-Q plots to check normality). (3) Use distribution-free methods (IQR rule: outliers outside Q1-1.5×IQR to Q3+1.5×IQR). (4) Domain knowledge. Always check normality assumption before blindly applying empirical rule!',
    keyPoints: [
      '68-95-99.7: probability within 1, 2, 3 standard deviations',
      'Outlier detection: |Z| > 3 suggests outlier',
      'Only valid if data is actually normal!',
      'Heavy tails, skewness violate assumptions',
      'Use Q-Q plots and robust methods to verify',
    ],
  },
];
