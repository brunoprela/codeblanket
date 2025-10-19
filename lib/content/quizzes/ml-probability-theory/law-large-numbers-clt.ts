/**
 * Quiz questions for Law of Large Numbers & CLT section
 */

export const lawnumberscltQuiz = [
  {
    id: 'q1',
    question:
      'Explain the difference between the Law of Large Numbers and the Central Limit Theorem. What does each tell us, and why are both important for machine learning?',
    hint: 'LLN is about convergence of means, CLT is about distribution shape.',
    sampleAnswer:
      'Law of Large Numbers (LLN): As sample size n increases, the sample mean X̄ converges to the population mean μ. It tells us the sample mean gets closer to the true mean with more data. Central Limit Theorem (CLT): The distribution of the sample mean approaches a normal distribution, regardless of the original distribution\'s shape. It tells us X̄ ~ N(μ, σ²/n) for large n. Key differences: (1) LLN is about convergence (where X̄ goes), CLT is about distribution shape (how X̄ is distributed). (2) LLN says "X̄ → μ", CLT says "X̄ ~ Normal". Both crucial for ML: LLN justifies using training data to estimate expectations - with enough data, empirical mean approximates true mean. CLT enables statistical inference - we can compute confidence intervals, hypothesis tests, and uncertainty estimates because we know sample means are normal. Together: LLN guarantees our estimates improve with data, CLT tells us how uncertainty decreases (σ/√n) and enables probabilistic statements about estimates.',
    keyPoints: [
      'LLN: sample mean converges to population mean',
      'CLT: sample mean distribution becomes normal',
      'LLN about convergence, CLT about distribution shape',
      'LLN justifies empirical estimation',
      'CLT enables confidence intervals and hypothesis tests',
    ],
  },
  {
    id: 'q2',
    question:
      'The CLT states that the sample mean has variance σ²/n, decreasing as 1/n. Explain why this means you need 4× more data to halve the standard error, not 2×.',
    sampleAnswer:
      'Sample mean variance is σ²/n, so standard error (standard deviation of sample mean) is σ/√n. To understand the relationship: if we have n samples with SE = σ/√n, and want half the standard error (SE/2), we need: σ/√n_new = (σ/√n)/2. Solving: √n_new = 2√n, so n_new = 4n. We need 4× the data to halve standard error! This is because SE ∝ 1/√n, not 1/n. The square root relationship means: (1) 4× data → 2× better estimates. (2) 100× data → 10× better estimates. (3) Diminishing returns as n grows. Practical implications for ML: (1) Going from 1k to 4k samples makes notable improvement. (2) Going from 1M to 4M samples makes minor improvement. (3) Collecting more data has diminishing returns. (4) Eventually, better models > more data. (5) This is why pre-training on huge datasets then fine-tuning works - the √n relationship means massive data gives good base, but task-specific data still valuable.',
    keyPoints: [
      'Standard error = σ/√n (square root relationship)',
      '4× data needed to halve standard error',
      'Due to √n, not linear n relationship',
      'Diminishing returns as n increases',
      'Explains why massive pre-training works',
    ],
  },
  {
    id: 'q3',
    question:
      'In stochastic gradient descent, mini-batch gradients are noisy estimates of the true gradient. Explain how the CLT applies here and what this means for choosing batch size.',
    sampleAnswer:
      "A mini-batch gradient is the average of individual sample gradients: ĝ = (1/b)Σᵢ gᵢ where b is batch size. By CLT, this average is approximately normal: ĝ ~ N(g_true, σ²/b) where g_true is the true gradient and σ² is variance of individual gradients. The noise decreases as σ/√b. Implications for batch size: (1) Larger batch (b=256) → less noise (σ/16), more stable but expensive. (2) Smaller batch (b=32) → more noise (σ/√32), less stable but faster iterations. (3) Batch size controls exploration: noise can help escape local minima. (4) Sweet spot typically 32-256: enough averaging for stability, enough noise for exploration. (5) Very large batches (>1000) reduce noise too much, hurt generalization. Trade-off: Small batch = more epochs needed but better exploration. Large batch = fewer epochs but may get stuck. Modern practice: adaptive batch size, starting small for exploration, increasing for final convergence. CLT explains why we can't just use batch=1 (too noisy) or batch=full dataset (too expensive, no exploration).",
    keyPoints: [
      'Mini-batch gradient ~ N(true_gradient, σ²/batch_size)',
      'Noise decreases as σ/√batch_size',
      'Larger batch: less noise, more computation',
      'Smaller batch: more noise, better exploration',
      'Optimal batch size balances stability vs exploration',
    ],
  },
];
