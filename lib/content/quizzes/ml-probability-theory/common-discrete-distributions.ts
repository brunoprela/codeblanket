/**
 * Quiz questions for Common Discrete Distributions section
 */

export const commondiscretedistributionsQuiz = [
  {
    id: 'q1',
    question:
      'Compare and contrast the Bernoulli and Binomial distributions. How are they related, and when would you use each in machine learning?',
    hint: 'Think about single trial vs multiple trials.',
    sampleAnswer:
      'Bernoulli models a single binary trial (one coin flip, one prediction). Binomial models the sum of n independent Bernoulli trials (total heads in 10 flips, total correct predictions in 100 samples). Mathematically: Bernoulli(p) has outcomes {0,1}. Binomial(n,p) has outcomes {0,1,...,n} and represents n independent Bernoulli(p) trials. A Binomial(1,p) = Bernoulli(p). Properties: Bernoulli has E[X]=p, Var=p(1-p). Binomial has E[X]=np, Var=np(1-p). In ML: Use Bernoulli for single prediction (is this email spam?). Use Binomial for aggregate counts (out of 100 emails, how many are spam?). Binomial appears in model evaluation (correct predictions out of test set), A/B testing (conversions out of visitors), and batch accuracy. Bernoulli appears in classification loss functions and individual prediction modeling.',
    keyPoints: [
      'Bernoulli: single binary trial, outcomes {0,1}',
      'Binomial: n Bernoulli trials, outcomes {0,...,n}',
      'Binomial(1,p) = Bernoulli(p)',
      'Bernoulli for single predictions, Binomial for counts',
      'Both fundamental to classification problems',
    ],
  },
  {
    id: 'q2',
    question:
      'The Poisson distribution has a unique property: mean equals variance (E[X] = Var(X) = λ). Explain why this matters and how you can test if count data follows a Poisson distribution.',
    sampleAnswer:
      'The Poisson property E[X]=Var(X)=λ arises from modeling independent events at constant rate. This is unique - most distributions have variance unrelated to mean. Why it matters: (1) One parameter λ determines everything. (2) We can test if data is Poisson by checking if sample mean ≈ sample variance. (3) Overdispersion (Var > Mean) suggests clustering/dependence - need different model (Negative Binomial). (4) Underdispersion (Var < Mean) suggests regularity/spacing. Testing: Compute sample mean m and variance s². If data is Poisson, expect s² ≈ m. Dispersion index D = s²/m should be ~1. If D >> 1 (overdispersed), events are clustered - not truly independent. If D << 1 (underdispersed), events are too regular. In ML: Check this before assuming Poisson for count data. Website clicks, bug counts, customer arrivals often overdispersed due to real-world dependencies.',
    keyPoints: [
      'Poisson: E[X] = Var(X) = λ',
      'Unique property from independent events assumption',
      'Test by checking if sample mean ≈ sample variance',
      'Overdispersion indicates clustering/dependence',
      'Important for validating Poisson model assumptions',
    ],
  },
  {
    id: 'q3',
    question:
      'The Geometric distribution has the memoryless property. Explain what this means, prove it mathematically, and discuss its implications for reinforcement learning.',
    sampleAnswer:
      "Memoryless property: P(X > s+t | X > s) = P(X > t). Past failures don't affect future probabilities. Proof: P(X > k) = (1-p)^k (k failures). P(X > s+t | X > s) = P(X > s+t) / P(X > s) = (1-p)^(s+t) / (1-p)^s = (1-p)^t = P(X > t). The conditioning \"cancels out\" past failures. Implications for RL: (1) If your policy hasn't succeeded in 100 episodes, expected episodes remaining = 1/p (same as if starting fresh). (2) Past failures don't inform future probability under geometric model. (3) This matches truly memoryless processes but not learning agents - agents improve over time, violating memoryless assumption. (4) Geometric is appropriate for random search, not for learning algorithms. (5) In reality, learning creates memory - past attempts inform future success probability. The memoryless property is both mathematically elegant and a limitation when modeling learning processes.",
    keyPoints: [
      'Memoryless: P(X > s+t | X > s) = P(X > t)',
      "Past doesn't affect future probabilities",
      'Appropriate for truly random processes',
      'NOT appropriate for learning algorithms',
      'Learning creates memory, violating assumption',
    ],
  },
];
