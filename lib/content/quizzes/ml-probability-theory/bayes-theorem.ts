/**
 * Quiz questions for Bayes' Theorem section
 */

export const bayestheoremQuiz = [
  {
    id: 'q1',
    question:
      "Explain each component of Bayes' Theorem: P(A|B) = P(B|A)×P(A)/P(B). Why is each term necessary, and what would happen if we ignored any of them?",
    hint: 'Think about prior beliefs, evidence strength, and normalization.',
    sampleAnswer:
      "Bayes' Theorem has four components: (1) P(A|B) is the posterior - what we want to find after seeing evidence B. (2) P(B|A) is the likelihood - how probable is our evidence if hypothesis A is true. This measures evidence strength. (3) P(A) is the prior - our initial belief before seeing evidence. This incorporates domain knowledge. (4) P(B) is the evidence/marginal probability - normalizes to ensure probabilities sum to 1. If we ignored: (1) Ignoring prior P(A) causes base rate neglect - we'd think rare diseases are likely with positive tests. (2) Ignoring likelihood P(B|A) means we don't consider how strong the evidence is. (3) Ignoring evidence P(B) means probabilities won't be calibrated correctly. Each component is essential: prior incorporates existing knowledge, likelihood weighs the evidence, and marginal probability ensures mathematical consistency. Together, they provide a principled way to update beliefs.",
    keyPoints: [
      'Posterior P(A|B): what we want to find',
      'Likelihood P(B|A): strength of evidence',
      'Prior P(A): initial belief before evidence',
      'Evidence P(B): normalizing constant',
      'All components necessary for valid inference',
    ],
  },
  {
    id: 'q2',
    question:
      "A medical test is 99% accurate (both sensitivity and specificity). A disease affects 0.1% of the population. If you test positive, what's the probability you have the disease? Walk through the calculation and explain why the answer surprises most people.",
    hint: "Use Bayes' Theorem with the base rate.",
    sampleAnswer:
      "Given: P(disease) = 0.001, P(+|disease) = 0.99, P(+|healthy) = 0.01. Using Bayes: P(disease|+) = P(+|disease)×P(disease) / P(+). First find P(+) using law of total probability: P(+) = P(+|disease)×P(disease) + P(+|healthy)×P(healthy) = 0.99×0.001 + 0.01×0.999 = 0.00099 + 0.00999 = 0.01098. So P(disease|+) = (0.99×0.001) / 0.01098 = 0.09 or about 9%. Even with a 99% accurate test, there's only a 9% chance of having the disease! This surprises people because they ignore the base rate (0.1% prevalence). When a disease is rare, most positive tests are false positives. Out of 100,000 people: ~100 have disease (99 test positive), ~99,900 are healthy (999 test positive). So 999/(99+999) ≈ 9% of positives actually have the disease. This is the prosecutor's fallacy - confusing P(+|disease) with P(disease|+).",
    keyPoints: [
      "Must use Bayes' Theorem, not just test accuracy",
      'Base rate (prior) critically affects posterior',
      'Rare diseases → most positives are false positives',
      'P(disease|+) ≠ P(+|disease) - common confusion',
      'Always consider prevalence/base rate',
    ],
  },
  {
    id: 'q3',
    question:
      "In Bayesian updating, the posterior from one observation becomes the prior for the next. Explain how this sequential updating works and why it's fundamental to machine learning.",
    hint: 'Think about learning as accumulating evidence.',
    sampleAnswer:
      'Bayesian updating works sequentially: (1) Start with prior P(H). (2) Observe evidence E₁, compute posterior P(H|E₁) using Bayes. (3) This posterior becomes the new prior for the next update. (4) Observe E₂, compute P(H|E₁,E₂) using P(H|E₁) as prior. (5) Repeat indefinitely. Each observation refines our belief. Example: Fair vs biased coin. Start with P(fair)=0.5. Flip heads → P(fair)=0.417. Flip heads again → P(fair)=0.347. Each observation updates our belief. This is fundamental to ML because: (1) Training is sequential updating - each batch updates model parameters. (2) Online learning continuously updates with new data. (3) Bayesian neural networks maintain uncertainty over parameters. (4) Reinforcement learning updates value estimates with each episode. (5) All learning is accumulating evidence and updating beliefs. The math guarantees consistency: final posterior depends only on total evidence, not order of observations. This principled framework for learning from data is why Bayesian methods are so powerful.',
    keyPoints: [
      'Sequential: posterior becomes next prior',
      'Each observation refines belief',
      "Order doesn't matter for final posterior",
      'Fundamental to all learning algorithms',
      'Principled framework for accumulating evidence',
    ],
  },
];
