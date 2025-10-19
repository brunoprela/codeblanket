import { MultipleChoiceQuestion } from '../../../types';

export const optimizationAlgorithmsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'optimization-algorithms-mc-1',
    question: 'What is the main advantage of Adam optimizer over vanilla SGD?',
    options: [
      'Adam uses less memory than SGD',
      'Adam adapts learning rates per parameter and includes momentum',
      'Adam always converges to better solutions than SGD',
      'Adam requires fewer epochs to train',
    ],
    correctAnswer: 1,
    explanation:
      'Adam adapts learning rates for each parameter based on gradient history (second moment) and includes momentum (first moment). This makes it robust and effective across different problems without manual tuning. While Adam often converges faster, SGD+Momentum can sometimes achieve better final generalization.',
  },
  {
    id: 'optimization-algorithms-mc-2',
    question:
      'In the momentum update rule v_t = β v_{t-1} + ∇L, what does a momentum coefficient β = 0.9 mean?',
    options: [
      '90% of the update comes from the current gradient',
      '90% of the previous velocity is retained in the new velocity',
      'Learning rate is multiplied by 0.9',
      'Gradients are scaled by 0.9',
    ],
    correctAnswer: 1,
    explanation:
      'β = 0.9 means that 90% of the previous velocity is retained, and 10% comes from the new gradient. This creates exponentially weighted moving average of gradients, dampening oscillations and accelerating convergence in consistent directions. Common values are β = 0.9 or 0.99.',
  },
  {
    id: 'optimization-algorithms-mc-3',
    question:
      'What is the purpose of bias correction in Adam optimizer (computing m̂_t and v̂_t)?',
    options: [
      'To prevent overfitting',
      'To correct for initialization bias in early training iterations',
      'To adjust for different batch sizes',
      'To normalize gradients across layers',
    ],
    correctAnswer: 1,
    explanation:
      'Bias correction (m̂_t = m_t/(1-β₁^t) and v̂_t = v_t/(1-β₂^t)) compensates for the fact that m and v are initialized at zero, which biases them toward zero in early iterations. Without correction, the optimizer would take very small steps initially. Correction ensures proper behavior from the start.',
  },
  {
    id: 'optimization-algorithms-mc-4',
    question:
      'In cosine annealing learning rate schedule, what happens to the learning rate over time?',
    options: [
      'It increases exponentially',
      'It stays constant',
      'It decreases following a cosine curve',
      'It oscillates randomly',
    ],
    correctAnswer: 2,
    explanation:
      'Cosine annealing smoothly decreases the learning rate following a cosine curve: η_t = η_min + (η_max - η_min)(1 + cos(πt/T))/2. This provides smooth, gradual decay that works well with modern architectures. Variants like SGDR can include periodic "warm restarts" to escape local minima.',
  },
  {
    id: 'optimization-algorithms-mc-5',
    question:
      'What is the key difference between Adam with L2 regularization and AdamW?',
    options: [
      'AdamW uses a different momentum coefficient',
      'AdamW decouples weight decay from adaptive learning rates',
      'AdamW trains faster than Adam',
      'AdamW uses less memory than Adam',
    ],
    correctAnswer: 1,
    explanation:
      'AdamW decouples weight decay from the gradient-based update. In standard Adam+L2, the L2 term is added to gradients before adaptive scaling, which interferes with the intended regularization. AdamW applies weight decay separately after the Adam update, providing consistent regularization across all parameters.',
  },
];
