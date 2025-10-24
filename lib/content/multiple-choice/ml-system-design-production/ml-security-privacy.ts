import { MultipleChoiceQuestion } from '@/types/curriculum';

export const mlSecurityPrivacyQuestions: MultipleChoiceQuestion[] = [
  {
    id: 'msp-mc-1',
    question:
      'Your production image classification model is vulnerable to adversarial attacks. Testing shows that adding imperceptible noise to images can cause misclassification. Which defense strategy provides the most robust protection?',
    options: [
      'Input validation: reject images with unusual pixel statistics',
      'Adversarial training: include adversarial examples in training data',
      'Defensive distillation: train a student model to smooth decision boundaries',
      'Ensemble methods: use multiple models and majority voting',
    ],
    correctAnswer: 1,
    explanation:
      'Adversarial training is the most robust defense: training on adversarial examples (generated using FGSM, PGD, etc.) makes the model inherently more robust to these attacks. This directly addresses the vulnerability. Input validation (option A) can be bypassed with carefully crafted adversarial examples that maintain valid statistics. Defensive distillation (option C) has been shown to provide limited robustness against strong attacks. Ensemble methods (option D) can be attacked by finding adversarial examples that fool all models in the ensemble.',
    difficulty: 'advanced',
    topic: 'ML Security & Privacy',
  },
  {
    id: 'msp-mc-2',
    question:
      "You're deploying a medical diagnosis model that must protect patient privacy while still allowing model improvements. The model needs to learn from distributed hospital data without centralizing sensitive information. Which approach best balances privacy and model quality?",
    options: [
      'Collect all data centrally but encrypt it',
      'Use federated learning with differential privacy guarantees',
      'Train separate models at each hospital and average the weights',
      'Use synthetic data generation to create privacy-safe training data',
    ],
    correctAnswer: 1,
    explanation:
      "Federated learning trains the model on decentralized data (at each hospital), sending only model updates to a central server, not raw data. Adding differential privacy (DP) to gradient updates provides mathematical privacy guarantees against membership inference attacks. This is the gold standard for private distributed learning. Option A (encrypted centralized data) doesn't prevent privacy breaches if the data is decrypted for training. Option C (average weights) is a simple form of federated learning but lacks DP guarantees. Option D (synthetic data) may not capture rare medical conditions well.",
    difficulty: 'advanced',
    topic: 'ML Security & Privacy',
  },
  {
    id: 'msp-mc-3',
    question:
      "Your ML model's predictions are being queried by an attacker attempting a model extraction attack: sending many queries to reconstruct your proprietary model. Which mitigation strategy is most effective?",
    options: [
      'Add random noise to predictions to obscure model behavior',
      'Implement rate limiting and query cost/pricing per user',
      'Use ensemble models that change over time',
      'Obfuscate model architecture by using complex ensembles',
    ],
    correctAnswer: 1,
    explanation:
      "Rate limiting and query pricing are the most practical defenses: they limit the number of queries an attacker can make, making extraction expensive and time-consuming. This doesn't degrade service for legitimate users with normal query patterns. Option A (random noise) hurts legitimate users' experience. Option C (changing ensembles) is operationally complex and may not prevent extraction if enough queries are made. Option D (obfuscation) provides security through obscurity, which is weak—determined attackers can still extract the model with enough queries.",
    difficulty: 'advanced',
    topic: 'ML Security & Privacy',
  },
  {
    id: 'msp-mc-4',
    question:
      "You're implementing differential privacy (DP) for a user behavior prediction model. Adding DP noise (ε=1.0) causes a 15% accuracy drop, which is unacceptable. What is the most appropriate approach to improve this tradeoff?",
    options: [
      'Increase epsilon to ε=5.0 to reduce noise and improve accuracy',
      "Use more training data to improve model's ability to learn despite noise",
      'Apply privacy budget allocation: use less privacy budget on less sensitive features',
      'Switch from DP-SGD to local differential privacy for better utility',
    ],
    correctAnswer: 1,
    explanation:
      'With more training data, the model can better learn the underlying patterns despite DP noise, improving the privacy-utility tradeoff. This is often the most effective approach. Increasing epsilon (option A) weakens privacy guarantees—ε=5.0 provides much less privacy than ε=1.0. Privacy budget allocation (option C) can help but requires careful feature sensitivity analysis. Local DP (option D) typically provides worse utility than central DP for the same privacy level because noise is added per-user rather than in aggregate.',
    difficulty: 'advanced',
    topic: 'ML Security & Privacy',
  },
  {
    id: 'msp-mc-5',
    question:
      "You need to verify that your proprietary model hasn't been stolen and deployed by a competitor. Which technique can help prove ownership if you discover unauthorized use?",
    options: [
      'Model fingerprinting: unique statistical properties that identify your model',
      'Watermarking: embed specific trigger inputs that produce identifiable outputs',
      'API monitoring: track unusual query patterns from competitor IPs',
      'Model hashing: compute and store cryptographic hashes of model weights',
    ],
    correctAnswer: 1,
    explanation:
      "Watermarking embeds specific backdoors (trigger-output pairs) into the model during training. If you query the suspected stolen model with trigger inputs and get the expected outputs, this proves it's your model. The triggers are designed to be hard to remove without significantly damaging model performance. Option A (fingerprinting) can be defeated by fine-tuning or transfer learning. Option C (API monitoring) only detects queries, not model theft. Option D (hashing) proves integrity but doesn't help identify stolen models—you need access to the competitor's model weights.",
    difficulty: 'advanced',
    topic: 'ML Security & Privacy',
  },
];
