/**
 * Multiple choice questions for Bias Detection & Mitigation section
 */

export const biasdetectionmitigationMultipleChoice = [
  {
    id: 'bias-det-mc-1',
    question:
      'Your resume screening AI has 80% acceptance rate for Group A, 40% for Group B. What type of bias is this?',
    options: [
      'Selection bias',
      'Demographic disparity',
      'Measurement bias',
      'Sampling bias',
    ],
    correctAnswer: 1,
    explanation:
      'This is demographic disparity—different outcomes (acceptance rates) for different demographic groups. Selection bias (A) is about training data collection. Measurement bias (C) is about data quality differences. Sampling bias (D) is about non-representative samples.',
  },
  {
    id: 'bias-det-mc-2',
    question:
      'To reduce gender bias, you remove "name" and "gender" fields from resumes. The AI still shows bias. Why?',
    options: [
      'The AI is inherently biased',
      'Other fields (hobbies, schools) act as gender proxies',
      'You need to remove more fields',
      'Gender removal only works with deep learning',
    ],
    correctAnswer: 1,
    explanation:
      'Even with direct gender indicators removed, other fields can act as proxies: hobbies ("football" vs "dance"), schools (all-male/all-female colleges), career gaps (maternity leave patterns). The AI learns gender from these indirect signals. This is why proxy variable detection is crucial.',
  },
  {
    id: 'bias-det-mc-3',
    question: 'Demographic parity means:',
    options: [
      'Equal accuracy across groups',
      'Equal acceptance rates across groups',
      'Equal error rates across groups',
      'Equal representation in training data',
    ],
    correctAnswer: 1,
    explanation:
      "Demographic parity (also called statistical parity) means equal acceptance/positive outcome rates across demographic groups. Option A describes equalized odds. Option C describes equal opportunity. Option D describes balanced training data, which helps achieve fairness but isn't the definition of demographic parity.",
  },
  {
    id: 'bias-det-mc-4',
    question:
      'Your training data is 90% Group A, 10% Group B. Which strategy does NOT help reduce bias?',
    options: [
      'Oversample Group B to 50%',
      'Undersample Group A to 50%',
      'Train longer to let the model learn Group B better',
      'Use weighted loss (higher weight for Group B)',
    ],
    correctAnswer: 2,
    explanation:
      "Training longer won't fix imbalanced data—the model will just overfit to Group A patterns. Options A, B, and D are valid mitigation strategies. Option A (oversample minority) and B (undersample majority) create balanced data. Option D (weighted loss) gives more importance to minority group errors.",
  },
  {
    id: 'bias-det-mc-5',
    question:
      'You deploy a bias mitigation system. Group A accuracy: 95%. Group B accuracy: 90%. Is this acceptable?',
    options: [
      'No—must have exactly equal accuracy',
      'Yes—5% gap is reasonable',
      'Depends on the application and risk',
      'No—Group B accuracy is too low',
    ],
    correctAnswer: 2,
    explanation:
      "Acceptability depends on application and risk. For medical diagnosis: 5% gap might be too large. For movie recommendations: 5% gap is fine. For hiring/lending: Regulated limits apply. There's no universal threshold—context matters. Option A (exactly equal) is often impossible and not required by fairness definitions.",
  },
];
