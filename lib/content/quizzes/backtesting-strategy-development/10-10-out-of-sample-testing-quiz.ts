import { MultipleChoiceQuestion } from '@/lib/types';

const outOfSampleTestingQuiz: MultipleChoiceQuestion[] = [
  {
    id: 'oos-1',
    question:
      'You have 10 years of daily data (2,520 observations). For proper out-of-sample testing, what is the minimum recommended size for the test set?',
    options: [
      '50 observations (2%) is sufficient for statistical power',
      'At least 500 observations (20%) and minimum 6 months',
      '100 observations (4%) as long as results are significant',
      'The entire last year (252 observations, 10%)',
    ],
    correctAnswer: 1,
    explanation:
      'Industry best practice requires at least 20% of data for the test set AND a minimum of 6 months (for daily strategies) to ensure results are robust across different market conditions. With 10 years of data, this means reserving at least 2 years (504 observations, 20%). Option A (2%) is far too small. Option C ignores the time-period requirement. Option D (10%) is below the 20% threshold. The test period must be long enough to encounter various market regimes and demonstrate consistent performance.',
    difficulty: 'intermediate',
  },
  {
    id: 'oos-2',
    question:
      'During strategy development, you accidentally glimpse the test set returns and notice your strategy would have performed poorly. What should you do?',
    options: [
      'Ignore what you saw and proceed with the original test plan',
      'Modify the strategy to fix the issues, then re-test',
      'Consider the test set compromised and create a new holdout period from recent data',
      'Report the poor performance and stop development',
    ],
    correctAnswer: 2,
    explanation:
      "Once you've seen test set results, that data is permanently compromised—your brain will unconsciously use that information in future decisions. The only valid solution is to lock away a NEW out-of-sample period from more recent data. Option A doesn't work because unconscious bias will creep in. Option B is explicit overfitting (modifying based on test performance). Option D is premature—the strategy might still be viable with proper OOS testing. This is why professional quant shops have strict access controls and use cryptographic hashes to verify test data integrity.",
    difficulty: 'advanced',
  },
  {
    id: 'oos-3',
    question:
      'A strategy shows Sharpe ratio of 2.1 in training, 1.8 in validation, and 0.9 in test. What is the training-to-test degradation and what does it indicate?',
    options: [
      '14% degradation (from 2.1 to 1.8), which is acceptable',
      '57% degradation (from 2.1 to 0.9), indicating severe overfitting',
      '50% degradation (from 1.8 to 0.9), which is concerning but acceptable',
      'The strategy improved by 0.9, so no degradation',
    ],
    correctAnswer: 1,
    explanation:
      "Degradation is calculated from training to test: (2.1 - 0.9) / 2.1 × 100% = 57%. This severe degradation indicates the strategy is heavily overfit to the training data. The in-sample performance was achieved by curve-fitting to historical noise that doesn't persist out-of-sample. Option A incorrectly calculates only train-to-validation. Option C uses validation-to-test, missing the full picture. A 57% drop typically warrants rejection—the strategy likely has minimal genuine edge. Professional standards: <10% is excellent, 10-30% acceptable, 30-50% concerning, >50% reject.",
    difficulty: 'intermediate',
  },
  {
    id: 'oos-4',
    question:
      'What is the main purpose of having THREE data splits (train/validation/test) instead of just two (train/test)?',
    options: [
      'To have more data for training the model',
      'The validation set is used for hyperparameter tuning without contaminating the test set',
      'The validation set is a backup in case the test set is compromised',
      'To increase statistical power through more testing',
    ],
    correctAnswer: 1,
    explanation:
      "The three-way split prevents test set contamination during model selection. Training data is for parameter learning, validation data is for choosing between models/hyperparameters, and test data is for final unbiased evaluation. Without a validation set, you'd tune models based on test performance, which contaminates the test set through repeated access. This is crucial in ML and quantitative trading—you need an unseen dataset for final evaluation. Option A is wrong (validation doesn't increase training data). Option C misunderstands the purpose. Option D is incorrect—you only test once on the test set.",
    difficulty: 'beginner',
  },
  {
    id: 'oos-5',
    question:
      'Your strategy passes all validation tests but fails the final out-of-sample test. What should you do?',
    options: [
      'Run the OOS test again on a different time period',
      'Adjust the strategy and re-run the OOS test',
      'Report the failure and either stop development or start fresh with new data',
      "Use the validation set as the 'new' test set since it passed",
    ],
    correctAnswer: 2,
    explanation:
      "Failed OOS tests must be reported and respected—this is the moment of truth that separates real edges from overfitting. You have three options: (1) abandon the strategy, (2) go back to square one with completely new holdout data (months/years later), or (3) deploy very cautiously with minimal capital after extended paper trading. You CANNOT re-test (Option A), adjust and re-test (Option B), or repurpose the validation set (Option D)—all of these invalidate the scientific process. Professional integrity requires reporting failures. Many published academic strategies fail OOS tests but aren't reported (publication bias), contributing to the replication crisis in quantitative finance.",
    difficulty: 'advanced',
  },
];

export default outOfSampleTestingQuiz;
