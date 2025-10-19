import { MultipleChoiceQuestion } from '../../../types';

export const trainTestSplitValidationMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'train-test-split-validation-mc-1',
      question:
        'Which of the following represents the CORRECT order of operations to prevent data leakage?',
      options: [
        'Scale data → Split data → Train model → Evaluate',
        'Split data → Scale training data → Scale test data using training statistics → Train model',
        'Split data → Scale all data together → Train model → Evaluate',
        'Train model → Split data → Scale data → Evaluate',
      ],
      correctAnswer: 1,
      explanation:
        'Data must be split FIRST, then preprocessing (like scaling) should be fitted on training data only and applied to test data using the training statistics. This prevents test set information from leaking into the training process.',
    },
    {
      id: 'train-test-split-validation-mc-2',
      question:
        'You have a binary classification dataset with 95% class 0 and 5% class 1. Which splitting strategy should you use?',
      options: [
        'Random split with shuffle=True',
        'Random split with shuffle=False',
        'Stratified split with stratify=y',
        'Sequential split in time order',
      ],
      correctAnswer: 2,
      explanation:
        'For imbalanced classification problems, stratified splitting (stratify=y) ensures that both train and test sets maintain the same class proportions (95%-5%) as the original dataset. Random splitting might create unrepresentative splits by chance.',
    },
    {
      id: 'train-test-split-validation-mc-3',
      question:
        'For a time series forecasting problem (e.g., predicting stock prices), what is the MOST appropriate splitting strategy?',
      options: [
        'Random split with shuffle=True to ensure IID data',
        'Stratified split based on price ranges',
        'Sequential split where training data comes before test data chronologically',
        'Split randomly but maintain temporal order within each split',
      ],
      correctAnswer: 2,
      explanation:
        'Time series data has temporal dependencies and must use sequential splitting where training data comes before test data chronologically. Random shuffling destroys the temporal structure and creates data leakage by allowing the model to "predict the past using the future".',
    },
    {
      id: 'train-test-split-validation-mc-4',
      question:
        'What is the PRIMARY purpose of the validation set in a train-validation-test split?',
      options: [
        'To provide additional training data when the training set is too small',
        'To tune hyperparameters and select between different models',
        'To give a final unbiased estimate of model performance',
        'To replace the test set when you need to evaluate multiple times',
      ],
      correctAnswer: 1,
      explanation:
        'The validation set is used for hyperparameter tuning and model selection during development. It can be used many times to compare different models and configurations. The test set (not validation) provides the final unbiased performance estimate and should only be used once.',
    },
    {
      id: 'train-test-split-validation-mc-5',
      question:
        'You notice your model achieves 98% accuracy on the test set but only 70% in production. What is the MOST likely cause?',
      options: [
        'The model is too simple (high bias)',
        'Data leakage during training - test set information influenced the model',
        'The test set was too large',
        'Random variation in performance metrics',
      ],
      correctAnswer: 1,
      explanation:
        'A large drop from test set to production performance typically indicates data leakage. Common causes include: scaling before splitting, using test set for feature selection, peeking at test set during model development, or including future information in time series features. This creates artificially inflated test set performance.',
    },
  ];
