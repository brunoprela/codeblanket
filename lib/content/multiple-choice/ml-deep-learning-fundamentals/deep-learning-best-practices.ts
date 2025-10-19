import { MultipleChoiceQuestion } from '../../../types';

export const deepLearningBestPracticesMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'best-practices-mc1',
      question:
        'When standardizing features (zero mean, unit variance) for train/val/test sets, which approach is correct?',
      options: [
        'Compute statistics independently on each set (train, val, test)',
        'Compute statistics on training set only, then apply to val/test',
        'Combine all sets, compute statistics, then split',
        'Only standardize the training set, leave val/test as-is',
      ],
      correctAnswer: 1,
      explanation:
        'You must compute statistics (mean, std) on the training set only, then apply those same statistics to validation and test sets. This prevents data leakage - test statistics should not influence preprocessing. Computing independently on each set would cause leakage and overoptimistic performance. Production inference must use the exact same training statistics.',
    },
    {
      id: 'best-practices-mc2',
      question: 'What is the purpose of the Learning Rate Finder technique?',
      options: [
        'To automatically find the optimal learning rate during training',
        'To systematically test different learning rates and identify the best range before full training',
        'To adjust the learning rate dynamically based on validation loss',
        'To reduce the learning rate when training plateaus',
      ],
      correctAnswer: 1,
      explanation:
        "The LR Finder runs a short test with exponentially increasing learning rates, plotting loss vs LR. You then manually select the LR at the steepest descent point before training. It's a pre-training diagnostic, not an automatic or dynamic adjustment. This avoids hours of trial-and-error and provides a principled way to choose learning rate.",
    },
    {
      id: 'best-practices-mc3',
      question:
        "When debugging a neural network that won't train, what is the FIRST diagnostic step you should take?",
      options: [
        'Increase the learning rate',
        'Add more layers to increase model capacity',
        'Try to overfit a single batch to verify the model can learn',
        'Increase regularization to prevent overfitting',
      ],
      correctAnswer: 2,
      explanation:
        "The first diagnostic is to verify your model can overfit a single batch. If it can't, there's a fundamental issue (bug in code, insufficient capacity, wrong loss function). If it can overfit one batch but not the full dataset, the problem is different (optimization, regularization). This simple test quickly identifies whether the issue is in implementation or training.",
    },
    {
      id: 'best-practices-mc4',
      question:
        'Why is random search often more effective than grid search for hyperparameter tuning?',
      options: [
        'Random search is faster to implement',
        'Random search explores more values for important hyperparameters',
        'Random search always finds better hyperparameters than grid search',
        'Random search requires fewer total trials',
      ],
      correctAnswer: 1,
      explanation:
        "Random search explores more values along each dimension, which is crucial because some hyperparameters matter much more than others. Grid search wastes trials on redundant combinations of unimportant parameters. While random search doesn't necessarily require fewer trials or guarantee better results, it's more efficient at exploring the important dimensions of the hyperparameter space.",
    },
    {
      id: 'best-practices-mc5',
      question: 'What is the purpose of gradient checking in deep learning?',
      options: [
        'To prevent exploding gradients during training',
        'To verify that analytical gradients match numerical gradients, detecting implementation bugs',
        'To monitor gradient norms and adjust learning rate',
        'To find the optimal gradient clipping threshold',
      ],
      correctAnswer: 1,
      explanation:
        "Gradient checking verifies your backpropagation implementation by comparing analytical gradients (from your backward pass) with numerical gradients (from finite differences). If they match (within tolerance), your backprop is correct. This is a debugging tool for implementation, not a training technique. It's computationally expensive so only run it during development.",
    },
  ];
