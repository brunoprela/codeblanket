import { MultipleChoiceQuestion } from '../../../types';

export const performanceoptimizationMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'performance-optimization-mc-1',
    question:
      'Why is vectorized code faster than Python loops in NumPy/Pandas?',
    options: [
      'Python is slow at everything',
      'Vectorized operations are compiled C code and avoid Python interpreter overhead',
      'Loops use more memory',
      'Vectorization uses multiple CPU cores',
    ],
    correctAnswer: 1,
    explanation:
      'Vectorized operations in NumPy are implemented in compiled C code and operate directly on arrays in memory, avoiding the Python interpreter overhead for each element. Loops in Python must interpret each operation, making them much slower.',
  },
  {
    id: 'performance-optimization-mc-2',
    question:
      'What is the memory advantage of using categorical dtype in pandas?',
    options: [
      'No advantage, just a different representation',
      'Stores unique values once and uses integer codes for references',
      'Compresses the strings',
      'Uses less precision',
    ],
    correctAnswer: 1,
    explanation:
      'Categorical dtype stores unique category values once and uses integer codes to reference them. For a column with many repeated strings (like country names), this saves enormous memory because it stores "USA" once instead of millions of times.',
  },
  {
    id: 'performance-optimization-mc-3',
    question: 'What is the difference between a view and a copy in NumPy?',
    options: [
      'Views are faster to create',
      'Views reference the original data; modifying a view modifies the original',
      'Copies are always better',
      'No practical difference',
    ],
    correctAnswer: 1,
    explanation:
      "A view is a reference to the original array's data - modifying it modifies the original. A copy is independent new data. Views are created by slicing (arr[1:5]), copies by .copy(). Understanding this prevents unexpected data modification.",
  },
  {
    id: 'performance-optimization-mc-4',
    question: 'Why should you avoid using df.iterrows() in pandas?',
    options: [
      'It causes errors',
      'It returns views instead of copies',
      'It is extremely slow because it returns each row as a Series (Python loop with overhead)',
      'It only works with small DataFrames',
    ],
    correctAnswer: 2,
    explanation:
      'iterrows() is very slow because it: 1) Uses a Python loop (no vectorization), 2) Creates a new Series object for each row (overhead), 3) Type inference for each row. Vectorized operations are typically 100-1000x faster.',
  },
  {
    id: 'performance-optimization-mc-5',
    question:
      'When is it appropriate to use .apply() instead of vectorized operations?',
    options: [
      'Never, vectorization is always better',
      'When the operation cannot be vectorized (complex Python function with no vectorized equivalent)',
      'For small DataFrames only',
      'When you want slower code',
    ],
    correctAnswer: 1,
    explanation:
      'Use .apply() only when you must use a complex Python function that has no vectorized equivalent. For example, calling an external API for each row, or a complex business logic function. For mathematical operations, always prefer vectorized operations.',
  },
];
