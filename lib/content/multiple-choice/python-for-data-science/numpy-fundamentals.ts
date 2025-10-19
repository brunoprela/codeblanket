import { MultipleChoiceQuestion } from '../../../types';

export const numpyFundamentalsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'numpy-fundamentals-mc-1',
    question:
      'What is the primary advantage of NumPy arrays over Python lists for numerical computations?',
    options: [
      'NumPy arrays can store mixed data types',
      'NumPy arrays are 10-100x faster and more memory efficient',
      'NumPy arrays can dynamically resize during operations',
      'NumPy arrays require less code to create',
    ],
    correctAnswer: 1,
    explanation:
      'NumPy arrays are significantly faster (10-100x) than Python lists for numerical operations because they are stored contiguously in memory, use homogeneous data types, and leverage optimized C/Fortran code. They also use less memory than Python lists.',
  },
  {
    id: 'numpy-fundamentals-mc-2',
    question:
      'Given the code: `arr = np.arange(12).reshape(3, 4)`, what is the output of `arr.shape`?',
    options: ['(12,)', '(3, 4)', '(4, 3)', '(1, 12)'],
    correctAnswer: 1,
    explanation:
      'The reshape(3, 4) method transforms the 1D array of 12 elements into a 2D array with 3 rows and 4 columns, resulting in a shape of (3, 4).',
  },
  {
    id: 'numpy-fundamentals-mc-3',
    question:
      'What is the difference between `arr.flatten()` and `arr.ravel()`?',
    options: [
      'flatten() is faster than ravel()',
      'ravel() only works on 2D arrays',
      'flatten() always returns a copy, ravel() returns a view when possible',
      'They are identical with no difference',
    ],
    correctAnswer: 2,
    explanation:
      "flatten() always returns a copy of the array, while ravel() returns a view when possible, making it more memory efficient. However, modifying a ravel() result may affect the original array if it's a view.",
  },
  {
    id: 'numpy-fundamentals-mc-4',
    question:
      'Which operation creates a COPY rather than a VIEW of the original array?',
    options: ['arr[1:5]', 'arr[:, 2]', 'arr[[1, 3, 5]]', 'arr.reshape(2, -1)'],
    correctAnswer: 2,
    explanation:
      'Fancy indexing (using a list or array of indices like arr[[1, 3, 5]]) creates a copy of the data. Basic slicing operations (arr[1:5], arr[:, 2]) and reshape typically create views that share memory with the original array.',
  },
  {
    id: 'numpy-fundamentals-mc-5',
    question:
      'For a neural network with 1 million parameters, which dtype would be most memory efficient while maintaining sufficient precision?',
    options: ['np.float64', 'np.float32', 'np.int32', 'np.float16'],
    correctAnswer: 1,
    explanation:
      'np.float32 (4 bytes per element) is the sweet spot for neural networks, providing sufficient precision for most applications while using half the memory of float64 (8 bytes). This would save 4 MB for 1 million parameters. float16 is too imprecise for many operations, and int32 cannot represent decimal values needed for weights.',
  },
];
