import { MultipleChoiceQuestion } from '../../../types';

export const numpyoperationsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'numpy-operations-mc-1',
    question:
      'What is the key difference between element-wise multiplication (*) and matrix multiplication (@) in NumPy?',
    options: [
      'There is no difference, they are identical',
      '(*) multiplies corresponding elements, (@) performs matrix multiplication',
      '(@) is faster than (*)',
      '(*) only works with 1D arrays',
    ],
    correctAnswer: 1,
    explanation:
      'Element-wise multiplication (*) multiplies corresponding elements in the same positions, while matrix multiplication (@) performs the mathematical matrix product. For 2x2 matrices, (*) gives 4 multiplications, while (@) involves dot products of rows and columns.',
  },
  {
    id: 'numpy-operations-mc-2',
    question:
      'Given arr = np.array([[1,2,3],[4,5,6],[7,8,9]]), what does arr.sum (axis=0) return?',
    options: ['[6, 15, 24]', '[12, 15, 18]', '45', '[[12, 15, 18]]'],
    correctAnswer: 1,
    explanation:
      'arr.sum (axis=0) aggregates along axis 0 (down columns), summing across rows. Column 0: 1+4+7=12, column 1: 2+5+8=15, column 2: 3+6+9=18. Result: [12, 15, 18].',
  },
  {
    id: 'numpy-operations-mc-3',
    question:
      'When combining boolean conditions in NumPy, which operators should you use instead of "and", "or", "not"?',
    options: ['&&, ||, !', '&, |, ~', 'AND, OR, NOT', '.and(), .or(), .not()'],
    correctAnswer: 1,
    explanation:
      'NumPy uses bitwise operators for element-wise boolean operations: & (and), | (or), ~ (not). Python\'s "and", "or", "not" don\'t work element-wise on arrays. Always use parentheses: (arr > 3) & (arr < 8).',
  },
  {
    id: 'numpy-operations-mc-4',
    question: 'What does np.where (arr > 5, 100, -100) do?',
    options: [
      'Returns indices where arr > 5',
      'Returns 100 where arr > 5, otherwise returns -100',
      'Returns True where arr > 5, False otherwise',
      'Raises an error because np.where requires only one argument',
    ],
    correctAnswer: 1,
    explanation:
      "np.where with three arguments is a vectorized ternary operator: it returns 100 for elements where the condition (arr > 5) is True, and -100 where it's False. With one argument, it returns indices.",
  },
  {
    id: 'numpy-operations-mc-5',
    question:
      'Why is np.random.seed() important when using random number generation?',
    options: [
      'It makes random number generation faster',
      'It ensures reproducibility by generating the same sequence of random numbers',
      'It is required before any random operation',
      'It generates truly random numbers instead of pseudo-random',
    ],
    correctAnswer: 1,
    explanation:
      'np.random.seed() initializes the random number generator with a specific state, ensuring that the same sequence of "random" numbers is generated each time. This is crucial for reproducibility in machine learning experiments, debugging, and testing.',
  },
];
