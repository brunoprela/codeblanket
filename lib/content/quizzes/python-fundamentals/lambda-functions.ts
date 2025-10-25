/**
 * Quiz questions for Lambda Functions section
 */

export const lambdafunctionsQuiz = [
  {
    id: 'q1',
    question: 'When should you use a lambda function vs a regular function?',
    sampleAnswer:
      'Use lambda for simple, one-line operations that are used once or passed to functions like map(), filter(), sorted(). Use regular functions for complex logic, reusable code, or anything needing documentation. For example, use lambda for sorting by a key (sorted (items, key=lambda x: x[1])), but use a regular function for data processing with multiple steps, validation, and error handling. Lambda is for convenience, not complexity.',
    keyPoints: [
      'Lambda: simple, one-line, single use',
      'Lambda: with map, filter, sorted',
      'Regular: complex logic, multiple steps',
      'Regular: reusable, needs documentation',
      'Lambda is convenience, not replacement',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain the difference between map() with lambda and list comprehension. Which is more Pythonic?',
    sampleAnswer:
      'Both transform lists but with different syntax. map() with lambda: result = list (map (lambda x: x**2, numbers)). List comprehension: result = [x**2 for x in numbers]. List comprehension is more Pythonic because it is more readable, self-contained, and supports filtering inline (e.g., [x**2 for x in numbers if x > 0]). Map with lambda requires wrapping in list() and filter() separately. However, map can be slightly faster for large datasets and more composable with other functional tools. In practice, Python community prefers list/dict comprehensions for readability.',
    keyPoints: [
      'map (lambda x: x**2, nums) vs [x**2 for x in nums]',
      'List comprehension more Pythonic and readable',
      'Comprehension supports inline filtering',
      'map() can be faster for very large data',
      'Python community prefers comprehensions',
    ],
  },
  {
    id: 'q3',
    question:
      'What are the limitations of lambda functions and why do they exist?',
    sampleAnswer:
      'Lambda limitations: (1) Only single expression, no statements, (2) No assignments inside, (3) No annotations, (4) No docstrings, (5) Harder to debug (shows as <lambda> in tracebacks). These limitations exist by design to keep lambdas simple. They force you to use named functions for complex logic, improving code readability and maintainability. If you need multiple steps, assignments, or complex logic, Python wants you to use def with a descriptive name. This prevents cryptic, unreadable code. Lambdas are for throwaway convenience, not as a replacement for proper functions.',
    keyPoints: [
      'Only single expression allowed',
      'No statements, assignments, or annotations',
      'No docstrings, harder to debug',
      'Limitations by design for simplicity',
      'Forces named functions for complex logic',
    ],
  },
];
