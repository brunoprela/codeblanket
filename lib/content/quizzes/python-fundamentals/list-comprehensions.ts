/**
 * Quiz questions for List Comprehensions section
 */

export const listcomprehensionsQuiz = [
  {
    id: 'q1',
    question:
      'Rewrite this loop as a list comprehension: result = []; for x in range(10): if x % 2 == 0: result.append (x ** 2)',
    sampleAnswer:
      'result = [x ** 2 for x in range(10) if x % 2 == 0]. The list comprehension first defines the expression (x ** 2), then the iteration (for x in range(10)), and finally the condition (if x % 2 == 0). This creates a list of squares of even numbers from 0-9: [0, 4, 16, 36, 64].',
    keyPoints: [
      'Syntax: [expression for item in iterable if condition]',
      'Expression comes first: x ** 2',
      'Then iteration: for x in range(10)',
      'Then filter: if x % 2 == 0',
      'More concise than loop',
    ],
  },
  {
    id: 'q2',
    question:
      'What is the difference between [] and () in list comprehension vs generator expression?',
    sampleAnswer:
      'Square brackets [] create a list comprehension that immediately creates the entire list in memory. Parentheses () create a generator expression that computes values lazily on-demand. For example, [x**2 for x in range(1000000)] creates a list of 1 million integers in memory, while (x**2 for x in range(1000000)) creates a generator object that computes values one at a time as needed. Generators are more memory-efficient for large datasets or when you only need values once.',
    keyPoints: [
      '[] creates list immediately (eager)',
      '() creates generator (lazy)',
      'List uses more memory',
      'Generator computes on-demand',
      'Use generators for large data or single iteration',
    ],
  },
  {
    id: 'q3',
    question:
      'When should you use a regular for loop instead of a list comprehension?',
    sampleAnswer:
      'Use a regular for loop when: (1) Logic is complex - list comprehensions should be simple and readable, if you need nested if/else or multiple conditions, use a loop, (2) You need to perform side effects - like printing, writing to files, or modifying external state, (3) The comprehension becomes too long (> 79 characters or wraps multiple lines) - readability matters more than conciseness, (4) You need to break early or use continue with complex logic. For example, "for x in nums: if complex_condition (x): process (x); do_other_stuff()" is clearer as a loop. List comprehensions are for transforming data into new lists, not for executing complex logic.',
    keyPoints: [
      'Use loop for complex logic (multiple conditions, nested if/else)',
      'Use loop for side effects (print, file I/O, mutations)',
      'Use loop if comprehension becomes too long (> 79 chars)',
      'Readability beats conciseness',
      'Comprehensions are for simple transformations',
    ],
  },
];
