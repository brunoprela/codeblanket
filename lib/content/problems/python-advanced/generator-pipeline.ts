/**
 * Chained Generator Pipeline
 * Problem ID: generator-pipeline
 * Order: 17
 */

import { Problem } from '../../../types';

export const generator_pipelineProblem: Problem = {
  id: 'generator-pipeline',
  title: 'Chained Generator Pipeline',
  difficulty: 'Medium',
  description: `Build a data pipeline using chained generators for memory-efficient processing.

Create generators that:
- read_numbers: yields numbers from a list
- filter_even: filters even numbers
- square: squares each number
- Chain them together

**Key Concept:** Generators enable lazy evaluation - no intermediate lists created.`,
  examples: [
    {
      input: '[1, 2, 3, 4, 5, 6]',
      output: '[4, 16, 36] (even numbers squared)',
    },
  ],
  constraints: [
    'Each stage must be a generator',
    'No intermediate lists',
    'Chain with function composition',
  ],
  hints: [
    'Each generator takes previous as input',
    'Use yield in loops',
    'Composition: square(filter_even(read_numbers()))',
  ],
  starterCode: `def read_numbers(numbers):
    """Yield numbers from list."""
    # Your code here
    pass

def filter_even(numbers):
    """Yield only even numbers."""
    # Your code here
    pass

def square(numbers):
    """Yield squared numbers."""
    # Your code here
    pass


# Build pipeline
data = [1, 2, 3, 4, 5, 6]
pipeline = square(filter_even(read_numbers(data)))
result = list(pipeline)
`,
  testCases: [
    {
      input: [],
      expected: [4, 16, 36],
    },
  ],
  solution: `def read_numbers(numbers):
    for num in numbers:
        yield num

def filter_even(numbers):
    for num in numbers:
        if num % 2 == 0:
            yield num

def square(numbers):
    for num in numbers:
        yield num ** 2


# Build pipeline
data = [1, 2, 3, 4, 5, 6]
pipeline = square(filter_even(read_numbers(data)))
result = list(pipeline)`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  order: 17,
  topic: 'Python Advanced',
};
