/**
 * Try-Except-Else-Finally
 * Problem ID: intermediate-exception-else
 * Order: 32
 */

import { Problem } from '../../../types';

export const intermediate_exception_elseProblem: Problem = {
  id: 'intermediate-exception-else',
  title: 'Try-Except-Else-Finally',
  difficulty: 'Medium',
  description: `Use all parts of exception handling: try, except, else, finally.

**Flow:**
- try: Code that might raise
- except: Handle exceptions
- else: Runs if no exception
- finally: Always runs

This tests:
- Exception handling flow
- else clause
- finally clause`,
  examples: [
    {
      input: 'Process with error handling',
      output: 'Proper cleanup in all cases',
    },
  ],
  constraints: ['Use all four parts', 'Handle cleanup properly'],
  hints: [
    'else runs when no exception',
    'finally always runs',
    'Order: try-except-else-finally',
  ],
  starterCode: `def divide_numbers(a, b):
    """
    Divide with proper exception handling.
    
    Returns:
        Result or error message
        
    Examples:
        >>> divide_numbers(10, 2)
        5.0
        >>> divide_numbers(10, 0)
        'Error: Division by zero'
    """
    try:
        result = a / b
    except ZeroDivisionError:
        return "Error: Division by zero"
    else:
        return result
    finally:
        pass  # Cleanup code


# Test
print(divide_numbers(10, 2))
`,
  testCases: [
    {
      input: [10, 2],
      expected: 5.0,
    },
    {
      input: [10, 0],
      expected: 'Error: Division by zero',
    },
  ],
  solution: `def divide_numbers(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        return "Error: Division by zero"
    else:
        return result
    finally:
        pass`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 32,
  topic: 'Python Intermediate',
};
