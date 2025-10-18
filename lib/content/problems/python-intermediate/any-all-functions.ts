/**
 * Using any() and all()
 * Problem ID: intermediate-any-all-functions
 * Order: 41
 */

import { Problem } from '../../../types';

export const intermediate_any_all_functionsProblem: Problem = {
  id: 'intermediate-any-all-functions',
  title: 'Using any() and all()',
  difficulty: 'Easy',
  description: `Use any() and all() for efficient boolean checks.

**any()**: Returns True if any element is truthy
**all()**: Returns True if all elements are truthy

Both short-circuit!

This tests:
- Built-in boolean functions
- Generator expressions
- Short-circuit evaluation`,
  examples: [
    {
      input: 'all([True, True, False])',
      output: 'False',
    },
  ],
  constraints: ['Use any() or all()', 'Can use with generator'],
  hints: ['any(iterable)', 'all(iterable)', 'Short-circuits (stops early)'],
  starterCode: `def has_even(numbers):
    """Check if any number is even"""
    return any(n % 2 == 0 for n in numbers)


def all_positive(numbers):
    """Check if all numbers are positive"""
    return all(n > 0 for n in numbers)


def test_any_all():
    """Test any and all"""
    # Test any
    result1 = has_even([1, 3, 5, 6, 7])  # True
    
    # Test all
    result2 = all_positive([1, 2, 3, 4])  # True
    result3 = all_positive([1, -2, 3])    # False
    
    # Count True results
    return sum([result1, result2, not result3])
`,
  testCases: [
    {
      input: [],
      expected: 3,
      functionName: 'test_any_all',
    },
  ],
  solution: `def has_even(numbers):
    return any(n % 2 == 0 for n in numbers)


def all_positive(numbers):
    return all(n > 0 for n in numbers)


def test_any_all():
    result1 = has_even([1, 3, 5, 6, 7])
    result2 = all_positive([1, 2, 3, 4])
    result3 = all_positive([1, -2, 3])
    
    return sum([result1, result2, not result3])`,
  timeComplexity: 'O(n) worst case, can be O(1) with short-circuit',
  spaceComplexity: 'O(1)',
  order: 41,
  topic: 'Python Intermediate',
};
