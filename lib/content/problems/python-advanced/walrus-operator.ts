/**
 * Walrus Operator (:=) for Assignment Expressions
 * Problem ID: advanced-walrus-operator
 * Order: 33
 */

import { Problem } from '../../../types';

export const walrus_operatorProblem: Problem = {
  id: 'advanced-walrus-operator',
  title: 'Walrus Operator (:=) for Assignment Expressions',
  difficulty: 'Easy',
  description: `Use the walrus operator (:=) for inline assignments in expressions.

Apply walrus operator in:
- While loop conditions
- List comprehensions with reuse
- If statements with assignments
- Complex conditionals

**Benefit:** Avoid duplicate evaluations and reduce code verbosity.`,
  examples: [
    {
      input: 'if (n := len(items)) > 10: print(f"Too many: {n}")',
      output: 'Assigns and uses n in one line',
    },
  ],
  constraints: [
    'Use := operator',
    'Valid in Python 3.8+',
    'Understand expression vs statement',
  ],
  hints: [
    'Syntax: (var := expression)',
    'Returns value of expression',
    'Useful in comprehensions and conditionals',
  ],
  starterCode: `def process_items(items):
    """Process items using walrus operator.
    
    Args:
        items: List of items
        
    Returns:
        List of processed lengths
    """
    # Use walrus operator in comprehension
    # Only include items where (length := len(item)) > 3
    pass


def read_until_stop(get_input):
    """Read input until "stop" using walrus operator.
    
    Args:
        get_input: Function that returns next input
        
    Returns:
        List of inputs (excluding "stop")
    """
    # Use walrus operator in while condition
    pass


def categorize_number(n):
    """Categorize number using walrus operator.
    
    Args:
        n: Number to categorize
        
    Returns:
        Category string
    """
    # Use walrus operator in if statements
    # if (abs_n := abs(n)) > 100: return "large"
    # elif abs_n > 10: return "medium"
    # else: return "small"
    pass


# Test
result = process_items(["hi", "hello", "hey", "goodbye"])
`,
  testCases: [
    {
      input: [],
      expected: [5, 7],
    },
  ],
  solution: `def process_items(items):
    return [length for item in items if (length := len(item)) > 3]


def read_until_stop(get_input):
    results = []
    while (value := get_input()) != "stop":
        results.append(value)
    return results


def categorize_number(n):
    if (abs_n := abs(n)) > 100:
        return "large"
    elif abs_n > 10:
        return "medium"
    else:
        return "small"


# Test
result = process_items(["hi", "hello", "hey", "goodbye"])`,
  timeComplexity:
    'O(n) for process_items and read_until_stop, O(1) for categorize',
  spaceComplexity: 'O(n)',
  order: 33,
  topic: 'Python Advanced',
};
