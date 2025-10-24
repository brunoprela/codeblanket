/**
 * Advanced String Formatting
 * Problem ID: intermediate-string-format-advanced
 * Order: 36
 */

import { Problem } from '../../../types';

export const intermediate_string_format_advancedProblem: Problem = {
  id: 'intermediate-string-format-advanced',
  title: 'Advanced String Formatting',
  difficulty: 'Easy',
  description: `Use advanced f-string formatting options.

Format options:
- Alignment: {var:<10}, {var:>10}, {var:^10}
- Precision: {num:.2f}
- Padding: {var:0>5}
- Date: {date:%Y-%m-%d}

This tests:
- F-string features
- Format specifiers
- String alignment`,
  examples: [
    {
      input: 'Format numbers and strings',
      output: 'Aligned and formatted output',
    },
  ],
  constraints: ['Use f-strings', 'Use format specifiers'],
  hints: [
    'f"{var:format_spec}"',
    '< left, > right, ^ center',
    '.2f for 2 decimal places',
  ],
  starterCode: `def format_table(name, score, percentage):
    """
    Format data in table format.
    
    Args:
        name: String
        score: Integer
        percentage: Float
        
    Returns:
        Formatted string
        
    Examples:
        >>> format_table("Alice", 95, 0.95)
        'Alice     |  95 | 95.00%'
    """
    # Left-align name (10 chars), right-align score (4 chars), 
    # format percentage with 2 decimals
    result = f"{name:<10}| {score:>4} | {percentage:>6.2%}"
    return result


# Test
print(format_table("Bob", 87, 0.87))
`,
  testCases: [
    {
      input: ['Bob', 87, 0.87],
      expected: 'Bob       |   87 | 87.00%',
    },
    {
      input: ['Alice', 95, 0.95],
      expected: 'Alice     |   95 | 95.00%',
    },
  ],
  solution: `def format_table(name, score, percentage):
    return f"{name:<10}| {score:>4} | {percentage:>6.2%}"


# More examples
def format_currency(amount):
    # Comma separator, 2 decimals
    return f"\${'$'}{amount:,.2f}"

def format_hex(number):
    # Hex with 0x prefix, padded
    return f"{number:#06x}"`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 36,
  topic: 'Python Intermediate',
};
