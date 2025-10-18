/**
 * sum() with Start Value
 * Problem ID: intermediate-sum-with-start
 * Order: 50
 */

import { Problem } from '../../../types';

export const intermediate_sum_with_startProblem: Problem = {
  id: 'intermediate-sum-with-start',
  title: 'sum() with Start Value',
  difficulty: 'Easy',
  description: `Use sum() with custom start value.

**Syntax:**
\`\`\`python
sum(iterable, start=0)
\`\`\`

Start value is added to the sum.

This tests:
- sum() function
- start parameter
- Accumulation`,
  examples: [
    {
      input: 'sum([1, 2, 3], start=10)',
      output: '16',
    },
  ],
  constraints: ['Use sum()', 'Provide start value'],
  hints: ['sum(iterable, start)', 'Default start is 0', 'Useful for offsets'],
  starterCode: `def calculate_total_with_tax(prices, tax_rate):
    """
    Calculate total with tax.
    
    Args:
        prices: List of prices
        tax_rate: Tax rate (e.g., 0.1 for 10%)
        
    Returns:
        Total with tax
        
    Examples:
        >>> calculate_total_with_tax([10, 20, 30], 0.1)
        66.0
    """
    subtotal = sum(prices)
    tax = subtotal * tax_rate
    total = subtotal + tax
    return total


# Test
print(calculate_total_with_tax([100, 200], 0.05))
`,
  testCases: [
    {
      input: [[100, 200], 0.05],
      expected: 315.0,
    },
    {
      input: [[10, 20, 30], 0.1],
      expected: 66.0,
    },
  ],
  solution: `def calculate_total_with_tax(prices, tax_rate):
    subtotal = sum(prices)
    tax = subtotal * tax_rate
    return subtotal + tax


# Using sum with start for different approach
def sum_with_bonus(values, bonus):
    return sum(values, start=bonus)`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  order: 50,
  topic: 'Python Intermediate',
};
