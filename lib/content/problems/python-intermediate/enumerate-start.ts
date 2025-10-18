/**
 * Enumerate with Custom Start
 * Problem ID: intermediate-enumerate-start
 * Order: 39
 */

import { Problem } from '../../../types';

export const intermediate_enumerate_startProblem: Problem = {
  id: 'intermediate-enumerate-start',
  title: 'Enumerate with Custom Start',
  difficulty: 'Easy',
  description: `Use enumerate with custom start index.

**Syntax:**
\`\`\`python
for i, item in enumerate(items, start=1):
    print(f"{i}. {item}")
\`\`\`

This tests:
- enumerate function
- start parameter
- Iteration patterns`,
  examples: [
    {
      input: 'enumerate(items, start=1)',
      output: 'Index starts at 1',
    },
  ],
  constraints: ['Use enumerate', 'Custom start'],
  hints: [
    'enumerate(iterable, start=n)',
    'Default start is 0',
    'Useful for 1-based numbering',
  ],
  starterCode: `def create_numbered_list(items):
    """
    Create numbered list starting from 1.
    
    Args:
        items: List of strings
        
    Returns:
        List of "1. item", "2. item", etc.
        
    Examples:
        >>> create_numbered_list(['a', 'b', 'c'])
        ['1. a', '2. b', '3. c']
    """
    pass


# Test
print(create_numbered_list(['apple', 'banana', 'cherry']))
`,
  testCases: [
    {
      input: [['apple', 'banana', 'cherry']],
      expected: ['1. apple', '2. banana', '3. cherry'],
    },
    {
      input: [['x', 'y']],
      expected: ['1. x', '2. y'],
    },
  ],
  solution: `def create_numbered_list(items):
    return [f"{i}. {item}" for i, item in enumerate(items, start=1)]


# Alternative
def create_numbered_list_custom_start(items, start=1):
    return [f"{i}. {item}" for i, item in enumerate(items, start=start)]`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  order: 39,
  topic: 'Python Intermediate',
};
