/**
 * Multiple Exception Types
 * Problem ID: intermediate-try-except-multiple
 * Order: 47
 */

import { Problem } from '../../../types';

export const intermediate_try_except_multipleProblem: Problem = {
  id: 'intermediate-try-except-multiple',
  title: 'Multiple Exception Types',
  difficulty: 'Easy',
  description: `Handle different exception types separately.

**Syntax:**
\`\`\`python
try:
    code()
except ValueError:
    handle_value_error()
except TypeError:
    handle_type_error()
\`\`\`

Or catch multiple:
\`\`\`python
except (ValueError, TypeError):
    handle_both()
\`\`\`

This tests:
- Exception handling
- Multiple except blocks
- Exception specificity`,
  examples: [
    {
      input: 'Try parsing user input',
      output: 'Different handling per error type',
    },
  ],
  constraints: [
    'Handle multiple exception types',
    'Specific handling per type',
  ],
  hints: [
    'Multiple except blocks',
    'Or tuple of exceptions',
    'More specific first',
  ],
  starterCode: `def safe_divide_and_convert(a, b):
    """
    Divide and convert to int, handling errors.
    
    Args:
        a, b: Values to divide
        
    Returns:
        Result or error message
        
    Examples:
        >>> safe_divide_and_convert(10, 2)
        5
        >>> safe_divide_and_convert(10, 0)
        'Error: Cannot divide by zero'
    """
    try:
        result = a / b
        return int(result)
    except ZeroDivisionError:
        return "Error: Cannot divide by zero"
    except (TypeError, ValueError):
        return "Error: Invalid input types"


# Test
print(safe_divide_and_convert(15, 3))
`,
  testCases: [
    {
      input: [15, 3],
      expected: 5,
    },
    {
      input: [10, 0],
      expected: 'Error: Cannot divide by zero',
    },
  ],
  solution: `def safe_divide_and_convert(a, b):
    try:
        result = a / b
        return int(result)
    except ZeroDivisionError:
        return "Error: Cannot divide by zero"
    except (TypeError, ValueError):
        return "Error: Invalid input types"`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 47,
  topic: 'Python Intermediate',
};
