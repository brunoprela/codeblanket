/**
 * Multiple Context Managers
 * Problem ID: intermediate-with-multiple-contexts
 * Order: 31
 */

import { Problem } from '../../../types';

export const intermediate_with_multiple_contextsProblem: Problem = {
  id: 'intermediate-with-multiple-contexts',
  title: 'Multiple Context Managers',
  difficulty: 'Easy',
  description: `Use multiple context managers in a single with statement.

**Syntax:**
\`\`\`python
with open('file1') as f1, open('file2') as f2:
    # Both files open
\`\`\`

This tests:
- Multiple with items
- Context manager chaining
- Resource management`,
  examples: [
    {
      input: 'Open multiple files at once',
      output: 'All automatically closed',
    },
  ],
  constraints: ['Use single with statement', 'Multiple context managers'],
  hints: [
    'Separate with comma',
    'with cm1 as v1, cm2 as v2:',
    'All exit in reverse order',
  ],
  starterCode: `def test_multiple_contexts():
    """Test multiple context managers"""
    from io import StringIO
    
    # Create mock file objects
    file1 = StringIO("Hello")
    file2 = StringIO("World")
    
    # Use both in one with statement
    with file1 as f1, file2 as f2:
        content1 = f1.read()
        content2 = f2.read()
    
    return len(content1) + len(content2)
`,
  testCases: [
    {
      input: [],
      expected: 10,
      functionName: 'test_multiple_contexts',
    },
  ],
  solution: `def test_multiple_contexts():
    from io import StringIO
    
    file1 = StringIO("Hello")
    file2 = StringIO("World")
    
    with file1 as f1, file2 as f2:
        content1 = f1.read()
        content2 = f2.read()
    
    return len(content1) + len(content2)`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 31,
  topic: 'Python Intermediate',
};
