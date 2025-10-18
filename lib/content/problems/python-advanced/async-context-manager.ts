/**
 * Async Context Manager
 * Problem ID: async-context-manager
 * Order: 12
 */

import { Problem } from '../../../types';

export const async_context_managerProblem: Problem = {
  id: 'async-context-manager',
  title: 'Async Context Manager',
  difficulty: 'Hard',
  description: `Implement an async context manager for async resource management.

The context manager should:
- Implement __aenter__ and __aexit__
- Work with async with statement
- Handle async setup and cleanup
- Properly handle exceptions

**Use Case:** Async database connections, async file I/O.`,
  examples: [
    {
      input: 'Async resource access',
      output: 'Async setup and cleanup',
    },
  ],
  constraints: [
    'Must be async context manager',
    'Use async/await',
    'Handle exceptions properly',
  ],
  hints: [
    'Implement __aenter__ and __aexit__',
    'Both methods are async',
    'Use await for async operations',
  ],
  starterCode: `import asyncio

class AsyncResource:
    """
    Async context manager for resource.
    """
    
    async def __aenter__(self):
        # Your code here
        pass
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Your code here
        pass


async def main():
    async with AsyncResource() as resource:
        print("Using resource")

asyncio.run(main())
`,
  testCases: [
    {
      input: [true],
      expected: 'Success',
    },
  ],
  solution: `import asyncio

class AsyncResource:
    async def __aenter__(self):
        print("Acquiring resource")
        await asyncio.sleep(0.1)  # Simulate async setup
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Releasing resource")
        await asyncio.sleep(0.1)  # Simulate async cleanup
        return False`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 12,
  topic: 'Python Advanced',
};
