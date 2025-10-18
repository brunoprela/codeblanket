/**
 * advanced-async-generator
 * Order: 42
 */

import { Problem } from '../../../types';

export const async_generatorProblem: Problem = {
  id: 'advanced-async-generator',
  title: 'Async Generator',
  difficulty: 'Hard',
  description: `Create an async generator that yields values asynchronously.

Async generators combine:
- Generator protocol (yield)
- Async protocol (await)
- Used with async for

**Example:** 
\`\`\`python
async for item in async_range(5):
    print(item)
\`\`\`

This tests:
- Async/await syntax
- Generator protocol
- Async iteration`,
  examples: [
    {
      input: 'Async range generator',
      output: 'Yields 0,1,2,3,4 asynchronously',
    },
  ],
  constraints: ['Must be async generator', 'Use yield not return'],
  hints: ['Use async def', 'Use yield for values', 'Can use await inside'],
  starterCode: `import asyncio

async def async_range(n):
    """
    Async generator that yields numbers.
    
    Args:
        n: Upper limit
        
    Examples:
        >>> async for i in async_range(3):
        ...     print(i)
        0
        1
        2
    """
    # Your code here
    pass


async def test_async_gen():
    """Test async generator"""
    results = []
    async for i in async_range(5):
        results.append(i)
    return results[2]  # Return middle value


# For testing, we need to run async function
def test_runner():
    """Synchronous test runner"""
    try:
        return asyncio.run(test_async_gen())
    except:
        return None
`,
  testCases: [
    {
      input: [],
      expected: 2,
      functionName: 'test_runner',
    },
  ],
  solution: `import asyncio

async def async_range(n):
    for i in range(n):
        await asyncio.sleep(0)  # Yield control
        yield i


async def test_async_gen():
    results = []
    async for i in async_range(5):
        results.append(i)
    return results[2]


def test_runner():
    try:
        return asyncio.run(test_async_gen())
    except:
        return None`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  order: 42,
  topic: 'Python Advanced',
};
