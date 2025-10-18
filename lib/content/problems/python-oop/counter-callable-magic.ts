/**
 * Callable Counter
 * Problem ID: counter-callable-magic
 * Order: 17
 */

import { Problem } from '../../../types';

export const counter_callable_magicProblem: Problem = {
  id: 'counter-callable-magic',
  title: 'Callable Counter',
  difficulty: 'Easy',
  category: 'python-oop',
  description: `Create a \`Counter\` class that tracks how many times it has been called.

Make the instance callable using \`__call__\` magic method.

Implement:
- \`__init__()\`: Initialize count to 0
- \`__call__()\`: Increment count and return current count
- \`get_count()\`: Return current count
- \`reset()\`: Reset count to 0

**Examples:**
\`\`\`python
counter = Counter()
print(counter())        # 1
print(counter())        # 2
print(counter())        # 3
print(counter.get_count())  # 3
counter.reset()
print(counter())        # 1
\`\`\``,
  starterCode: `class Counter:
    def __init__(self):
        """Initialize counter."""
        pass
    
    def __call__(self):
        """Increment and return count."""
        pass
    
    def get_count(self):
        """Return current count."""
        pass
    
    def reset(self):
        """Reset count to 0."""
        pass`,
  testCases: [
    {
      input: [['Counter'], ['call'], ['call'], ['call']],
      expected: 3,
    },
    {
      input: [['Counter'], ['call'], ['get_count']],
      expected: 2,
    },
    {
      input: [['Counter'], ['call'], ['call'], ['reset'], ['call']],
      expected: 1,
    },
  ],
  hints: [
    '__call__ makes instances callable like functions',
    'Increment self.count in __call__',
    'Return the new count after incrementing',
  ],
  solution: `class Counter:
    def __init__(self):
        """Initialize counter."""
        self.count = 0
    
    def __call__(self):
        """Increment and return count."""
        self.count += 1
        return self.count
    
    def get_count(self):
        """Return current count."""
        return self.count
    
    def reset(self):
        """Reset count to 0."""
        self.count = 0


# Test
counter = Counter()
print(counter())        # 1
print(counter())        # 2
print(counter())        # 3
print(counter.get_count())  # 3
counter.reset()
print(counter())        # 1

# Can pass as function!
def apply_twice(func):
    func()
    func()

apply_twice(counter)  # counter is callable!`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 17,
  topic: 'Python Object-Oriented Programming',
};
