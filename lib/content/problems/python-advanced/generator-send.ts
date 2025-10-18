/**
 * Generator with Send - Running Average
 * Problem ID: generator-send
 * Order: 14
 */

import { Problem } from '../../../types';

export const generator_sendProblem: Problem = {
  id: 'generator-send',
  title: 'Generator with Send - Running Average',
  difficulty: 'Medium',
  description: `Implement a generator that calculates a running average using send().

The generator should:
- Accept values via send()
- Maintain running total and count
- Yield current average after each value
- Handle first next() call (prime)

**Use Case:** Real-time statistics, streaming data analysis.`,
  examples: [
    {
      input: 'send(10), send(20), send(30)',
      output: 'Yields 10.0, 15.0, 20.0',
    },
  ],
  constraints: [
    'Use generator with send()',
    'Calculate average correctly',
    'Handle initialization',
  ],
  hints: [
    'First yield returns None (for priming)',
    'Receive value with yield',
    'Update total and count',
  ],
  starterCode: `def running_average():
    """
    Generator that calculates running average.
    
    Yields:
        Current average
    """
    # Your code here
    pass


avg = running_average()
next(avg)  # Prime the generator

print(avg.send(10))  # 10.0
print(avg.send(20))  # 15.0
print(avg.send(30))  # 20.0


# Test helper function
def test_running_average(values):
    """Test running average with list of values"""
    avg = running_average()
    next(avg)  # Prime the generator
    results = []
    for val in values:
        results.append(avg.send(val))
    return results

result = test_running_average([10, 20, 30])
`,
  testCases: [
    {
      input: [],
      expected: [10.0, 15.0, 20.0],
    },
  ],
  solution: `def running_average():
    total = 0
    count = 0
    average = None
    
    while True:
        value = yield average
        if value is not None:
            total += value
            count += 1
            average = total / count


# Test helper function
def test_running_average(values):
    """Test running average with list of values"""
    avg = running_average()
    next(avg)  # Prime the generator
    results = []
    for val in values:
        results.append(avg.send(val))
    return results

result = test_running_average([10, 20, 30])`,
  timeComplexity: 'O(1) per value',
  spaceComplexity: 'O(1)',
  order: 14,
  topic: 'Python Advanced',
};
