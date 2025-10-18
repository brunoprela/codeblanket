/**
 * Teemo Attacking
 * Problem ID: fundamentals-teemo-attacking
 * Order: 100
 */

import { Problem } from '../../../types';

export const teemo_attackingProblem: Problem = {
  id: 'fundamentals-teemo-attacking',
  title: 'Teemo Attacking',
  difficulty: 'Easy',
  description: `Calculate total time target is poisoned.

Each attack poisons for duration seconds.
If attacked again before poison expires, timer resets (doesn't stack).

**Example:** timeSeries=[1,4], duration=2
→ 4 (poisoned: 1-2, 4-5)

This tests:
- Interval merging
- Time calculation
- Overlap handling`,
  examples: [
    {
      input: 'timeSeries = [1,4], duration = 2',
      output: '4',
    },
    {
      input: 'timeSeries = [1,2], duration = 2',
      output: '3',
      explanation: 'Overlap at time 2',
    },
  ],
  constraints: ['1 <= len(timeSeries) <= 10^4', '1 <= duration <= 10^9'],
  hints: [
    'Compare attack time with previous end time',
    'Add min(duration, gap) for each attack',
    'Last attack always adds full duration',
  ],
  starterCode: `def find_poisoned_duration(time_series, duration):
    """
    Calculate total poisoned time.
    
    Args:
        time_series: Attack times (sorted)
        duration: Poison duration
        
    Returns:
        Total poisoned time
        
    Examples:
        >>> find_poisoned_duration([1,4], 2)
        4
    """
    pass


# Test
print(find_poisoned_duration([1,2,3,4,5], 5))
`,
  testCases: [
    {
      input: [[1, 4], 2],
      expected: 4,
    },
    {
      input: [[1, 2], 2],
      expected: 3,
    },
    {
      input: [[1, 2, 3, 4, 5], 5],
      expected: 9,
    },
  ],
  solution: `def find_poisoned_duration(time_series, duration):
    if not time_series:
        return 0
    
    total = 0
    
    for i in range(len(time_series) - 1):
        # Add either full duration or time until next attack
        total += min(duration, time_series[i + 1] - time_series[i])
    
    # Last attack always adds full duration
    total += duration
    
    return total`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',
  order: 100,
  topic: 'Python Fundamentals',
};
